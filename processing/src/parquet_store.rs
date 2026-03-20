//! Parquet-based detection store.
//!
//! Each processing container writes detections to **Parquet files** via
//! an in-memory DuckDB instance.  No cross-process file locking is
//! ever needed — each container owns its own in-memory DB and writes
//! to uniquely-named Parquet files under `/data/detections/`.
//!
//! The web server reads these Parquet files via DuckDB's `read_parquet`
//! for fast columnar analytics.
//!
//! ## Flush strategy
//!
//! Detections are buffered in an in-memory DuckDB table and flushed to
//! a Parquet file when:
//!
//! * The buffer reaches [`FLUSH_THRESHOLD`] rows, **or**
//! * [`flush()`] is called explicitly (e.g. at shutdown or at the end
//!   of each polling cycle).
//!
//! Each Parquet file is written atomically: first to a `.tmp` file,
//! then renamed to the final name.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::OnceLock;

use anyhow::{Context, Result};
use duckdb::params;
use tracing::{debug, info};

use gaia_common::detection::Detection;

// ─── Configuration ───────────────────────────────────────────────────────────

/// Number of buffered detections before an automatic flush.
const FLUSH_THRESHOLD: usize = 500;

// ─── Global state ────────────────────────────────────────────────────────────

/// The store is created once and shared across worker + reporting threads.
/// `duckdb::Connection` is `Send` so `Mutex<Store>` is `Send + Sync`.
static STORE: OnceLock<Mutex<Store>> = OnceLock::new();

struct Store {
    conn: duckdb::Connection,
    output_dir: PathBuf,
    instance: String,
    buffered: usize,
    /// Monotonically increasing sequence number within this process.
    /// Combined with epoch-millis to produce unique `id` values.
    seq: u64,
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Initialise the Parquet store.
///
/// * `output_dir` — directory where `.parquet` files are written
///   (e.g. `/data/detections`).
/// * `instance` — slug identifying this container (used in filenames).
///
/// Must be called once before any [`write_detection`] or [`flush`] calls.
pub fn initialize(output_dir: &Path, instance: &str) -> Result<()> {
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("Cannot create detections dir: {}", output_dir.display()))?;

    let conn = duckdb::Connection::open_in_memory()
        .context("Cannot open in-memory DuckDB")?;

    conn.execute_batch(
        "CREATE TABLE buffer (
            id          BIGINT   NOT NULL,
            Date        VARCHAR  NOT NULL,
            Time        VARCHAR  NOT NULL,
            Domain      VARCHAR  NOT NULL,
            Sci_Name    VARCHAR  NOT NULL,
            Com_Name    VARCHAR  NOT NULL,
            Confidence  DOUBLE   NOT NULL,
            Lat         DOUBLE   NOT NULL,
            Lon         DOUBLE   NOT NULL,
            Cutoff      DOUBLE   NOT NULL,
            Week        INTEGER  NOT NULL,
            Sens        DOUBLE   NOT NULL,
            Overlap     DOUBLE   NOT NULL,
            File_Name   VARCHAR  NOT NULL,
            Source_Node VARCHAR  NOT NULL,
            Excluded    INTEGER  NOT NULL,
            Model_Slug  VARCHAR  NOT NULL,
            Model_Name  VARCHAR  NOT NULL
        );",
    )
    .context("Cannot create DuckDB buffer table")?;

    let _ = STORE.set(Mutex::new(Store {
        conn,
        output_dir: output_dir.to_path_buf(),
        instance: instance.to_string(),
        buffered: 0,
        seq: 0,
    }));

    info!(
        "Parquet store initialised → {}  (instance={instance:?})",
        output_dir.display()
    );
    Ok(())
}

/// Buffer a single detection.
///
/// The detection is inserted into the in-memory DuckDB table.  When the
/// buffer reaches [`FLUSH_THRESHOLD`] rows it is automatically flushed
/// to a Parquet file.
pub fn write_detection(
    d: &Detection,
    lat: f64,
    lon: f64,
    cutoff: f64,
    sensitivity: f64,
    overlap: f64,
    file_name: &str,
    source_node: &str,
) -> Result<()> {
    let store = STORE.get().context("Parquet store not initialised")?;
    let mut s = store
        .lock()
        .map_err(|e| anyhow::anyhow!("Parquet store lock poisoned: {e}"))?;

    // Generate a unique, sortable ID: epoch-millis shifted left 16 bits
    // + intra-process sequence counter.  This gives ~65 K IDs per
    // millisecond and sorts chronologically.
    let epoch_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    s.seq += 1;
    let id = ((epoch_ms & 0xFFFF_FFFF_FFFF) << 16) | (s.seq & 0xFFFF);

    s.conn.execute(
        "INSERT INTO buffer VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        params![
            id as i64,
            d.date,
            d.time,
            d.domain,
            d.scientific_name,
            d.common_name,
            d.confidence,
            lat,
            lon,
            cutoff,
            d.week as i32,
            sensitivity,
            overlap,
            file_name,
            source_node,
            d.excluded as i32,
            d.model_slug,
            d.model_name,
        ],
    )
    .context("Failed to buffer detection in DuckDB")?;

    s.buffered += 1;
    if s.buffered >= FLUSH_THRESHOLD {
        flush_locked(&mut s)?;
    }
    Ok(())
}

/// Flush any buffered detections to a Parquet file.
///
/// Safe to call even when the buffer is empty (no-op).
pub fn flush() -> Result<()> {
    let store = STORE.get().context("Parquet store not initialised")?;
    let mut s = store
        .lock()
        .map_err(|e| anyhow::anyhow!("Parquet store lock poisoned: {e}"))?;
    flush_locked(&mut s)
}

/// Return how many detections are currently buffered (for diagnostics).
pub fn buffered_count() -> usize {
    STORE
        .get()
        .and_then(|m| m.lock().ok())
        .map(|s| s.buffered)
        .unwrap_or(0)
}

// ─── Internals ───────────────────────────────────────────────────────────────

/// Flush the in-memory buffer to a Parquet file (caller holds the lock).
fn flush_locked(s: &mut Store) -> Result<()> {
    if s.buffered == 0 {
        return Ok(());
    }

    let ts = chrono::Utc::now().format("%Y%m%d-%H%M%S%.3f");
    let filename = format!("{}-{ts}.parquet", s.instance);
    let final_path = s.output_dir.join(&filename);
    let tmp_path = s.output_dir.join(format!(".{filename}.tmp"));

    s.conn
        .execute(
            &format!(
                "COPY buffer TO '{}' (FORMAT PARQUET, COMPRESSION ZSTD)",
                tmp_path.display()
            ),
            [],
        )
        .with_context(|| format!("Failed to write Parquet: {}", tmp_path.display()))?;

    std::fs::rename(&tmp_path, &final_path).with_context(|| {
        format!(
            "Failed to rename {} → {}",
            tmp_path.display(),
            final_path.display()
        )
    })?;

    debug!("Flushed {} detections → {filename}", s.buffered);
    s.conn
        .execute_batch("DELETE FROM buffer")
        .context("Failed to clear DuckDB buffer")?;
    s.buffered = 0;
    Ok(())
}
