//! Database layer (libsql / Turso).
//!
//! All database access goes through the libsql crate.  A single
//! `Database` and a single `Connection` are cached for the lifetime
//! of the process.  Because libsql connections are `Send + Sync`,
//! the connection can be shared across threads without a Mutex —
//! libsql serialises writes internally and, with WAL mode, supports
//! concurrent readers.
//!
//! WAL mode and `synchronous = NORMAL` are set **once** during
//! [`initialize()`].  Regular connection access (`conn()`) never
//! touches write-lock-requiring PRAGMAs, eliminating the
//! "database is locked" errors that occurred when every operation
//! tried to re-set `journal_mode=WAL`.
//!
//! Write-heavy paths use plain **autocommit** — each individual
//! `INSERT`/`UPDATE` is its own implicit transaction.  Cross-process
//! contention (multiple containers accessing the same WAL file) is
//! handled by `PRAGMA busy_timeout` (30 s) **plus** application-level
//! retry with exponential backoff (up to 5 attempts) for the hot
//! write paths.

use std::path::Path;
use std::sync::OnceLock;
use std::time::Duration;

use anyhow::{Context, Result};
use libsql::params;
use tracing::{debug, info, warn};

use gaia_common::detection::Detection;

// ── Async runtime for sync callers ───────────────────────────────────

/// Single-threaded tokio runtime used exclusively for database I/O.
/// Created lazily on first use.
static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();

fn rt() -> &'static tokio::runtime::Runtime {
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("Cannot create tokio runtime for database")
    })
}

// ── Cached connection ────────────────────────────────────────────────────

/// Busy-timeout in milliseconds.  Applied once at connection creation.
const BUSY_TIMEOUT_MS: u32 = 30_000;

/// Maximum application-level retries for SQLITE_BUSY / locked errors.
const MAX_BUSY_RETRIES: u32 = 5;

/// Check whether a libsql error is a transient "database is locked" /
/// SQLITE_BUSY error that is worth retrying.
fn is_busy_error(e: &libsql::Error) -> bool {
    let msg = e.to_string();
    msg.contains("database is locked") || msg.contains("database table is locked")
}

/// Cached `Database` handle — opened once, kept alive for the whole process.
static DB: OnceLock<libsql::Database> = OnceLock::new();

/// Cached `Connection` — created once from the cached `Database`.
/// libsql connections are `Send + Sync`; concurrent operations are
/// serialised internally by libsql.
static CONN: OnceLock<libsql::Connection> = OnceLock::new();

/// Resolve the effective database URL.
///
/// `TURSO_DATABASE_URL` takes precedence over the `db_path` argument.
fn effective_db_path(db_path: &Path) -> String {
    std::env::var("TURSO_DATABASE_URL")
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| {
            db_path
                .to_str()
                .expect("Non-UTF-8 database path")
                .to_string()
        })
}

/// Return the cached `Database`, opening it on first call.
async fn get_or_open_db(db_path: &Path) -> Result<&'static libsql::Database> {
    if let Some(db) = DB.get() {
        return Ok(db);
    }
    let url = effective_db_path(db_path);
    info!("Opening database: {url}");
    let new_db = libsql::Builder::new_local(&url)
        .build()
        .await
        .with_context(|| format!("Cannot open database: {url}"))?;
    let _ = DB.set(new_db);
    Ok(DB.get().expect("DB was just set"))
}

/// Return the cached `Connection`, creating it on first call.
///
/// Only sets `busy_timeout` — WAL and synchronous are configured
/// once in [`initialize()`].
async fn get_or_open_conn(db_path: &Path) -> Result<&'static libsql::Connection> {
    if let Some(c) = CONN.get() {
        return Ok(c);
    }
    let db = get_or_open_db(db_path).await?;
    let c = db.connect().context("Cannot connect to database")?;
    c.execute_batch(&format!("PRAGMA busy_timeout={BUSY_TIMEOUT_MS};"))
        .await
        .context("Failed to set busy_timeout")?;
    let _ = CONN.set(c);
    Ok(CONN.get().expect("CONN was just set"))
}

/// Get the cached connection (sync wrapper).  Panics if called before
/// [`initialize()`].
fn conn(db_path: &Path) -> Result<&'static libsql::Connection> {
    if let Some(c) = CONN.get() {
        return Ok(c);
    }
    rt().block_on(get_or_open_conn(db_path))
}

/// Get the cached connection, returning `None` on failure.
fn conn_opt(db_path: &Path) -> Option<&'static libsql::Connection> {
    conn(db_path).ok()
}

/// Public access to the cached connection for other crates (e.g.
/// `compress.rs`).
pub fn open_conn_pub(db_path: &Path) -> Result<libsql::Connection> {
    // Return a *new* connection from the same Database for callers
    // that need their own (e.g. long-running compression loops).
    let db = rt().block_on(get_or_open_db(db_path))?;
    let c = db.connect().context("Cannot create connection")?;
    rt().block_on(async {
        c.execute_batch(&format!("PRAGMA busy_timeout={BUSY_TIMEOUT_MS};"))
            .await
            .context("Failed to set busy_timeout")
    })?;
    Ok(c)
}

/// Execute an async operation using the module-level runtime.
/// Public for use by other crates (e.g. compress.rs).
pub fn block_on<F: std::future::Future>(f: F) -> F::Output {
    rt().block_on(f)
}

/// Labels that the BirdNET model emits which are not actual bird species.
/// These are counted in the `urban_noise` table instead of `detections`.
pub const URBAN_NOISE_LABELS: &[&str] = &[
    "Engine",
    "Dog",
    "Human",
    "Human vocal",
    "Human whistle",
    "Human_Human",
    "Power tools",
    "Siren",
    "Gun",
    "Fireworks",
    "Noise",
    "Environmental",
];

/// Returns `true` if the given scientific name is a known non-bird /
/// urban-noise label.
pub fn is_urban_noise(sci_name: &str) -> bool {
    URBAN_NOISE_LABELS
        .iter()
        .any(|&label| sci_name.eq_ignore_ascii_case(label))
}

/// Create the `detections` table (and indices) if it doesn't exist.
///
/// **Must** be called once at startup. Sets WAL mode and
/// `synchronous=NORMAL` — no other code path needs to touch those
/// PRAGMAs.
pub fn initialize(db_path: &Path) -> Result<()> {
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let conn = rt().block_on(get_or_open_conn(db_path))?;

    rt().block_on(async {
        // Set WAL + synchronous once for the whole process lifetime.
        conn.execute_batch(
            "PRAGMA journal_mode=WAL; \
             PRAGMA synchronous=NORMAL;"
        )
        .await
        .context("Failed to set WAL mode")?;

        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS detections (
                Date       DATE,
                Time       TIME,
                Domain     VARCHAR(50) NOT NULL DEFAULT 'birds',
                Sci_Name   VARCHAR(100) NOT NULL,
                Com_Name   VARCHAR(100) NOT NULL,
                Confidence FLOAT,
                Lat        FLOAT,
                Lon        FLOAT,
                Cutoff     FLOAT,
                Week       INT,
                Sens       FLOAT,
                Overlap    FLOAT,
                File_Name  VARCHAR(100) NOT NULL,
                Source_Node VARCHAR(200) NOT NULL DEFAULT '',
                Excluded   INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS detections_Com_Name    ON detections (Com_Name);
            CREATE INDEX IF NOT EXISTS detections_Sci_Name    ON detections (Sci_Name);
            CREATE INDEX IF NOT EXISTS detections_Domain      ON detections (Domain);
            CREATE INDEX IF NOT EXISTS detections_Date_Time   ON detections (Date DESC, Time DESC);

            CREATE TABLE IF NOT EXISTS urban_noise (
                Date       DATE    NOT NULL,
                Hour       INT     NOT NULL,
                Category   VARCHAR(50) NOT NULL,
                Count      INT     NOT NULL DEFAULT 1,
                UNIQUE(Date, Hour, Category)
            );
            CREATE INDEX IF NOT EXISTS urban_noise_date ON urban_noise (Date DESC);

            CREATE TABLE IF NOT EXISTS settings (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS exclusion_overrides (
                Sci_Name      VARCHAR(100) PRIMARY KEY,
                overridden_at TEXT NOT NULL DEFAULT (datetime('now')),
                notes         TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS file_processing_log (
                filename   TEXT    NOT NULL,
                instance   TEXT    NOT NULL,
                processed_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (filename, instance)
            );

            CREATE TABLE IF NOT EXISTS processing_instances (
                instance       TEXT PRIMARY KEY,
                registered_at  TEXT NOT NULL DEFAULT (datetime('now')),
                last_heartbeat TEXT NOT NULL DEFAULT (datetime('now'))
            );
        ",
        )
        .await
        .context("Failed to create tables")?;

        // Migration: add Source_Node to existing databases that lack it.
        migrate_add_column(&conn, "detections", "Source_Node", "VARCHAR(200) NOT NULL DEFAULT ''").await;
        // Migration: add Excluded column to existing databases.
        migrate_add_column(&conn, "detections", "Excluded", "INTEGER NOT NULL DEFAULT 0").await;
        // Migration: add last_heartbeat to existing processing_instances tables.
        let added = conn.execute_batch(
            "ALTER TABLE processing_instances ADD COLUMN last_heartbeat TEXT NOT NULL DEFAULT '';"
        ).await;
        if added.is_ok() {
            conn.execute_batch(
                "UPDATE processing_instances SET last_heartbeat = registered_at WHERE last_heartbeat = '';"
            ).await.ok();
        }
        // Migration: add Model_Slug and Model_Name columns.
        migrate_add_column(&conn, "detections", "Model_Slug", "VARCHAR(100) NOT NULL DEFAULT ''").await;
        migrate_add_column(&conn, "detections", "Model_Name", "VARCHAR(200) NOT NULL DEFAULT ''").await;

        Ok::<(), anyhow::Error>(())
    })?;

    info!("Database schema verified");
    Ok(())
}

/// Idempotent column addition — ignores "duplicate column" errors.
async fn migrate_add_column(conn: &libsql::Connection, table: &str, column: &str, typedef: &str) {
    let _ = conn
        .execute_batch(&format!("ALTER TABLE {table} ADD COLUMN {column} {typedef};"))
        .await;
}

/// Insert a single detection row.
///
/// Runs as a single autocommit statement — SQLite handles the
/// implicit transaction.  Cross-process contention is absorbed by
/// `PRAGMA busy_timeout` (30 s) **plus** application-level retry
/// with exponential backoff.
pub fn insert_detection(
    db_path: &Path,
    d: &Detection,
    lat: f64,
    lon: f64,
    cutoff: f64,
    sensitivity: f64,
    overlap: f64,
    file_name: &str,
    source_node: &str,
) -> Result<()> {
    let conn = conn(db_path)?;
    for attempt in 0..=MAX_BUSY_RETRIES {
        let res: Result<(), libsql::Error> = rt().block_on(async {
            conn.execute(
                "INSERT INTO detections (Date, Time, Domain, Sci_Name, Com_Name, Confidence, \
                 Lat, Lon, Cutoff, Week, Sens, Overlap, File_Name, Source_Node, Excluded, \
                 Model_Slug, Model_Name) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)",
                params![
                    d.date.clone(),
                    d.time.clone(),
                    d.domain.clone(),
                    d.scientific_name.clone(),
                    d.common_name.clone(),
                    d.confidence,
                    lat,
                    lon,
                    cutoff,
                    d.week as i64,
                    sensitivity,
                    overlap,
                    file_name.to_string(),
                    source_node.to_string(),
                    d.excluded as i64,
                    d.model_slug.clone(),
                    d.model_name.clone(),
                ],
            )
            .await?;
            Ok(())
        });
        match res {
            Ok(()) => return Ok(()),
            Err(e) if is_busy_error(&e) && attempt < MAX_BUSY_RETRIES => {
                let delay = Duration::from_millis(100 * 2u64.pow(attempt));
                warn!("insert_detection: busy (attempt {}/{}), retrying in {delay:?}",
                      attempt + 1, MAX_BUSY_RETRIES);
                std::thread::sleep(delay);
            }
            Err(e) => return Err(anyhow::Error::from(e).context("Failed to insert detection")),
        }
    }
    unreachable!()
}

/// Count today's detections of a species in a given domain.
#[allow(dead_code)]
pub fn todays_count_for(db_path: &Path, domain: &str, sci_name: &str) -> u32 {
    let today = chrono::Local::now().format("%Y-%m-%d").to_string();
    _count(
        db_path,
        "SELECT COUNT(*) FROM detections WHERE Date = DATE(?1) \
         AND Domain = ?2 AND Sci_Name = ?3",
        vec![
            libsql::Value::from(today),
            libsql::Value::from(domain.to_string()),
            libsql::Value::from(sci_name.to_string()),
        ],
    )
}

/// Count this week's detections of a species in a given domain.
#[allow(dead_code)]
pub fn weeks_count_for(db_path: &Path, domain: &str, sci_name: &str) -> u32 {
    let today = chrono::Local::now().format("%Y-%m-%d").to_string();
    _count(
        db_path,
        "SELECT COUNT(*) FROM detections WHERE Date >= DATE(?1, '-7 day') \
         AND Domain = ?2 AND Sci_Name = ?3",
        vec![
            libsql::Value::from(today),
            libsql::Value::from(domain.to_string()),
            libsql::Value::from(sci_name.to_string()),
        ],
    )
}

fn _count(db_path: &Path, sql: &str, p: Vec<libsql::Value>) -> u32 {
    let conn = match conn_opt(db_path) {
        Some(c) => c,
        None => return 0,
    };
    rt().block_on(async {
        let mut rows = match conn.query(sql, p).await {
            Ok(r) => r,
            Err(_) => return 0,
        };
        match rows.next().await {
            Ok(Some(row)) => row.get::<u32>(0).unwrap_or(0),
            _ => 0,
        }
    })
}

/// Increment the urban-noise counter for a category / date / hour.
///
/// Uses `INSERT … ON CONFLICT` (UPSERT) so a single row is stored
/// per (Date, Hour, Category) tuple.  Each statement runs in
/// autocommit — no explicit transaction needed.  Retries on busy.
pub fn increment_urban_noise(db_path: &Path, date: &str, hour: u32, category: &str) -> Result<()> {
    let conn = conn(db_path)?;
    for attempt in 0..=MAX_BUSY_RETRIES {
        let res: Result<(), libsql::Error> = rt().block_on(async {
            let result = conn.execute(
                "INSERT INTO urban_noise (Date, Hour, Category, Count) VALUES (?1, ?2, ?3, 1) \
                 ON CONFLICT(Date, Hour, Category) DO UPDATE SET Count = Count + 1",
                params![date.to_string(), hour as i64, category.to_string()],
            ).await;

            if result.is_err() {
                // Fallback for older schemas without the UNIQUE constraint.
                let updated = conn.execute(
                    "UPDATE urban_noise SET Count = Count + 1 WHERE Date = ?1 AND Hour = ?2 AND Category = ?3",
                    params![date.to_string(), hour as i64, category.to_string()],
                ).await?;
                if updated == 0 {
                    conn.execute(
                        "INSERT INTO urban_noise (Date, Hour, Category, Count) VALUES (?1, ?2, ?3, 1)",
                        params![date.to_string(), hour as i64, category.to_string()],
                    ).await?;
                }
            }

            Ok(())
        });
        match res {
            Ok(()) => return Ok(()),
            Err(e) if is_busy_error(&e) && attempt < MAX_BUSY_RETRIES => {
                let delay = Duration::from_millis(100 * 2u64.pow(attempt));
                warn!("increment_urban_noise: busy (attempt {}/{}), retrying in {delay:?}",
                      attempt + 1, MAX_BUSY_RETRIES);
                std::thread::sleep(delay);
            }
            Err(e) => return Err(anyhow::Error::from(e).context("increment_urban_noise failed")),
        }
    }
    unreachable!()
}

// ─── Settings ────────────────────────────────────────────────────────────────

/// Read a single setting from the `settings` table.
/// Returns `None` if the table or key doesn't exist.
pub fn get_setting(db_path: &Path, key: &str) -> Option<String> {
    let conn = conn_opt(db_path)?;
    rt().block_on(async {
        let mut rows = conn
            .query("SELECT value FROM settings WHERE key = ?1", params![key.to_string()])
            .await
            .ok()?;
        let row = rows.next().await.ok()??;
        row.get::<String>(0).ok()
    })
}

/// Read a setting as `f64`, returning `None` on missing / parse error.
pub fn get_setting_f64(db_path: &Path, key: &str) -> Option<f64> {
    get_setting(db_path, key).and_then(|v| v.parse().ok())
}

/// Refresh a `Config` with any overrides stored in the settings table.
///
/// This is called on each poll cycle so that changes made via the web UI
/// take effect without restarting the processing container.
pub fn apply_settings_overrides(config: &mut gaia_common::config::Config) {
    let db = &config.db_path.clone();
    if let Some(v) = get_setting_f64(db, "sensitivity") {
        config.sensitivity = v;
    }
    if let Some(v) = get_setting_f64(db, "confidence") {
        config.confidence = v;
    }
    if let Some(v) = get_setting_f64(db, "sf_thresh") {
        config.sf_thresh = v;
    }
    if let Some(v) = get_setting_f64(db, "overlap") {
        config.overlap = v;
    }
    if let Some(v) = get_setting(db, "colormap") {
        config.colormap = v;
    }
}

/// Load all scientific names from the `exclusion_overrides` table.
///
/// These species have been manually confirmed by an ornithologist and
/// should bypass the species-range occurrence-threshold filter.
pub fn load_exclusion_overrides(db_path: &Path) -> Vec<String> {
    let conn = match conn(db_path) {
        Ok(c) => c,
        Err(_) => return vec![],
    };
    rt().block_on(async {
        let mut rows = match conn.query("SELECT Sci_Name FROM exclusion_overrides", ()).await {
            Ok(r) => r,
            Err(_) => return vec![],
        };
        let mut out = Vec::new();
        while let Ok(Some(row)) = rows.next().await {
            if let Ok(name) = row.get::<String>(0) {
                out.push(name);
            }
        }
        out
    })
}

// ─── File processing coordination ────────────────────────────────────────────

/// Register a processing instance so other instances know about it.
/// Called once at startup.
pub fn register_instance(db_path: &Path, instance: &str) -> Result<()> {
    let conn = conn(db_path)?;
    rt().block_on(async {
        conn.execute(
            "INSERT OR REPLACE INTO processing_instances (instance, registered_at, last_heartbeat) \
             VALUES (?1, datetime('now'), datetime('now'))",
            params![instance.to_string()],
        )
        .await?;
        let mut rows = conn
            .query("SELECT COUNT(*) FROM processing_instances", ())
            .await?;
        let total: u32 = rows
            .next()
            .await?
            .and_then(|r| r.get::<u32>(0).ok())
            .unwrap_or(1);
        debug!(
            "Registered processing instance {:?} ({total} instance(s) total)",
            instance
        );
        Ok::<(), libsql::Error>(())
    })?;
    Ok(())
}

/// Update the heartbeat timestamp for a running instance.
/// Should be called every poll cycle so `all_instances_done` can
/// distinguish live instances from stale ones.
pub fn update_heartbeat(db_path: &Path, instance: &str) {
    let conn = match conn(db_path) {
        Ok(c) => c,
        Err(_) => return,
    };
    rt().block_on(async {
        conn.execute(
            "UPDATE processing_instances SET last_heartbeat = datetime('now') WHERE instance = ?1",
            params![instance.to_string()],
        )
        .await
        .ok();
    });
}

/// Remove processing instances whose heartbeat is older than the given
/// threshold (in minutes).  Returns the number of pruned rows.
pub fn prune_stale_instances(db_path: &Path, stale_minutes: u32) -> usize {
    let conn = match conn(db_path) {
        Ok(c) => c,
        Err(_) => return 0,
    };
    let removed = rt().block_on(async {
        conn.execute(
            &format!(
                "DELETE FROM processing_instances \
                 WHERE last_heartbeat < datetime('now', '-{stale_minutes} minutes')"
            ),
            (),
        )
        .await
        .unwrap_or(0)
    });
    if removed > 0 {
        info!(
            "Pruned {removed} stale processing instance(s) (no heartbeat in >{stale_minutes} min)"
        );
    }
    removed as usize
}

/// Check whether this instance has already processed a specific file.
///
/// Used at startup to avoid re-downloading files that were already
/// processed in a previous run (the in-memory `dispatched` set is
/// lost on restart, but the DB log survives).
pub fn is_file_processed(db_path: &Path, filename: &str, instance: &str) -> bool {
    let conn = match conn(db_path) {
        Ok(c) => c,
        Err(_) => return false,
    };
    rt().block_on(async {
        let mut rows = match conn
            .query(
                "SELECT COUNT(*) FROM file_processing_log WHERE filename = ?1 AND instance = ?2",
                params![filename.to_string(), instance.to_string()],
            )
            .await
        {
            Ok(r) => r,
            Err(_) => return false,
        };
        match rows.next().await {
            Ok(Some(row)) => row.get::<u32>(0).unwrap_or(0) > 0,
            _ => false,
        }
    })
}

/// Record that this instance has finished processing a file.
///
/// Runs as a single autocommit `INSERT OR IGNORE`.  Cross-process
/// contention is absorbed by `PRAGMA busy_timeout` (30 s) **plus**
/// application-level retry with exponential backoff.
pub fn mark_file_processed(db_path: &Path, filename: &str, instance: &str) -> Result<()> {
    let conn = conn(db_path)?;

    for attempt in 0..=MAX_BUSY_RETRIES {
        let res: Result<(), libsql::Error> = rt().block_on(async {
            conn.execute(
                "INSERT OR IGNORE INTO file_processing_log (filename, instance) VALUES (?1, ?2)",
                params![filename.to_string(), instance.to_string()],
            )
            .await?;

            let mut rows = conn
                .query(
                    "SELECT COUNT(*) FROM file_processing_log WHERE filename = ?1",
                    params![filename.to_string()],
                )
                .await?;
            let done: u32 = rows
                .next()
                .await?
                .and_then(|r| r.get::<u32>(0).ok())
                .unwrap_or(0);

            let mut rows2 = conn
                .query("SELECT COUNT(*) FROM processing_instances", ())
                .await?;
            let total: u32 = rows2
                .next()
                .await?
                .and_then(|r| r.get::<u32>(0).ok())
                .unwrap_or(0);

            debug!(
                "mark_file_processed: {filename} by {:?} ({done}/{total} instances done)",
                instance
            );
            Ok(())
        });
        match res {
            Ok(()) => return Ok(()),
            Err(e) if is_busy_error(&e) && attempt < MAX_BUSY_RETRIES => {
                let delay = Duration::from_millis(100 * 2u64.pow(attempt));
                warn!("mark_file_processed: busy (attempt {}/{}), retrying in {delay:?}",
                      attempt + 1, MAX_BUSY_RETRIES);
                std::thread::sleep(delay);
            }
            Err(e) => return Err(anyhow::Error::from(e).context("mark_file_processed failed")),
        }
    }
    unreachable!()
}

/// Check whether every **active** processing instance has processed this file.
///
/// Only instances with a heartbeat within the last 5 minutes are
/// considered.  This prevents stale / stopped containers from blocking
/// deletion forever.
pub fn all_instances_done(db_path: &Path, filename: &str) -> bool {
    let conn = match conn(db_path) {
        Ok(c) => c,
        Err(_) => return true, // fail-open: delete if we can't check
    };

    rt().block_on(async {
        // Only count instances that are still alive (heartbeat within last
        // 5 minutes).  Stale entries from previous runs are ignored.
        let remaining: u32 = {
            let mut rows = match conn
                .query(
                    "SELECT COUNT(*) FROM processing_instances \
                     WHERE last_heartbeat >= datetime('now', '-5 minutes') \
                       AND instance NOT IN \
                         (SELECT instance FROM file_processing_log WHERE filename = ?1)",
                    params![filename.to_string()],
                )
                .await
            {
                Ok(r) => r,
                Err(_) => return true,
            };
            match rows.next().await {
                Ok(Some(row)) => row.get::<u32>(0).unwrap_or(0),
                _ => 0,
            }
        };

        let active: u32 = {
            let mut rows = match conn
                .query(
                    "SELECT COUNT(*) FROM processing_instances \
                     WHERE last_heartbeat >= datetime('now', '-5 minutes')",
                    (),
                )
                .await
            {
                Ok(r) => r,
                Err(_) => return true,
            };
            match rows.next().await {
                Ok(Some(row)) => row.get::<u32>(0).unwrap_or(0),
                _ => 0,
            }
        };

        let done = remaining == 0;
        debug!(
            "all_instances_done({filename}): {remaining}/{active} active instance(s) still pending → {}",
            if done { "ready to delete" } else { "waiting" }
        );
        done
    })
}

/// Remove old entries from the processing log (files older than 1 hour).
pub fn cleanup_processing_log(db_path: &Path) {
    let conn = match conn(db_path) {
        Ok(c) => c,
        Err(_) => return,
    };
    let removed = rt().block_on(async {
        conn.execute(
            "DELETE FROM file_processing_log WHERE processed_at < datetime('now', '-1 hour')",
            (),
        )
        .await
        .unwrap_or(0)
    });
    if removed > 0 {
        debug!("cleanup_processing_log: purged {removed} stale entries");
    }
}

/// One-time migration: backfill the `Domain` column with per-species
/// taxonomic class values parsed from the labels CSV.
///
/// For models whose labels CSV has a `class` column (e.g. BirdNET+ V3.0),
/// this updates existing rows where `Domain` still holds the coarse
/// model-wide value (e.g. `"birds"`) to use the correct per-species
/// class (e.g. `"Aves"`, `"Mammalia"`, `"Insecta"`, …).
///
/// The update is idempotent: rows that already have the correct class
/// are not touched, and model_slug scoping prevents cross-model clashes.
pub fn migrate_domain_classes(
    db_path: &Path,
    model_slug: &str,
    old_domain: &str,
    class_map: &std::collections::HashMap<String, String>,
) {
    if class_map.is_empty() {
        return;
    }

    let conn = match conn(db_path) {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!("migrate_domain_classes: cannot open DB: {e}");
            return;
        }
    };

    let mut updated: usize = 0;
    for (sci_name, class) in class_map {
        if class == old_domain {
            continue; // nothing to change
        }
        let n = rt().block_on(async {
            conn.execute(
                "UPDATE detections SET Domain = ?1 \
                 WHERE Sci_Name = ?2 AND Domain = ?3 AND Model_Slug = ?4",
                params![
                    class.clone(),
                    sci_name.clone(),
                    old_domain.to_string(),
                    model_slug.to_string()
                ],
            )
            .await
            .unwrap_or(0)
        });
        updated += n as usize;
    }
    if updated > 0 {
        info!(
            "migrate_domain_classes: backfilled {updated} detection(s) \
             for model {model_slug} ('{old_domain}' → per-species class)",
        );
    }
}
