//! One-time migration: normalise scientific names in existing Parquet files.
//!
//! Reads all `*.parquet` files from the detections directory, normalises
//! `Sci_Name` (underscores → spaces, proper capitalisation) and picks
//! the best `Com_Name` per species (preferring a real common name over
//! a copy of the scientific name).  Writes the corrected data back as
//! new Parquet files and removes the originals.
//!
//! A marker file (`.sci_name_normalised`) is written after a successful
//! migration so the process is idempotent — it runs at most once.

use std::path::Path;

use anyhow::{Context, Result};
use tracing::info;

/// Marker file written after a successful migration.
const MARKER: &str = ".sci_name_normalised";

/// Run the Parquet normalisation migration if it hasn't been done yet.
///
/// Call this at startup in both the processing and web containers.
/// The detections directory is the folder containing `*.parquet` files
/// (e.g. `/data/detections`).
pub fn run_if_needed(detections_dir: &Path) -> Result<()> {
    let marker = detections_dir.join(MARKER);
    if marker.exists() {
        return Ok(()); // already migrated
    }

    // Check if there are any Parquet files to migrate.
    let has_files = std::fs::read_dir(detections_dir)
        .map(|rd| {
            rd.flatten()
                .any(|e| e.path().extension().map(|x| x == "parquet").unwrap_or(false))
        })
        .unwrap_or(false);

    if !has_files {
        // Nothing to migrate — write marker and return.
        std::fs::write(&marker, "no files to migrate\n").ok();
        return Ok(());
    }

    info!(
        "Running Sci_Name normalisation migration on {}",
        detections_dir.display()
    );

    let t0 = std::time::Instant::now();
    let glob = format!("{}/*.parquet", detections_dir.display());

    // Open a temporary DuckDB instance for the migration.
    let conn = duckdb::Connection::open_in_memory()
        .context("Cannot open migration DuckDB")?;

    // Read all existing Parquet files into a temporary table.
    conn.execute_batch(&format!(
        "CREATE TABLE migration AS SELECT * FROM read_parquet('{glob}', union_by_name=true)"
    ))
    .context("Cannot read existing Parquet files for migration")?;

    let count_before: u64 = conn
        .query_row("SELECT COUNT(*) FROM migration", [], |row| row.get(0))
        .unwrap_or(0);

    if count_before == 0 {
        info!("Migration: no rows found, skipping");
        std::fs::write(&marker, "0 rows\n").ok();
        return Ok(());
    }

    // ── Build the best common-name lookup ────────────────────────────
    //
    // For each normalised Sci_Name, find the best Com_Name:
    // prefer a real common name (Com_Name != Sci_Name) over a fallback
    // that just copies the scientific name.
    //
    // We also normalise Sci_Name in-place: replace underscores with
    // spaces, fix capitalisation (only genus capitalised).

    // Step 1: Add a normalised column.
    // DuckDB doesn't have a built-in "title case genus only" function,
    // so we use: UPPER(LEFT(word, 1)) || LOWER(SUBSTR(word, 2)) for
    // the genus, and LOWER for the rest.  Since Sci_Name may contain
    // underscores, we first replace them with spaces.
    conn.execute_batch(
        "ALTER TABLE migration ADD COLUMN Sci_Norm VARCHAR; \
         UPDATE migration SET Sci_Norm = \
             CONCAT( \
                 UPPER(LEFT(TRIM(REPLACE(Sci_Name, '_', ' ')), 1)), \
                 LOWER(SUBSTR(TRIM(REPLACE(Sci_Name, '_', ' ')), 2)) \
             );"
    ).context("Cannot normalise Sci_Name")?;

    // Step 2: Build a lookup of best common name per normalised Sci_Name.
    conn.execute_batch(
        "CREATE TABLE best_names AS \
         SELECT Sci_Norm, \
             COALESCE( \
                 MAX(CASE WHEN TRIM(Com_Name) != '' \
                          AND TRIM(Com_Name) != Sci_Name \
                          AND TRIM(Com_Name) != Sci_Norm \
                     THEN Com_Name ELSE NULL END), \
                 MAX(Com_Name) \
             ) AS Best_Com_Name \
         FROM migration \
         GROUP BY Sci_Norm"
    ).context("Cannot build best-name lookup")?;

    // Step 3: Update all rows — normalise Sci_Name and set best Com_Name.
    conn.execute_batch(
        "UPDATE migration SET \
             Sci_Name = Sci_Norm, \
             Com_Name = COALESCE( \
                 (SELECT b.Best_Com_Name FROM best_names b WHERE b.Sci_Norm = migration.Sci_Norm), \
                 Com_Name \
             )"
    ).context("Cannot update migration table")?;

    // Drop helper columns/tables.
    conn.execute_batch(
        "ALTER TABLE migration DROP COLUMN Sci_Norm; \
         DROP TABLE best_names;"
    ).ok();

    let count_after: u64 = conn
        .query_row("SELECT COUNT(*) FROM migration", [], |row| row.get(0))
        .unwrap_or(0);

    // ── Write the corrected data back ────────────────────────────────
    let out_path = detections_dir.join("_migrated.parquet");
    let out_tmp = detections_dir.join("._migrated.parquet.tmp");
    conn.execute(
        &format!(
            "COPY migration TO '{}' (FORMAT PARQUET, COMPRESSION ZSTD)",
            out_tmp.display()
        ),
        [],
    )
    .context("Cannot write migrated Parquet")?;

    std::fs::rename(&out_tmp, &out_path)
        .context("Cannot rename migrated Parquet file")?;

    // ── Remove old Parquet files ─────────────────────────────────────
    let mut removed = 0u32;
    for entry in std::fs::read_dir(detections_dir)
        .context("Cannot read detections dir")?
        .flatten()
    {
        let path = entry.path();
        if path.extension().map(|x| x == "parquet").unwrap_or(false)
            && path.file_name() != Some(std::ffi::OsStr::new("_migrated.parquet"))
        {
            std::fs::remove_file(&path).ok();
            removed += 1;
        }
    }

    info!(
        "Migration complete: {count_before} → {count_after} rows, \
         removed {removed} old file(s), wrote _migrated.parquet \
         ({:.1}s)",
        t0.elapsed().as_secs_f64()
    );

    // Write marker so we don't run again.
    std::fs::write(&marker, format!("{count_after} rows migrated\n")).ok();
    Ok(())
}
