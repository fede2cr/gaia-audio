//! SQLite database layer.
//!
//! Reused from `birdnet-server/src/db.rs`, with an added `Domain` column.

use std::path::Path;

use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use tracing::info;

use gaia_common::detection::Detection;

/// Create the `detections` table (and indices) if it doesn't exist.
pub fn initialize(db_path: &Path) -> Result<()> {
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let conn = Connection::open(db_path)
        .with_context(|| format!("Cannot open database: {}", db_path.display()))?;

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
            Source_Node VARCHAR(200) NOT NULL DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS detections_Com_Name    ON detections (Com_Name);
        CREATE INDEX IF NOT EXISTS detections_Sci_Name    ON detections (Sci_Name);
        CREATE INDEX IF NOT EXISTS detections_Domain      ON detections (Domain);
        CREATE INDEX IF NOT EXISTS detections_Date_Time   ON detections (Date DESC, Time DESC);
    ",
    )
    .context("Failed to create detections table")?;

    // Migration: add Source_Node to existing databases that lack it.
    migrate_add_source_node(&conn);

    info!("Database schema verified");
    Ok(())
}

/// Add the `Source_Node` column if it doesn't exist (idempotent).
fn migrate_add_source_node(conn: &Connection) {
    // SQLite's ALTER TABLE ADD COLUMN is a no-op if the column already exists
    // — except it returns an error. We simply ignore that.
    let _ = conn.execute_batch(
        "ALTER TABLE detections ADD COLUMN Source_Node VARCHAR(200) NOT NULL DEFAULT '';",
    );
}

/// Insert a single detection row.
pub fn insert_detection(
    db_path: &Path,
    detection: &Detection,
    lat: f64,
    lon: f64,
    cutoff: f64,
    sensitivity: f64,
    overlap: f64,
    file_name: &str,
    source_node: &str,
) -> Result<()> {
    for attempt in 0..3 {
        match try_insert(
            db_path, detection, lat, lon, cutoff, sensitivity, overlap, file_name, source_node,
        ) {
            Ok(()) => return Ok(()),
            Err(e) => {
                tracing::warn!("Database busy (attempt {attempt}): {e}");
                std::thread::sleep(std::time::Duration::from_secs(2));
            }
        }
    }
    anyhow::bail!("Failed to insert detection after 3 attempts")
}

fn try_insert(
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
    let conn = Connection::open(db_path)?;
    conn.execute(
        "INSERT INTO detections (Date, Time, Domain, Sci_Name, Com_Name, Confidence, \
         Lat, Lon, Cutoff, Week, Sens, Overlap, File_Name, Source_Node) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
        params![
            d.date,
            d.time,
            d.domain,
            d.scientific_name,
            d.common_name,
            d.confidence,
            lat,
            lon,
            cutoff,
            d.week,
            sensitivity,
            overlap,
            file_name,
            source_node,
        ],
    )?;
    Ok(())
}

/// Count today's detections of a species in a given domain.
#[allow(dead_code)]
pub fn todays_count_for(db_path: &Path, domain: &str, sci_name: &str) -> u32 {
    let today = chrono::Local::now().format("%Y-%m-%d").to_string();
    _count(
        db_path,
        &format!(
            "SELECT COUNT(*) FROM detections WHERE Date = DATE('{today}') \
             AND Domain = '{domain}' AND Sci_Name = '{sci_name}'"
        ),
    )
}

/// Count this week's detections of a species in a given domain.
#[allow(dead_code)]
pub fn weeks_count_for(db_path: &Path, domain: &str, sci_name: &str) -> u32 {
    let today = chrono::Local::now().format("%Y-%m-%d").to_string();
    _count(
        db_path,
        &format!(
            "SELECT COUNT(*) FROM detections WHERE Date >= DATE('{today}', '-7 day') \
             AND Domain = '{domain}' AND Sci_Name = '{sci_name}'"
        ),
    )
}

fn _count(db_path: &Path, sql: &str) -> u32 {
    Connection::open(db_path)
        .ok()
        .and_then(|conn| conn.query_row(sql, [], |row| row.get::<_, u32>(0)).ok())
        .unwrap_or(0)
}
