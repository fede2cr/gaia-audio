//! SQLite database layer.
//!
//! Reused from `birdnet-server/src/db.rs`, with an added `Domain` column.

use std::path::Path;

use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use tracing::info;

use gaia_common::detection::Detection;

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
    ",
    )
    .context("Failed to create tables")?;

    // Migration: add Source_Node to existing databases that lack it.
    migrate_add_source_node(&conn);
    // Migration: add Excluded column to existing databases.
    migrate_add_excluded(&conn);

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

/// Add the `Excluded` column if it doesn't exist (idempotent).
fn migrate_add_excluded(conn: &Connection) {
    let _ = conn.execute_batch(
        "ALTER TABLE detections ADD COLUMN Excluded INTEGER NOT NULL DEFAULT 0;",
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
         Lat, Lon, Cutoff, Week, Sens, Overlap, File_Name, Source_Node, Excluded) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
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
            d.excluded as i32,
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

/// Increment the urban-noise counter for a category / date / hour.
///
/// Uses `INSERT OR REPLACE` with an UPSERT pattern so a single row is
/// stored per (Date, Hour, Category) tuple.
pub fn increment_urban_noise(db_path: &Path, date: &str, hour: u32, category: &str) -> Result<()> {
    let conn = Connection::open(db_path)?;
    let result = conn.execute(
        "INSERT INTO urban_noise (Date, Hour, Category, Count) VALUES (?1, ?2, ?3, 1) \
         ON CONFLICT(Date, Hour, Category) DO UPDATE SET Count = Count + 1",
        params![date, hour, category],
    );

    if let Err(_) = result {
        // Fallback for older schemas without the UNIQUE constraint.
        let updated = conn.execute(
            "UPDATE urban_noise SET Count = Count + 1 WHERE Date = ?1 AND Hour = ?2 AND Category = ?3",
            params![date, hour, category],
        )?;
        if updated == 0 {
            conn.execute(
                "INSERT INTO urban_noise (Date, Hour, Category, Count) VALUES (?1, ?2, ?3, 1)",
                params![date, hour, category],
            )?;
        }
    }

    Ok(())
}

// ─── Settings ────────────────────────────────────────────────────────────────

/// Read a single setting from the `settings` table.
/// Returns `None` if the table or key doesn't exist.
pub fn get_setting(db_path: &Path, key: &str) -> Option<String> {
    let conn = Connection::open(db_path).ok()?;
    conn.execute_batch("PRAGMA busy_timeout=1000;").ok();
    conn.query_row(
        "SELECT value FROM settings WHERE key = ?1",
        params![key],
        |row| row.get(0),
    )
    .ok()
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
    let conn = match Connection::open(db_path) {
        Ok(c) => c,
        Err(_) => return vec![],
    };
    conn.execute_batch("PRAGMA busy_timeout=1000;").ok();
    let mut stmt = match conn.prepare("SELECT Sci_Name FROM exclusion_overrides") {
        Ok(s) => s,
        Err(_) => return vec![],
    };
    stmt.query_map([], |row| row.get::<_, String>(0))
        .ok()
        .map(|rows| rows.filter_map(|r| r.ok()).collect())
        .unwrap_or_default()
}
