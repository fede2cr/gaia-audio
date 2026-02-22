//! BirdNET-Pi backup importer.
//!
//! Reads a BirdNET-Pi backup `.tar` file and imports:
//! - Detection records from the embedded `birds.db` SQLite database  
//! - Audio clips (`.mp3`) from `By_Date/` into the Gaia extracted directory
//! - Spectrograms (`.mp3.png`) from `By_Date/` alongside the clips
//!
//! The tar is read in a streaming fashion so memory usage stays bounded
//! even for multi-GB archives.

use std::collections::HashSet;
use std::io::Read;
use std::path::{Path, PathBuf};

use rusqlite::{params, Connection};

/// Pre-import report – shows what the backup contains before committing.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ImportReport {
    /// Path to the tar on disk.
    pub tar_path: String,
    /// Size of the tar file in bytes.
    pub tar_size_bytes: u64,
    /// Total detection rows in the embedded `birds.db`.
    pub total_detections: u64,
    /// Detections from today's date only.
    pub today_detections: u64,
    /// Distinct species across all time.
    pub total_species: u32,
    /// Distinct species detected today.
    pub today_species: u32,
    /// Date range: earliest detection date.
    pub date_min: Option<String>,
    /// Date range: latest detection date.
    pub date_max: Option<String>,
    /// Number of audio files (mp3) in the archive.
    pub audio_file_count: u64,
    /// Number of spectrogram PNGs in the archive.
    pub spectrogram_count: u64,
    /// Source lat/lon from the backup config.
    pub latitude: Option<f64>,
    pub longitude: Option<f64>,
    /// Top 10 species by detection count.
    pub top_species: Vec<(String, u64)>,
}

/// Progress report emitted during import.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ImportProgress {
    pub detections_imported: u64,
    pub files_extracted: u64,
    pub phase: String,
}

/// Result of a completed import.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ImportResult {
    pub detections_imported: u64,
    pub files_extracted: u64,
    pub skipped_existing: u64,
    pub errors: Vec<String>,
}

/// Analyse a BirdNET-Pi backup tar and produce a report *without* importing.
pub fn analyse_backup(tar_path: &Path) -> Result<ImportReport, String> {
    let meta = std::fs::metadata(tar_path)
        .map_err(|e| format!("Cannot stat {}: {e}", tar_path.display()))?;

    // We need to do two passes: one for the DB (extract to temp), one for file counts.
    // But we can do it in one pass by extracting the DB while counting files.
    let file = std::fs::File::open(tar_path)
        .map_err(|e| format!("Cannot open {}: {e}", tar_path.display()))?;

    let mut archive = tar::Archive::new(file);

    let tmp_dir = std::env::temp_dir().join("gaia_import_analysis");
    std::fs::create_dir_all(&tmp_dir)
        .map_err(|e| format!("Cannot create temp dir: {e}"))?;

    let mut audio_count: u64 = 0;
    let mut spectrogram_count: u64 = 0;
    let mut db_extracted = false;
    let mut conf_extracted = false;

    for entry_result in archive.entries().map_err(|e| format!("Cannot read tar: {e}"))? {
        let mut entry = match entry_result {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!("Skipping corrupt tar entry: {e}");
                continue;
            }
        };

        let path_str = entry
            .path()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default();

        if path_str == "birds.db" && !db_extracted {
            entry
                .unpack(tmp_dir.join("birds.db"))
                .map_err(|e| format!("Cannot extract birds.db: {e}"))?;
            db_extracted = true;
        } else if path_str == "birdnet.conf" && !conf_extracted {
            entry
                .unpack(tmp_dir.join("birdnet.conf"))
                .map_err(|e| format!("Cannot extract birdnet.conf: {e}"))?;
            conf_extracted = true;
        } else if path_str.ends_with(".mp3.png") {
            spectrogram_count += 1;
        } else if path_str.ends_with(".mp3") {
            audio_count += 1;
        }
    }

    if !db_extracted {
        return Err("No birds.db found in the backup tar".into());
    }

    // Analyse the embedded DB
    let db_path = tmp_dir.join("birds.db");
    let conn = Connection::open(&db_path)
        .map_err(|e| format!("Cannot open extracted birds.db: {e}"))?;

    let today = chrono::Local::now().format("%Y-%m-%d").to_string();

    let total_detections: u64 = conn
        .query_row("SELECT COUNT(*) FROM detections", [], |r| r.get(0))
        .unwrap_or(0);

    let today_detections: u64 = conn
        .query_row(
            "SELECT COUNT(*) FROM detections WHERE Date = ?1",
            params![today],
            |r| r.get(0),
        )
        .unwrap_or(0);

    let total_species: u32 = conn
        .query_row(
            "SELECT COUNT(DISTINCT Com_Name) FROM detections",
            [],
            |r| r.get(0),
        )
        .unwrap_or(0);

    let today_species: u32 = conn
        .query_row(
            "SELECT COUNT(DISTINCT Com_Name) FROM detections WHERE Date = ?1",
            params![today],
            |r| r.get(0),
        )
        .unwrap_or(0);

    let date_min: Option<String> = conn
        .query_row("SELECT MIN(Date) FROM detections", [], |r| r.get(0))
        .ok();

    let date_max: Option<String> = conn
        .query_row("SELECT MAX(Date) FROM detections", [], |r| r.get(0))
        .ok();

    // Top 10 species
    let mut top_stmt = conn
        .prepare(
            "SELECT Com_Name, COUNT(*) AS cnt FROM detections \
             GROUP BY Com_Name ORDER BY cnt DESC LIMIT 10",
        )
        .map_err(|e| format!("Query error: {e}"))?;

    let top_species: Vec<(String, u64)> = top_stmt
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
        .map_err(|e| format!("Query error: {e}"))?
        .filter_map(|r| r.ok())
        .collect();

    // Read lat/lon from config if available
    let (latitude, longitude) = if conf_extracted {
        parse_lat_lon(&tmp_dir.join("birdnet.conf"))
    } else {
        (None, None)
    };

    // Cleanup temp
    let _ = std::fs::remove_dir_all(&tmp_dir);

    Ok(ImportReport {
        tar_path: tar_path.to_string_lossy().to_string(),
        tar_size_bytes: meta.len(),
        total_detections,
        today_detections,
        total_species,
        today_species,
        date_min,
        date_max,
        audio_file_count: audio_count,
        spectrogram_count,
        latitude,
        longitude,
        top_species,
    })
}

/// Perform the full import of a BirdNET-Pi backup into the Gaia database.
///
/// - `tar_path`: path to the `.tar` backup file
/// - `gaia_db_path`: path to the Gaia `detections.db` (will be created if needed)
/// - `extracted_dir`: directory where audio clips and spectrograms are stored
pub fn import_backup(
    tar_path: &Path,
    gaia_db_path: &Path,
    extracted_dir: &Path,
) -> Result<ImportResult, String> {
    let mut result = ImportResult {
        detections_imported: 0,
        files_extracted: 0,
        skipped_existing: 0,
        errors: Vec::new(),
    };

    // ── Phase 1: Extract birds.db to temp and import detections ──────
    tracing::info!("Phase 1: Importing detections from backup DB…");

    let file = std::fs::File::open(tar_path)
        .map_err(|e| format!("Cannot open tar: {e}"))?;
    let mut archive = tar::Archive::new(file);

    let tmp_dir = std::env::temp_dir().join("gaia_import_work");
    std::fs::create_dir_all(&tmp_dir)
        .map_err(|e| format!("Cannot create temp dir: {e}"))?;

    // First pass: extract just the DB
    for entry_result in archive.entries().map_err(|e| format!("Tar read error: {e}"))? {
        let mut entry = match entry_result {
            Ok(e) => e,
            Err(_) => continue,
        };

        let path_str = entry
            .path()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default();

        if path_str == "birds.db" {
            entry
                .unpack(tmp_dir.join("birds.db"))
                .map_err(|e| format!("Cannot extract birds.db: {e}"))?;
            break;
        }
    }

    let source_db = tmp_dir.join("birds.db");
    if !source_db.exists() {
        return Err("No birds.db found in the backup tar".into());
    }

    // Ensure Gaia DB exists with schema
    ensure_gaia_schema(gaia_db_path)?;

    // Read existing file names to avoid duplicates
    let existing_files = get_existing_filenames(gaia_db_path)?;

    // Import detections
    let src_conn = Connection::open(&source_db)
        .map_err(|e| format!("Cannot open source DB: {e}"))?;

    let dst_conn = Connection::open(gaia_db_path)
        .map_err(|e| format!("Cannot open Gaia DB: {e}"))?;
    dst_conn
        .execute_batch("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;")
        .map_err(|e| format!("Pragma error: {e}"))?;

    let mut src_stmt = src_conn
        .prepare(
            "SELECT Date, Time, Sci_Name, Com_Name, Confidence, \
             Lat, Lon, Cutoff, Week, Sens, Overlap, File_Name \
             FROM detections ORDER BY Date, Time",
        )
        .map_err(|e| format!("Source query error: {e}"))?;

    // Batch insert in transactions of 5000 rows
    let mut batch_count = 0u64;
    dst_conn
        .execute_batch("BEGIN TRANSACTION")
        .map_err(|e| format!("Transaction error: {e}"))?;

    let rows = src_stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,  // Date
                row.get::<_, String>(1)?,  // Time
                row.get::<_, String>(2)?,  // Sci_Name
                row.get::<_, String>(3)?,  // Com_Name
                row.get::<_, f64>(4)?,     // Confidence
                row.get::<_, f64>(5)?,     // Lat
                row.get::<_, f64>(6)?,     // Lon
                row.get::<_, f64>(7)?,     // Cutoff
                row.get::<_, i32>(8)?,     // Week
                row.get::<_, f64>(9)?,     // Sens
                row.get::<_, f64>(10)?,    // Overlap
                row.get::<_, String>(11)?, // File_Name
            ))
        })
        .map_err(|e| format!("Query error: {e}"))?;

    for row_result in rows {
        let (date, time, sci, com, conf, lat, lon, cutoff, week, sens, overlap, fname) =
            match row_result {
                Ok(r) => r,
                Err(e) => {
                    result.errors.push(format!("Row read error: {e}"));
                    continue;
                }
            };

        // Skip duplicates (same File_Name already in Gaia DB)
        if existing_files.contains(&fname) {
            result.skipped_existing += 1;
            continue;
        }

        if let Err(e) = dst_conn.execute(
            "INSERT INTO detections (Date, Time, Domain, Sci_Name, Com_Name, \
             Confidence, Lat, Lon, Cutoff, Week, Sens, Overlap, File_Name) \
             VALUES (?1, ?2, 'birds', ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![date, time, sci, com, conf, lat, lon, cutoff, week, sens, overlap, fname],
        ) {
            result.errors.push(format!("Insert error for {fname}: {e}"));
            continue;
        }

        result.detections_imported += 1;
        batch_count += 1;

        if batch_count % 5000 == 0 {
            dst_conn
                .execute_batch("COMMIT; BEGIN TRANSACTION")
                .map_err(|e| format!("Commit error: {e}"))?;
            tracing::info!(
                "Imported {} detections so far…",
                result.detections_imported
            );
        }
    }

    dst_conn
        .execute_batch("COMMIT")
        .map_err(|e| format!("Final commit error: {e}"))?;

    drop(src_stmt);
    drop(src_conn);
    drop(dst_conn);

    tracing::info!(
        "Phase 1 complete: {} detections imported, {} skipped (existing)",
        result.detections_imported,
        result.skipped_existing
    );

    // ── Phase 2: Extract audio and spectrogram files ─────────────────
    tracing::info!("Phase 2: Extracting audio files and spectrograms…");

    std::fs::create_dir_all(extracted_dir)
        .map_err(|e| format!("Cannot create extracted dir: {e}"))?;

    let file2 = std::fs::File::open(tar_path)
        .map_err(|e| format!("Cannot reopen tar: {e}"))?;
    let mut archive2 = tar::Archive::new(file2);

    for entry_result in archive2
        .entries()
        .map_err(|e| format!("Tar read error: {e}"))?
    {
        let mut entry = match entry_result {
            Ok(e) => e,
            Err(_) => continue,
        };

        let path_str = entry
            .path()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default();

        // Only extract By_Date files (mp3 + png)
        if !path_str.starts_with("By_Date/") {
            continue;
        }

        let is_audio = path_str.ends_with(".mp3");
        let is_spectrogram = path_str.ends_with(".mp3.png");
        if !is_audio && !is_spectrogram {
            continue;
        }

        // Preserve the By_Date structure inside the extracted dir
        let dest = extracted_dir.join(&path_str);
        if dest.exists() {
            result.skipped_existing += 1;
            continue;
        }

        if let Some(parent) = dest.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                result
                    .errors
                    .push(format!("Cannot create dir {}: {e}", parent.display()));
                continue;
            }
        }

        if let Err(e) = entry.unpack(&dest) {
            result
                .errors
                .push(format!("Cannot extract {}: {e}", path_str));
            continue;
        }

        result.files_extracted += 1;

        if result.files_extracted % 10000 == 0 {
            tracing::info!("Extracted {} files so far…", result.files_extracted);
        }
    }

    // Cleanup temp
    let _ = std::fs::remove_dir_all(&tmp_dir);

    tracing::info!(
        "Import complete: {} detections, {} files extracted, {} skipped, {} errors",
        result.detections_imported,
        result.files_extracted,
        result.skipped_existing,
        result.errors.len()
    );

    Ok(result)
}

/// Ensure the Gaia detections table exists.
pub fn ensure_gaia_schema(db_path: &Path) -> Result<(), String> {
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("Cannot create DB dir: {e}"))?;
    }

    let conn =
        Connection::open(db_path).map_err(|e| format!("Cannot open Gaia DB: {e}"))?;

    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS detections (
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
            File_Name  VARCHAR(100) NOT NULL
        );
        CREATE INDEX IF NOT EXISTS detections_Com_Name    ON detections (Com_Name);
        CREATE INDEX IF NOT EXISTS detections_Sci_Name    ON detections (Sci_Name);
        CREATE INDEX IF NOT EXISTS detections_Domain      ON detections (Domain);
        CREATE INDEX IF NOT EXISTS detections_Date_Time   ON detections (Date DESC, Time DESC);",
    )
    .map_err(|e| format!("Schema error: {e}"))?;

    Ok(())
}

/// Load all existing File_Name values from the Gaia DB for dedup.
fn get_existing_filenames(db_path: &Path) -> Result<HashSet<String>, String> {
    let conn = Connection::open(db_path)
        .map_err(|e| format!("Cannot open Gaia DB: {e}"))?;

    let mut stmt = conn
        .prepare("SELECT File_Name FROM detections")
        .map_err(|e| format!("Query error: {e}"))?;

    let names: HashSet<String> = stmt
        .query_map([], |row| row.get(0))
        .map_err(|e| format!("Query error: {e}"))?
        .filter_map(|r| r.ok())
        .collect();

    Ok(names)
}

/// Parse LATITUDE and LONGITUDE from a birdnet.conf file.
fn parse_lat_lon(conf_path: &Path) -> (Option<f64>, Option<f64>) {
    let text = match std::fs::read_to_string(conf_path) {
        Ok(t) => t,
        Err(_) => return (None, None),
    };

    let mut lat = None;
    let mut lon = None;

    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        if let Some((key, val)) = line.split_once('=') {
            let key = key.trim();
            let val = val.trim().trim_matches('"');
            match key {
                "LATITUDE" => lat = val.parse().ok(),
                "LONGITUDE" => lon = val.parse().ok(),
                _ => {}
            }
        }
    }

    (lat, lon)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_lat_lon() {
        let dir = std::env::temp_dir().join("gaia_import_test");
        std::fs::create_dir_all(&dir).unwrap();
        let conf = dir.join("test.conf");
        std::fs::write(
            &conf,
            "# comment\nLATITUDE=9.9346\nLONGITUDE=\"-84.0706\"\n",
        )
        .unwrap();

        let (lat, lon) = parse_lat_lon(&conf);
        assert!((lat.unwrap() - 9.9346).abs() < 0.001);
        assert!((lon.unwrap() - (-84.0706)).abs() < 0.001);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Run with: BACKUP_PATH=/path/to/backup.tar cargo test -p gaia-web --features ssr -- --ignored test_analyse_real_backup --nocapture
    #[test]
    #[ignore]
    fn test_analyse_real_backup() {
        let path = std::env::var("BACKUP_PATH")
            .expect("Set BACKUP_PATH env var to the .tar backup file");
        let report = analyse_backup(Path::new(&path)).expect("analyse_backup failed");

        println!("\n╔══════════════════════════════════════╗");
        println!("║   BirdNET-Pi Backup Analysis Report  ║");
        println!("╠══════════════════════════════════════╣");
        println!("║ Archive:        {}", report.tar_path);
        println!("║ Size:           {:.1} GB", report.tar_size_bytes as f64 / 1_073_741_824.0);
        println!("╠══════════════════════════════════════╣");
        println!("║ Total detections: {:>8}", report.total_detections);
        println!("║ Today detections: {:>8}", report.today_detections);
        println!("║ Total species:    {:>8}", report.total_species);
        println!("║ Today species:    {:>8}", report.today_species);
        println!("║ Date range:       {} → {}", 
            report.date_min.as_deref().unwrap_or("?"),
            report.date_max.as_deref().unwrap_or("?"));
        println!("╠══════════════════════════════════════╣");
        println!("║ Audio files (mp3): {:>7}", report.audio_file_count);
        println!("║ Spectrograms:      {:>7}", report.spectrogram_count);
        if let Some(lat) = report.latitude {
            println!("║ Location:          {:.4}°, {:.4}°", lat, report.longitude.unwrap_or(0.0));
        }
        println!("╠══════════════════════════════════════╣");
        println!("║ Top 10 Species:");
        for (i, (name, count)) in report.top_species.iter().enumerate() {
            println!("║  {:>2}. {:<30} {:>6}", i + 1, name, count);
        }
        println!("╚══════════════════════════════════════╝\n");
    }
}
