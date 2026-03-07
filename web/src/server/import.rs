//! BirdNET-Pi backup importer.
//!
//! Supports two import modes:
//!
//! 1. **Network streaming** (preferred): fetches the backup directly from a
//!    BirdNET-Pi node via `curl` HTTP POST and processes the tar stream
//!    on-the-fly — no temporary archive written to disk.
//! 2. **File-based** (legacy): reads a pre-downloaded `.tar` backup from the
//!    `/backups` volume.
//!
//! Both modes handle deduplication: detections and audio clips that have
//! already been imported — even if subsequently compressed from `.mp3`/`.wav`
//! to `.opus` by the processing server — are detected and skipped.

use std::collections::HashSet;
use std::io::Read;
use std::path::Path;

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

// ─── Discovered BirdNET-Pi node (reuse the shared model type) ────────────
pub use crate::model::BirdnetNode;

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

    // Read existing file names to avoid duplicates (opus-aware)
    let existing_files = get_existing_filenames(gaia_db_path)?;

    import_detections_from_db(&source_db, gaia_db_path, &existing_files, &mut result)?;

    tracing::info!(
        "Phase 1 complete: {} detections imported, {} skipped (existing)",
        result.detections_imported,
        result.skipped_existing
    );

    // ── Phase 2: Extract audio and spectrogram files ─────────────────
    tracing::info!("Phase 2: Extracting audio files and spectrograms…");

    extract_media_from_tar(tar_path, extracted_dir, &mut result)?;

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

// ─── Network streaming import ────────────────────────────────────────────────

/// Import observations by streaming the backup directly from a BirdNET-Pi node.
///
/// 1. POSTs to `http://{address}:{port}/scripts/backup.php` via `curl`
/// 2. Processes the tar response as a stream — audio clips and spectrograms
///    are extracted directly to `extracted_dir` without writing a temporary
///    archive.
/// 3. The small `birds.db` is extracted to `/tmp`, then detection rows are
///    imported into the Gaia database with opus-aware deduplication.
///
/// This function is blocking and designed to run inside `spawn_blocking`.
pub fn stream_import(
    address: &str,
    port: u16,
    gaia_db_path: &Path,
    extracted_dir: &Path,
) -> Result<ImportResult, String> {
    let url = format!("http://{}:{}/scripts/backup.php", address, port);

    let mut result = ImportResult {
        detections_imported: 0,
        files_extracted: 0,
        skipped_existing: 0,
        errors: Vec::new(),
    };

    tracing::info!("Starting streaming import from {url}");

    // Ensure Gaia DB and directories exist
    ensure_gaia_schema(gaia_db_path)?;
    std::fs::create_dir_all(extracted_dir)
        .map_err(|e| format!("Cannot create extracted dir: {e}"))?;

    // Build the opus-aware dedup set BEFORE we start downloading
    let existing = get_existing_filenames(gaia_db_path)?;

    // ── Stream the tar from the BirdNET-Pi node via curl ─────────────
    let mut child = std::process::Command::new("curl")
        .args([
            "-s",                           // silent
            "-f",                           // fail on HTTP errors
            "-L",                           // follow redirects
            "-d", "user=birdnet&password=", // POST credentials
            &url,
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("Cannot start curl (is it installed?): {e}"))?;

    let stdout = child
        .stdout
        .take()
        .ok_or("Cannot capture curl stdout")?;

    let mut archive = tar::Archive::new(stdout);

    let tmp_dir = std::env::temp_dir().join("gaia_stream_import");
    std::fs::create_dir_all(&tmp_dir)
        .map_err(|e| format!("Cannot create temp dir: {e}"))?;

    // ── Single-pass: extract files and DB simultaneously ─────────────
    for entry_result in archive
        .entries()
        .map_err(|e| format!("Tar stream error: {e}"))?
    {
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

        // Extract birds.db to temp for later detection import
        if path_str == "birds.db" {
            entry
                .unpack(tmp_dir.join("birds.db"))
                .map_err(|e| format!("Cannot extract birds.db: {e}"))?;
            tracing::info!("Extracted birds.db from stream");
            continue;
        }

        // Extract birdnet.conf to temp (for future use)
        if path_str == "birdnet.conf" {
            let _ = entry.unpack(tmp_dir.join("birdnet.conf"));
            continue;
        }

        // Only process By_Date/ audio + spectrograms
        if !path_str.starts_with("By_Date/") {
            continue;
        }

        let is_audio = path_str.ends_with(".mp3") && !path_str.ends_with(".mp3.png");
        let is_spectrogram = path_str.ends_with(".mp3.png");
        if !is_audio && !is_spectrogram {
            continue;
        }

        let dest = extracted_dir.join(&path_str);

        // Skip if file already exists (original or opus-converted variant)
        if file_already_imported(&dest) {
            result.skipped_existing += 1;
            continue;
        }

        // Create parent directory and extract
        if let Some(parent) = dest.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                result
                    .errors
                    .push(format!("Cannot create dir: {e}"));
                continue;
            }
        }

        if let Err(e) = entry.unpack(&dest) {
            result
                .errors
                .push(format!("Cannot extract {path_str}: {e}"));
            continue;
        }

        result.files_extracted += 1;
        if result.files_extracted % 5000 == 0 {
            tracing::info!("Extracted {} files so far…", result.files_extracted);
        }
    }

    // ── Wait for curl to finish ──────────────────────────────────────
    let status = child
        .wait()
        .map_err(|e| format!("curl wait error: {e}"))?;

    if !status.success() {
        let stderr_msg = child
            .stderr
            .take()
            .map(|mut s| {
                let mut buf = String::new();
                s.read_to_string(&mut buf).ok();
                buf
            })
            .unwrap_or_default();
        return Err(format!(
            "curl failed (exit {}): {stderr_msg}",
            status.code().unwrap_or(-1)
        ));
    }

    // ── Import detections from the extracted birds.db ────────────────
    let source_db = tmp_dir.join("birds.db");
    if source_db.exists() {
        tracing::info!("Importing detections from birds.db…");
        import_detections_from_db(&source_db, gaia_db_path, &existing, &mut result)?;
    } else {
        tracing::warn!("No birds.db found in the streamed backup");
    }

    let _ = std::fs::remove_dir_all(&tmp_dir);

    tracing::info!(
        "Stream import complete: {} detections, {} files, {} skipped, {} errors",
        result.detections_imported,
        result.files_extracted,
        result.skipped_existing,
        result.errors.len()
    );

    Ok(result)
}

// ─── mDNS discovery ──────────────────────────────────────────────────────────

/// Discover BirdNET-Pi nodes on the local network.
///
/// Browses `_http._tcp.local.` for 4 seconds and returns nodes whose
/// instance name contains "birdnet" (case-insensitive).
pub fn discover_birdnet_nodes() -> Vec<BirdnetNode> {
    use mdns_sd::{ServiceDaemon, ServiceEvent};
    use std::time::{Duration, Instant};

    let daemon = match ServiceDaemon::new() {
        Ok(d) => d,
        Err(e) => {
            tracing::warn!("Cannot start mDNS daemon: {e}");
            return vec![];
        }
    };

    let receiver = match daemon.browse("_http._tcp.local.") {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("Cannot browse mDNS: {e}");
            let _ = daemon.shutdown();
            return vec![];
        }
    };

    let mut nodes: Vec<BirdnetNode> = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(4);

    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            break;
        }
        match receiver.recv_timeout(remaining) {
            Ok(ServiceEvent::ServiceResolved(info)) => {
                let instance = info
                    .get_fullname()
                    .split('.')
                    .next()
                    .unwrap_or("")
                    .to_string();

                // Filter for BirdNET-Pi nodes
                if !instance.to_lowercase().contains("birdnet") {
                    continue;
                }

                let port = info.get_port();
                // Prefer IPv4 addresses
                let addrs: Vec<std::net::IpAddr> = info
                    .get_addresses()
                    .iter()
                    .map(|a| a.to_ip_addr())
                    .filter(|a| a.is_ipv4())
                    .collect();

                if let Some(addr) = addrs.first() {
                    let address = addr.to_string();
                    // Avoid duplicate entries
                    if !nodes.iter().any(|n| n.address == address) {
                        tracing::info!(
                            "mDNS: found BirdNET-Pi node '{}' at {}:{}",
                            instance, address, port
                        );
                        nodes.push(BirdnetNode {
                            name: instance,
                            address,
                            port,
                        });
                    }
                }
            }
            Ok(_) => {}
            Err(_) => break,
        }
    }

    let _ = daemon.stop_browse("_http._tcp.local.");
    let _ = daemon.shutdown();

    nodes
}

// ─── Shared helpers ──────────────────────────────────────────────────────────

/// Import detection rows from a BirdNET-Pi `birds.db` into the Gaia database.
///
/// `existing` is a set of `File_Name` values already in the Gaia DB, expanded
/// with pre-compression variants (`.mp3`/`.wav`) so that re-importing a backup
/// after Opus compression still deduplicates correctly.
fn import_detections_from_db(
    source_db: &Path,
    gaia_db_path: &Path,
    existing: &HashSet<String>,
    result: &mut ImportResult,
) -> Result<(), String> {
    let src_conn = Connection::open(source_db)
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

        // Skip if this file was already imported (even if now .opus)
        if existing.contains(&fname) {
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

    Ok(())
}

/// Extract audio and spectrogram files from a tar on disk into `extracted_dir`,
/// with opus-aware deduplication.
fn extract_media_from_tar(
    tar_path: &Path,
    extracted_dir: &Path,
    result: &mut ImportResult,
) -> Result<(), String> {
    std::fs::create_dir_all(extracted_dir)
        .map_err(|e| format!("Cannot create extracted dir: {e}"))?;

    let file = std::fs::File::open(tar_path)
        .map_err(|e| format!("Cannot reopen tar: {e}"))?;
    let mut archive = tar::Archive::new(file);

    for entry_result in archive
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

        if !path_str.starts_with("By_Date/") {
            continue;
        }

        let is_audio = path_str.ends_with(".mp3") && !path_str.ends_with(".mp3.png");
        let is_spectrogram = path_str.ends_with(".mp3.png");
        if !is_audio && !is_spectrogram {
            continue;
        }

        let dest = extracted_dir.join(&path_str);

        // Check if file already exists (original or opus-converted variant)
        if file_already_imported(&dest) {
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

    Ok(())
}

/// Check whether a media file (or its Opus-converted counterpart) already
/// exists on disk.
///
/// After the compression background task converts `.mp3` → `.opus` and
/// renames spectrograms from `.mp3.png` → `.opus.png`, a plain `dest.exists()`
/// would miss these and cause a duplicate import.
fn file_already_imported(dest: &Path) -> bool {
    if dest.exists() {
        return true;
    }

    let name = match dest.file_name() {
        Some(n) => n.to_string_lossy(),
        None => return false,
    };

    // audio.mp3 → check for audio.opus
    if let Some(stem) = name.strip_suffix(".mp3") {
        return dest.with_file_name(format!("{stem}.opus")).exists();
    }

    // audio.mp3.png → check for audio.opus.png
    if let Some(stem) = name.strip_suffix(".mp3.png") {
        return dest.with_file_name(format!("{stem}.opus.png")).exists();
    }

    // audio.wav → check for audio.opus
    if let Some(stem) = name.strip_suffix(".wav") {
        return dest.with_file_name(format!("{stem}.opus")).exists();
    }

    // audio.wav.png → check for audio.opus.png
    if let Some(stem) = name.strip_suffix(".wav.png") {
        return dest.with_file_name(format!("{stem}.opus.png")).exists();
    }

    false
}

/// Ensure the Gaia detections table exists.
pub fn ensure_gaia_schema(db_path: &Path) -> Result<(), String> {
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("Cannot create DB dir: {e}"))?;
    }

    let conn =
        Connection::open(db_path).map_err(|e| format!("Cannot open Gaia DB: {e}"))?;

    // Enable WAL once via a read-write connection so that later read-only
    // connections inherit it without needing write access.
    conn.execute_batch("PRAGMA journal_mode=WAL;")
        .map_err(|e| format!("WAL pragma error: {e}"))?;

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
        );",
    )
    .map_err(|e| format!("Schema error: {e}"))?;

    // Migration: add Source_Node to existing databases that lack it.
    let _ = conn.execute_batch(
        "ALTER TABLE detections ADD COLUMN Source_Node VARCHAR(200) NOT NULL DEFAULT '';",
    );
    // Migration: add Excluded column to existing databases.
    let _ = conn.execute_batch(
        "ALTER TABLE detections ADD COLUMN Excluded INTEGER NOT NULL DEFAULT 0;",
    );
    // Migration: add model tracking columns.
    let _ = conn.execute_batch(
        "ALTER TABLE detections ADD COLUMN Model_Slug VARCHAR(100) NOT NULL DEFAULT '';",
    );
    let _ = conn.execute_batch(
        "ALTER TABLE detections ADD COLUMN Model_Name VARCHAR(200) NOT NULL DEFAULT '';",
    );

    Ok(())
}

/// Load all existing File_Name values from the Gaia DB for dedup.
///
/// After Opus compression, `File_Name` values may have changed from
/// `.mp3`/`.wav` to `.opus`.  We expand the set with the pre-compression
/// variants so that re-importing a BirdNET-Pi backup still deduplicates.
fn get_existing_filenames(db_path: &Path) -> Result<HashSet<String>, String> {
    let conn = Connection::open(db_path)
        .map_err(|e| format!("Cannot open Gaia DB: {e}"))?;

    let mut stmt = conn
        .prepare("SELECT File_Name FROM detections")
        .map_err(|e| format!("Query error: {e}"))?;

    let mut names: HashSet<String> = stmt
        .query_map([], |row| row.get(0))
        .map_err(|e| format!("Query error: {e}"))?
        .filter_map(|r| r.ok())
        .collect();

    // For every .opus filename, also add .mp3 and .wav so the source
    // BirdNET-Pi filenames (which are always .mp3) still match.
    let opus_variants: Vec<String> = names
        .iter()
        .filter(|n| n.ends_with(".opus"))
        .flat_map(|n| {
            let stem = &n[..n.len() - 5]; // strip ".opus"
            [format!("{stem}.mp3"), format!("{stem}.wav")]
        })
        .collect();

    names.extend(opus_variants);

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

    #[test]
    fn test_file_already_imported_opus_variant() {
        let dir = std::env::temp_dir().join("gaia_import_opus_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // Create an .opus file (as if compression already ran)
        let opus_file = dir.join("birds-Blackbird-92-birdnet-0715.opus");
        std::fs::write(&opus_file, b"fake opus").unwrap();

        // The original .mp3 does NOT exist on disk
        let mp3_path = dir.join("birds-Blackbird-92-birdnet-0715.mp3");
        assert!(!mp3_path.exists());

        // But file_already_imported should detect the .opus variant
        assert!(file_already_imported(&mp3_path));

        // Same for spectrograms: .opus.png exists, .mp3.png does not
        let opus_png = dir.join("birds-Blackbird-92-birdnet-0715.opus.png");
        std::fs::write(&opus_png, b"fake png").unwrap();
        let mp3_png = dir.join("birds-Blackbird-92-birdnet-0715.mp3.png");
        assert!(file_already_imported(&mp3_png));

        // Non-existent file with no variant → false
        let unknown = dir.join("nonexistent.mp3");
        assert!(!file_already_imported(&unknown));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_existing_filenames_includes_opus_variants() {
        let dir = std::env::temp_dir().join("gaia_import_dedup_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("test.db");

        // Create a DB with an .opus filename (post-compression)
        let conn = Connection::open(&db_path).unwrap();
        conn.execute_batch(
            "CREATE TABLE detections (File_Name TEXT NOT NULL);
             INSERT INTO detections VALUES ('birds-Robin-42.opus');
             INSERT INTO detections VALUES ('birds-Wren-10.mp3');",
        )
        .unwrap();
        drop(conn);

        let names = get_existing_filenames(&db_path).unwrap();

        // The .opus entry should also produce .mp3 and .wav variants
        assert!(names.contains("birds-Robin-42.opus"));
        assert!(names.contains("birds-Robin-42.mp3"));
        assert!(names.contains("birds-Robin-42.wav"));

        // The .mp3 entry stays as-is (no expansion needed)
        assert!(names.contains("birds-Wren-10.mp3"));
        // .mp3 is NOT expanded to .opus (source BirdNET-Pi always uses .mp3)
        assert!(!names.contains("birds-Wren-10.opus"));

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
