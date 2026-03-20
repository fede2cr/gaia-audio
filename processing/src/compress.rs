//! Periodic compression of extracted audio clips to Opus.
//!
//! Walks `{extracted_dir}/By_Date/` looking for `.wav` and `.mp3` files,
//! converts each to Opus via `ffmpeg`, renames the companion spectrogram
//! PNG, and updates the `File_Name` column in the database so the web
//! dashboard keeps working.
//!
//! Designed to run on a background thread 4× per day (every 6 hours).
//! Each run is idempotent — already-compressed files (`.opus`) are skipped.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};
use libsql::params;
use tracing::{debug, error, info, warn};

/// Opus encoding bitrate.  96 kbps is transparent quality for bird /
/// wildlife audio and speech.
const OPUS_BITRATE: &str = "96k";

/// Run one compression sweep over the extracted directory.
///
/// Returns the number of files successfully compressed.
pub fn compress_sweep(
    extracted_dir: &Path,
    db_path: &Path,
    shutdown: &AtomicBool,
) -> Result<u64> {
    let by_date = extracted_dir.join("By_Date");
    if !by_date.is_dir() {
        debug!("No By_Date directory yet — nothing to compress");
        return Ok(0);
    }

    // Verify ffmpeg is available
    if !ffmpeg_available() {
        warn!("ffmpeg not found in $PATH — skipping compression sweep");
        return Ok(0);
    }

    let candidates = collect_candidates(&by_date)?;
    if candidates.is_empty() {
        info!("Compression sweep: no WAV/MP3 clips to convert");
        return Ok(0);
    }

    info!(
        "Compression sweep: {} candidate file(s) in {}",
        candidates.len(),
        by_date.display()
    );

    let conn = crate::db::open_conn_pub(db_path)
        .context("Cannot open database for compression")?;

    let mut converted = 0u64;

    for src_path in &candidates {
        if shutdown.load(Ordering::Relaxed) {
            info!("Compression sweep interrupted by shutdown");
            break;
        }

        match convert_one(&conn, src_path) {
            Ok(()) => converted += 1,
            Err(e) => {
                warn!("Failed to compress {}: {e:#}", src_path.display());
            }
        }
    }

    info!("Compression sweep complete: {converted}/{} file(s) converted", candidates.len());
    Ok(converted)
}

/// Loop that runs [`compress_sweep`] every `interval` (typically 6 h).
///
/// Blocks until `shutdown` is signalled.
pub fn compress_loop(
    extracted_dir: PathBuf,
    db_path: PathBuf,
    interval: std::time::Duration,
    shutdown: &AtomicBool,
) {
    info!(
        "Compression thread started (interval={}h, bitrate={OPUS_BITRATE})",
        interval.as_secs() / 3600
    );

    // Run one initial sweep shortly after startup (give processing 60 s
    // to finish its first cycle).
    sleep_interruptible(std::time::Duration::from_secs(60), shutdown);

    loop {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }

        match compress_sweep(&extracted_dir, &db_path, shutdown) {
            Ok(n) if n > 0 => info!("Compressed {n} file(s) this cycle"),
            Ok(_) => {}
            Err(e) => error!("Compression sweep error: {e:#}"),
        }

        sleep_interruptible(interval, shutdown);
    }

    info!("Compression thread stopped");
}

// ── Inline single-clip conversion ────────────────────────────────────────

/// Convert a single extracted WAV clip to Opus immediately after
/// extraction + spectrogram generation.
///
/// Returns the path to the new `.opus` file on success, or `None` if
/// ffmpeg is unavailable or the conversion fails.  The caller should
/// use the returned path (or fall back to the original WAV path) when
/// storing the filename in the database.
pub fn compress_inline(wav_path: &Path) -> Option<PathBuf> {
    if !ffmpeg_available() {
        debug!("ffmpeg not available — skipping inline Opus conversion");
        return None;
    }

    let src_name = wav_path.file_name()?.to_string_lossy().to_string();
    if !src_name.ends_with(".wav") {
        return None;
    }

    let opus_name = format!("{}.opus", &src_name[..src_name.len() - 4]);
    let opus_path = wav_path.with_file_name(&opus_name);

    // If the Opus file already exists, just clean up the WAV source.
    if opus_path.exists() {
        debug!("Opus already exists, removing WAV source: {}", wav_path.display());
        std::fs::remove_file(wav_path).ok();
        return Some(opus_path);
    }

    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-i",
            &wav_path.to_string_lossy(),
            "-c:a",
            "libopus",
            "-b:a",
            OPUS_BITRATE,
            "-vn",
            "-loglevel",
            "error",
        ])
        .arg(&opus_path)
        .status()
        .ok()?;

    if !status.success() {
        warn!("Inline Opus conversion failed for {}", wav_path.display());
        return None;
    }

    // Rename companion spectrogram: .wav.png → .opus.png
    let old_spec = wav_path.with_file_name(format!("{src_name}.png"));
    if old_spec.exists() {
        let new_spec = wav_path.with_file_name(format!("{opus_name}.png"));
        if let Err(e) = std::fs::rename(&old_spec, &new_spec) {
            warn!(
                "Cannot rename spectrogram {} → {}: {e}",
                old_spec.display(),
                new_spec.display()
            );
        }
    }

    // Remove original WAV
    if let Err(e) = std::fs::remove_file(wav_path) {
        warn!("Cannot remove original WAV {}: {e}", wav_path.display());
    }

    debug!("Inline compressed: {src_name} → {opus_name}");
    Some(opus_path)
}

// ── Internals ────────────────────────────────────────────────────────────

/// Recursively collect `.wav` and `.mp3` files under `dir`.
fn collect_candidates(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    walk_dir(dir, &mut out)?;
    Ok(out)
}

fn walk_dir(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    let entries = std::fs::read_dir(dir)
        .with_context(|| format!("Cannot read {}", dir.display()))?;

    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        let ft = match entry.file_type() {
            Ok(t) => t,
            Err(_) => continue,
        };
        if ft.is_dir() {
            walk_dir(&entry.path(), out)?;
        } else if ft.is_file() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            // Skip spectrogram PNGs (they end in .wav.png or .mp3.png)
            if name.ends_with(".wav") || name.ends_with(".mp3") {
                out.push(entry.path());
            }
        }
    }
    Ok(())
}

/// Convert a single `.wav` or `.mp3` file to `.opus`, rename its
/// spectrogram, and update the DB `File_Name`.
fn convert_one(conn: &libsql::Connection, src_path: &Path) -> Result<()> {
    let src_name = src_path
        .file_name()
        .context("No filename")?
        .to_string_lossy();

    // Determine the old extension and build the new filename.
    let (_old_ext, new_name) = if src_name.ends_with(".wav") {
        (".wav", format!("{}.opus", &src_name[..src_name.len() - 4]))
    } else if src_name.ends_with(".mp3") {
        (".mp3", format!("{}.opus", &src_name[..src_name.len() - 4]))
    } else {
        anyhow::bail!("Unexpected extension: {src_name}");
    };

    let dest_path = src_path.with_file_name(&new_name);

    // If the opus file already exists, just clean up the source
    if dest_path.exists() {
        debug!("Opus already exists, removing source: {}", src_path.display());
        std::fs::remove_file(src_path).ok();
        // Still try to update DB in case a previous run crashed mid-way
        update_db_filename(conn, &src_name, &new_name);
        return Ok(());
    }

    // Convert via ffmpeg
    let status = Command::new("ffmpeg")
        .args([
            "-y",           // overwrite
            "-i",
            &src_path.to_string_lossy(),
            "-c:a",
            "libopus",
            "-b:a",
            OPUS_BITRATE,
            "-vn",          // no video
            "-loglevel",
            "error",
        ])
        .arg(&dest_path)
        .status()
        .context("Cannot run ffmpeg")?;

    if !status.success() {
        anyhow::bail!("ffmpeg exited with {status}");
    }

    // Rename companion spectrogram: .wav.png → .opus.png (or .mp3.png → .opus.png)
    let old_spec = src_path.with_file_name(format!("{src_name}.png"));
    if old_spec.exists() {
        let new_spec = src_path.with_file_name(format!("{new_name}.png"));
        if let Err(e) = std::fs::rename(&old_spec, &new_spec) {
            warn!(
                "Cannot rename spectrogram {} → {}: {e}",
                old_spec.display(),
                new_spec.display()
            );
        }
    }

    // Update database
    update_db_filename(conn, &src_name, &new_name);

    // Remove original source file
    if let Err(e) = std::fs::remove_file(src_path) {
        warn!("Cannot remove original {}: {e}", src_path.display());
    }

    debug!("Compressed: {src_name} → {new_name}");
    Ok(())
}

/// Update `File_Name` in the detections table.
///
/// This is best-effort: imported BirdNET-Pi clips may have the same
/// basename stored in the DB, while Gaia clips use a different naming
/// convention.  We update all matching rows.
fn update_db_filename(conn: &libsql::Connection, old_name: &str, new_name: &str) {
    match crate::db::block_on(conn.execute(
        "UPDATE detections SET File_Name = ?1 WHERE File_Name = ?2",
        params![new_name.to_string(), old_name.to_string()],
    )) {
        Ok(n) if n > 0 => debug!("Updated {n} DB row(s): {old_name} → {new_name}"),
        Ok(_) => {
            // No matching rows — file may have been extracted without a
            // detection (e.g. urban noise), or the naming doesn't match.
            // This is normal and not an error.
        }
        Err(e) => warn!("DB update failed for {old_name}: {e}"),
    }
}

/// Check that ffmpeg is reachable.
fn ffmpeg_available() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Sleep in 1-second increments so we respond to shutdown quickly.
fn sleep_interruptible(duration: std::time::Duration, shutdown: &AtomicBool) {
    let mut remaining = duration;
    while remaining > std::time::Duration::ZERO && !shutdown.load(Ordering::Relaxed) {
        let step = remaining.min(std::time::Duration::from_secs(1));
        std::thread::sleep(step);
        remaining = remaining.saturating_sub(step);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collect_candidates_empty() {
        let dir = std::env::temp_dir().join("gaia_compress_test_empty");
        std::fs::create_dir_all(&dir).unwrap();
        let c = collect_candidates(&dir).unwrap();
        assert!(c.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_collect_candidates_filters() {
        let dir = std::env::temp_dir().join("gaia_compress_test_filter");
        let sub = dir.join("2026-01-01").join("Blackbird");
        std::fs::create_dir_all(&sub).unwrap();

        // These should be collected
        std::fs::write(sub.join("det.wav"), b"").unwrap();
        std::fs::write(sub.join("det.mp3"), b"").unwrap();

        // These should NOT be collected
        std::fs::write(sub.join("det.opus"), b"").unwrap();
        std::fs::write(sub.join("det.wav.png"), b"").unwrap();
        std::fs::write(sub.join("det.mp3.png"), b"").unwrap();
        std::fs::write(sub.join("notes.txt"), b"").unwrap();

        let c = collect_candidates(&dir).unwrap();
        assert_eq!(c.len(), 2);
        assert!(c.iter().all(|p| {
            let name = p.file_name().unwrap().to_string_lossy();
            name.ends_with(".wav") || name.ends_with(".mp3")
        }));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_update_db_filename() {
        let dir = std::env::temp_dir().join("gaia_compress_test_db");
        std::fs::create_dir_all(&dir).unwrap();
        let db = dir.join("test.db");
        let conn = crate::db::open_conn_pub(&db).unwrap();
        crate::db::block_on(conn.execute_batch(
            "CREATE TABLE detections (File_Name TEXT NOT NULL);
             INSERT INTO detections (File_Name) VALUES ('clip.wav');
             INSERT INTO detections (File_Name) VALUES ('clip.wav');
             INSERT INTO detections (File_Name) VALUES ('other.mp3');",
        ))
        .unwrap();

        update_db_filename(&conn, "clip.wav", "clip.opus");
        let count: u32 = crate::db::block_on(async {
            let mut rows = conn
                .query("SELECT COUNT(*) FROM detections WHERE File_Name = 'clip.opus'", ())
                .await
                .unwrap();
            rows.next().await.unwrap().unwrap().get::<u32>(0).unwrap()
        });
        assert_eq!(count, 2);

        // mp3 row untouched
        let mp3: u32 = crate::db::block_on(async {
            let mut rows = conn
                .query("SELECT COUNT(*) FROM detections WHERE File_Name = 'other.mp3'", ())
                .await
                .unwrap();
            rows.next().await.unwrap().unwrap().get::<u32>(0).unwrap()
        });
        assert_eq!(mp3, 1);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
