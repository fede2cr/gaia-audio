//! Zenodo model downloader – fetches and extracts model archives on demand.
//!
//! When a manifest includes a `[download]` section and the expected model
//! files are not yet present on disk, this module downloads the appropriate
//! variant zip from Zenodo, verifies its MD5 checksum, and extracts the
//! contents into the model directory.

use std::path::Path;

use anyhow::{Context, Result};
use tracing::{info, warn};

use crate::manifest::ResolvedManifest;

const ZENODO_FILES_URL: &str = "https://zenodo.org/api/records";

/// Name of the marker file used to implement exponential backoff across
/// container restarts.  The file contains the next retry timestamp.
const BACKOFF_MARKER: &str = ".download_backoff";

/// Maximum backoff between download attempts (across restarts).
const MAX_RESTART_BACKOFF_SECS: u64 = 600; // 10 minutes

/// Ensure the model files for `manifest` are present, downloading from
/// Zenodo if necessary.
///
/// `variant` is the selected variant name (e.g. "fp16", "fp32", "int8").
///
/// This function:
/// 1. Applies variant overrides to the manifest (tflite_file, labels_file, etc.)
/// 2. Checks if the model file already exists on disk
/// 3. If missing, downloads the variant's zip from Zenodo and extracts it
pub fn ensure_model_files(manifest: &mut ResolvedManifest, variant: &str) -> Result<()> {
    // Apply variant overrides first
    manifest.apply_variant(variant)?;

    let download = match &manifest.manifest.download {
        Some(d) => d,
        None => return Ok(()),
    };

    let variant_info = &download.variants[variant];

    // Check if the primary model file already exists
    let tflite_path = manifest.tflite_path();
    if tflite_path.exists() {
        info!(
            "Model file already present: {} (variant={})",
            tflite_path.display(),
            variant
        );
        // Clear any leftover backoff marker on success
        clear_backoff_marker(&manifest.base_dir);
        return Ok(());
    }

    // ── honour backoff from a previous failed attempt ────────────────
    wait_for_backoff(&manifest.base_dir);

    info!(
        "Model file not found at {}, downloading variant '{}' from Zenodo record {}…",
        tflite_path.display(),
        variant,
        download.zenodo_record_id
    );

    let url = format!(
        "{}/{}/files/{}/content",
        ZENODO_FILES_URL, download.zenodo_record_id, variant_info.zenodo_file
    );

    if let Err(e) = download_and_extract(&url, &manifest.base_dir, variant_info.md5.as_deref()) {
        write_backoff_marker(&manifest.base_dir);
        return Err(e);
    }

    // Verify the model file now exists
    if !tflite_path.exists() {
        // List files in `base_dir` to help the user diagnose the mismatch.
        let available: Vec<String> = std::fs::read_dir(&manifest.base_dir)
            .ok()
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .map(|e| e.file_name().to_string_lossy().to_string())
                    .collect()
            })
            .unwrap_or_default();

        write_backoff_marker(&manifest.base_dir);
        anyhow::bail!(
            "Downloaded and extracted '{}' but expected model file not found: {}.\n\
             Files present in {}: {:?}\n\
             Check that [model].tflite_file (or the variant override) matches \
             a file inside the Zenodo zip.",
            variant_info.zenodo_file,
            tflite_path.display(),
            manifest.base_dir.display(),
            available
        );
    }

    info!(
        "Model download complete: {} (variant={})",
        tflite_path.display(),
        variant
    );
    clear_backoff_marker(&manifest.base_dir);
    Ok(())
}

// ── ONNX auto-conversion ─────────────────────────────────────────────────

/// If the manifest declares an `onnx_file` and the ONNX model does not
/// already exist, attempt to convert the model to ONNX.
///
/// **Preferred path** (BirdNET V2.4+): download the Keras `.h5` model from
/// Zenodo and convert the *classifier sub-model* (without the RFFT-based
/// mel spectrogram layers) using `scripts/convert_keras_to_onnx.py`.
/// The mel preprocessing is handled at runtime in Rust (`mel.rs`).
///
/// **Fallback**: if no `keras_zenodo_file` is configured, attempt a direct
/// TFLite → ONNX conversion via `tf2onnx` (works for simple models but
/// fails on BirdNET V2.4 due to unsupported RFFT/SPLIT_V ops).
///
/// This is best-effort: when Python or the required packages are not
/// installed the function logs a warning and returns `Ok(())` so the server
/// can still attempt to fall back to TFLite.
pub fn ensure_onnx_file(manifest: &ResolvedManifest) -> Result<()> {
    let onnx_path = match manifest.onnx_path() {
        Some(p) => p,
        None => return Ok(()), // no onnx_file configured
    };

    if onnx_path.exists() {
        info!("ONNX model already present: {}", onnx_path.display());
        return Ok(());
    }

    // ── Keras-based conversion (preferred) ───────────────────────────
    if let Some(download) = &manifest.manifest.download {
        if let Some(keras_file) = &download.keras_zenodo_file {
            return convert_keras_to_onnx(
                manifest,
                &download.zenodo_record_id,
                keras_file,
                download.keras_md5.as_deref(),
                &onnx_path,
            );
        }
    }

    // ── Fallback: direct TFLite → ONNX via tf2onnx CLI ──────────────
    convert_tflite_to_onnx(manifest, &onnx_path)
}

/// Download the Keras model from Zenodo, extract it, and convert the
/// classifier sub-model to ONNX.
fn convert_keras_to_onnx(
    manifest: &ResolvedManifest,
    zenodo_record_id: &str,
    keras_zenodo_file: &str,
    keras_md5: Option<&str>,
    onnx_path: &Path,
) -> Result<()> {
    // Download the Keras zip into a temporary subdirectory.
    let keras_dir = manifest.base_dir.join(".keras_tmp");
    std::fs::create_dir_all(&keras_dir)
        .with_context(|| format!("Cannot create {}", keras_dir.display()))?;

    let h5_path = keras_dir.join("audio-model.h5");
    if !h5_path.exists() {
        let url = format!(
            "{}/{}/files/{}/content",
            ZENODO_FILES_URL, zenodo_record_id, keras_zenodo_file
        );
        info!(
            "Downloading Keras model for ONNX conversion: {} → {}",
            keras_zenodo_file,
            keras_dir.display()
        );
        download_and_extract(&url, &keras_dir, keras_md5)?;

        if !h5_path.exists() {
            let available: Vec<String> = std::fs::read_dir(&keras_dir)
                .ok()
                .map(|entries| {
                    entries
                        .filter_map(|e| e.ok())
                        .map(|e| e.file_name().to_string_lossy().to_string())
                        .collect()
                })
                .unwrap_or_default();
            anyhow::bail!(
                "Downloaded Keras zip but audio-model.h5 not found.\n\
                 Files present: {:?}",
                available
            );
        }
    }

    // Determine the ONNX output filename.
    let onnx_filename = onnx_path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    info!(
        "Converting Keras classifier → ONNX: {} → {}",
        h5_path.display(),
        onnx_path.display()
    );

    // Find the conversion script (repo location or container install path).
    let script = ["scripts/convert_keras_to_onnx.py",
                   "/usr/local/share/gaia/scripts/convert_keras_to_onnx.py"]
        .iter()
        .find(|p| Path::new(p).exists())
        .copied()
        .unwrap_or("scripts/convert_keras_to_onnx.py");

    let output = std::process::Command::new("python3")
        .args([
            script,
            &keras_dir.to_string_lossy(),
            "-o",
            &onnx_filename,
        ])
        .output();

    match output {
        Ok(result) if result.status.success() => {
            // The script writes the ONNX into keras_dir — move it to the
            // final location alongside the other model files.
            let generated = keras_dir.join(&onnx_filename);
            if generated.exists() && generated != onnx_path {
                std::fs::rename(&generated, onnx_path).with_context(|| {
                    format!(
                        "Cannot move {} → {}",
                        generated.display(),
                        onnx_path.display()
                    )
                })?;
            }
            let size = std::fs::metadata(onnx_path)
                .map(|m| m.len())
                .unwrap_or(0);
            info!(
                "Keras → ONNX conversion complete: {} ({:.1} MB)",
                onnx_path.display(),
                size as f64 / 1_048_576.0
            );
            // Clean up temporary Keras files.
            let _ = std::fs::remove_dir_all(&keras_dir);
            Ok(())
        }
        Ok(result) => {
            let stderr = String::from_utf8_lossy(&result.stderr);
            let stdout = String::from_utf8_lossy(&result.stdout);
            warn!(
                "Keras → ONNX conversion failed (exit {}):\n{}\n{}",
                result.status,
                stdout.lines().take(10).collect::<Vec<_>>().join("\n"),
                stderr.lines().take(10).collect::<Vec<_>>().join("\n"),
            );
            warn!("The server will attempt to load the TFLite model directly");
            Ok(())
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            warn!(
                "python3 not found — cannot convert Keras model to ONNX.\n\
                 Install Python 3 with tf_keras and tf2onnx, or convert manually:\n\
                 python3 scripts/convert_keras_to_onnx.py {}",
                keras_dir.display()
            );
            Ok(())
        }
        Err(e) => {
            warn!("Failed to run Keras → ONNX conversion: {e}");
            Ok(())
        }
    }
}

/// Attempt a direct TFLite → ONNX conversion via tf2onnx CLI.
/// (Fallback for simple models; fails on BirdNET V2.4 due to RFFT ops.)
fn convert_tflite_to_onnx(manifest: &ResolvedManifest, onnx_path: &Path) -> Result<()> {
    let tflite_path = manifest.tflite_path();
    if !tflite_path.exists() {
        warn!(
            "Cannot convert to ONNX: TFLite source not found at {}",
            tflite_path.display()
        );
        return Ok(());
    }

    info!(
        "Converting TFLite → ONNX: {} → {}",
        tflite_path.display(),
        onnx_path.display()
    );

    let output = std::process::Command::new("python3")
        .args([
            "-m",
            "tf2onnx.convert",
            "--tflite",
            &tflite_path.to_string_lossy(),
            "--output",
            &onnx_path.to_string_lossy(),
        ])
        .output();

    match output {
        Ok(result) if result.status.success() => {
            let size = std::fs::metadata(onnx_path)
                .map(|m| m.len())
                .unwrap_or(0);
            info!(
                "ONNX conversion complete: {} ({:.1} MB)",
                onnx_path.display(),
                size as f64 / 1_048_576.0
            );
            Ok(())
        }
        Ok(result) => {
            let stderr = String::from_utf8_lossy(&result.stderr);
            warn!(
                "tf2onnx conversion failed (exit {}): {}",
                result.status,
                stderr.lines().take(10).collect::<Vec<_>>().join("\n")
            );
            warn!("The server will attempt to load the TFLite model directly");
            Ok(())
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            warn!(
                "python3 not found – cannot auto-convert TFLite to ONNX. \
                 Install Python 3 and tf2onnx (`pip install tf2onnx`) to enable \
                 automatic conversion, or convert manually with: \
                 python scripts/convert_tflite_to_onnx.py {}",
                tflite_path.display()
            );
            Ok(())
        }
        Err(e) => {
            warn!("Failed to run tf2onnx conversion: {e}");
            Ok(())
        }
    }
}

// ── Backoff marker helpers ───────────────────────────────────────────────────
//
// When a download or post-download check fails the container will be
// restarted by the compose `restart: unless-stopped` policy.  Without a
// backoff the new instance would immediately re-download and fail in a
// tight loop, hammering the Zenodo server.
//
// We persist a small marker file containing the earliest timestamp (as
// seconds since UNIX epoch) the next attempt should run at, plus the
// current backoff duration.  Each failure doubles the backoff up to
// `MAX_RESTART_BACKOFF_SECS`.

fn backoff_marker_path(base_dir: &Path) -> std::path::PathBuf {
    base_dir.join(BACKOFF_MARKER)
}

/// Read the marker and sleep until the recorded deadline, if any.
fn wait_for_backoff(base_dir: &Path) {
    let path = backoff_marker_path(base_dir);
    let content = match std::fs::read_to_string(&path) {
        Ok(c) => c,
        Err(_) => return, // no marker → no wait
    };

    // Format: "resume_epoch_secs backoff_secs"
    let mut parts = content.split_whitespace();
    let resume_at: u64 = parts
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let now = std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    if now < resume_at {
        let wait = resume_at - now;
        warn!(
            "Previous download attempt failed — backing off for {}s before retrying",
            wait
        );
        std::thread::sleep(std::time::Duration::from_secs(wait));
    }
}

/// Write (or update) the marker, doubling the backoff each time.
fn write_backoff_marker(base_dir: &Path) {
    let path = backoff_marker_path(base_dir);

    // Read the previous backoff duration, if any, and double it.
    let prev_backoff: u64 = std::fs::read_to_string(&path)
        .ok()
        .and_then(|c| c.split_whitespace().nth(1)?.parse().ok())
        .unwrap_or(0);

    let next_backoff = if prev_backoff == 0 {
        INITIAL_BACKOFF.as_secs()
    } else {
        (prev_backoff * 2).min(MAX_RESTART_BACKOFF_SECS)
    };

    let now = std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let resume_at = now + next_backoff;

    warn!(
        "Writing download backoff marker: next attempt in {}s (at epoch {})",
        next_backoff, resume_at
    );

    let _ = std::fs::write(&path, format!("{resume_at} {next_backoff}"));
}

/// Remove the backoff marker (called on success).
fn clear_backoff_marker(base_dir: &Path) {
    let _ = std::fs::remove_file(backoff_marker_path(base_dir));
}

/// Maximum number of download attempts before giving up.
const MAX_RETRIES: u32 = 5;

/// Initial backoff delay between retries.
const INITIAL_BACKOFF: std::time::Duration = std::time::Duration::from_secs(5);

/// User-Agent sent to Zenodo (they block requests without one).
const USER_AGENT: &str = "gaia-processing/0.1";

/// Build the shared HTTP client used for Zenodo downloads.
fn build_client() -> Result<reqwest::blocking::Client> {
    reqwest::blocking::Client::builder()
        .user_agent(USER_AGENT)
        .timeout(std::time::Duration::from_secs(600)) // 10 min for large models
        .build()
        .context("Cannot build HTTP client")
}

/// Download a zip from `url`, verify MD5, and extract into `dest_dir`.
///
/// The download is resumable: data is streamed to a `.part` file and, on
/// failure, subsequent retries use an HTTP `Range` header to continue where
/// they left off instead of starting from scratch.  Retries use exponential
/// backoff to avoid overloading the server.
fn download_and_extract(url: &str, dest_dir: &Path, expected_md5: Option<&str>) -> Result<()> {
    std::fs::create_dir_all(dest_dir)
        .with_context(|| format!("Cannot create model directory: {}", dest_dir.display()))?;

    // Use a .part file so incomplete downloads are obvious and resumable.
    let part_path = dest_dir.join(".download.part");

    let client = build_client()?;

    download_with_resume(&client, url, &part_path)?;

    // Read the completed file and verify MD5.
    let bytes =
        std::fs::read(&part_path).with_context(|| format!("Cannot read {}", part_path.display()))?;

    info!("Downloaded {:.1} MB", bytes.len() as f64 / 1_048_576.0);

    if let Some(expected) = expected_md5 {
        let digest = format!("{:x}", md5::compute(&bytes));
        if digest != expected {
            // Remove the corrupt partial file so the next run starts fresh.
            let _ = std::fs::remove_file(&part_path);
            anyhow::bail!(
                "MD5 checksum mismatch: expected {}, got {}. \
                 The download may be corrupted.",
                expected,
                digest
            );
        }
        info!("MD5 checksum verified ✓");
    }

    // Extract zip
    extract_zip(&bytes, dest_dir)?;

    // Clean up the .part file after successful extraction.
    let _ = std::fs::remove_file(&part_path);

    Ok(())
}

/// Download `url` into `part_path`, resuming from where a previous attempt
/// left off.  Retries up to [`MAX_RETRIES`] times with exponential backoff.
fn download_with_resume(
    client: &reqwest::blocking::Client,
    url: &str,
    part_path: &Path,
) -> Result<()> {
    let mut backoff = INITIAL_BACKOFF;

    for attempt in 1..=MAX_RETRIES {
        // How many bytes we already have on disk.
        let existing_len = part_path.metadata().map(|m| m.len()).unwrap_or(0);

        info!(
            "GET {} (attempt {}/{}, resume from {} bytes)",
            url, attempt, MAX_RETRIES, existing_len
        );

        let mut request = client.get(url);
        if existing_len > 0 {
            request = request.header(reqwest::header::RANGE, format!("bytes={}-", existing_len));
        }

        let response = match request.send() {
            Ok(r) => r,
            Err(e) => {
                warn!(
                    "Download request failed (attempt {}/{}): {}",
                    attempt, MAX_RETRIES, e
                );
                if attempt == MAX_RETRIES {
                    return Err(e).with_context(|| format!("Failed to download after {MAX_RETRIES} attempts: {url}"));
                }
                info!("Retrying in {:?}…", backoff);
                std::thread::sleep(backoff);
                backoff *= 2;
                continue;
            }
        };

        let status = response.status();

        // 416 Range Not Satisfiable → the server says we already have the
        // full file (existing_len >= content length).  Treat as success.
        if status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE && existing_len > 0 {
            info!("Server indicates the file is already fully downloaded");
            return Ok(());
        }

        if !status.is_success() && status != reqwest::StatusCode::PARTIAL_CONTENT {
            warn!(
                "Download failed HTTP {} (attempt {}/{})",
                status, attempt, MAX_RETRIES
            );
            if attempt == MAX_RETRIES {
                anyhow::bail!(
                    "Download failed (HTTP {}) after {} attempts: {}",
                    status,
                    MAX_RETRIES,
                    url
                );
            }
            info!("Retrying in {:?}…", backoff);
            std::thread::sleep(backoff);
            backoff *= 2;
            continue;
        }

        // If the server returned 200 (not 206), it doesn't support Range
        // requests – we must start from scratch.
        let append = status == reqwest::StatusCode::PARTIAL_CONTENT;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .append(append)
            .truncate(!append)
            .open(part_path)
            .with_context(|| format!("Cannot open {}", part_path.display()))?;

        // Stream the body in chunks instead of holding it all in memory.
        match stream_to_file(response, &mut file) {
            Ok(()) => return Ok(()),
            Err(e) => {
                warn!(
                    "Download stream interrupted (attempt {}/{}): {}",
                    attempt, MAX_RETRIES, e
                );
                if attempt == MAX_RETRIES {
                    return Err(e).context(format!(
                        "Download stream failed after {MAX_RETRIES} attempts: {url}"
                    ));
                }
                info!("Retrying in {:?}…", backoff);
                std::thread::sleep(backoff);
                backoff *= 2;
            }
        }
    }

    unreachable!()
}

/// Copy response body to `file`, chunk by chunk.
fn stream_to_file(
    response: reqwest::blocking::Response,
    file: &mut std::fs::File,
) -> Result<()> {
    use std::io::{Read, Write};

    let mut reader = response;
    let mut buf = vec![0u8; 256 * 1024]; // 256 KB chunks
    loop {
        let n = reader.read(&mut buf).context("Error reading response body")?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])
            .context("Error writing to .part file")?;
    }
    file.flush().context("Error flushing .part file")?;
    Ok(())
}

/// Extract a zip archive from `bytes` into `dest_dir`.
fn extract_zip(bytes: &[u8], dest_dir: &Path) -> Result<()> {
    let cursor = std::io::Cursor::new(bytes);
    let mut archive = zip::ZipArchive::new(cursor).context("Failed to open zip archive")?;

    let mut extracted = 0usize;
    for i in 0..archive.len() {
        let mut file = archive.by_index(i).context("Cannot read zip entry")?;
        let raw_name = file.name().to_string();

        // Skip directories, macOS resource forks, and hidden files
        if file.is_dir() || raw_name.starts_with("__MACOSX") || raw_name.contains("/._") {
            continue;
        }

        // Flatten: strip any leading directory components so files land
        // directly in dest_dir (Zenodo zips often wrap in a top-level dir).
        let file_name = match Path::new(&raw_name).file_name() {
            Some(f) => f.to_string_lossy().to_string(),
            None => {
                warn!("Skipping zip entry with no file name: {}", raw_name);
                continue;
            }
        };

        let out_path = dest_dir.join(&file_name);
        info!("  extracting: {} → {}", raw_name, out_path.display());

        let mut out_file = std::fs::File::create(&out_path)
            .with_context(|| format!("Cannot create {}", out_path.display()))?;

        std::io::copy(&mut file, &mut out_file)
            .with_context(|| format!("Cannot write {}", out_path.display()))?;

        extracted += 1;
    }

    info!("Extracted {} file(s) into {}", extracted, dest_dir.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zenodo_url_format() {
        let url = format!(
            "{}/{}/files/{}/content",
            ZENODO_FILES_URL, "15050749", "BirdNET_v2.4_tflite_fp16.zip"
        );
        assert_eq!(
            url,
            "https://zenodo.org/api/records/15050749/files/BirdNET_v2.4_tflite_fp16.zip/content"
        );
    }
}
