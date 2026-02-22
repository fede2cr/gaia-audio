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
        return Ok(());
    }

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

    download_and_extract(&url, &manifest.base_dir, variant_info.md5.as_deref())?;

    // Verify the model file now exists
    if !tflite_path.exists() {
        anyhow::bail!(
            "Downloaded and extracted '{}' but expected model file not found: {}.\n\
             Check that [model].tflite_file (or the variant override) matches \
             a file inside the Zenodo zip.",
            variant_info.zenodo_file,
            tflite_path.display()
        );
    }

    info!(
        "Model download complete: {} (variant={})",
        tflite_path.display(),
        variant
    );
    Ok(())
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
