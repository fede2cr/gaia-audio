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

/// Download a zip from `url`, verify MD5, and extract into `dest_dir`.
fn download_and_extract(url: &str, dest_dir: &Path, expected_md5: Option<&str>) -> Result<()> {
    info!("GET {}", url);

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(600)) // 10 min for large models
        .build()
        .context("Cannot build HTTP client")?;

    let response = client
        .get(url)
        .send()
        .with_context(|| format!("Failed to download: {url}"))?;

    if !response.status().is_success() {
        anyhow::bail!("Download failed (HTTP {}): {}", response.status(), url);
    }

    let bytes = response
        .bytes()
        .context("Failed to read download response body")?;

    info!("Downloaded {:.1} MB", bytes.len() as f64 / 1_048_576.0);

    // Verify MD5 if provided
    if let Some(expected) = expected_md5 {
        let digest = format!("{:x}", md5::compute(&bytes));
        if digest != expected {
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
    let cursor = std::io::Cursor::new(&bytes);
    let mut archive = zip::ZipArchive::new(cursor).context("Failed to open zip archive")?;

    std::fs::create_dir_all(dest_dir)
        .with_context(|| format!("Cannot create model directory: {}", dest_dir.display()))?;

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
