//! HTTP client that polls the capture server for new WAV recordings.
//!
//! Replaces the inotify-based filesystem watcher from `birdnet-server`.
//! This enables the processing server to run on a different host/container
//! from the capture server.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::SyncSender;
use std::time::Duration;

use anyhow::{Context, Result};
use tracing::{debug, error, info, warn};

use gaia_common::config::Config;
use gaia_common::protocol::RecordingInfo;

use crate::analysis;
use crate::model::LoadedModel;
use crate::ReportPayload;

/// Poll the capture server for new recordings and process them.
///
/// Blocks until `shutdown` is set.
pub fn poll_and_process(
    models: &[LoadedModel],
    config: &Config,
    report_tx: &SyncSender<ReportPayload>,
    shutdown: &AtomicBool,
) -> Result<()> {
    let base_url = &config.capture_server_url;
    let poll_interval = Duration::from_secs(config.poll_interval_secs);
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .context("Cannot create HTTP client")?;

    let tmp_dir = config.recs_dir.join("processing_tmp");
    std::fs::create_dir_all(&tmp_dir)?;

    // Track which files we've already processed this session to avoid
    // re-downloading when the capture server hasn't deleted them yet.
    let mut processed: HashSet<String> = HashSet::new();

    info!(
        "Polling capture server at {} every {}s",
        base_url, config.poll_interval_secs
    );

    loop {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }

        // Without models we cannot analyse anything.  Sleep and retry so
        // the capture server keeps buffering recordings.
        if models.is_empty() {
            warn!("No models loaded – skipping poll cycle");
            std::thread::sleep(poll_interval);
            continue;
        }

        // ── list available recordings ────────────────────────────────
        let recordings = match list_recordings(&client, base_url) {
            Ok(r) => r,
            Err(e) => {
                warn!("Cannot reach capture server: {e}");
                std::thread::sleep(poll_interval);
                continue;
            }
        };

        if recordings.is_empty() {
            debug!("Recording queue empty – nothing to analyse, sleeping");
            std::thread::sleep(poll_interval);
            continue;
        }

        info!("Found {} new recording(s) to process", recordings.len());

        for rec in &recordings {
            if shutdown.load(Ordering::Relaxed) {
                break;
            }

            if processed.contains(&rec.filename) {
                continue;
            }

            debug!("New recording: {} ({} bytes)", rec.filename, rec.size);

            // ── download ─────────────────────────────────────────────
            let local_path = tmp_dir.join(&rec.filename);
            match download_recording(&client, base_url, &rec.filename, &local_path) {
                Ok(()) => {}
                Err(e) => {
                    error!("Failed to download {}: {e}", rec.filename);
                    continue;
                }
            }

            // ── process ──────────────────────────────────────────────
            if let Err(e) =
                analysis::process_file(&local_path, models, config, report_tx)
            {
                error!("Error processing {}: {e:#}", rec.filename);
            }

            // ── clean up local temp file ─────────────────────────────
            std::fs::remove_file(&local_path).ok();

            // ── ask capture server to delete ─────────────────────────
            if let Err(e) = delete_recording(&client, base_url, &rec.filename) {
                warn!("Failed to delete {} from capture server: {e}", rec.filename);
            }

            processed.insert(rec.filename.clone());
        }

        // Prevent unbounded growth of the processed set
        if processed.len() > 10_000 {
            processed.clear();
        }

        std::thread::sleep(poll_interval);
    }

    info!("Polling loop stopped");
    Ok(())
}

// ── HTTP helpers ─────────────────────────────────────────────────────────

fn list_recordings(
    client: &reqwest::blocking::Client,
    base_url: &str,
) -> Result<Vec<RecordingInfo>> {
    let url = format!("{base_url}/api/recordings");
    let resp = client.get(&url).send().context("GET /api/recordings")?;

    if !resp.status().is_success() {
        anyhow::bail!("GET /api/recordings returned {}", resp.status());
    }

    let recordings: Vec<RecordingInfo> = resp.json().context("Parse recordings JSON")?;
    Ok(recordings)
}

fn download_recording(
    client: &reqwest::blocking::Client,
    base_url: &str,
    filename: &str,
    out_path: &Path,
) -> Result<()> {
    let url = format!("{base_url}/api/recordings/{filename}");
    let resp = client.get(&url).send().context("GET recording")?;

    if !resp.status().is_success() {
        anyhow::bail!("GET {} returned {}", url, resp.status());
    }

    let bytes = resp.bytes()?;
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(out_path, &bytes)?;
    info!("Downloaded {} → {}", filename, out_path.display());
    Ok(())
}

fn delete_recording(
    client: &reqwest::blocking::Client,
    base_url: &str,
    filename: &str,
) -> Result<()> {
    let url = format!("{base_url}/api/recordings/{filename}");
    let resp = client.delete(&url).send().context("DELETE recording")?;

    if resp.status().is_success() || resp.status() == reqwest::StatusCode::NOT_FOUND {
        debug!("Deleted {filename} from capture server");
        Ok(())
    } else {
        anyhow::bail!("DELETE {} returned {}", url, resp.status())
    }
}

/// Download a specific recording to a local path. Utility for one-shot use.
#[allow(dead_code)]
pub fn fetch_recording(
    config: &Config,
    filename: &str,
    out_path: &PathBuf,
) -> Result<()> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()?;
    download_recording(&client, &config.capture_server_url, filename, out_path)
}
