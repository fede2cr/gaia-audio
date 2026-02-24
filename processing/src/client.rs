//! HTTP client that polls capture servers for new WAV recordings.
//!
//! When mDNS discovery is available the processing node automatically
//! finds all capture nodes on the network.  Otherwise it falls back to
//! the single `CAPTURE_SERVER_URL` from `gaia.conf`.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::SyncSender;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use tracing::{debug, error, info, warn};

use gaia_common::config::Config;
use gaia_common::discovery::{DiscoveryHandle, ServiceRole};
use gaia_common::protocol::RecordingInfo;

use crate::analysis;
use crate::model::LoadedModel;
use crate::ReportPayload;

/// How often to re-scan mDNS for new/removed capture nodes.
const REDISCOVERY_INTERVAL: Duration = Duration::from_secs(60);

/// Poll all known capture servers for new recordings and process them.
///
/// If `discovery` is `Some`, capture nodes are located (and periodically
/// refreshed) via mDNS.  If mDNS finds no capture nodes the config's
/// `capture_server_url` is used as a fallback.
///
/// Blocks until `shutdown` is set.
pub fn poll_and_process(
    models: &[LoadedModel],
    config: &Config,
    discovery: Option<&DiscoveryHandle>,
    report_tx: &SyncSender<ReportPayload>,
    shutdown: &AtomicBool,
) -> Result<()> {
    let poll_interval = Duration::from_secs(config.poll_interval_secs);
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .context("Cannot create HTTP client")?;

    let tmp_dir = config.recs_dir.join("processing_tmp");
    std::fs::create_dir_all(&tmp_dir)?;

    // Track which files we've already processed this session.
    // Key = "base_url:filename" to avoid collisions across capture nodes.
    let mut processed: HashSet<String> = HashSet::new();

    // Build initial list of capture URLs
    let mut capture_urls = resolve_capture_urls(discovery, config);
    info!(
        "Polling {} capture server(s) every {}s: {:?}",
        capture_urls.len(),
        config.poll_interval_secs,
        capture_urls
    );

    // Quick reachability check at startup so operators see a clear
    // confirmation (or failure) in the logs immediately.
    for url in &capture_urls {
        match list_recordings(&client, url) {
            Ok(r) => info!("[{url}] Reachable – {} recording(s) queued", r.len()),
            Err(e) => warn!("[{url}] Not reachable at startup: {e:#}"),
        }
    }

    let mut last_discovery = Instant::now();

    loop {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }

        // Without models we cannot analyse anything.
        if models.is_empty() {
            warn!("No models loaded – skipping poll cycle");
            std::thread::sleep(poll_interval);
            continue;
        }

        // ── periodic mDNS re-discovery ───────────────────────────────
        if last_discovery.elapsed() >= REDISCOVERY_INTERVAL {
            let new_urls = resolve_capture_urls(discovery, config);
            if new_urls != capture_urls {
                info!("Capture node list updated: {:?}", new_urls);
                capture_urls = new_urls;
            }
            last_discovery = Instant::now();
        }

        // ── poll each capture server ─────────────────────────────────
        let mut found_any = false;

        for base_url in &capture_urls {
            if shutdown.load(Ordering::Relaxed) {
                break;
            }

            let recordings = match list_recordings(&client, base_url) {
                Ok(r) => r,
                Err(e) => {
                    warn!("Cannot reach capture server {}: {e}", base_url);
                    continue;
                }
            };

            if recordings.is_empty() {
                continue;
            }

            found_any = true;
            info!(
                "[{}] Found {} recording(s) to process",
                base_url,
                recordings.len()
            );

            for rec in &recordings {
                if shutdown.load(Ordering::Relaxed) {
                    break;
                }

                let key = format!("{}:{}", base_url, rec.filename);
                if processed.contains(&key) {
                    continue;
                }

                debug!(
                    "[{}] New recording: {} ({} bytes)",
                    base_url, rec.filename, rec.size
                );

                // ── download ─────────────────────────────────────────
                let local_path = tmp_dir.join(&rec.filename);
                match download_recording(&client, base_url, &rec.filename, &local_path) {
                    Ok(()) => {}
                    Err(e) => {
                        error!("Failed to download {}: {e}", rec.filename);
                        continue;
                    }
                }

                // ── process ──────────────────────────────────────────
                if let Err(e) =
                    analysis::process_file(&local_path, models, config, report_tx)
                {
                    error!("Error processing {}: {e:#}", rec.filename);
                }

                // ── clean up local temp file ─────────────────────────
                std::fs::remove_file(&local_path).ok();

                // ── ask capture server to delete ─────────────────────
                if let Err(e) = delete_recording(&client, base_url, &rec.filename) {
                    warn!(
                        "Failed to delete {} from {}: {e}",
                        rec.filename, base_url
                    );
                }

                processed.insert(key);
            }
        }

        if !found_any {
            debug!("No recordings on any capture node – sleeping");
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

/// Resolve the list of capture server URLs.
///
/// Tries mDNS first (with a retry); falls back to the config value when
/// mDNS is unavailable or discovers no capture nodes.
fn resolve_capture_urls(
    discovery: Option<&DiscoveryHandle>,
    config: &Config,
) -> Vec<String> {
    if let Some(dh) = discovery {
        // Try twice: the first scan may miss the capture node if it
        // registered just moments before us and the mDNS cache hasn't
        // propagated yet.
        for attempt in 1..=2 {
            let timeout = if attempt == 1 { 5 } else { 3 };
            let peers = dh.discover_peers(
                ServiceRole::Capture,
                Duration::from_secs(timeout),
            );
            if !peers.is_empty() {
                let urls: Vec<String> = peers
                    .iter()
                    .filter_map(|p| p.http_url())
                    .collect();
                info!("mDNS discovered {} capture node(s): {:?}", urls.len(), urls);
                return urls;
            }
            if attempt == 1 {
                debug!("mDNS scan {attempt}: no peers yet, retrying…");
            }
        }
        info!("No capture nodes found via mDNS, falling back to config URL");
    }
    vec![config.capture_server_url.clone()]
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
