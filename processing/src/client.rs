//! HTTP client that polls capture servers for new recordings (WAV/Opus).
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

use crate::WorkItem;

/// How often to re-scan mDNS for new/removed capture nodes.
const REDISCOVERY_INTERVAL: Duration = Duration::from_secs(60);

/// Poll all known capture servers for new recordings, download them,
/// and dispatch work items to the worker pool.
///
/// This function only handles downloading and dispatching — the actual
/// analysis is performed by worker threads that receive `WorkItem`s via
/// the `work_tx` channel.
///
/// Blocks until `shutdown` is set.
pub fn poll_and_dispatch(
    config: &mut Config,
    discovery: Option<&DiscoveryHandle>,
    work_tx: &SyncSender<WorkItem>,
    shutdown: &AtomicBool,
) -> Result<()> {
    let poll_interval = Duration::from_secs(config.poll_interval_secs);
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .context("Cannot create HTTP client")?;

    let instance_suffix = if config.processing_instance.is_empty() {
        "processing_tmp".to_string()
    } else {
        format!("processing_tmp_{}", config.processing_instance)
    };
    let tmp_dir = config.recs_dir.join(&instance_suffix);
    std::fs::create_dir_all(&tmp_dir)?;

    // Track which files we've already dispatched this session.
    // Key = "base_url:filename" to avoid collisions across capture nodes.
    let mut dispatched: HashSet<String> = HashSet::new();

    // Track how many NEW items we actually dispatched per iteration so we
    // can distinguish "new work to do" from "recordings on disk but
    // already dispatched".
    let mut dispatched_this_round: usize;

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
    // Prune processing instances that haven't heartbeated in a while.
    let pruned = crate::kv::prune_stale_instances(10);
    if pruned > 0 {
        info!("Pruned {pruned} stale processing instance(s) from previous runs");
    }
    let mut last_discovery = Instant::now();

    loop {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }

        // ── refresh settings from DB ─────────────────────────────
        crate::kv::apply_settings_overrides(config);

        // ── heartbeat so coordination layer knows we're alive ────
        crate::kv::update_heartbeat("default");

        // ── idle when no models are enabled ──────────────────────
        // The container stays running but does not download or
        // process files until at least one model is activated in
        // Settings.  The set is checked every poll interval.
        let enabled_models = crate::kv::get_enabled_models_state();
        {
            static IDLE_LOGGED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if matches!(enabled_models.as_ref(), Some(v) if v.is_empty()) {
                if !IDLE_LOGGED.load(Ordering::Relaxed) {
                    info!("No models enabled — idling (container stays running)");
                    IDLE_LOGGED.store(true, Ordering::Relaxed);
                }
                std::thread::sleep(poll_interval);
                continue;
            } else if IDLE_LOGGED.swap(false, Ordering::Relaxed) {
                info!(
                    "Models re-enabled ({} active) — resuming polling",
                    enabled_models.as_ref().map_or(0, |v| v.len())
                );
            }
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
        dispatched_this_round = 0;

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

            debug!(
                "[{}] Found {} recording(s) to process",
                base_url,
                recordings.len()
            );

            for rec in &recordings {
                if shutdown.load(Ordering::Relaxed) {
                    break;
                }

                let key = format!("{}:{}", base_url, rec.filename);
                if dispatched.contains(&key) {
                    continue;
                }

                // Skip files this instance already processed in a previous
                // run.  The dispatched HashSet is in-memory only and lost on
                // restart, but the Redis processed set persists (with TTL).
                if crate::kv::is_file_processed(&rec.filename, "default") {
                    debug!(
                        "[{}] Skipping {} — already processed by this instance",
                        base_url, rec.filename
                    );
                    dispatched.insert(key);
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

                // ── dispatch to worker pool ──────────────────────────
                let item = WorkItem {
                    local_path,
                    filename: rec.filename.clone(),
                    base_url: base_url.clone(),
                    config_snapshot: config.clone(),
                };
                if work_tx.send(item).is_err() {
                    warn!("Work channel closed — stopping dispatch");
                    return Ok(());
                }

                dispatched.insert(key);
                dispatched_this_round += 1;
            }
        }

        if dispatched_this_round > 0 {
            info!(
                "Dispatched {dispatched_this_round} new recording(s) for processing"
            );
        }

        // Prevent unbounded growth of the dispatched set
        if dispatched.len() > 10_000 {
            dispatched.clear();
            crate::kv::cleanup_processing_log();
            crate::kv::prune_stale_instances(10);
        }

        // Only skip the sleep when we actually dispatched new work
        // this round — there may be more files arriving soon.  When
        // recordings exist on the capture node but have already been
        // dispatched, sleeping prevents a busy-loop that would spam
        // the logs and waste CPU.
        if dispatched_this_round == 0 {
            debug!("No new recordings to dispatch – sleeping {poll_interval:?}");
            std::thread::sleep(poll_interval);
        }
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
    debug!(
        "[{base_url}] GET /api/recordings → {} file(s)",
        recordings.len()
    );
    Ok(recordings)
}

fn download_recording(
    client: &reqwest::blocking::Client,
    base_url: &str,
    filename: &str,
    out_path: &Path,
) -> Result<()> {
    let url = format!("{base_url}/api/recordings/{filename}");
    let t0 = Instant::now();
    let resp = client.get(&url).send().context("GET recording")?;

    if !resp.status().is_success() {
        anyhow::bail!("GET {} returned {}", url, resp.status());
    }

    let bytes = resp.bytes()?;
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(out_path, &bytes)?;
    let elapsed = t0.elapsed();
    let size_mb = bytes.len() as f64 / 1_048_576.0;
    let rate = if elapsed.as_secs_f64() > 0.0 {
        size_mb / elapsed.as_secs_f64()
    } else {
        0.0
    };
    debug!(
        "Downloaded {} → {} ({:.2} MB in {:.1}s, {:.1} MB/s)",
        filename,
        out_path.display(),
        size_mb,
        elapsed.as_secs_f64(),
        rate
    );
    info!("Downloaded {} → {}", filename, out_path.display());
    Ok(())
}

#[allow(dead_code)]
fn delete_recording(
    client: &reqwest::blocking::Client,
    base_url: &str,
    filename: &str,
) -> Result<()> {
    let url = format!("{base_url}/api/recordings/{filename}");
    let resp = client.delete(&url).send().context("DELETE recording")?;

    if resp.status().is_success() || resp.status() == reqwest::StatusCode::NOT_FOUND {
        debug!(
            "DELETE {filename} from capture server → {} ({})",
            resp.status(),
            if resp.status().is_success() { "removed" } else { "already gone" }
        );
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
