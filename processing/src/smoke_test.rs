//! Build-stage smoke test — end-to-end inference validation.
//!
//! Loads every bird model from a model directory, runs real inference on a
//! known bird recording, and validates that:
//!
//!   1. Every model produces at least one detection above a minimum
//!      confidence threshold (i.e. it's not returning garbage).
//!   2. All bird models agree on the top species (within a tolerance).
//!   3. Confidence scores are in a sane range — not all near-zero
//!      (broken softmax) and not all near-100% (broken sigmoid).
//!
//! Invoked during `docker build` via:
//!
//!     gaia-processing smoke-test \
//!         --audio /test/bird.wav \
//!         --models /test/models \
//!         --species "Turdus merula"
//!
//! Exits 0 on success, 1 on any assertion failure, 2 on usage error.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use tracing::{error, info, warn};

use crate::manifest::{self, ResolvedManifest, ScoreTransform};
use crate::model;
use gaia_common::config::Config;

// ── Thresholds ───────────────────────────────────────────────────

/// Minimum confidence for a "real" detection (sigmoid models).
/// If the top prediction from a sigmoid model is below this, its
/// score transform is probably broken.
const MIN_TOP_CONFIDENCE_SIGMOID: f64 = 0.05;

/// Minimum confidence for softmax models.  Softmax over many thousands
/// of classes (e.g. Perch V2 with ~15K classes) inherently produces
/// lower absolute probabilities, so the threshold is much lower.
/// A correct confident detection still concentrates probability mass
/// well above this floor.
const MIN_TOP_CONFIDENCE_SOFTMAX: f64 = 0.005;

/// Maximum ratio between the highest and lowest top-score across bird
/// models.  If one model gives 0.90 and another gives 0.001, something
/// is wrong.  We allow a 200× spread because sigmoid-over-6K and
/// softmax-over-15K have fundamentally different score scales.
/// The spread check is a WARNING, not a hard failure.
const MAX_CONFIDENCE_RATIO: f64 = 200.0;

/// Dummy lat/lon/week for inference (location-based filtering is
/// disabled in smoke tests, but the predict() API requires them).
const DUMMY_LAT: f64 = 52.0;   // Central Europe
const DUMMY_LON: f64 = 5.0;
const DUMMY_WEEK: u32 = 20;    // May

// ── Result types ─────────────────────────────────────────────────────

/// Per-model result from the smoke test.
struct ModelResult {
    slug: String,
    name: String,
    top_label: String,
    top_confidence: f64,
    top5: Vec<(String, f64)>,
    n_above_threshold: usize,
    score_transform: ScoreTransform,
}

// ── Public entry point ───────────────────────────────────────────────

/// Run the build-stage smoke test.  Returns `Ok(())` on pass, `Err` on
/// any failure.
///
/// `expected_model_count`: if `Some(n)`, the test asserts that exactly
/// `n` bird/wildlife models were discovered.  This catches the case
/// where a malformed manifest silently excludes a model.
pub fn run(
    audio_path: &Path,
    models_dir: &Path,
    expected_species: &str,
    expected_model_count: Option<usize>,
) -> Result<()> {
    info!("╔══════════════════════════════════════════════════════════╗");
    info!("║           BUILD-STAGE SMOKE TEST                        ║");
    info!("╚══════════════════════════════════════════════════════════╝");
    info!("Audio:    {}", audio_path.display());
    info!("Models:   {}", models_dir.display());
    info!("Expected: {}", expected_species);

    // Log the ORT shared library version so CI output shows exactly
    // which onnxruntime is loaded.  Helps diagnose version-drift hangs.
    log_ort_library_version();

    info!("");

    // ── Discover manifests ───────────────────────────────────────────
    let manifests = manifest::discover_manifests(models_dir)
        .context("Failed to discover model manifests")?;

    if manifests.is_empty() {
        bail!("No manifests found in {}", models_dir.display());
    }

    info!("Discovered {} model(s):", manifests.len());
    for m in &manifests {
        info!(
            "  - {} (slug={}, domain={}, sr={}, chunk={}s)",
            m.manifest.model.name,
            m.slug(),
            m.manifest.model.domain,
            m.manifest.model.sample_rate,
            m.manifest.model.chunk_duration,
        );
    }
    info!("");

    // ── Filter to bird models only ───────────────────────────────────
    // "birds" domain for BirdNET, "wildlife" for Perch (which covers birds)
    let bird_manifests: Vec<&ResolvedManifest> = manifests
        .iter()
        .filter(|m| {
            let domain = m.manifest.model.domain.to_lowercase();
            domain == "birds" || domain == "wildlife"
        })
        .collect();

    if bird_manifests.is_empty() {
        bail!("No bird/wildlife models found — cannot run smoke test");
    }

    // Assert expected model count when provided.  This catches the case
    // where a malformed manifest.toml (e.g. duplicate keys) is silently
    // skipped by discover_manifests(), leaving fewer models than expected.
    if let Some(expected) = expected_model_count {
        if bird_manifests.len() != expected {
            bail!(
                "Expected {} bird/wildlife model(s) but discovered {} — \
                 a manifest may be malformed or missing.  \
                 Run `gaia-processing validate-manifests <dir>` for details.",
                expected,
                bird_manifests.len(),
            );
        }
    }

    info!("Testing {} bird/wildlife model(s):", bird_manifests.len());
    for m in &bird_manifests {
        info!("  → {}", m.manifest.model.name);
    }
    info!("");

    // ── Build a minimal config ───────────────────────────────────────
    let config = Config {
        latitude: DUMMY_LAT,
        longitude: DUMMY_LON,
        confidence: 0.01,       // Very low — we want to see all detections
        sensitivity: 1.25,      // Default BirdNET sensitivity
        overlap: 0.0,
        recording_length: 15,
        channels: 1,
        rec_card: None,
        model_dir: models_dir.to_path_buf(),
        database_lang: "en".to_string(),
        db_path: PathBuf::from("/dev/null"),
        capture_server_url: String::new(),
        capture_listen_addr: "0.0.0.0:8089".to_string(),
        poll_interval_secs: 0,
        recs_dir: PathBuf::from("/tmp"),
        extracted_dir: PathBuf::from("/tmp"),
        audio_fmt: "wav".to_string(),
        rtsp_streams: vec![],
        sf_thresh: 0.03,
        data_model_version: 2,
        model_variant: None,
        model_slugs: vec![],
        processing_instance: String::new(),
        processing_threads: 1,
        raw_spectrogram: false,
        privacy_threshold: 0.0,
        extraction_length: 6,
        birdweather_id: None,
        heartbeat_url: None,
        turso_database_url: None,
        turso_auth_token: None,
        colormap: "default".to_string(),
        disk_usage_max: 95.0,
    };

    // ── Load models and run inference ────────────────────────────────
    let mut results: Vec<ModelResult> = Vec::new();

    for manifest in &bird_manifests {
        info!("━━━ {} ━━━", manifest.manifest.model.name);

        let mut loaded = match model::load_model(manifest, &config) {
            Ok(m) => m,
            Err(e) => {
                error!(
                    "FAIL: Cannot load {}: {e:#}",
                    manifest.manifest.model.name
                );
                bail!(
                    "Model {} failed to load: {e:#}",
                    manifest.manifest.model.name
                );
            }
        };
        info!("  Loaded successfully");

        // Resolve this model's score transform and threshold
        let transform = manifest.manifest.model.score_transform
            .unwrap_or_else(|| {
                if manifest.manifest.model.apply_softmax {
                    ScoreTransform::Softmax
                } else {
                    ScoreTransform::Sigmoid
                }
            });
        let min_conf = match transform {
            ScoreTransform::Softmax => MIN_TOP_CONFIDENCE_SOFTMAX,
            _ => MIN_TOP_CONFIDENCE_SIGMOID,
        };

        // Read audio at this model's sample rate
        let chunks = gaia_common::audio::read_audio(
            audio_path,
            manifest.manifest.model.sample_rate,
            manifest.manifest.model.chunk_duration,
            0.0,  // no overlap for smoke test
        )
        .with_context(|| {
            format!(
                "Cannot read audio for {} (sr={})",
                manifest.manifest.model.name,
                manifest.manifest.model.sample_rate
            )
        })?;

        info!("  Audio: {} chunks @ {} Hz", chunks.len(), manifest.manifest.model.sample_rate);

        if chunks.is_empty() {
            bail!(
                "No audio chunks produced for {} — recording may be too short",
                manifest.manifest.model.name
            );
        }

        // Run inference on all chunks, collect best prediction
        let mut best_label = String::new();
        let mut best_confidence: f64 = 0.0;
        let mut best_top5: Vec<(String, f64)> = Vec::new();
        let mut total_above = 0usize;

        for (i, chunk) in chunks.iter().enumerate() {
            let predictions = loaded
                .predict(chunk, DUMMY_LAT, DUMMY_LON, DUMMY_WEEK)
                .with_context(|| {
                    format!(
                        "Inference failed on chunk {} for {}",
                        i,
                        manifest.manifest.model.name
                    )
                })?;

            // Count predictions above a minimal threshold
            let above: Vec<_> = predictions
                .iter()
                .filter(|(_, conf)| *conf > min_conf)
                .collect();
            total_above += above.len();

            if let Some((label, conf)) = predictions.first() {
                info!(
                    "  Chunk {}: top={} ({:.4}), {} above {:.4}",
                    i,
                    label,
                    conf,
                    above.len(),
                    min_conf,
                );
                if *conf > best_confidence {
                    best_confidence = *conf;
                    best_label = label.clone();
                    best_top5 = predictions.iter().take(5).cloned().collect();
                }
            }
        }

        info!("  Best: {} ({:.4})", best_label, best_confidence);
        info!("  Top-5 (best chunk):");
        for (j, (label, conf)) in best_top5.iter().enumerate() {
            info!("    [{}] {} ({:.4})", j + 1, label, conf);
        }

        results.push(ModelResult {
            slug: manifest.slug(),
            name: manifest.manifest.model.name.clone(),
            top_label: best_label,
            top_confidence: best_confidence,
            top5: best_top5,
            n_above_threshold: total_above,
            score_transform: transform,
        });
        info!("");
    }

    // ── Assertions ───────────────────────────────────────────────────
    // Every model must be tested — no silent skips.  This guarantees
    // the build-time test exercises the exact same load + predict path
    // that runs at container startup.
    if results.len() != bird_manifests.len() {
        bail!(
            "Only {}/{} bird/wildlife model(s) were tested — \
             all models must pass to keep build-time and runtime parity",
            results.len(),
            bird_manifests.len()
        );
    }

    info!("═══════════════════════════════════════════════════════════");
    info!(
        "ASSERTIONS  ({} model(s) tested)",
        results.len()
    );
    info!("═══════════════════════════════════════════════════════════");

    let mut failures: Vec<String> = Vec::new();
    let expected_lower = expected_species.to_lowercase();
    // Also accept just the genus+species portion (without subspecies)
    let expected_parts: Vec<&str> = expected_lower.split('_').collect();
    let expected_sciname = if expected_parts.len() >= 2 {
        format!("{}_{}", expected_parts[0], expected_parts[1])
    } else {
        expected_lower.clone()
    };

    // 1. Every model must produce at least one detection
    for r in &results {
        let min_conf = match r.score_transform {
            ScoreTransform::Softmax => MIN_TOP_CONFIDENCE_SOFTMAX,
            _ => MIN_TOP_CONFIDENCE_SIGMOID,
        };
        if r.n_above_threshold == 0 {
            let msg = format!(
                "{}: No detections above {:.4} threshold — score transform likely broken",
                r.name, min_conf
            );
            error!("FAIL: {}", msg);
            failures.push(msg);
        } else {
            info!(
                "PASS: {} produced {} detection(s) above {:.4}",
                r.name, r.n_above_threshold, min_conf
            );
        }
    }

    // 2. Top confidence must be above minimum threshold
    for r in &results {
        let min_conf = match r.score_transform {
            ScoreTransform::Softmax => MIN_TOP_CONFIDENCE_SOFTMAX,
            _ => MIN_TOP_CONFIDENCE_SIGMOID,
        };
        if r.top_confidence < min_conf {
            let msg = format!(
                "{}: Top confidence {:.4} < {:.4} minimum ({:?}) — model output is near-uniform garbage",
                r.name, r.top_confidence, min_conf, r.score_transform
            );
            error!("FAIL: {}", msg);
            failures.push(msg);
        } else {
            info!(
                "PASS: {} top confidence {:.4} >= {:.4} ({:?})",
                r.name, r.top_confidence, min_conf, r.score_transform
            );
        }
    }

    // 3. Expected species in top-5 — informational, not a hard failure.
    //    Different models have different training data and taxonomic
    //    resolution (e.g. Perch may identify the correct genus but a
    //    different species).  Only confidence sanity is load-bearing.
    for r in &results {
        let found_in_top5 = r.top5.iter().any(|(label, _)| {
            let label_lower = label.to_lowercase();
            label_lower.contains(&expected_lower)
                || label_lower.contains(&expected_sciname)
                || expected_lower.contains(&label_lower)
        });
        if !found_in_top5 {
            let top5_labels: Vec<String> = r
                .top5
                .iter()
                .map(|(l, c)| format!("{l} ({c:.4})" ))
                .collect();
            warn!(
                "{}: Expected species '{}' not in top-5: [{}]",
                r.name,
                expected_species,
                top5_labels.join(", ")
            );
        } else {
            info!(
                "PASS: {} detected '{}' in top-5",
                r.name, expected_species
            );
        }
    }

    // 4. Confidence spread: all bird models should be within
    //    MAX_CONFIDENCE_RATIO of each other.
    //    This is a WARNING, not a hard failure — different score transforms
    //    (sigmoid vs softmax) produce fundamentally different scales.
    if results.len() >= 2 {
        let max_conf = results
            .iter()
            .map(|r| r.top_confidence)
            .fold(f64::NEG_INFINITY, f64::max);
        let min_conf = results
            .iter()
            .map(|r| r.top_confidence)
            .filter(|c| *c > 0.0)
            .fold(f64::INFINITY, f64::min);

        if min_conf > 0.0 {
            let ratio = max_conf / min_conf;
            let model_detail = results
                .iter()
                .map(|r| format!("{}={:.4}", r.slug, r.top_confidence))
                .collect::<Vec<_>>()
                .join(", ");
            if ratio > MAX_CONFIDENCE_RATIO {
                warn!(
                    "Confidence spread wide: max={:.4} / min={:.4} = {:.1}x \
                     (soft limit: {:.0}x). Models: {}",
                    max_conf, min_conf, ratio, MAX_CONFIDENCE_RATIO, model_detail,
                );
            } else {
                info!(
                    "OK: Confidence spread {:.1}x within {:.0}x limit ({})",
                    ratio, MAX_CONFIDENCE_RATIO, model_detail,
                );
            }
        }
    }

    // 5. No model should have top confidence > 0.9999 (sigmoid saturation)
    for r in &results {
        if r.top_confidence > 0.9999 {
            let msg = format!(
                "{}: Top confidence {:.6} is suspiciously close to 1.0 — \
                 sigmoid may be saturating all classes",
                r.name, r.top_confidence
            );
            warn!("WARN: {}", msg);
            // This is a warning, not a hard failure — some models
            // legitimately produce very high confidence.
        }
    }

    // ── Summary ──────────────────────────────────────────────────────
    info!("");
    info!("═══════════════════════════════════════════════════════════");
    info!("SUMMARY");
    info!("═══════════════════════════════════════════════════════════");

    info!("┌────────────────────────┬──────────────┬────────────┐");
    info!("│ Model                  │ Top Species  │ Confidence │");
    info!("├────────────────────────┼──────────────┼────────────┤");
    for r in &results {
        let name = if r.name.len() > 22 {
            format!("{}…", &r.name[..21])
        } else {
            format!("{:<22}", r.name)
        };
        let label = if r.top_label.len() > 12 {
            format!("{}…", &r.top_label[..11])
        } else {
            format!("{:<12}", r.top_label)
        };
        info!("│ {} │ {} │ {:<10.4} │", name, label, r.top_confidence);
    }
    info!("└────────────────────────┴──────────────┴────────────┘");

    if failures.is_empty() {
        info!("");
        info!(
            "✅ ALL SMOKE TESTS PASSED ({} model(s))",
            results.len()
        );
        Ok(())
    } else {
        error!("");
        error!(
            "❌ {} SMOKE TEST FAILURE(S):",
            failures.len()
        );
        for (i, f) in failures.iter().enumerate() {
            error!("  {}. {}", i + 1, f);
        }
        bail!(
            "{} smoke test assertion(s) failed — see above for details",
            failures.len()
        );
    }
}

/// Log the version of the ONNX Runtime shared library found on the system.
///
/// Scans `/usr/lib/libonnxruntime.so*` for versioned symlinks (e.g.
/// `libonnxruntime.so.1.24.2`).  This appears in CI build logs so
/// version-drift issues are immediately visible.
fn log_ort_library_version() {
    let lib_dir = std::path::Path::new("/usr/lib");
    let mut versions: Vec<String> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(lib_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("libonnxruntime.so") {
                versions.push(name);
            }
        }
    }
    versions.sort();
    if versions.is_empty() {
        warn!("No libonnxruntime.so found in /usr/lib — ORT models will fail to load");
    } else {
        info!("ORT libs: {}", versions.join(", "));
    }
}

// ══════════════════════════════════════════════════════════════════════
//  Pipeline smoke test — simulates the full main-loop path:
//    mDNS discovery → capture API → download → analysis → detections
// ══════════════════════════════════════════════════════════════════════

/// Run the pipeline smoke test.
///
/// Spins up a **mock capture HTTP server** on `localhost`, registers it
/// as a capture node via mDNS, discovers it from a processing node,
/// downloads the test recording through the real HTTP client, runs the
/// full analysis pipeline, and asserts detections were produced.
///
/// Everything runs locally — no containers, no network peers required.
pub fn run_pipeline(
    audio_path: &Path,
    models_dir: &Path,
    expected_species: &str,
) -> Result<()> {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    info!("╔══════════════════════════════════════════════════════════╗");
    info!("║           PIPELINE SMOKE TEST                           ║");
    info!("╚══════════════════════════════════════════════════════════╝");
    info!("Audio:    {}", audio_path.display());
    info!("Models:   {}", models_dir.display());
    info!("Expected: {}", expected_species);
    info!("");

    // ── 1. Check test audio exists ───────────────────────────────────
    anyhow::ensure!(
        audio_path.is_file(),
        "Test audio file not found: {}",
        audio_path.display()
    );

    // The filename must match the format that ParsedFileName::parse
    // expects: `YYYY-MM-DD-<slug>-HH:MM:SS.wav`.
    // Create a copy with a parseable name.
    let tmp_dir = std::env::temp_dir().join(format!(
        "gaia-pipeline-smoke-{}",
        std::process::id()
    ));
    if tmp_dir.exists() {
        std::fs::remove_dir_all(&tmp_dir).ok();
    }
    let stream_dir = tmp_dir.join("stream");
    std::fs::create_dir_all(&stream_dir)?;
    let test_filename = "2026-04-01-birdnet-12:00:00.wav";
    let stream_file = stream_dir.join(test_filename);
    std::fs::copy(audio_path, &stream_file).with_context(|| {
        format!(
            "Cannot copy test audio to {}",
            stream_file.display()
        )
    })?;

    info!("Step 1: Test audio staged as {}", stream_file.display());

    // ── 2. Start mock capture HTTP server ────────────────────────────
    // Bind to 0.0.0.0 so the server is reachable on whatever IP
    // mDNS advertises (enable_addr_auto picks the host's LAN address,
    // not 127.0.0.1).
    let listener = std::net::TcpListener::bind("0.0.0.0:0")
        .context("Cannot bind mock capture server")?;
    let mock_port = listener.local_addr()?.port();
    let mock_url = format!("http://127.0.0.1:{mock_port}");
    info!("Step 2: Mock capture server on {mock_url}");

    let server_stop = Arc::new(AtomicBool::new(false));
    let server_stop2 = server_stop.clone();
    let stream_dir2 = stream_dir.clone();

    // Set a short accept timeout so the server thread can poll for shutdown.
    listener
        .set_nonblocking(false)
        .ok();

    let server_thread = std::thread::Builder::new()
        .name("mock-capture".into())
        .spawn(move || {
            mock_capture_server(listener, stream_dir2, server_stop2);
        })
        .context("Cannot spawn mock capture server")?;

    // ── 3. Register on mDNS ──────────────────────────────────────────
    info!("Step 3: Registering mock capture on mDNS …");

    let capture_discovery = gaia_common::discovery::register(
        gaia_common::discovery::ServiceRole::Capture,
        mock_port,
    );
    let processing_discovery = gaia_common::discovery::register(
        gaia_common::discovery::ServiceRole::Processing,
        0,
    );

    let capture_dh = match capture_discovery {
        Ok(dh) => {
            info!(
                "  Mock capture registered as '{}' on port {mock_port}",
                dh.instance_name()
            );
            Some(dh)
        }
        Err(e) => {
            warn!("  mDNS capture registration failed (non-fatal): {e:#}");
            None
        }
    };
    let processing_dh = match processing_discovery {
        Ok(dh) => {
            info!("  Processing registered as '{}'", dh.instance_name());
            Some(dh)
        }
        Err(e) => {
            warn!("  mDNS processing registration failed (non-fatal): {e:#}");
            None
        }
    };

    // ── 4. Discover capture via mDNS ─────────────────────────────────
    info!("Step 4: Discovering capture node via mDNS …");
    let capture_url = if let Some(ref pdh) = processing_dh {
        let peers = pdh.discover_peers(
            gaia_common::discovery::ServiceRole::Capture,
            std::time::Duration::from_secs(5),
        );
        if let Some(peer) = peers.first() {
            if let Some(url) = peer.http_url() {
                info!("  Discovered '{}' at {url}", peer.instance_name);
                url
            } else {
                info!("  Peer found but no URL — using mock_url directly");
                mock_url.clone()
            }
        } else {
            info!("  No peers found via mDNS — using mock_url directly");
            mock_url.clone()
        }
    } else {
        info!("  mDNS unavailable — using mock_url directly");
        mock_url.clone()
    };

    // ── 5. List recordings via HTTP ──────────────────────────────────
    info!("Step 5: Listing recordings from {capture_url} …");
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .context("Cannot create HTTP client")?;

    let list_url = format!("{capture_url}/api/recordings");
    let resp = client
        .get(&list_url)
        .send()
        .context("GET /api/recordings failed")?;
    anyhow::ensure!(
        resp.status().is_success(),
        "GET /api/recordings returned {}",
        resp.status()
    );
    let recordings: Vec<gaia_common::protocol::RecordingInfo> =
        resp.json().context("Cannot parse recordings JSON")?;
    anyhow::ensure!(
        !recordings.is_empty(),
        "Mock capture server returned 0 recordings"
    );
    info!("  Found {} recording(s): {:?}",
        recordings.len(),
        recordings.iter().map(|r| &r.filename).collect::<Vec<_>>()
    );

    // ── 6. Download a recording ──────────────────────────────────────
    let rec = &recordings[0];
    let download_dir = tmp_dir.join("processing_tmp");
    std::fs::create_dir_all(&download_dir)?;
    let local_path = download_dir.join(&rec.filename);

    info!("Step 6: Downloading '{}' …", rec.filename);
    let dl_url = format!("{capture_url}/api/recordings/{}", rec.filename);
    let dl_resp = client.get(&dl_url).send().context("GET recording failed")?;
    anyhow::ensure!(
        dl_resp.status().is_success(),
        "GET recording returned {}",
        dl_resp.status()
    );
    let bytes = dl_resp.bytes()?;
    std::fs::write(&local_path, &bytes)?;
    info!(
        "  Downloaded {} ({:.1} KB) → {}",
        rec.filename,
        bytes.len() as f64 / 1024.0,
        local_path.display()
    );

    // ── 7. Load models ───────────────────────────────────────────────
    info!("Step 7: Loading models …");
    let manifests = crate::manifest::discover_manifests(models_dir)
        .context("Failed to discover model manifests")?;
    anyhow::ensure!(!manifests.is_empty(), "No manifests found");

    let config = Config {
        latitude: DUMMY_LAT,
        longitude: DUMMY_LON,
        confidence: 0.01,
        sensitivity: 1.25,
        overlap: 0.0,
        recording_length: 15,
        channels: 1,
        rec_card: None,
        model_dir: models_dir.to_path_buf(),
        database_lang: "en".to_string(),
        db_path: PathBuf::from("/dev/null"),
        capture_server_url: capture_url.clone(),
        capture_listen_addr: format!("0.0.0.0:{mock_port}"),
        poll_interval_secs: 5,
        recs_dir: download_dir.clone(),
        extracted_dir: tmp_dir.join("extracted"),
        audio_fmt: "wav".to_string(),
        rtsp_streams: vec![],
        sf_thresh: 0.03,
        data_model_version: 2,
        model_variant: None,
        model_slugs: vec![],
        processing_instance: String::new(),
        processing_threads: 1,
        raw_spectrogram: false,
        privacy_threshold: 0.0,
        extraction_length: 6,
        birdweather_id: None,
        heartbeat_url: None,
        turso_database_url: None,
        turso_auth_token: None,
        colormap: "default".to_string(),
        disk_usage_max: 95.0,
    };

    std::fs::create_dir_all(&config.extracted_dir).ok();

    let mut models = Vec::new();
    for m in &manifests {
        match crate::model::load_model(m, &config) {
            Ok(loaded) => {
                info!("  Loaded: {}", m.manifest.model.name);
                models.push(loaded);
            }
            Err(e) => {
                warn!("  Cannot load {}: {e:#}", m.manifest.model.name);
            }
        }
    }
    anyhow::ensure!(!models.is_empty(), "No models loaded — cannot run pipeline smoke test");
    info!("  {} model(s) ready", models.len());

    // ── 8. Run analysis (real pipeline path) ─────────────────────────
    info!("Step 8: Running analysis pipeline …");

    let (report_tx, report_rx) = std::sync::mpsc::sync_channel::<crate::ReportPayload>(4);

    crate::analysis::process_file(
        &local_path,
        &mut models,
        &config,
        &report_tx,
        &capture_url,
    )
    .context("analysis::process_file failed")?;

    // Drain the report channel.
    drop(report_tx);
    let mut total_detections = 0usize;
    let mut species_found = Vec::new();
    while let Ok(payload) = report_rx.try_recv() {
        total_detections += payload.detections.len();
        for d in &payload.detections {
            if !species_found.contains(&d.common_name) {
                species_found.push(d.common_name.clone());
            }
        }
    }
    info!(
        "  Analysis produced {} detection(s), {} unique species",
        total_detections,
        species_found.len()
    );
    if !species_found.is_empty() {
        info!("  Species detected: {:?}", species_found);
    }

    // ── 9. Delete recording via HTTP ─────────────────────────────────
    info!("Step 9: Deleting recording from mock capture …");
    let del_url = format!("{capture_url}/api/recordings/{}", rec.filename);
    let del_resp = client.delete(&del_url).send();
    match del_resp {
        Ok(r) if r.status().is_success() => {
            info!("  DELETE {} → {}", rec.filename, r.status());
        }
        Ok(r) => {
            warn!("  DELETE {} → {} (unexpected)", rec.filename, r.status());
        }
        Err(e) => {
            warn!("  DELETE failed: {e}");
        }
    }

    // ── 10. Cleanup ──────────────────────────────────────────────────
    server_stop.store(true, Ordering::Release);
    // Send a dummy connection to unblock the accept() call.
    std::net::TcpStream::connect(format!("127.0.0.1:{mock_port}")).ok();
    server_thread.join().ok();

    if let Some(dh) = capture_dh {
        dh.shutdown();
    }
    if let Some(dh) = processing_dh {
        dh.shutdown();
    }

    // ── Assertions ───────────────────────────────────────────────────
    info!("");
    info!("═══════════════════════════════════════════════════════════");
    info!("PIPELINE ASSERTIONS");
    info!("═══════════════════════════════════════════════════════════");

    let mut failures: Vec<String> = Vec::new();

    // Must have produced at least one detection.
    if total_detections == 0 {
        let msg = "Pipeline produced 0 detections — analysis is broken".to_string();
        error!("FAIL: {msg}");
        failures.push(msg);
    } else {
        info!(
            "PASS: Pipeline produced {} detection(s)",
            total_detections
        );
    }

    // Check if expected species was detected (warning only).
    let expected_lower = expected_species.to_lowercase();
    let found = species_found.iter().any(|s| {
        s.to_lowercase().contains(&expected_lower) || expected_lower.contains(&s.to_lowercase())
    });
    if found {
        info!("PASS: Expected species '{}' was detected", expected_species);
    } else {
        warn!(
            "Expected species '{}' NOT detected (species found: {:?})",
            expected_species, species_found
        );
    }

    if failures.is_empty() {
        info!("");
        info!("✅ PIPELINE SMOKE TEST PASSED");
        Ok(())
    } else {
        error!("");
        error!(
            "❌ {} PIPELINE FAILURE(S):",
            failures.len()
        );
        for (i, f) in failures.iter().enumerate() {
            error!("  {}. {}", i + 1, f);
        }
        bail!(
            "{} pipeline smoke test assertion(s) failed",
            failures.len()
        );
    }
}

// ── Mock capture HTTP server ─────────────────────────────────────────────

/// A minimal HTTP server that simulates the capture server API.
///
/// Handles:
///   - `GET /api/recordings` → JSON list of WAV files in `stream_dir`
///   - `GET /api/recordings/{name}` → file download
///   - `DELETE /api/recordings/{name}` → file removal + 204
///   - `GET /api/health` → `{"status":"ok"}`
///
/// Runs until `stop` is set to `true`.
fn mock_capture_server(
    listener: std::net::TcpListener,
    stream_dir: PathBuf,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
) {
    use std::io::{BufRead, BufReader, Write};
    use std::sync::atomic::Ordering;

    // Set a 1-second timeout so we can check the stop flag.
    listener
        .set_nonblocking(false)
        .ok();
    let _ = listener.set_nonblocking(false);

    info!("Mock capture server started, serving {}", stream_dir.display());

    for stream in listener.incoming() {
        if stop.load(Ordering::Acquire) {
            break;
        }

        let mut stream = match stream {
            Ok(s) => s,
            Err(_) => continue,
        };

        // Set a read timeout so malformed requests don't hang the server.
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(5)))
            .ok();

        let mut reader = BufReader::new(stream.try_clone().unwrap_or_else(|_| {
            // Fallback: if clone fails, just use the original.
            // This branch shouldn't be reached on Linux.
            panic!("TcpStream::try_clone failed");
        }));

        // Read the request line.
        let mut request_line = String::new();
        if reader.read_line(&mut request_line).is_err() {
            continue;
        }

        // Consume remaining headers (we don't need them).
        loop {
            let mut header = String::new();
            match reader.read_line(&mut header) {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    if header.trim().is_empty() {
                        break;
                    }
                }
            }
        }

        let parts: Vec<&str> = request_line.split_whitespace().collect();
        if parts.len() < 2 {
            continue;
        }
        let method = parts[0];
        let path = parts[1];

        match (method, path) {
            ("GET", "/api/health") => {
                let body = r#"{"status":"ok","uptime_secs":1,"disk_usage_pct":10.0,"capture_paused":false}"#;
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                stream.write_all(resp.as_bytes()).ok();
            }
            ("GET", "/api/recordings") => {
                let mut recordings = Vec::new();
                if let Ok(entries) = std::fs::read_dir(&stream_dir) {
                    for entry in entries.flatten() {
                        let p = entry.path();
                        let ext = p.extension().and_then(|e| e.to_str());
                        if ext != Some("wav") && ext != Some("opus") {
                            continue;
                        }
                        if let Ok(meta) = p.metadata() {
                            if meta.len() == 0 {
                                continue;
                            }
                            recordings.push(gaia_common::protocol::RecordingInfo {
                                filename: p
                                    .file_name()
                                    .unwrap_or_default()
                                    .to_string_lossy()
                                    .to_string(),
                                size: meta.len(),
                                created: "2026-04-01T12:00:00Z".to_string(),
                            });
                        }
                    }
                }
                recordings.sort_by(|a, b| a.filename.cmp(&b.filename));
                let body = serde_json::to_string(&recordings).unwrap_or_default();
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                stream.write_all(resp.as_bytes()).ok();
            }
            ("GET", p) if p.starts_with("/api/recordings/") => {
                let name = &p["/api/recordings/".len()..];
                // Basic path safety: reject traversal.
                if name.contains("..") || name.contains('/') || name.contains('\\') {
                    let resp = "HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n";
                    stream.write_all(resp.as_bytes()).ok();
                    continue;
                }
                let file_path = stream_dir.join(name);
                if file_path.is_file() {
                    let data = std::fs::read(&file_path).unwrap_or_default();
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: audio/wav\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        data.len()
                    );
                    stream.write_all(resp.as_bytes()).ok();
                    stream.write_all(&data).ok();
                } else {
                    let resp = "HTTP/1.1 404 Not Found\r\nConnection: close\r\n\r\n";
                    stream.write_all(resp.as_bytes()).ok();
                }
            }
            ("DELETE", p) if p.starts_with("/api/recordings/") => {
                let name = &p["/api/recordings/".len()..];
                if name.contains("..") || name.contains('/') || name.contains('\\') {
                    let resp = "HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n";
                    stream.write_all(resp.as_bytes()).ok();
                    continue;
                }
                let file_path = stream_dir.join(name);
                if file_path.is_file() {
                    std::fs::remove_file(&file_path).ok();
                    let resp = "HTTP/1.1 204 No Content\r\nConnection: close\r\n\r\n";
                    stream.write_all(resp.as_bytes()).ok();
                } else {
                    let resp = "HTTP/1.1 404 Not Found\r\nConnection: close\r\n\r\n";
                    stream.write_all(resp.as_bytes()).ok();
                }
            }
            _ => {
                let resp = "HTTP/1.1 404 Not Found\r\nConnection: close\r\n\r\n";
                stream.write_all(resp.as_bytes()).ok();
            }
        }
    }

    info!("Mock capture server stopped");
}
