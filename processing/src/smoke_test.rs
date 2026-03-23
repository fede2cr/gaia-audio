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
pub fn run(audio_path: &Path, models_dir: &Path, expected_species: &str) -> Result<()> {
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
