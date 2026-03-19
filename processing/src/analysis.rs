//! Full analysis pipeline – from WAV file to confident detections.
//!
//! Evolved from `birdnet-server/src/analysis.rs`.  Now works with multiple
//! models (one per domain) and tags each detection with its domain.

use std::path::Path;

use anyhow::{Context, Result};
use tracing::{debug, info, warn};

use gaia_common::audio;
use gaia_common::config::Config;
use gaia_common::detection::{Detection, ParsedFileName};

use crate::live_status::{self, LivePrediction};
use crate::model::{self, LoadedModel, Prediction};
use crate::ReportPayload;

/// Process a single WAV file through all loaded models.
pub fn process_file(
    file_path: &Path,
    models: &mut [LoadedModel],
    config: &Config,
    report_tx: &std::sync::mpsc::SyncSender<ReportPayload>,
    source_node: &str,
) -> Result<()> {
    // Skip empty files
    let meta = std::fs::metadata(file_path)?;
    if meta.len() == 0 {
        std::fs::remove_file(file_path).ok();
        return Ok(());
    }

    info!("Analysing {}", file_path.display());

    let file = ParsedFileName::parse(file_path)
        .with_context(|| format!("Cannot parse filename: {}", file_path.display()))?;

    let mut all_detections = Vec::new();
    // Collect the top raw predictions across models for the live feed.
    let mut live_predictions: Vec<LivePrediction> = Vec::new();

    for model in models.iter_mut() {
        let (detections, top_preds) = run_analysis(&file, model, config)?;
        all_detections.extend(detections);
        live_predictions.extend(top_preds);
    }

    // ── Update live analysis status ──────────────────────────────────
    // Read a short chunk of audio at 24 kHz for the live spectrogram.
    {
        let live_sr = 24_000u32;
        match gaia_common::audio::read_audio(file_path, live_sr, 3.0, 0.0) {
            Ok(chunks) => {
                let samples: Vec<f32> = chunks.into_iter().flatten().collect();
                // Keep only the top 5 predictions by confidence.
                live_predictions.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
                live_predictions.truncate(5);
                let captured_at = file.file_date.format("%Y-%m-%dT%H:%M:%S").to_string();
                live_status::update(
                    &file.file_path.file_name().unwrap_or_default().to_string_lossy(),
                    &samples,
                    live_sr,
                    live_predictions,
                    config.confidence,
                    &config.colormap,
                    source_node,
                    &captured_at,
                );
            }
            Err(e) => {
                warn!("Cannot read audio for live spectrogram: {e:#}");
            }
        }
    }

    report_tx
        .send(ReportPayload {
            file,
            detections: all_detections,
            source_node: source_node.to_string(),
        })
        .map_err(|_| anyhow::anyhow!("Reporting channel closed"))?;

    Ok(())
}

/// Core analysis logic for a single model.
///
/// Returns the confident detections and the top raw predictions (for live feed).
fn run_analysis(
    file: &ParsedFileName,
    model: &mut LoadedModel,
    config: &Config,
) -> Result<(Vec<Detection>, Vec<LivePrediction>)> {
    let domain = model.domain().to_string();
    let model_slug = model.manifest.slug();
    let model_name = model.manifest.manifest.model.name.clone();
    // Tag for log messages: "BirdNET V2.4/birds" or "Google Perch 2.0/wildlife"
    let tag = format!("{model_name}/{domain}");

    // ── custom species lists ─────────────────────────────────────────
    let base = std::env::var("GAIA_DIR").unwrap_or_else(|_| "/app".to_string());
    let include_list =
        model::load_species_list(Path::new(&base).join("include_species_list.txt").as_path());
    let exclude_list =
        model::load_species_list(Path::new(&base).join("exclude_species_list.txt").as_path());
    let mut whitelist =
        model::load_species_list(Path::new(&base).join("whitelist_species_list.txt").as_path());

    // Merge in DB-based exclusion overrides (species confirmed via the
    // web UI by an ornithologist).  These bypass the occurrence threshold
    // just like the file-based whitelist.
    let db_overrides = crate::db::load_exclusion_overrides(&config.db_path);
    for sp in db_overrides {
        if !whitelist.contains(&sp) {
            whitelist.push(sp);
        }
    }

    // ── language map ─────────────────────────────────────────────────
    let mut names =
        model::load_language(&model.manifest.language_dir(), &config.database_lang)
            .unwrap_or_default();
    // Fallback: when no language JSON exists (e.g. BirdNET+ V3.0), use
    // common names parsed from the CSV labels file.
    if names.is_empty() {
        names = model.csv_common_names().clone();
    }

    // ── read audio ───────────────────────────────────────────────────
    let chunks = match audio::read_audio(
        &file.file_path,
        model.sample_rate(),
        model.chunk_duration(),
        config.overlap,
    ) {
        Ok(c) => c,
        Err(e) => {
            tracing::error!("[{tag}] Error reading audio: {e}");
            return Ok((vec![], vec![]));
        }
    };

    // ── run inference on each chunk ──────────────────────────────────
    let mut raw_detections: Vec<Vec<Prediction>> = Vec::with_capacity(chunks.len());
    for (i, chunk) in chunks.iter().enumerate() {
        let preds = model.predict(chunk, config.latitude, config.longitude, file.week())?;
        // Log top-3 raw scores so operators can tell whether the model
        // produces meaningful output.
        if let Some(top) = preds.first() {
            let top3: Vec<String> = preds.iter().take(3)
                .map(|(name, conf)| format!("{name}={conf:.4}"))
                .collect();
            debug!("[{tag}] chunk {i}: top raw = [{}]", top3.join(", "));
            if top.1 >= config.confidence {
                info!(
                    "[{tag}] chunk {i}: {} ({:.1}%) ≥ threshold {:.0}%",
                    top.0, top.1 * 100.0, config.confidence * 100.0
                );
            }
        }
        raw_detections.push(preds);
    }

    // ── filter human speech (birds models only) ──────────────────────
    let filtered = if domain == "birds" {
        filter_humans(&raw_detections, config)
    } else {
        raw_detections
    };

    // ── assemble time-labeled detections ─────────────────────────────
    let mut labeled: Vec<(f64, f64, Vec<Prediction>)> = Vec::new();
    let mut pred_start = 0.0_f64;
    for preds in &filtered {
        let pred_end = pred_start + model.chunk_duration();
        labeled.push((pred_start, pred_end, preds.clone()));
        pred_start = pred_end - config.overlap;
    }

    // ── species-range model (location-based filtering) ──────────────
    let predicted_species_list = model.get_species_list(
        config.latitude,
        config.longitude,
        file.week(),
    );
    if !predicted_species_list.is_empty() {
        info!(
            "[{tag}] Species range model: {} species expected at ({}, {}) week {}",
            predicted_species_list.len(),
            config.latitude,
            config.longitude,
            file.week(),
        );
    } else if config.latitude != -1.0 && config.longitude != -1.0 {
        info!(
            "[{tag}] No species-range model loaded — accepting all species"
        );
    }

    // ── apply confidence threshold + species filters ─────────────────

    let mut confident_detections = Vec::new();
    for (start, end, entries) in &labeled {
        if let Some((sci_name, confidence)) = entries.first() {
            debug!(
                "[{tag}] {start:.1}-{end:.1}: {sci_name} ({} = {confidence:.4})",
                names.get(sci_name.as_str()).unwrap_or(sci_name)
            );
        }

        for (sci_name, confidence) in entries {
            if *confidence < config.confidence {
                continue;
            }

            let com_name = names
                .get(sci_name.as_str())
                .cloned()
                .unwrap_or_else(|| sci_name.clone());

            if !include_list.is_empty() && !include_list.contains(sci_name) {
                warn!("[{tag}] Excluded (not in include list): {sci_name}");
                continue;
            }
            if !exclude_list.is_empty() && exclude_list.contains(sci_name) {
                warn!("[{tag}] Excluded (in exclude list): {sci_name}");
                continue;
            }

            // Species-range filter: if the location model says this species
            // is unlikely here, still record the detection but flag it as
            // excluded so an ornithologist can review it later.
            let excluded = !predicted_species_list.is_empty()
                && !predicted_species_list.contains(sci_name)
                && !whitelist.contains(sci_name);

            if excluded {
                warn!(
                    "[{tag}] Recording excluded detection (below occurrence threshold): {sci_name} ({:.1}%)",
                    confidence * 100.0
                );
            }

            let mut det = Detection::new(
                &domain,
                file.file_date,
                *start,
                *end,
                sci_name,
                &com_name,
                *confidence,
            );
            det.excluded = excluded;
            det.model_slug = model_slug.clone();
            det.model_name = model_name.clone();
            confident_detections.push(det);
        }
    }

    let included = confident_detections.iter().filter(|d| !d.excluded).count();
    let excluded = confident_detections.iter().filter(|d| d.excluded).count();
    info!(
        "[{tag}] {}: {} detection(s) ({included} included, {excluded} excluded)",
        file.file_path.display(),
        confident_detections.len()
    );

    // Collect top raw predictions for the live feed (all chunks, top
    // entry per chunk regardless of confidence threshold).
    let mut top_preds: Vec<LivePrediction> = Vec::new();
    for (_start, _end, entries) in &labeled {
        if let Some((sci_name, confidence)) = entries.first() {
            let com_name = names
                .get(sci_name.as_str())
                .cloned()
                .unwrap_or_else(|| sci_name.clone());
            top_preds.push(LivePrediction {
                scientific_name: sci_name.clone(),
                common_name: com_name,
                confidence: *confidence,
                model_slug: model_slug.clone(),
                model_name: model_name.clone(),
            });
        }
    }

    Ok((confident_detections, top_preds))
}

// ── privacy filter ───────────────────────────────────────────────────────

fn filter_humans(predictions: &[Vec<Prediction>], config: &Config) -> Vec<Vec<Prediction>> {
    let human_cutoff = (6000.0 * config.privacy_threshold / 100.0).max(10.0) as usize;

    let human_mask: Vec<bool> = predictions
        .iter()
        .map(|preds| {
            preds
                .iter()
                .take(human_cutoff)
                .any(|(name, _)| name.contains("Human"))
        })
        .collect();

    let neighbour_mask: Vec<bool> = (0..predictions.len())
        .map(|i| {
            (i > 0 && human_mask[i - 1]) || (i + 1 < human_mask.len() && human_mask[i + 1])
        })
        .collect();

    predictions
        .iter()
        .enumerate()
        .map(|(i, preds)| {
            if human_mask[i] || neighbour_mask[i] {
                debug!("Overwriting prediction (human): {:?}", preds.first());
                vec![("Human_Human".to_string(), 0.0)]
            } else {
                preds.iter().take(10).cloned().collect()
            }
        })
        .collect()
}
