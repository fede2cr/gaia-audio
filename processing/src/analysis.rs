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

use crate::model::{self, LoadedModel, Prediction};
use crate::ReportPayload;

/// Process a single WAV file through all loaded models.
pub fn process_file(
    file_path: &Path,
    models: &[LoadedModel],
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

    for model in models {
        let detections = run_analysis(&file, model, config)?;
        all_detections.extend(detections);
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
fn run_analysis(
    file: &ParsedFileName,
    model: &LoadedModel,
    config: &Config,
) -> Result<Vec<Detection>> {
    let domain = model.domain();

    // ── custom species lists ─────────────────────────────────────────
    let base = std::env::var("GAIA_DIR").unwrap_or_else(|_| "/app".to_string());
    let include_list =
        model::load_species_list(Path::new(&base).join("include_species_list.txt").as_path());
    let exclude_list =
        model::load_species_list(Path::new(&base).join("exclude_species_list.txt").as_path());
    let whitelist =
        model::load_species_list(Path::new(&base).join("whitelist_species_list.txt").as_path());

    // ── language map ─────────────────────────────────────────────────
    let names =
        model::load_language(&model.manifest.language_dir(), &config.database_lang)
            .unwrap_or_default();

    // ── read audio ───────────────────────────────────────────────────
    let chunks = match audio::read_audio(
        &file.file_path,
        model.sample_rate(),
        model.chunk_duration(),
        config.overlap,
    ) {
        Ok(c) => c,
        Err(e) => {
            tracing::error!("[{domain}] Error reading audio: {e}");
            return Ok(vec![]);
        }
    };

    // ── run inference on each chunk ──────────────────────────────────
    let mut raw_detections: Vec<Vec<Prediction>> = Vec::with_capacity(chunks.len());
    for chunk in &chunks {
        let preds = model.predict(chunk, config.latitude, config.longitude, file.week())?;
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

    // ── apply confidence threshold + species filters ─────────────────
    let predicted_species_list: Vec<String> = vec![];

    let mut confident_detections = Vec::new();
    for (start, end, entries) in &labeled {
        if let Some((sci_name, confidence)) = entries.first() {
            debug!(
                "[{domain}] {start:.1}-{end:.1}: {sci_name} ({} = {confidence:.4})",
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
                warn!("[{domain}] Excluded (not in include list): {sci_name}");
                continue;
            }
            if !exclude_list.is_empty() && exclude_list.contains(sci_name) {
                warn!("[{domain}] Excluded (in exclude list): {sci_name}");
                continue;
            }
            if !predicted_species_list.is_empty()
                && !predicted_species_list.contains(sci_name)
                && !whitelist.contains(sci_name)
            {
                warn!("[{domain}] Excluded (below occurrence threshold): {sci_name}");
                continue;
            }

            let det = Detection::new(
                domain,
                file.file_date,
                *start,
                *end,
                sci_name,
                &com_name,
                *confidence,
            );
            confident_detections.push(det);
        }
    }

    info!(
        "[{domain}] {}: {} confident detection(s)",
        file.file_path.display(),
        confident_detections.len()
    );
    Ok(confident_detections)
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
