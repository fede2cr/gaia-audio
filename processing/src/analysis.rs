//! Full analysis pipeline – from WAV file to confident detections.
//!
//! Evolved from `birdnet-server/src/analysis.rs`.  Now works with multiple
//! models (one per domain) and tags each detection with its domain.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{Context, Result};
use chrono::Datelike;
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

    // Read the set of enabled models from Redis (managed in Settings).
    // An empty set means the Redis key is missing — treat as "all models
    // are enabled" (backward-compat with deployments not yet seeded).
    let enabled = crate::kv::get_enabled_models();
    let all_enabled = enabled.is_empty();

    // ── shared species-range data ────────────────────────────────────
    // Pre-compute the species-range list from models that have a
    // metadata model (e.g. BirdNET V2.4).  Models without their own
    // metadata model (e.g. Perch) will use this shared list so that
    // they also benefit from geographic filtering.
    //
    // We also collect the full set of bird label names from those
    // models so that, for models lacking a `class` column in their
    // labels (like Perch), we can tell birds from non-birds and only
    // apply the geo filter to bird species.
    let mut shared_species_range: Vec<String> = Vec::new();
    let mut known_bird_labels: HashSet<String> = HashSet::new();
    // Shared common-name map: merge names from all models so that any
    // model (e.g. Perch) that lacks its own language file still gets
    // proper common names from models that have one (e.g. BirdNET).
    let mut shared_common_names: HashMap<String, String> = HashMap::new();
    for model in models.iter_mut() {
        if !all_enabled && !enabled.contains(&model.manifest.slug()) {
            continue;
        }
        // Collect common names from every model.
        let model_names = model::load_language(
            &model.manifest.language_dir(), &config.database_lang,
        ).unwrap_or_default();
        for (sci, com) in &model_names {
            let norm = gaia_common::detection::normalize_sci_name(sci);
            shared_common_names.entry(norm).or_insert_with(|| com.clone());
        }
        // Also include CSV-parsed common names.
        for (sci, com) in model.csv_common_names() {
            let norm = gaia_common::detection::normalize_sci_name(sci);
            shared_common_names.entry(norm).or_insert_with(|| com.clone());
        }
        if model.has_species_range_model() {
            let list = model.get_species_list(
                config.latitude,
                config.longitude,
                file.week(),
            );
            if !list.is_empty() && shared_species_range.is_empty() {
                shared_species_range = list;
            }
            // All labels from a model that ships a species-range model are
            // bird species (BirdNET V2.4's label set is bird-only).
            for label in model.labels() {
                // BirdNET labels: "Sci Name_Common Name" or just "Sci Name".
                let sci = label.split('_').next().unwrap_or(label);
                known_bird_labels.insert(sci.to_string());
            }
        }
    }

    for model in models.iter_mut() {
        if !all_enabled && !enabled.contains(&model.manifest.slug()) {
            debug!("Skipping disabled model: {}", model.manifest.manifest.model.name);
            continue;
        }
        let (detections, top_preds) = run_analysis(
            &file, model, config,
            &shared_species_range, &known_bird_labels,
            &shared_common_names,
        )?;
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
///
/// `shared_species_range`: species-range list pre-computed from models
/// that have a metadata model (e.g. BirdNET V2.4).  Used as a fallback
/// when this model lacks its own metadata model.
///
/// `known_bird_labels`: scientific names from bird-only models so that
/// we can distinguish bird vs. non-bird species in models that lack a
/// taxonomic `class` column (e.g. Perch).
///
/// `shared_common_names`: merged common-name map from all models so
/// that models without their own language file (e.g. Perch) can still
/// display proper common names like "Keel-billed Toucan" instead of
/// falling back to the scientific name.
fn run_analysis(
    file: &ParsedFileName,
    model: &mut LoadedModel,
    config: &Config,
    shared_species_range: &[String],
    known_bird_labels: &HashSet<String>,
    shared_common_names: &HashMap<String, String>,
) -> Result<(Vec<Detection>, Vec<LivePrediction>)> {
    let domain = model.domain().to_string();
    let class_map = model.csv_classes().clone();
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

    // Merge in Redis-based exclusion overrides (species confirmed via the
    // web UI by an ornithologist).  These bypass the occurrence threshold
    // just like the file-based whitelist.
    let db_overrides = crate::kv::load_exclusion_overrides();
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
    // Merge in shared common names from other models.  This ensures that
    // models without their own language file (e.g. Perch) still show
    // "Keel-billed Toucan" instead of "Ramphastos sulfuratus" as the
    // common name.  Existing names take priority — only fill gaps.
    for (sci, com) in shared_common_names {
        names.entry(sci.clone()).or_insert_with(|| com.clone());
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
            // Log at info for the first chunk so operators always see
            // what the model is producing.  Subsequent chunks stay at
            // debug to avoid flooding.
            if i == 0 {
                info!("[{tag}] chunk {i}: top = [{}]", top3.join(", "));
            } else {
                debug!("[{tag}] chunk {i}: top = [{}]", top3.join(", "));
            }
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
    let own_species_list = model.get_species_list(
        config.latitude,
        config.longitude,
        file.week(),
    );
    // Use the model's own species-range list if available; otherwise
    // fall back to the shared list from another model (e.g. BirdNET's
    // metadata model providing geographic filtering for Perch).
    let predicted_species_list = if !own_species_list.is_empty() {
        &own_species_list
    } else {
        shared_species_range
    };
    let using_shared = own_species_list.is_empty() && !shared_species_range.is_empty();

    if !predicted_species_list.is_empty() {
        if using_shared {
            info!(
                "[{tag}] Using shared species range filter: {} species expected at ({}, {}) week {}",
                predicted_species_list.len(),
                config.latitude,
                config.longitude,
                file.week(),
            );
        } else {
            info!(
                "[{tag}] Species range model: {} species expected at ({}, {}) week {}",
                predicted_species_list.len(),
                config.latitude,
                config.longitude,
                file.week(),
            );
        }
    } else if config.latitude != -1.0 && config.longitude != -1.0 {
        info!(
            "[{tag}] No species-range model loaded — accepting all species"
        );
    }

    // Log static range file availability (first file only).
    {
        let static_ranges = crate::species_range::global();
        if !static_ranges.is_empty() {
            let at_loc = static_ranges.species_at(
                config.latitude, config.longitude, file.file_date.month(),
            );
            debug!(
                "[{tag}] Static range file: {} total, {} at location month {}",
                static_ranges.len(), at_loc.len(), file.file_date.month(),
            );
        }
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
            //
            // The metadata model (V2.4) only knows about birds.  When
            // BirdNET+ V3.0 detects non-bird species (Mammalia, Insecta,
            // …) they would always be marked excluded because they are
            // absent from the bird-only occurrence list.  Skip the check
            // for species whose taxonomic class is not Aves.
            //
            // For models without a `class` column (e.g. Perch), fall back
            // to checking if the species appears in the label set of a
            // bird-only model (known_bird_labels).  Non-bird Perch
            // species (crickets, frogs, …) would not be in that set and
            // should pass through unfiltered unless the static range file
            // says otherwise.
            let is_bird = if let Some(cls) = class_map.get(sci_name.as_str()) {
                cls == "Aves"
            } else if !known_bird_labels.is_empty() {
                known_bird_labels.contains(sci_name.as_str())
            } else {
                // No class info and no bird reference set — assume bird
                // for backward compatibility (matches previous default).
                true
            };

            // Check the neural species-range model (birds).
            let excluded_by_range_model = is_bird
                && !predicted_species_list.is_empty()
                && !predicted_species_list.contains(sci_name)
                && !whitelist.contains(sci_name);

            // Check the static CSV range file (non-birds: insects, frogs, bats, etc.).
            let static_ranges = crate::species_range::global();
            let excluded_by_static = if !is_bird && !static_ranges.is_empty() {
                match static_ranges.check(sci_name, config.latitude, config.longitude, file.file_date.month()) {
                    Some(false) => !whitelist.contains(sci_name),
                    _ => false, // in range, or not in file → accept
                }
            } else {
                false
            };

            let excluded = excluded_by_range_model || excluded_by_static;

            if excluded {
                let reason = if excluded_by_static {
                    "out of static range"
                } else {
                    "below occurrence threshold"
                };
                warn!(
                    "[{tag}] Recording excluded detection ({reason}): {sci_name} ({:.1}%)",
                    confidence * 100.0
                );
            }

            // Use per-species taxonomic class when the labels CSV
            // provides one (e.g. BirdNET+ V3.0: Aves, Mammalia, …),
            // otherwise fall back to the model-wide domain.
            let det_domain = class_map
                .get(sci_name.as_str())
                .map_or(&domain, |c| c);

            let mut det = Detection::new(
                det_domain,
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
    // Skip chunks where the top score is effectively zero — these are
    // no-signal chunks (sigmoid baseline clamped to 0, or softmax with
    // negligible probability) and would clutter the live feed.
    let mut top_preds: Vec<LivePrediction> = Vec::new();
    for (_start, _end, entries) in &labeled {
        if let Some((sci_name, confidence)) = entries.first() {
            if *confidence < 0.01 {
                continue; // no meaningful signal in this chunk
            }
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
