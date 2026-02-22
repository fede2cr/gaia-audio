//! TFLite model loading and inference – generalized for multiple domains.
//!
//! Evolved from `birdnet-server/src/model.rs`.  Instead of hard-coding model
//! variants, each model is described by a [`manifest::ResolvedManifest`] that
//! specifies sample rate, chunk duration, label format, etc.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use tract_tflite::prelude::*;
use tracing::info;

use crate::manifest::ResolvedManifest;
use gaia_common::config::Config;

// ── public types ─────────────────────────────────────────────────────────

/// A loaded model ready for inference, built from a manifest.
pub struct LoadedModel {
    runner: TypedRunnableModel<TypedModel>,
    meta_model: Option<MetaDataModel>,
    labels: Vec<String>,
    pub manifest: ResolvedManifest,
    sensitivity: f64,
}

/// Species-occurrence metadata model (filters by location/week).
struct MetaDataModel {
    runner: TypedRunnableModel<TypedModel>,
    labels: Vec<String>,
    sf_thresh: f64,
    cached_params: Option<(f64, f64, u32)>,
    cached_list: Vec<String>,
}

/// One prediction: (label, confidence).
pub type Prediction = (String, f64);

// ── model loading ────────────────────────────────────────────────────────

/// Load a model from a resolved manifest.
pub fn load_model(resolved: &ResolvedManifest, config: &Config) -> Result<LoadedModel> {
    let tflite_path = resolved.tflite_path();
    info!("Loading model from {}", tflite_path.display());

    let runner = tract_tflite::tflite()
        .model_for_path(&tflite_path)
        .with_context(|| format!("Cannot load TFLite model: {}", tflite_path.display()))?
        .into_optimized()
        .context("Model optimisation failed")?
        .into_runnable()
        .context("Cannot make model runnable")?;

    let labels = load_labels(&resolved.labels_path())?;

    let meta_model = load_meta_model(resolved, &labels, config.sf_thresh)?;

    let sensitivity = config.sensitivity.clamp(0.5, 1.5);
    let adjusted_sensitivity = (1.0 - (sensitivity - 1.0)).clamp(0.5, 1.5);

    Ok(LoadedModel {
        runner,
        meta_model,
        labels,
        manifest: resolved.clone(),
        sensitivity: adjusted_sensitivity,
    })
}

impl LoadedModel {
    /// The model's domain (e.g. "birds", "bats").
    pub fn domain(&self) -> &str {
        self.manifest.domain()
    }

    /// Target sample rate for this model.
    pub fn sample_rate(&self) -> u32 {
        self.manifest.manifest.model.sample_rate
    }

    /// Chunk duration (seconds) expected by this model.
    pub fn chunk_duration(&self) -> f64 {
        self.manifest.manifest.model.chunk_duration
    }

    /// Whether this model uses V1-style metadata input.
    pub fn v1_metadata(&self) -> bool {
        self.manifest.manifest.model.v1_metadata
    }

    /// Run inference on a single audio chunk.
    ///
    /// Returns a sorted list of `(label, confidence)` pairs,
    /// highest confidence first.
    pub fn predict(
        &self,
        chunk: &[f32],
        lat: f64,
        lon: f64,
        week: u32,
    ) -> Result<Vec<Prediction>> {
        let n = chunk.len();
        let input: Tensor = tract_ndarray::Array2::from_shape_vec((1, n), chunk.to_vec())
            .context("Cannot reshape audio chunk")?
            .into();

        let result = if self.v1_metadata() {
            let mdata = convert_v1_metadata(lat, lon, week);
            let mdata_tensor: Tensor =
                tract_ndarray::Array2::from_shape_vec((1, 6), mdata.to_vec())
                    .context("Cannot reshape metadata")?
                    .into();
            self.runner
                .run(tvec![input.into(), mdata_tensor.into()])
                .context("V1 inference failed")?
        } else {
            self.runner
                .run(tvec![input.into()])
                .context("Inference failed")?
        };

        let output = result[0]
            .to_array_view::<f32>()
            .context("Cannot read output tensor")?;

        let logits: Vec<f32> = output.iter().copied().collect();
        let scores = if self.manifest.manifest.model.apply_softmax {
            softmax(&logits)
        } else {
            self.scale_logits(&logits)
        };

        let mut predictions: Vec<Prediction> = self
            .labels
            .iter()
            .zip(scores.iter())
            .map(|(label, &score)| (label.clone(), score as f64))
            .collect();

        predictions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(predictions)
    }

    /// Apply sigmoid scaling with sensitivity adjustment.
    fn scale_logits(&self, logits: &[f32]) -> Vec<f32> {
        logits
            .iter()
            .map(|&x| 1.0 / (1.0 + (-self.sensitivity as f32 * x).exp()))
            .collect()
    }

    /// Get the list of species that the metadata model predicts for the
    /// given location/week.  Returns an empty list when no meta-model is
    /// loaded (meaning "accept everything").
    #[allow(dead_code)]
    pub fn get_species_list(&mut self, lat: f64, lon: f64, week: u32) -> Vec<String> {
        match &mut self.meta_model {
            Some(meta) => meta.get_species_list(lat, lon, week),
            None => vec![],
        }
    }
}

// ── softmax ──────────────────────────────────────────────────────────────

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

// ── metadata model ───────────────────────────────────────────────────────

fn load_meta_model(
    resolved: &ResolvedManifest,
    _labels: &[String],
    sf_thresh: f64,
) -> Result<Option<MetaDataModel>> {
    let meta_path = match resolved.metadata_tflite_path() {
        Some(p) if p.exists() => p,
        _ => return Ok(None),
    };

    info!("Loading metadata model: {}", meta_path.display());
    let runner = tract_tflite::tflite()
        .model_for_path(&meta_path)
        .context("Cannot load metadata model")?
        .into_optimized()?
        .into_runnable()?;

    let labels = load_labels(&resolved.labels_path())?;

    Ok(Some(MetaDataModel {
        runner,
        labels,
        sf_thresh,
        cached_params: None,
        cached_list: vec![],
    }))
}

impl MetaDataModel {
    fn get_species_list(&mut self, lat: f64, lon: f64, week: u32) -> Vec<String> {
        let params = (lat, lon, week);
        if self.cached_params == Some(params) {
            return self.cached_list.clone();
        }

        let input: Tensor =
            tract_ndarray::Array2::from_shape_vec((1, 3), vec![lat as f32, lon as f32, week as f32])
                .expect("metadata input shape")
                .into();

        let result = match self.runner.run(tvec![input.into()]) {
            Ok(r) => r,
            Err(e) => {
                tracing::error!("Metadata model inference failed: {e}");
                return vec![];
            }
        };

        let output = match result[0].to_array_view::<f32>() {
            Ok(o) => o,
            Err(e) => {
                tracing::error!("Cannot read metadata output: {e}");
                return vec![];
            }
        };

        let filter: Vec<f32> = output.iter().copied().collect();

        let mut scored: Vec<(f32, &str)> = filter
            .iter()
            .zip(self.labels.iter())
            .map(|(&score, label)| (score, label.as_str()))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let list: Vec<String> = scored
            .iter()
            .filter(|(score, _)| *score >= self.sf_thresh as f32)
            .map(|(_, label)| label.split('_').next().unwrap_or(label).to_string())
            .collect();

        self.cached_params = Some(params);
        self.cached_list = list.clone();
        list
    }
}

// ── helpers ──────────────────────────────────────────────────────────────

/// Load label file.  Each line is one label.
/// Labels of the form `Sci Name_Common Name` are normalised to `Sci Name`.
fn load_labels(label_path: &Path) -> Result<Vec<String>> {
    let text = std::fs::read_to_string(label_path)
        .with_context(|| format!("Cannot read labels: {}", label_path.display()))?;

    let labels: Vec<String> = text
        .lines()
        .map(|line| {
            let line = line.trim();
            if line.matches('_').count() == 1 {
                line.split('_').next().unwrap_or(line).to_string()
            } else {
                line.to_string()
            }
        })
        .collect();

    info!("Loaded {} labels from {}", labels.len(), label_path.display());
    Ok(labels)
}

/// Load the JSON language file that maps `scientific_name → common_name`.
pub fn load_language(lang_dir: &Path, lang: &str) -> Result<HashMap<String, String>> {
    let file = lang_dir.join(format!("labels_{lang}.json"));
    let text = std::fs::read_to_string(&file)
        .with_context(|| format!("Cannot read language file: {}", file.display()))?;
    let map: HashMap<String, String> =
        serde_json::from_str(&text).context("Invalid language JSON")?;
    Ok(map)
}

/// Load a custom species list (include / exclude / whitelist).
pub fn load_species_list(path: &Path) -> Vec<String> {
    match std::fs::read_to_string(path) {
        Ok(text) => text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.trim().split('_').next().unwrap_or(l.trim()).to_string())
            .collect(),
        Err(_) => vec![],
    }
}

/// Convert lat/lon/week into the 6-element metadata vector for BirdNET V1.
fn convert_v1_metadata(lat: f64, lon: f64, week: u32) -> [f32; 6] {
    let w = if (1..=48).contains(&week) {
        (week as f64 * 7.5_f64.to_radians()).cos() + 1.0
    } else {
        -1.0
    };

    let (mask0, mask1, mask2) = if lat == -1.0 || lon == -1.0 {
        (0.0, 0.0, if w == -1.0 { 0.0 } else { 1.0 })
    } else {
        (1.0, 1.0, if w == -1.0 { 0.0 } else { 1.0 })
    };

    [lat as f32, lon as f32, w as f32, mask0, mask1, mask2]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_v1_metadata_normal() {
        let m = convert_v1_metadata(42.0, -72.0, 10);
        assert!((m[0] - 42.0).abs() < 1e-5);
        assert!((m[1] - (-72.0)).abs() < 1e-5);
        assert!((m[2] - 1.2588).abs() < 0.01);
        assert_eq!(m[3], 1.0);
        assert_eq!(m[4], 1.0);
        assert_eq!(m[5], 1.0);
    }

    #[test]
    fn test_v1_metadata_missing_location() {
        let m = convert_v1_metadata(-1.0, -1.0, 10);
        assert_eq!(m[3], 0.0);
        assert_eq!(m[4], 0.0);
        assert_eq!(m[5], 1.0);
    }

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }
}
