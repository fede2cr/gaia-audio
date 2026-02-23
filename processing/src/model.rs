//! TFLite / ONNX model loading and inference – generalized for multiple domains.
//!
//! Evolved from `birdnet-server/src/model.rs`.  Instead of hard-coding model
//! variants, each model is described by a [`manifest::ResolvedManifest`] that
//! specifies sample rate, chunk duration, label format, etc.
//!
//! When a manifest specifies an `onnx_file` **and** that file exists on disk,
//! the model is loaded via `tract-onnx` instead of `tract-tflite`.  This
//! avoids unsupported-operator issues (e.g. `SPLIT_V` in BirdNET V2.4).

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use anyhow::{bail, Context, Result};
use tract_tflite::prelude::*;
use tract_onnx::prelude::InferenceModelExt as _;
use tracing::info;

use crate::manifest::ResolvedManifest;
use gaia_common::config::Config;

// ── constants ────────────────────────────────────────────────────────────

/// TFLite FlatBuffer schema identifier at bytes 4..8.
const TFLITE_SCHEMA_ID: &[u8; 4] = b"TFL3";

/// Minimum plausible size for a real TFLite model (header + at least one
/// tensor).  Anything smaller is almost certainly corrupt or truncated.
const MIN_TFLITE_SIZE: u64 = 1024;

// ── public types ─────────────────────────────────────────────────────────

/// A loaded model ready for inference, built from a manifest.
pub struct LoadedModel {
    runner: TypedRunnableModel<TypedModel>,
    meta_model: Option<MetaDataModel>,
    labels: Vec<String>,
    pub manifest: ResolvedManifest,
    sensitivity: f64,
    /// When `true` the ONNX model is a classifier-only sub-model that
    /// expects a mel-spectrogram input `[1, 96, 511, 2]` instead of raw
    /// audio `[1, N]`.  The mel computation is handled by [`crate::mel`].
    onnx_classifier: bool,
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

/// Validate a TFLite file *before* handing it to tract.
///
/// Checks performed (cheapest first):
///   1. File exists
///   2. File is not empty / not suspiciously small
///   3. FlatBuffer identifier bytes == `TFL3`
///   4. Root offset (first 4 bytes, little-endian u32) points inside the file
///   5. File is not accidentally a zip archive or HTML error page
fn validate_tflite_file(path: &Path) -> Result<()> {
    // 1. existence
    if !path.exists() {
        bail!(
            "TFLite file not found: {}. \
             Check that the model has been downloaded and extracted correctly.",
            path.display()
        );
    }

    // 2. size
    let meta = fs::metadata(path)
        .with_context(|| format!("Cannot stat {}", path.display()))?;

    if meta.len() == 0 {
        bail!("TFLite file is empty (0 bytes): {}", path.display());
    }
    if meta.len() < MIN_TFLITE_SIZE {
        bail!(
            "TFLite file is suspiciously small ({} bytes): {}. \
             Expected at least {} bytes for a valid model.",
            meta.len(),
            path.display(),
            MIN_TFLITE_SIZE,
        );
    }

    // Read the first 32 bytes – enough for all header checks.
    let header = {
        use std::io::Read;
        let mut f = fs::File::open(path)
            .with_context(|| format!("Cannot open {}", path.display()))?;
        let mut buf = [0u8; 32];
        let n = f.read(&mut buf)
            .with_context(|| format!("Cannot read header of {}", path.display()))?;
        buf[..n].to_vec()
    };

    if header.len() < 8 {
        bail!(
            "TFLite file too short to contain a valid header ({} bytes): {}",
            header.len(),
            path.display(),
        );
    }

    // 5a. Reject zip archives (PK\x03\x04 magic)
    if header.starts_with(b"PK\x03\x04") {
        bail!(
            "File appears to be a zip archive, not a TFLite model: {}. \
             The downloaded zip may not have been extracted.",
            path.display(),
        );
    }

    // 5b. Reject HTML error pages
    if header.starts_with(b"<!") || header.starts_with(b"<h") || header.starts_with(b"<H") {
        bail!(
            "File appears to be an HTML page, not a TFLite model: {}. \
             The download server may have returned an error page.",
            path.display(),
        );
    }

    // 3. FlatBuffer schema identifier at offset 4..8
    if header[4..8] != *TFLITE_SCHEMA_ID {
        let id = &header[4..8];
        bail!(
            "Invalid TFLite schema identifier in {}: expected {:?} (TFL3), \
             got {:?}. The file may be corrupt or not a TFLite model.",
            path.display(),
            TFLITE_SCHEMA_ID,
            id,
        );
    }

    // 4. Root table offset (bytes 0..4, little-endian u32) must point
    //    inside the file.
    let root_offset = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
    if root_offset as u64 >= meta.len() {
        bail!(
            "TFLite root table offset ({}) exceeds file size ({} bytes) in {}. \
             The file is likely truncated or corrupt.",
            root_offset,
            meta.len(),
            path.display(),
        );
    }

    info!(
        "Validated TFLite file: {} ({:.1} MB)",
        path.display(),
        meta.len() as f64 / (1024.0 * 1024.0),
    );
    Ok(())
}

/// Load a model from a resolved manifest.
///
/// Prefers ONNX when `onnx_file` is configured **and** the file exists;
/// otherwise falls back to TFLite.
pub fn load_model(resolved: &ResolvedManifest, config: &Config) -> Result<LoadedModel> {
    // ── choose format ────────────────────────────────────────────────
    let (runner, onnx_classifier) = if let Some(onnx_path) = resolved.onnx_path() {
        if onnx_path.exists() {
            info!("Loading ONNX classifier from {}", onnx_path.display());
            (load_onnx_runner(&onnx_path)?, true)
        } else {
            info!(
                "ONNX file configured but missing ({}), falling back to TFLite",
                onnx_path.display()
            );
            (load_tflite_runner(&resolved.tflite_path())?, false)
        }
    } else {
        (load_tflite_runner(&resolved.tflite_path())?, false)
    };

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
        onnx_classifier,
    })
}

/// Load and optimise a TFLite model file.
fn load_tflite_runner(path: &Path) -> Result<TypedRunnableModel<TypedModel>> {
    validate_tflite_file(path)
        .with_context(|| format!("Pre-flight check failed for {}", path.display()))?;
    info!("Loading TFLite model from {}", path.display());

    tract_tflite::tflite()
        .model_for_path(path)
        .with_context(|| format!("Cannot load TFLite model: {}", path.display()))?
        .into_optimized()
        .context("TFLite model optimisation failed")?
        .into_runnable()
        .context("Cannot make TFLite model runnable")
}

/// Load and optimise an ONNX model file.
fn load_onnx_runner(path: &Path) -> Result<TypedRunnableModel<TypedModel>> {
    tract_onnx::onnx()
        .model_for_path(path)
        .with_context(|| format!("Cannot load ONNX model: {}", path.display()))?
        .into_optimized()
        .context("ONNX model optimisation failed")?
        .into_runnable()
        .context("Cannot make ONNX model runnable")
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
    ///
    /// When the model is an ONNX classifier (split at the mel-spectrogram
    /// boundary), the mel preprocessing is computed in Rust via
    /// [`crate::mel::birdnet_mel_spectrogram`] before feeding the classifier.
    pub fn predict(
        &self,
        chunk: &[f32],
        lat: f64,
        lon: f64,
        week: u32,
    ) -> Result<Vec<Prediction>> {
        let result = if self.onnx_classifier {
            // ── ONNX classifier: audio → Rust mel → CNN ──────────
            let mel = crate::mel::birdnet_mel_spectrogram(chunk);
            let input: Tensor =
                tract_ndarray::Array4::from_shape_vec((1, 96, 511, 2), mel)
                    .context("Cannot reshape mel spectrogram")?
                    .into();
            self.runner
                .run(tvec![input.into()])
                .context("ONNX classifier inference failed")?
        } else if self.v1_metadata() {
            // ── TFLite V1 with metadata sidecar ──────────────────
            let n = chunk.len();
            let input: Tensor =
                tract_ndarray::Array2::from_shape_vec((1, n), chunk.to_vec())
                    .context("Cannot reshape audio chunk")?
                    .into();
            let mdata = convert_v1_metadata(lat, lon, week);
            let mdata_tensor: Tensor =
                tract_ndarray::Array2::from_shape_vec((1, 6), mdata.to_vec())
                    .context("Cannot reshape metadata")?
                    .into();
            self.runner
                .run(tvec![input.into(), mdata_tensor.into()])
                .context("V1 inference failed")?
        } else {
            // ── TFLite standard ──────────────────────────────────
            let n = chunk.len();
            let input: Tensor =
                tract_ndarray::Array2::from_shape_vec((1, n), chunk.to_vec())
                    .context("Cannot reshape audio chunk")?
                    .into();
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

    validate_tflite_file(&meta_path)
        .with_context(|| format!("Pre-flight check failed for metadata model {}", meta_path.display()))?;

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

    // ── validate_tflite_file tests ───────────────────────────────────

    #[test]
    fn test_validate_nonexistent_file() {
        let r = validate_tflite_file(Path::new("/tmp/does_not_exist_gaia_test.tflite"));
        assert!(r.is_err());
        let msg = format!("{}", r.unwrap_err());
        assert!(msg.contains("not found"), "got: {msg}");
    }

    #[test]
    fn test_validate_empty_file() {
        let dir = std::env::temp_dir().join("gaia_test_validate");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("empty.tflite");
        fs::write(&path, b"").unwrap();
        let r = validate_tflite_file(&path);
        assert!(r.is_err());
        let msg = format!("{}", r.unwrap_err());
        assert!(msg.contains("empty"), "got: {msg}");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_validate_too_small() {
        let dir = std::env::temp_dir().join("gaia_test_validate");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("tiny.tflite");
        fs::write(&path, b"hello").unwrap();
        let r = validate_tflite_file(&path);
        assert!(r.is_err());
        let msg = format!("{}", r.unwrap_err());
        assert!(msg.contains("suspiciously small"), "got: {msg}");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_validate_zip_file_rejected() {
        let dir = std::env::temp_dir().join("gaia_test_validate");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("fake.tflite");
        // PK\x03\x04 header + padding
        let mut data = vec![0x50, 0x4B, 0x03, 0x04];
        data.resize(2048, 0);
        fs::write(&path, &data).unwrap();
        let r = validate_tflite_file(&path);
        assert!(r.is_err());
        let msg = format!("{}", r.unwrap_err());
        assert!(msg.contains("zip archive"), "got: {msg}");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_validate_html_rejected() {
        let dir = std::env::temp_dir().join("gaia_test_validate");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("error.tflite");
        let mut data = b"<!DOCTYPE html><html>403 Forbidden</html>".to_vec();
        data.resize(2048, 0);
        fs::write(&path, &data).unwrap();
        let r = validate_tflite_file(&path);
        assert!(r.is_err());
        let msg = format!("{}", r.unwrap_err());
        assert!(msg.contains("HTML"), "got: {msg}");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_validate_bad_schema_id() {
        let dir = std::env::temp_dir().join("gaia_test_validate");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("bad_id.tflite");
        // Valid-looking root offset (16) but wrong schema id
        let mut data = vec![0; 2048];
        data[0..4].copy_from_slice(&16u32.to_le_bytes()); // root offset
        data[4..8].copy_from_slice(b"XXXX");               // wrong id
        fs::write(&path, &data).unwrap();
        let r = validate_tflite_file(&path);
        assert!(r.is_err());
        let msg = format!("{}", r.unwrap_err());
        assert!(msg.contains("schema identifier"), "got: {msg}");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_validate_truncated_root_offset() {
        let dir = std::env::temp_dir().join("gaia_test_validate");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("truncated.tflite");
        let mut data = vec![0; 2048];
        data[0..4].copy_from_slice(&999999u32.to_le_bytes()); // offset past EOF
        data[4..8].copy_from_slice(b"TFL3");
        fs::write(&path, &data).unwrap();
        let r = validate_tflite_file(&path);
        assert!(r.is_err());
        let msg = format!("{}", r.unwrap_err());
        assert!(msg.contains("truncated") || msg.contains("exceeds"), "got: {msg}");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_validate_good_file() {
        let dir = std::env::temp_dir().join("gaia_test_validate");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("good.tflite");
        let mut data = vec![0; 2048];
        data[0..4].copy_from_slice(&16u32.to_le_bytes()); // valid root offset
        data[4..8].copy_from_slice(b"TFL3");               // correct schema
        fs::write(&path, &data).unwrap();
        assert!(validate_tflite_file(&path).is_ok());
        let _ = fs::remove_file(&path);
    }

    /// Smoke-test ONNX model loading via tract-onnx.
    ///
    /// Only runs when the classifier ONNX file exists at the expected path.
    #[test]
    fn test_load_onnx_classifier() {
        let onnx_path = std::path::Path::new("/tmp/birdnet_v2.4_classifier.onnx");
        if !onnx_path.exists() {
            eprintln!("Skipping ONNX test: {onnx_path:?} not found");
            return;
        }
        let runner = load_onnx_runner(onnx_path)
            .expect("Failed to load ONNX model");

        // Run inference with zeros input (1, 96, 511, 2)
        let input = tract_ndarray::Array4::<f32>::zeros((1, 96, 511, 2));
        let input_tensor: Tensor = input.into();
        let result = runner
            .run(tvec![input_tensor.into()])
            .expect("ONNX inference failed");

        let output = result[0]
            .to_array_view::<f32>()
            .expect("Cannot read output");
        assert_eq!(output.shape(), &[1, 6522], "unexpected output shape");
        eprintln!("ONNX output sum: {:.4}", output.iter().sum::<f32>());
    }

    /// End-to-end test: Rust mel spectrogram → ONNX classifier → compare
    /// predictions with the Python/Keras reference.
    ///
    /// This validates the full inference pipeline that will run in
    /// production: audio → mel.rs preprocessing → tract-onnx classifier.
    #[test]
    fn test_end_to_end_mel_onnx() {
        let audio_path = std::path::Path::new("/tmp/test_audio_raw.f32");
        let pred_ref_path = std::path::Path::new("/tmp/test_pred_ref.f32");
        let onnx_path = std::path::Path::new("/tmp/birdnet_v2.4_classifier.onnx");

        if !audio_path.exists() || !pred_ref_path.exists() || !onnx_path.exists() {
            eprintln!("Skipping end-to-end test: reference files not found");
            return;
        }

        // 1. Load audio
        let audio_bytes = std::fs::read(audio_path).unwrap();
        let audio: Vec<f32> = audio_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(audio.len(), 144000);

        // 2. Compute mel spectrogram in Rust
        let mel = crate::mel::birdnet_mel_spectrogram(&audio);
        assert_eq!(mel.len(), 96 * 511 * 2);

        // 3. Run ONNX classifier
        let runner = load_onnx_runner(onnx_path)
            .expect("Failed to load ONNX model");

        let input = tract_ndarray::Array4::from_shape_vec((1, 96, 511, 2), mel)
            .expect("Cannot reshape mel spectrogram");
        let input_tensor: Tensor = input.into();
        let result = runner
            .run(tvec![input_tensor.into()])
            .expect("ONNX inference failed");

        let output = result[0]
            .to_array_view::<f32>()
            .expect("Cannot read output");
        assert_eq!(output.shape(), &[1, 6522]);

        let rust_pred: Vec<f32> = output.iter().copied().collect();

        // 4. Load Python/Keras reference predictions
        let ref_bytes = std::fs::read(pred_ref_path).unwrap();
        let ref_pred: Vec<f32> = ref_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(ref_pred.len(), 6522);

        // 5. Compare predictions
        let mut max_diff = 0.0f32;
        let mut sum_diff = 0.0f64;
        for (&r, &p) in rust_pred.iter().zip(ref_pred.iter()) {
            let d = (r - p).abs();
            if d > max_diff {
                max_diff = d;
            }
            sum_diff += d as f64;
        }
        let mean_diff = sum_diff / 6522.0;

        // Top-5 from Rust
        let mut rust_top: Vec<(usize, f32)> = rust_pred
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        rust_top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        rust_top.truncate(5);

        // Top-5 from reference
        let mut ref_top: Vec<(usize, f32)> = ref_pred
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        ref_top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        ref_top.truncate(5);

        eprintln!("=== End-to-end Rust mel → ONNX vs Python/Keras reference ===");
        eprintln!("Max diff: {max_diff:.6}");
        eprintln!("Mean diff: {mean_diff:.8}");
        eprintln!("Rust top-5:");
        for (i, (idx, score)) in rust_top.iter().enumerate() {
            eprintln!("  #{}: index {idx:5}, confidence {score:.6}", i + 1);
        }
        eprintln!("Reference top-5:");
        for (i, (idx, score)) in ref_top.iter().enumerate() {
            eprintln!("  #{}: index {idx:5}, confidence {score:.6}", i + 1);
        }

        // Check that top-1 species matches
        assert_eq!(
            rust_top[0].0, ref_top[0].0,
            "Top-1 species index mismatch: Rust={} vs Ref={}",
            rust_top[0].0, ref_top[0].0
        );

        // Allow some tolerance due to mel spectrogram float differences
        // propagating through the neural network
        assert!(
            max_diff < 0.1,
            "Prediction max diff too large: {max_diff}"
        );
        assert!(
            mean_diff < 0.01,
            "Prediction mean diff too large: {mean_diff}"
        );
    }
}
