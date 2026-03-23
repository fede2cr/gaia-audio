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
    /// tract-onnx or tract-tflite runner.  `None` when the model was
    /// loaded through the ORT fallback (tract could not handle it).
    runner: Option<TypedRunnableModel<TypedModel>>,
    /// ONNX Runtime session – used when tract cannot load the model
    /// (e.g. unsupported DFT/STFT operators in BirdNET V3), **or** when
    /// `GAIA_ACCEL` is set for GPU-accelerated inference (rocm or cuda).
    /// Requires `libonnxruntime.so` to be available at runtime.
    ort_session: Option<crate::accel::OrtSession>,
    meta_model: Option<MetaDataModel>,
    labels: Vec<String>,
    /// Common-name map parsed from CSV labels (sci_name → com_name).
    /// Used as fallback when no JSON language file is available (e.g.
    /// BirdNET+ V3.0).
    csv_common_names: HashMap<String, String>,
    /// Taxonomic class map parsed from CSV labels (sci_name → class).
    /// Populated for models whose label file has a `class` column
    /// (e.g. BirdNET+ V3.0: Aves, Mammalia, Insecta, …).
    csv_classes: HashMap<String, String>,
    pub manifest: ResolvedManifest,
    sensitivity: f64,
    /// When `true` the ONNX model is a classifier-only sub-model that
    /// expects a mel-spectrogram input `[1, 96, 511, 2]` instead of raw
    /// audio `[1, N]`.  The mel computation is handled by [`crate::mel`].
    onnx_classifier: bool,
    /// One-time diagnostic flag: log raw logit statistics on the first
    /// inference so operators can verify the model's output scale.
    first_predict_logged: bool,
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
    let (runner, ort_session, onnx_classifier) = if let Some(onnx_path) = resolved.onnx_path() {
        if onnx_path.exists() {
            let is_classifier = resolved.manifest.model.onnx_is_classifier;
            let prefer_ort = resolved.manifest.model.prefer_ort;
            info!(
                "Loading ONNX model from {} (classifier={}, prefer_ort={})",
                onnx_path.display(),
                is_classifier,
                prefer_ort,
            );

            // 0. If the manifest says prefer_ort, skip tract entirely.
            //    Some models load in tract but produce incorrect inference
            //    (e.g. patched DFT → all-zero output, MatMul-replaced DFT
            //    → input-independent predictions).
            //
            //    Use ONNX Runtime CPU-only with explicit thread limits:
            //    without them, ORT deadlocks in CreateSession for models
            //    with DFT/STFT ops.
            if prefer_ort {
                let cache_dir = onnx_path.parent().unwrap_or(Path::new("/tmp")).join(".ort-cache");
                match crate::accel::OrtSession::new_cpu(&onnx_path, &cache_dir) {
                    Ok(sess) => {
                        info!(
                            "ORT active (prefer_ort) for {}",
                            onnx_path.display()
                        );
                        (None, Some(sess), is_classifier)
                    }
                    Err(e) => {
                        tracing::error!(
                            "prefer_ort is set but ORT session failed: {e:#}"
                        );
                        return Err(e.context(
                            "prefer_ort is set but ORT session creation failed \
                             (is libonnxruntime.so installed?)"
                        ));
                    }
                }
            } else {

            // 1. Try tract-onnx.
            let tract_result = load_onnx_runner(&onnx_path);

            // 2. If tract failed, try the baked (patched) copy.
            let tract_result = match tract_result {
                ok @ Ok(_) => ok,
                Err(e) => {
                    let fallback = onnx_path.file_name().map(|f| {
                        Path::new(crate::download::BAKED_MODELS_DIR).join(f)
                    });
                    if let Some(baked) = fallback.filter(|p| p.exists() && *p != onnx_path) {
                        tracing::warn!(
                            "ONNX load failed ({e:#}); retrying with baked model at {}",
                            baked.display()
                        );
                        if let Err(copy_err) = std::fs::copy(&baked, &onnx_path) {
                            tracing::warn!(
                                "Could not overwrite {} with baked model: {copy_err}",
                                onnx_path.display()
                            );
                        }
                        load_onnx_runner(&onnx_path)
                    } else {
                        Err(e)
                    }
                }
            };

            // 3. If tract still failed, try ONNX Runtime CPU fallback.
            //    This handles models with operators tract cannot optimise
            //    (e.g. DFT/STFT in BirdNET V3).
            //
            //    We force CPU-only here: models that fail tract have
            //    exotic ops (DFT, STFT) that MIGraphX / TensorRT
            //    cannot compile efficiently — attempting GPU compilation
            //    can hang for a very long time with no benefit.
            match tract_result {
                Ok(r) => (Some(r), None, is_classifier),
                Err(tract_err) => {
                    tracing::warn!(
                        "tract-onnx failed for {} ({tract_err:#}); \
                         trying ONNX Runtime CPU-only fallback",
                        onnx_path.display()
                    );
                    let cache_dir = onnx_path.parent().unwrap_or(Path::new("/tmp")).join(".ort-cache");
                    match crate::accel::OrtSession::new_cpu(&onnx_path, &cache_dir) {
                        Ok(sess) => {
                            info!(
                                "ONNX Runtime fallback active for {}",
                                onnx_path.display()
                            );
                            (None, Some(sess), is_classifier)
                        }
                        Err(ort_err) => {
                            tracing::error!(
                                "ONNX Runtime fallback also failed: {ort_err:#}"
                            );
                            return Err(tract_err.context(
                                "tract-onnx failed and ONNX Runtime fallback \
                                 is unavailable (is libonnxruntime.so installed?)"
                            ));
                        }
                    }
                }
            }
            } // else (tract path)
        } else {
            info!(
                "ONNX file configured but missing ({}), falling back to TFLite",
                onnx_path.display()
            );
            (Some(load_tflite_runner(&resolved.tflite_path())?), None, false)
        }
    } else {
        (Some(load_tflite_runner(&resolved.tflite_path())?), None, false)
    };
    let (labels, csv_common_names, csv_classes) = load_labels(&resolved.labels_path())?;

    let meta_model = match load_meta_model(resolved, &labels, config.sf_thresh) {
        Ok(m) => m,
        Err(e) => {
            tracing::warn!(
                "Metadata model failed to load – location-based filtering \
                 will be disabled: {e:#}"
            );
            None
        }
    };

    // BirdNET-Analyzer uses SIGMOID_SENSITIVITY (default 1.0) directly
    // as the slope of flat_sigmoid: 1/(1+exp(-sensitivity * clip(x,-20,20))).
    // Higher values → steeper sigmoid → higher reported confidences.
    let sensitivity = config.sensitivity.clamp(0.5, 1.5);

    // ── ORT session ───────────────────────────────────────────────────
    // The ORT session is only used when tract could not load the model
    // (e.g. unsupported DFT/STFT ops).  If tract succeeded, we keep the
    // tract runner — it's fast and avoids the potentially slow MIGraphX /
    // TensorRT compilation step.  When ORT *is* active and GAIA_ACCEL is
    // set, it will automatically try the GPU EP chain → CPU.

    Ok(LoadedModel {
        runner,
        ort_session,
        meta_model,
        labels,
        csv_common_names,
        csv_classes,
        manifest: resolved.clone(),
        sensitivity,
        onnx_classifier,
        first_predict_logged: false,
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

/// Validate that an ONNX file can be loaded, optimised, and made
/// runnable by tract-onnx.
///
/// This exercises the exact same code path as the runtime model
/// loader.  It is meant to be called at container **build** time
/// (via `gaia-processing validate-model <path>`) so that
/// incompatibilities (unsupported ops, variable Reshape shapes, etc.)
/// are caught before the image is published.
pub fn validate_onnx_with_tract(path: &Path) -> Result<()> {
    let _runner = load_onnx_runner(path)?;
    Ok(())
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

    /// Common-name map extracted from CSV labels.
    ///
    /// Non-empty for models whose label file is a CSV with a `com_name`
    /// column (e.g. BirdNET+ V3.0).  Can be used as a fallback when no
    /// JSON language file ships with the model.
    pub fn csv_common_names(&self) -> &HashMap<String, String> {
        &self.csv_common_names
    }

    /// Taxonomic class map extracted from CSV labels.
    ///
    /// Non-empty for models whose label file has a `class` column
    /// (e.g. BirdNET+ V3.0: Aves, Mammalia, Insecta, …).  Returns
    /// `sci_name → class`.  When the map is empty (V2.4, Perch, …)
    /// callers should fall back to [`Self::domain()`].
    pub fn csv_classes(&self) -> &HashMap<String, String> {
        &self.csv_classes
    }

    /// Full list of scientific names the model was trained on.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Whether this model has a species-range (metadata) model loaded.
    pub fn has_species_range_model(&self) -> bool {
        self.meta_model.is_some()
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
        &mut self,
        chunk: &[f32],
        lat: f64,
        lon: f64,
        week: u32,
    ) -> Result<Vec<Prediction>> {
        // ── ORT path (GPU-accelerated or CPU fallback) ───────────────
        if let Some(ort) = &mut self.ort_session {
            let out_idx = self.manifest.manifest.model.prediction_output_index;
            let logits = if self.onnx_classifier {
                let mel = crate::mel::birdnet_mel_spectrogram(chunk);
                ort.predict(mel, vec![1, 96, 511, 2], out_idx)?
            } else {
                let n = chunk.len();
                ort.predict(chunk.to_vec(), vec![1, n], out_idx)?
            };

            self.log_first_prediction(&logits);
            let scores = self.transform_scores(&logits);

            let mut predictions: Vec<Prediction> = self
                .labels
                .iter()
                .zip(scores.iter())
                .map(|(label, &score)| (label.clone(), score as f64))
                .collect();
            predictions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            return Ok(predictions);
        }

        // ── tract path (tract-onnx / tract-tflite) ──────────────────
        let runner = self.runner.as_ref()
            .context("No inference backend available (tract did not load and ORT is absent)")?;

        let result = if self.onnx_classifier {
            // ── ONNX classifier: audio → Rust mel → CNN ──────────
            let mel = crate::mel::birdnet_mel_spectrogram(chunk);
            let input: Tensor =
                tract_ndarray::Array4::from_shape_vec((1, 96, 511, 2), mel)
                    .context("Cannot reshape mel spectrogram")?
                    .into();
            runner
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
            runner
                .run(tvec![input.into(), mdata_tensor.into()])
                .context("V1 inference failed")?
        } else {
            // ── TFLite standard ──────────────────────────────────
            let n = chunk.len();
            let input: Tensor =
                tract_ndarray::Array2::from_shape_vec((1, n), chunk.to_vec())
                    .context("Cannot reshape audio chunk")?
                    .into();
            runner
                .run(tvec![input.into()])
                .context("Inference failed")?
        };

        let output = result[self.manifest.manifest.model.prediction_output_index]
            .to_array_view::<f32>()
            .context("Cannot read output tensor")?;

        let logits: Vec<f32> = output.iter().copied().collect();
        self.log_first_prediction(&logits);
        let scores = self.transform_scores(&logits);

        let mut predictions: Vec<Prediction> = self
            .labels
            .iter()
            .zip(scores.iter())
            .map(|(label, &score)| (label.clone(), score as f64))
            .collect();

        predictions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(predictions)
    }

    /// Log raw logit statistics once per model so operators can verify
    /// the model's output scale and diagnose zero-detection issues.
    fn log_first_prediction(&mut self, logits: &[f32]) {
        if self.first_predict_logged {
            return;
        }
        self.first_predict_logged = true;

        let name = &self.manifest.manifest.model.name;
        let n_logits = logits.len();
        let n_labels = self.labels.len();
        let transform = self.effective_transform();

        if n_logits != n_labels {
            tracing::warn!(
                "[{name}] Logit/label mismatch: model produced {n_logits} logits \
                 but labels file has {n_labels} entries.  Predictions will be \
                 truncated to the shorter of the two — check labels_file in \
                 the manifest."
            );
        }

        if logits.is_empty() {
            info!("[{name}] First inference: 0 logits (model produced no output!)");
            return;
        }

        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_logit = logits.iter().cloned().fold(f32::INFINITY, f32::min);
        let positive = logits.iter().filter(|&&x| x > 0.0).count();

        // Top-5 raw logits (before any transform).
        let mut sorted: Vec<f32> = logits.to_vec();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let top5: Vec<String> = sorted.iter().take(5).map(|x| format!("{x:.4}")).collect();

        info!(
            "[{name}] First inference: {n_logits} logits, {n_labels} labels, \
             range=[{min_logit:.4}, {max_logit:.4}], {positive} positive, \
             top5=[{}], transform={transform:?}",
            top5.join(", ")
        );

        // Also show what the transform does to the top logit.
        let top_score = match transform {
            crate::manifest::ScoreTransform::Sigmoid => {
                let s = self.sensitivity as f32;
                let c = max_logit.clamp(-20.0, 20.0);
                let raw = 1.0 / (1.0 + (-s * c).exp());
                if raw <= 0.5 { 0.0 } else { raw }
            }
            crate::manifest::ScoreTransform::Softmax => {
                // Compute softmax to show what the top score will be.
                let sm = softmax(logits);
                sm.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            }
            crate::manifest::ScoreTransform::CenteredSigmoid => {
                // Approximate: re-center top logit by the global mean.
                let n = logits.len() as f32;
                let mean = logits.iter().sum::<f32>() / n;
                let var = logits.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
                let std = var.sqrt().max(1.0);
                let z = ((max_logit - mean) / std).clamp(-20.0, 20.0);
                1.0 / (1.0 + (-z).exp())
            }
            crate::manifest::ScoreTransform::None => max_logit.clamp(0.0, 1.0),
        };
        if !top_score.is_nan() {
            info!(
                "[{name}] Top logit {max_logit:.4} → score {top_score:.4} \
                 (sensitivity={:.2})",
                self.sensitivity,
            );
        }
    }

    /// Resolve which score transform to use for this model.
    ///
    /// Priority: `score_transform` field > legacy `apply_softmax` bool > sigmoid default.
    fn effective_transform(&self) -> crate::manifest::ScoreTransform {
        use crate::manifest::ScoreTransform;
        if let Some(t) = self.manifest.manifest.model.score_transform {
            return t;
        }
        if self.manifest.manifest.model.apply_softmax {
            return ScoreTransform::Softmax;
        }
        ScoreTransform::Sigmoid
    }

    /// Transform raw model logits into 0..1 confidence scores.
    fn transform_scores(&self, logits: &[f32]) -> Vec<f32> {
        use crate::manifest::ScoreTransform;
        match self.effective_transform() {
            ScoreTransform::Softmax => softmax(logits),
            ScoreTransform::Sigmoid => self.sigmoid_scale(logits),
            ScoreTransform::CenteredSigmoid => centered_sigmoid(logits),
            ScoreTransform::None => {
                // Clamp raw logits to [0, 1].
                logits.iter().map(|&x| x.clamp(0.0, 1.0)).collect()
            }
        }
    }

    /// Apply sigmoid scaling with sensitivity.
    ///
    /// Matches BirdNET-Analyzer `flat_sigmoid`:
    ///   `1 / (1 + exp(-sensitivity * clip(x, -20, 20)))`
    /// where `sensitivity` = `cfg.SIGMOID_SENSITIVITY` (default 1.0).
    ///
    /// Scores ≤ 0.5 are clamped to 0: `sigmoid(0) = 0.5` is the
    /// mathematical "no signal" baseline, so anything at or below it
    /// carries no positive information and should not be treated as a
    /// detection at any confidence threshold.
    fn sigmoid_scale(&self, logits: &[f32]) -> Vec<f32> {
        let s = self.sensitivity as f32;
        logits
            .iter()
            .map(|&x| {
                let clamped = x.clamp(-20.0, 20.0);
                let score = 1.0 / (1.0 + (-s * clamped).exp());
                if score <= 0.5 { 0.0 } else { score }
            })
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

// ── centered sigmoid ─────────────────────────────────────────────────────

/// Z-score centered sigmoid: normalise logits to zero-mean / unit-variance
/// (with a floor of 1.0 on the std to avoid over-amplification), then
/// apply the standard sigmoid.  This ensures the per-chunk average logit
/// maps to 50% and only genuinely strong activations reach >80%.
fn centered_sigmoid(logits: &[f32]) -> Vec<f32> {
    let n = logits.len() as f32;
    if n == 0.0 {
        return vec![];
    }
    let mean = logits.iter().sum::<f32>() / n;
    let variance = logits.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
    let std = variance.sqrt().max(1.0);
    logits
        .iter()
        .map(|&x| {
            let z = ((x - mean) / std).clamp(-20.0, 20.0);
            1.0 / (1.0 + (-z).exp())
        })
        .collect()
}

// ── metadata model ───────────────────────────────────────────────────────

fn load_meta_model(
    resolved: &ResolvedManifest,
    _labels: &[String],
    sf_thresh: f64,
) -> Result<Option<MetaDataModel>> {
    // ── prefer ONNX when configured and present ──────────────────────
    if let Some(onnx_path) = resolved.metadata_onnx_path() {
        if onnx_path.exists() {
            info!("Loading ONNX metadata model: {}", onnx_path.display());
            let runner = load_onnx_runner(&onnx_path)
                .with_context(|| format!("Cannot load ONNX metadata model: {}", onnx_path.display()))?;
            let (labels, _, _) = load_labels(&resolved.metadata_labels_path())?;
            return Ok(Some(MetaDataModel {
                runner,
                labels,
                sf_thresh,
                cached_params: None,
                cached_list: vec![],
            }));
        } else {
            info!(
                "ONNX metadata model configured but missing ({}), trying TFLite",
                onnx_path.display()
            );
        }
    }

    // ── fall back to TFLite ──────────────────────────────────────────
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

    let (labels, _, _) = load_labels(&resolved.metadata_labels_path())?;

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

        // BirdNET-Analyzer's explore() compares the meta-model output
        // DIRECTLY to LOCATION_FILTER_THRESHOLD (no sigmoid).  The model
        // already outputs occurrence probabilities in [0, 1].
        let raw: Vec<f32> = output.iter().copied().collect();

        let mut scored: Vec<(f32, &str)> = raw
            .iter()
            .zip(self.labels.iter())
            .map(|(&score, label)| (score, label.as_str()))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Log the value range and top/bottom species for diagnostics.
        if let (Some(top), Some(bot)) = (scored.first(), scored.last()) {
            tracing::info!(
                "Meta-model raw output: {}/{} labels, range [{:.6}, {:.6}], \
                 top={} ({:.6}), bottom={} ({:.6})",
                raw.len(),
                self.labels.len(),
                bot.0,
                top.0,
                top.1,
                top.0,
                bot.1,
                bot.0,
            );
        }

        let list: Vec<String> = scored
            .iter()
            .filter(|(score, _)| *score >= self.sf_thresh as f32)
            .map(|(_, label)| label.split('_').next().unwrap_or(label).to_string())
            .collect();

        tracing::info!(
            "Species range filter: {} of {} species pass sf_thresh={:.3} at ({}, {}) week {}",
            list.len(),
            self.labels.len(),
            self.sf_thresh,
            lat,
            lon,
            week,
        );

        self.cached_params = Some(params);
        self.cached_list = list.clone();
        list
    }
}

// ── helpers ──────────────────────────────────────────────────────────────

/// Load label file.
///
/// Supports two formats:
///   - **Plain text** (`.txt`): one label per line.
///     Labels of the form `Sci Name_Common Name` are normalised to `Sci Name`.
///   - **CSV** (`.csv`): comma- or semicolon-separated values.
///     The delimiter is auto-detected from the first non-empty line.
///     When a header row is detected (starting with `ebird`, `species`,
///     or `idx`), it is used to locate the `sci_name` / `com_name`
///     columns; otherwise the first column is taken as the label.
///
/// Returns `(labels, common_names)`.  The common-name map is populated
/// only for CSV files that have a recognisable `com_name` column;
/// for plain-text labels the map is empty.
///
/// When the CSV also contains a `class` column (e.g. BirdNET+ V3.0:
/// `Aves`, `Mammalia`, `Insecta`, …), a third map `sci_name → class`
/// is returned so that callers can use per-species taxonomic class
/// instead of the coarse per-model domain.
fn load_labels(
    label_path: &Path,
) -> Result<(Vec<String>, HashMap<String, String>, HashMap<String, String>)> {
    let text = std::fs::read_to_string(label_path)
        .with_context(|| format!("Cannot read labels: {}", label_path.display()))?;

    // Strip the UTF-8 BOM if present (BirdNET+ V3.0 labels.csv starts with one).
    let text = text.strip_prefix('\u{feff}').unwrap_or(&text);

    let is_csv = label_path
        .extension()
        .map_or(false, |ext| ext.eq_ignore_ascii_case("csv"));

    let mut common_names: HashMap<String, String> = HashMap::new();
    let mut classes: HashMap<String, String> = HashMap::new();

    let labels: Vec<String> = if is_csv {
        // Auto-detect delimiter: semicolon if the first line contains `;`,
        // otherwise comma.
        let first_line = text.lines().find(|l| !l.trim().is_empty()).unwrap_or("");
        let delim = if first_line.contains(';') { ';' } else { ',' };

        // Detect header row and find the scientific name column index.
        let mut lines = text.lines().map(|l| l.trim()).filter(|l| !l.is_empty());
        let first = lines.next().unwrap_or("");
        let lower_first = first.to_lowercase();
        // Recognised header prefixes (BirdNET, Perch, general CSV).
        // Also treat single-token lines without a space as non-species
        // headers (e.g. Perch's "inat2024_fsd50k" dataset identifier),
        // since valid scientific names are always binomial ("Genus species").
        let is_header = lower_first.starts_with("ebird")
            || lower_first.starts_with("species")
            || lower_first.starts_with("idx")
            || lower_first.starts_with("inat")
            || (!first.contains(delim) && !first.contains(' '));

        // Find the column index for sci_name (or use 0 as default).
        let sci_col = if is_header {
            first
                .split(delim)
                .position(|col| {
                    let c = col.trim().to_lowercase();
                    c == "sci_name" || c == "scientific_name"
                })
                .unwrap_or(0)
        } else {
            0
        };

        // Find the column index for com_name / common_name (if present).
        let com_col = if is_header {
            first
                .split(delim)
                .position(|col| {
                    let c = col.trim().to_lowercase();
                    c == "com_name" || c == "common_name"
                })
        } else {
            None
        };

        // Find the column index for `class` (taxonomic class, e.g. Aves).
        let class_col = if is_header {
            first
                .split(delim)
                .position(|col| col.trim().eq_ignore_ascii_case("class"))
        } else {
            None
        };

        let data_lines: Box<dyn Iterator<Item = &str>> = if is_header {
            Box::new(lines)
        } else {
            // First line is data, not a header — include it.
            Box::new(std::iter::once(first).chain(lines))
        };

        data_lines
            .map(|line| {
                let cols: Vec<&str> = line.split(delim).collect();
                let sci = cols.get(sci_col).unwrap_or(&line).trim().to_string();
                if let Some(ci) = com_col {
                    if let Some(cn) = cols.get(ci) {
                        let cn = cn.trim();
                        if !cn.is_empty() {
                            common_names.insert(sci.clone(), cn.to_string());
                        }
                    }
                }
                if let Some(ci) = class_col {
                    if let Some(cls) = cols.get(ci) {
                        let cls = cls.trim();
                        if !cls.is_empty() {
                            classes.insert(sci.clone(), cls.to_string());
                        }
                    }
                }
                sci
            })
            .collect()
    } else {
        text.lines()
            .map(|line| {
                let line = line.trim();
                if line.matches('_').count() == 1 {
                    let sci = line.split('_').next().unwrap_or(line).to_string();
                    let com = line.split('_').nth(1).unwrap_or("");
                    if !com.is_empty() {
                        common_names.insert(sci.clone(), com.to_string());
                    }
                    sci
                } else {
                    line.to_string()
                }
            })
            .collect()
    };

    info!(
        "Loaded {} labels ({} with common names, {} with class) from {}",
        labels.len(),
        common_names.len(),
        classes.len(),
        label_path.display(),
    );
    Ok((labels, common_names, classes))
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
