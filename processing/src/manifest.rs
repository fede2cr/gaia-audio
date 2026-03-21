//! Model manifest – describes a TFLite model's properties.
//!
//! Each model lives in its own directory with a `manifest.toml`:
//!
//! ```toml
//! [model]
//! name = "BirdNET V2.4"
//! domain = "birds"
//! sample_rate = 48000
//! chunk_duration = 3.0
//! tflite_file = "audio-model-fp16.tflite"
//! labels_file = "en_us.txt"
//! v1_metadata = false
//!
//! [metadata_model]
//! enabled = true
//! tflite_file = "meta-model.tflite"
//!
//! [download]
//! zenodo_record_id = "15050749"
//! default_variant = "fp16"
//!
//! [download.variants.fp32]
//! zenodo_file = "BirdNET_v2.4_tflite.zip"
//! md5 = "c13f7fd28a5f7a3b092cd993087f93f7"
//! tflite_file = "audio-model.tflite"
//!
//! [download.variants.fp16]
//! zenodo_file = "BirdNET_v2.4_tflite_fp16.zip"
//! md5 = "4cd35da63e442d974faf2121700192b5"
//! tflite_file = "audio-model-fp16.tflite"
//!
//! [download.variants.int8]
//! zenodo_file = "BirdNET_v2.4_tflite_int8.zip"
//! md5 = "69becc3e8eb1c72d1d9dae7f21062c74"
//! tflite_file = "audio-model-int8.tflite"
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;
use tracing::info;

/// How raw model outputs (logits) should be transformed into 0..1
/// confidence scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoreTransform {
    /// BirdNET-style sigmoid:  `1/(1+exp(-sensitivity*clip(x,-20,20)))`
    /// with scores ≤ 0.5 clamped to 0.  This is the default for
    /// backward compatibility.
    #[default]
    Sigmoid,
    /// Standard softmax normalisation across all classes.
    Softmax,
    /// Z-score centered sigmoid: `sigmoid((x - mean) / max(std, 1.0))`.
    ///
    /// Normalises each chunk's logits so that the per-chunk average maps
    /// to 50% confidence.  Species whose logit is significantly above
    /// the mean get high confidence; species near the average stay near
    /// 50%.  The standard-deviation divisor (floored at 1.0) prevents
    /// over-amplification when all logits are similar.
    ///
    /// Use for models with many output classes where raw logits cluster
    /// at similar values (e.g. Google Perch 2.0 with ~15K classes).
    CenteredSigmoid,
    /// No transformation — raw logits are clamped to 0..1 and used
    /// as-is.  Only suitable when the model already outputs values in
    /// the 0..1 range.
    None,
}

/// Top-level manifest structure.
#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    pub model: ModelSection,
    #[serde(default)]
    pub metadata_model: Option<MetadataSection>,
    #[serde(default)]
    pub language: Option<LanguageSection>,
    #[serde(default)]
    pub download: Option<DownloadSection>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelSection {
    pub name: String,
    /// Short machine-friendly identifier (e.g. `"birdnet"`).  Derived
    /// from `name` when not explicitly set in the manifest.
    #[serde(default)]
    pub slug: Option<String>,
    pub domain: String,
    pub sample_rate: u32,
    pub chunk_duration: f64,
    pub tflite_file: String,
    pub labels_file: String,
    /// Optional ONNX model file.  When present **and** the file exists on
    /// disk, tract-onnx is used instead of tract-tflite.  This avoids
    /// unsupported-operator issues (e.g. SPLIT_V in BirdNET V2.4).
    #[serde(default)]
    pub onnx_file: Option<String>,
    /// BirdNET V1 requires a metadata input tensor.
    #[serde(default)]
    pub v1_metadata: bool,
    /// Whether to apply softmax to raw logits.
    /// **Deprecated** — prefer `score_transform`.  Kept for backward
    /// compatibility: when `score_transform` is unset and this is `true`,
    /// softmax is used.
    #[serde(default)]
    pub apply_softmax: bool,
    /// How to convert raw model logits into 0..1 confidence scores.
    ///   - `"sigmoid"` (default): BirdNET-style flat sigmoid.
    ///   - `"softmax"`: standard softmax across all classes.
    ///   - `"none"`: clamp to 0..1, no transformation (Perch).
    #[serde(default)]
    pub score_transform: Option<ScoreTransform>,
    /// When `true`, the ONNX model is a classifier sub-model that expects
    /// a precomputed mel-spectrogram input `[1, 96, 511, 2]` instead of
    /// raw audio `[1, N]`.  Only BirdNET V2.4's extracted classifier needs
    /// this; most ONNX models (e.g. Perch) accept raw audio directly.
    #[serde(default)]
    pub onnx_is_classifier: bool,
    /// Which ONNX output tensor contains the class predictions.
    ///
    /// Multi-output models place predictions at different indices:
    ///   - BirdNET V2.4:  1 output  → predictions at index **0** (default)
    ///   - BirdNET V3.0:  2 outputs → predictions at index **1**
    ///                    (output 0 = 1280-dim embeddings)
    ///   - Perch V2:      4 outputs → predictions at index **3**
    ///                    (0 = embedding, 1 = spatial_emb, 2 = spectrogram)
    ///
    /// Defaults to `0` for backward compatibility with single-output models.
    #[serde(default)]
    pub prediction_output_index: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MetadataSection {
    #[serde(default = "default_true")]
    pub enabled: bool,
    pub tflite_file: String,
    /// Optional ONNX variant of the metadata model.  When present **and**
    /// the file exists on disk, tract-onnx is used instead of tract-tflite.
    /// This avoids unsupported-operator issues (e.g. STRIDED_SLICE with
    /// shrink_axis_mask in BirdNET V2.4's metadata model).
    #[serde(default)]
    pub onnx_file: Option<String>,
    /// Optional separate labels file for the metadata model.
    /// When set, the meta-model uses this file instead of the main
    /// model's labels.  Useful when reusing a V2.4 meta-model with
    /// V3.0 (which has a different species set).
    #[serde(default)]
    pub labels_file: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LanguageSection {
    /// Subdirectory containing `labels_{lang}.json` files.
    #[serde(default = "default_l18n")]
    pub dir: String,
}

/// Download configuration – automatic model fetching from Zenodo or
/// direct URLs (e.g. HuggingFace).
#[derive(Debug, Clone, Deserialize)]
pub struct DownloadSection {
    /// Zenodo record ID (e.g. "15050749").  Required for Zenodo downloads;
    /// omit for models using only `direct_files`.
    #[serde(default)]
    pub zenodo_record_id: Option<String>,
    /// Default variant if `MODEL_VARIANT` is not set (e.g. "fp16").
    #[serde(default = "default_variant")]
    pub default_variant: String,
    /// Map of variant name → download info.  Empty for non-Zenodo models.
    #[serde(default)]
    pub variants: HashMap<String, VariantInfo>,
    /// Optional Keras zip filename on Zenodo for ONNX conversion.
    /// When specified, `ensure_onnx_file()` will download this zip,
    /// extract `audio-model.h5`, and convert the classifier sub-model
    /// to ONNX using `scripts/convert_keras_to_onnx.py`.
    #[serde(default)]
    pub keras_zenodo_file: Option<String>,
    /// Expected MD5 hex digest of the Keras zip file.
    #[serde(default)]
    pub keras_md5: Option<String>,
    /// Direct file downloads: maps local filename → remote URL.
    ///
    /// Files are downloaded individually (not from a Zenodo zip).  Useful
    /// for models hosted on HuggingFace or other direct-download sources.
    ///
    /// ```toml
    /// [download.direct_files]
    /// "model.onnx" = "https://huggingface.co/org/repo/resolve/main/model.onnx"
    /// "labels.csv" = "https://huggingface.co/org/repo/resolve/main/labels.csv"
    /// ```
    #[serde(default)]
    pub direct_files: HashMap<String, String>,
}

/// Information about a single model variant available on Zenodo.
#[derive(Debug, Clone, Deserialize)]
pub struct VariantInfo {
    /// Filename of the zip on Zenodo (e.g. "BirdNET_v2.4_tflite_fp16.zip").
    pub zenodo_file: String,
    /// Expected MD5 hex digest of the zip file.
    #[serde(default)]
    pub md5: Option<String>,
    /// Override for `[model].tflite_file` when this variant is selected.
    #[serde(default)]
    pub tflite_file: Option<String>,    /// Override for `[model].onnx_file` when this variant is selected.
    #[serde(default)]
    pub onnx_file: Option<String>,    /// Override for `[model].labels_file` when this variant is selected.
    #[serde(default)]
    pub labels_file: Option<String>,
    /// Override for `[metadata_model].tflite_file` when this variant is selected.
    #[serde(default)]
    pub metadata_tflite_file: Option<String>,
}

fn default_variant() -> String {
    "fp16".to_string()
}

fn default_true() -> bool {
    true
}
fn default_l18n() -> String {
    "l18n".to_string()
}

/// A resolved manifest with absolute paths.
#[derive(Debug, Clone)]
pub struct ResolvedManifest {
    pub manifest: Manifest,
    /// Directory containing the manifest and model files.
    pub base_dir: PathBuf,
}

impl ResolvedManifest {
    pub fn tflite_path(&self) -> PathBuf {
        self.base_dir.join(&self.manifest.model.tflite_file)
    }

    /// Path to the ONNX model file, if configured.
    pub fn onnx_path(&self) -> Option<PathBuf> {
        self.manifest
            .model
            .onnx_file
            .as_ref()
            .map(|f| self.base_dir.join(f))
    }

    pub fn labels_path(&self) -> PathBuf {
        self.base_dir.join(&self.manifest.model.labels_file)
    }

    pub fn metadata_tflite_path(&self) -> Option<PathBuf> {
        self.manifest
            .metadata_model
            .as_ref()
            .filter(|m| m.enabled)
            .map(|m| self.base_dir.join(&m.tflite_file))
    }

    /// Path to the ONNX metadata model file, if configured.
    pub fn metadata_onnx_path(&self) -> Option<PathBuf> {
        self.manifest
            .metadata_model
            .as_ref()
            .filter(|m| m.enabled)
            .and_then(|m| m.onnx_file.as_ref())
            .map(|f| self.base_dir.join(f))
    }

    /// Labels file for the metadata model.
    ///
    /// When `[metadata_model].labels_file` is set, returns that path;
    /// otherwise falls back to the main model's labels.
    pub fn metadata_labels_path(&self) -> PathBuf {
        self.manifest
            .metadata_model
            .as_ref()
            .and_then(|m| m.labels_file.as_ref())
            .map(|f| self.base_dir.join(f))
            .unwrap_or_else(|| self.labels_path())
    }

    pub fn language_dir(&self) -> PathBuf {
        let sub = self
            .manifest
            .language
            .as_ref()
            .map(|l| l.dir.as_str())
            .unwrap_or("l18n");
        self.base_dir.join(sub)
    }

    pub fn domain(&self) -> &str {
        &self.manifest.model.domain
    }

    /// Short machine-friendly identifier for the model.
    ///
    /// Uses the explicit `slug` from the manifest when present, otherwise
    /// derives one from the model name (lower-case, non-alphanumeric
    /// characters replaced with `-`, collapsed, trimmed).
    pub fn slug(&self) -> String {
        if let Some(ref s) = self.manifest.model.slug {
            if !s.is_empty() {
                return s.clone();
            }
        }
        // Derive from name: "BirdNET V2.4" → "birdnet-v2-4"
        let raw: String = self
            .manifest
            .model
            .name
            .to_lowercase()
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
            .collect();
        // Collapse consecutive dashes and trim leading/trailing
        let mut slug = String::with_capacity(raw.len());
        let mut prev_dash = true; // suppress leading dash
        for c in raw.chars() {
            if c == '-' {
                if !prev_dash {
                    slug.push('-');
                }
                prev_dash = true;
            } else {
                slug.push(c);
                prev_dash = false;
            }
        }
        slug.trim_end_matches('-').to_string()
    }

    /// Apply variant overrides from the `[download]` section.
    ///
    /// If the selected variant provides `tflite_file`, `labels_file`, or
    /// `metadata_tflite_file` overrides, they replace the corresponding
    /// fields in the manifest.
    pub fn apply_variant(&mut self, variant_name: &str) -> Result<()> {
        let download = match &self.manifest.download {
            Some(d) => d.clone(),
            None => return Ok(()),
        };

        // Models with only direct_files (no Zenodo variants) skip
        // variant resolution entirely — there's nothing to override.
        if download.variants.is_empty() {
            return Ok(());
        }

        let variant = download
            .variants
            .get(variant_name)
            .with_context(|| {
                format!(
                    "Unknown model variant '{}'. Available: {:?}",
                    variant_name,
                    download.variants.keys().collect::<Vec<_>>()
                )
            })?;

        if let Some(ref tf) = variant.tflite_file {
            self.manifest.model.tflite_file = tf.clone();
        }
        if let Some(ref of_) = variant.onnx_file {
            self.manifest.model.onnx_file = Some(of_.clone());
        }
        if let Some(ref lf) = variant.labels_file {
            self.manifest.model.labels_file = lf.clone();
        }
        if let Some(ref mf) = variant.metadata_tflite_file {
            if let Some(ref mut meta) = self.manifest.metadata_model {
                meta.tflite_file = mf.clone();
            }
        }

        Ok(())
    }

    /// Resolve the effective variant name from config or manifest default.
    pub fn effective_variant(&self, config_variant: Option<&str>) -> Option<String> {
        self.manifest.download.as_ref().map(|d| {
            config_variant
                .unwrap_or(&d.default_variant)
                .to_string()
        })
    }
}

/// Load a single manifest from a directory.
pub fn load_manifest(dir: &Path) -> Result<ResolvedManifest> {
    let manifest_path = dir.join("manifest.toml");
    let text = std::fs::read_to_string(&manifest_path)
        .with_context(|| format!("Cannot read {}", manifest_path.display()))?;
    let manifest: Manifest =
        toml::from_str(&text).with_context(|| format!("Invalid manifest: {}", manifest_path.display()))?;
    info!(
        "Loaded model manifest: {} (domain={}, sr={}, chunk={}s)",
        manifest.model.name,
        manifest.model.domain,
        manifest.model.sample_rate,
        manifest.model.chunk_duration,
    );
    Ok(ResolvedManifest {
        manifest,
        base_dir: dir.to_path_buf(),
    })
}

/// Auto-discover all model manifests under `root_dir`.
///
/// Each immediate subdirectory that contains a `manifest.toml` is loaded.
/// When multiple directories yield the same slug (e.g. an old `birds/` and
/// a new `birdnet/`), only the first one found is kept — duplicates are
/// logged and skipped.
pub fn discover_manifests(root_dir: &Path) -> Result<Vec<ResolvedManifest>> {
    let mut manifests = Vec::new();

    if !root_dir.exists() {
        tracing::warn!(
            "Model directory does not exist: {}. No models will be loaded.",
            root_dir.display()
        );
        return Ok(manifests);
    }

    // Check if root_dir itself has a manifest (single-model setup)
    let root_manifest = root_dir.join("manifest.toml");
    if root_manifest.exists() {
        manifests.push(load_manifest(root_dir)?);
        return Ok(manifests);
    }

    // Otherwise scan subdirectories
    let mut seen_slugs = std::collections::HashSet::new();
    for entry in std::fs::read_dir(root_dir)
        .with_context(|| format!("Cannot read {}", root_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() && path.join("manifest.toml").exists() {
            match load_manifest(&path) {
                Ok(m) => {
                    let slug = m.slug();
                    if seen_slugs.contains(&slug) {
                        tracing::warn!(
                            "Duplicate slug '{}' in {} — skipping (already loaded from another directory)",
                            slug,
                            path.display(),
                        );
                    } else {
                        seen_slugs.insert(slug);
                        manifests.push(m);
                    }
                }
                Err(e) => tracing::warn!("Skipping {}: {e}", path.display()),
            }
        }
    }

    if manifests.is_empty() {
        tracing::warn!(
            "No model manifests found in {}. \
             Each model subdirectory needs a manifest.toml. \
             The processing server will start but cannot analyse audio until models are available.",
            root_dir.display()
        );
    } else {
        info!("Discovered {} model(s)", manifests.len());
    }
    Ok(manifests)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_manifest() {
        let toml = r#"
[model]
name = "BirdNET V2.4"
domain = "birds"
sample_rate = 48000
chunk_duration = 3.0
tflite_file = "model.tflite"
labels_file = "labels.txt"

[metadata_model]
enabled = true
tflite_file = "meta.tflite"
"#;
        let m: Manifest = toml::from_str(toml).unwrap();
        assert_eq!(m.model.domain, "birds");
        assert_eq!(m.model.sample_rate, 48000);
        assert!(m.metadata_model.unwrap().enabled);
    }

    #[test]
    fn test_minimal_manifest() {
        let toml = r#"
[model]
name = "BatDetect2"
domain = "bats"
sample_rate = 256000
chunk_duration = 1.0
tflite_file = "batdetect.tflite"
labels_file = "bat_labels.txt"
"#;
        let m: Manifest = toml::from_str(toml).unwrap();
        assert_eq!(m.model.domain, "bats");
        assert_eq!(m.model.sample_rate, 256000);
        assert!(!m.model.v1_metadata);
        assert!(m.metadata_model.is_none());
        assert!(m.download.is_none());
    }

    #[test]
    fn test_manifest_with_download() {
        let toml = r#"
[model]
name = "BirdNET V2.4"
domain = "birds"
sample_rate = 48000
chunk_duration = 3.0
tflite_file = "model_fp16.tflite"
labels_file = "labels.txt"

[download]
zenodo_record_id = "15050749"
default_variant = "fp16"

[download.variants.fp32]
zenodo_file = "BirdNET_v2.4_tflite.zip"
md5 = "c13f7fd28a5f7a3b092cd993087f93f7"
tflite_file = "model_fp32.tflite"

[download.variants.fp16]
zenodo_file = "BirdNET_v2.4_tflite_fp16.zip"
md5 = "4cd35da63e442d974faf2121700192b5"

[download.variants.int8]
zenodo_file = "BirdNET_v2.4_tflite_int8.zip"
md5 = "69becc3e8eb1c72d1d9dae7f21062c74"
tflite_file = "model_int8.tflite"
"#;
        let m: Manifest = toml::from_str(toml).unwrap();
        let dl = m.download.as_ref().unwrap();
        assert_eq!(dl.zenodo_record_id.as_deref(), Some("15050749"));
        assert_eq!(dl.default_variant, "fp16");
        assert_eq!(dl.variants.len(), 3);
        assert_eq!(dl.variants["fp32"].zenodo_file, "BirdNET_v2.4_tflite.zip");
        assert_eq!(
            dl.variants["fp32"].tflite_file.as_deref(),
            Some("model_fp32.tflite")
        );
        // fp16 has no override → uses the [model] default
        assert!(dl.variants["fp16"].tflite_file.is_none());
    }

    #[test]
    fn test_apply_variant_overrides() {
        let toml = r#"
[model]
name = "Test"
domain = "test"
sample_rate = 48000
chunk_duration = 3.0
tflite_file = "default.tflite"
labels_file = "labels.txt"

[download]
zenodo_record_id = "12345"

[download.variants.fp32]
zenodo_file = "test.zip"
tflite_file = "big_model.tflite"
labels_file = "big_labels.txt"

[download.variants.int8]
zenodo_file = "test_int8.zip"
tflite_file = "small_model.tflite"
"#;
        let m: Manifest = toml::from_str(toml).unwrap();
        let mut resolved = ResolvedManifest {
            manifest: m,
            base_dir: PathBuf::from("/tmp/models/test"),
        };

        // Apply fp32 variant
        resolved.apply_variant("fp32").unwrap();
        assert_eq!(resolved.manifest.model.tflite_file, "big_model.tflite");
        assert_eq!(resolved.manifest.model.labels_file, "big_labels.txt");

        // Reset and apply int8
        resolved.manifest.model.tflite_file = "default.tflite".into();
        resolved.manifest.model.labels_file = "labels.txt".into();
        resolved.apply_variant("int8").unwrap();
        assert_eq!(resolved.manifest.model.tflite_file, "small_model.tflite");
        // labels_file not overridden by int8 variant
        assert_eq!(resolved.manifest.model.labels_file, "labels.txt");
    }

    #[test]
    fn test_effective_variant() {
        let toml = r#"
[model]
name = "Test"
domain = "test"
sample_rate = 48000
chunk_duration = 3.0
tflite_file = "model.tflite"
labels_file = "labels.txt"

[download]
zenodo_record_id = "12345"
default_variant = "fp16"

[download.variants.fp16]
zenodo_file = "test.zip"
"#;
        let m: Manifest = toml::from_str(toml).unwrap();
        let resolved = ResolvedManifest {
            manifest: m,
            base_dir: PathBuf::from("/tmp"),
        };

        // No config override → uses default
        assert_eq!(resolved.effective_variant(None), Some("fp16".to_string()));
        // Config override
        assert_eq!(
            resolved.effective_variant(Some("int8")),
            Some("int8".to_string())
        );
    }
}
