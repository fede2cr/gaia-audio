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
    pub domain: String,
    pub sample_rate: u32,
    pub chunk_duration: f64,
    pub tflite_file: String,
    pub labels_file: String,
    /// BirdNET V1 requires a metadata input tensor.
    #[serde(default)]
    pub v1_metadata: bool,
    /// Whether to apply softmax to raw logits (e.g. Perch).
    #[serde(default)]
    pub apply_softmax: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MetadataSection {
    #[serde(default = "default_true")]
    pub enabled: bool,
    pub tflite_file: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LanguageSection {
    /// Subdirectory containing `labels_{lang}.json` files.
    #[serde(default = "default_l18n")]
    pub dir: String,
}

/// Zenodo download configuration – allows automatic model fetching.
#[derive(Debug, Clone, Deserialize)]
pub struct DownloadSection {
    /// Zenodo record ID (e.g. "15050749").
    pub zenodo_record_id: String,
    /// Default variant if `MODEL_VARIANT` is not set (e.g. "fp16").
    #[serde(default = "default_variant")]
    pub default_variant: String,
    /// Map of variant name → download info.
    pub variants: HashMap<String, VariantInfo>,
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
    pub tflite_file: Option<String>,
    /// Override for `[model].labels_file` when this variant is selected.
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
    for entry in std::fs::read_dir(root_dir)
        .with_context(|| format!("Cannot read {}", root_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() && path.join("manifest.toml").exists() {
            match load_manifest(&path) {
                Ok(m) => manifests.push(m),
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
        assert_eq!(dl.zenodo_record_id, "15050749");
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
