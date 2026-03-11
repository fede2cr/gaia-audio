//! GPU-accelerated inference via ONNX Runtime with MIGraphX execution provider.
//!
//! This module is compiled only when the `rocm` feature is enabled and
//! activated at runtime via the `GAIA_ACCEL=rocm` environment variable
//! (injected by gaia-core when ROCm hardware is detected).
//!
//! ## Architecture
//!
//! - [`AccelSession`] wraps an `ort::session::Session` configured with the
//!   MIGraphX EP (falling back to ROCm EP, then CPU).
//! - The session caches the compiled MIGraphX plan on disk so subsequent
//!   runs skip the slow compilation step.
//! - [`is_rocm_requested`] checks the `GAIA_ACCEL` env var at startup
//!   to decide whether to use this module or the default tract backend.
//!
//! ## MIGraphX vs ROCm EP
//!
//! MIGraphX compiles the full ONNX graph into an optimised HIP kernel
//! sequence, giving the best throughput on AMD GPUs.  The ROCm EP
//! (MIOpen-based) is the fallback when MIGraphX is unavailable.

#[cfg(feature = "rocm")]
use std::path::Path;

#[cfg(feature = "rocm")]
use anyhow::{Context, Result};

#[cfg(feature = "rocm")]
use tracing::info;

// ── Runtime check ────────────────────────────────────────────────────────

/// Returns `true` when the operator has requested ROCm acceleration
/// via the `GAIA_ACCEL` environment variable.
///
/// This is set by gaia-core's `maybe_inject_rocm_args()` when AMD GPUs
/// with `/dev/kfd` are detected on the host.
pub fn is_rocm_requested() -> bool {
    std::env::var("GAIA_ACCEL")
        .map(|v| v.eq_ignore_ascii_case("rocm"))
        .unwrap_or(false)
}

// ── Accelerated session ──────────────────────────────────────────────────

/// An ONNX Runtime session configured for GPU-accelerated inference.
///
/// When `rocm` feature is enabled, this wraps `ort::session::Session` with
/// the MIGraphX execution provider.  Provides a `predict()` method matching
/// the interface expected by `model.rs`.
#[cfg(feature = "rocm")]
pub struct AccelSession {
    session: ort::session::Session,
    /// Number of output classes (labels).
    #[allow(dead_code)]
    num_classes: usize,
}

#[cfg(feature = "rocm")]
impl AccelSession {
    /// Create a new accelerated session for the given ONNX model file.
    ///
    /// Tries execution providers in order:
    ///   1. MIGraphX (best AMD GPU performance, compiles ONNX → HIP)
    ///   2. ROCm (MIOpen-based, broader op coverage)
    ///   3. CPU (always works, fallback)
    ///
    /// The `cache_dir` is used by MIGraphX to store compiled plans so
    /// subsequent loads are fast.
    pub fn new(onnx_path: &Path, cache_dir: &Path, num_classes: usize) -> Result<Self> {
        info!(
            "Creating ONNX Runtime session with MIGraphX EP for {}",
            onnx_path.display()
        );

        // Ensure the MIGraphX cache directory exists.
        std::fs::create_dir_all(cache_dir).ok();

        // Build a cache path for the compiled MIGraphX model (.mxr).
        let model_stem = onnx_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model");
        let cache_file = cache_dir.join(format!("{model_stem}.mxr"));
        let cache_str = cache_file.to_string_lossy().to_string();

        // Configure MIGraphX EP with model caching.
        let mut migraphx = ort::ep::MIGraphX::default()
            .with_device_id(0)
            .with_fp16(true)
            .with_save_model(&cache_str);

        // If a previously compiled model exists, load it to skip recompilation.
        if cache_file.exists() {
            migraphx = migraphx.with_load_model(&cache_str);
        }

        let session = ort::session::Session::builder()
            .context("Failed to create ORT session builder")?
            .with_execution_providers([
                // 1. MIGraphX — compiles the graph into optimised HIP kernels.
                migraphx.build(),
                // 2. ROCm EP — MIOpen-based, broader compatibility.
                ort::ep::ROCm::default()
                    .with_device_id(0)
                    .build(),
                // 3. CPU fallback — always available.
                ort::ep::CPU::default()
                    .build(),
            ])
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .commit_from_file(onnx_path)
            .with_context(|| format!("Failed to load ONNX model: {}", onnx_path.display()))?;

        // Log which EP was actually selected.
        info!(
            "ONNX Runtime session created for {} ({} inputs, {} outputs)",
            onnx_path.display(),
            session.inputs().len(),
            session.outputs().len(),
        );

        Ok(Self {
            session,
            num_classes,
        })
    }

    /// Run inference on a batch of f32 input data.
    ///
    /// `input_data` is the flattened tensor; `input_shape` is its dimensions.
    /// Returns the raw output logits/scores as a `Vec<f32>`.
    pub fn predict(&mut self, input_data: Vec<f32>, input_shape: Vec<usize>) -> Result<Vec<f32>> {
        let input_name = self.session.inputs()[0].name().to_string();

        let shape_i64: Vec<i64> = input_shape.iter().map(|&d| d as i64).collect();
        let input_tensor = ort::value::Tensor::from_array(
            (shape_i64, input_data.into_boxed_slice()),
        )
        .context("Failed to create input tensor")?;

        let outputs = self
            .session
            .run(ort::inputs![input_name => input_tensor])
            .context("ORT inference failed")?;

        let (_shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .context("Cannot extract f32 output tensor")?;

        Ok(data.to_vec())
    }

    /// Number of output classes.
    #[allow(dead_code)]
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }
}

// ── Stub when feature is disabled ────────────────────────────────────────

/// When the `rocm` feature is not compiled in but `GAIA_ACCEL=rocm` is
/// set, the processing server logs a warning and falls back to CPU.
#[cfg(not(feature = "rocm"))]
pub fn warn_if_requested() {
    if is_rocm_requested() {
        tracing::warn!(
            "GAIA_ACCEL=rocm is set but this binary was built without the 'rocm' feature. \
             GPU acceleration is not available — falling back to CPU inference (tract-onnx). \
             Rebuild with `cargo build --features rocm` for MIGraphX support."
        );
    }
}
