//! ONNX Runtime integration – adaptive GPU / CPU inference.
//!
//! [`OrtSession`] wraps an `ort::session::Session` and selects the best
//! available execution provider at session creation time:
//!
//! | `GAIA_ACCEL` env var | EP chain tried                      |
//! |----------------------|-------------------------------------|
//! | `rocm`               | MIGraphX → ROCm → CPU              |
//! | anything else / unset| CPU only                            |
//!
//! The `ort` crate is compiled with **`load-dynamic`**: if
//! `libonnxruntime.so` is not installed at runtime, session creation
//! returns an error and the caller falls through to tract-onnx or
//! skips the model.
//!
//! MIGraphX caches compiled HIP plans on disk so subsequent starts
//! skip the slow compilation step.

use std::path::Path;

use anyhow::{Context, Result};
use tracing::info;

// ── Runtime check ────────────────────────────────────────────────────────

/// Returns `true` when the operator has requested ROCm acceleration
/// via the `GAIA_ACCEL` environment variable.
///
/// This is set by gaia-core's `inject_rocm_args()` when AMD GPUs with
/// `/dev/kfd` are detected on the host.
pub fn is_rocm_requested() -> bool {
    std::env::var("GAIA_ACCEL")
        .map(|v| v.eq_ignore_ascii_case("rocm"))
        .unwrap_or(false)
}

// ── Adaptive ORT session ─────────────────────────────────────────────────

/// An ONNX Runtime session that automatically selects the best execution
/// provider.
///
/// When `GAIA_ACCEL=rocm` is set, the session is configured with:
///   1. MIGraphX EP (compiles ONNX → optimised HIP kernels)
///   2. ROCm EP (MIOpen-based, broader op coverage)
///   3. CPU fallback
///
/// Otherwise only the CPU EP is registered.
///
/// Requires `libonnxruntime.so` at runtime.  If the library is absent,
/// `new()` returns `Err` and the caller can skip this path.
pub struct OrtSession {
    session: ort::session::Session,
}

impl OrtSession {
    /// Create an ORT session for the given ONNX file.
    ///
    /// `cache_dir` is used by MIGraphX to store compiled plans (`.mxr`
    /// files).  Ignored when running on CPU only.
    pub fn new(onnx_path: &Path, cache_dir: &Path) -> Result<Self> {
        let use_gpu = is_rocm_requested();

        if use_gpu {
            info!(
                "Creating ONNX Runtime session (MIGraphX → ROCm → CPU) for {}",
                onnx_path.display()
            );
        } else {
            info!(
                "Creating ONNX Runtime session (CPU) for {}",
                onnx_path.display()
            );
        }

        let mut builder = ort::session::Session::builder()
            .context("Failed to create ORT session builder (is libonnxruntime.so installed?)")?;

        builder = if use_gpu {
            // Ensure the MIGraphX cache directory exists.
            std::fs::create_dir_all(cache_dir).ok();

            let model_stem = onnx_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("model");
            let cache_file = cache_dir.join(format!("{model_stem}.mxr"));
            let cache_str = cache_file.to_string_lossy().to_string();

            let mut migraphx = ort::ep::MIGraphX::default()
                .with_device_id(0)
                .with_fp16(true)
                .with_save_model(&cache_str);

            if cache_file.exists() {
                migraphx = migraphx.with_load_model(&cache_str);
            }

            builder
                .with_execution_providers([
                    migraphx.build(),
                    ort::ep::ROCm::default().with_device_id(0).build(),
                    ort::ep::CPU::default().build(),
                ])
                .map_err(|e| anyhow::anyhow!("{e}"))?
        } else {
            builder
                .with_execution_providers([ort::ep::CPU::default().build()])
                .map_err(|e| anyhow::anyhow!("{e}"))?
        };

        let session = builder
            .commit_from_file(onnx_path)
            .with_context(|| format!("ORT: failed to load {}", onnx_path.display()))?;

        info!(
            "ORT session created for {} ({} inputs, {} outputs, gpu={})",
            onnx_path.display(),
            session.inputs().len(),
            session.outputs().len(),
            use_gpu,
        );

        Ok(Self { session })
    }

    /// Run inference on a batch of f32 input data.
    ///
    /// `input_data` is the flattened tensor; `input_shape` is its
    /// dimensions.  Returns the raw output logits/scores.
    pub fn predict(&mut self, input_data: Vec<f32>, input_shape: Vec<usize>) -> Result<Vec<f32>> {
        let input_name = self.session.inputs()[0].name().to_string();

        let shape_i64: Vec<i64> = input_shape.iter().map(|&d| d as i64).collect();
        let input_tensor =
            ort::value::Tensor::from_array((shape_i64, input_data.into_boxed_slice()))
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
}
