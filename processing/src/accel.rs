//! ONNX Runtime integration – adaptive GPU / CPU inference.
//!
//! [`OrtSession`] wraps an `ort::session::Session` and presents a
//! uniform `predict()` API to the rest of the codebase.
//!
//! | `GAIA_ACCEL` env var | EP chain tried                          |
//! |----------------------|-----------------------------------------|
//! | `rocm`               | MIGraphX → ROCm → CPU                  |
//! | `cuda`               | TensorRT → CUDA → CPU                  |
//! | anything else / unset| CPU only                                |
//!
//! The `ort` crate is compiled with **`load-dynamic`**: if
//! `libonnxruntime.so` is not installed at runtime, session creation
//! returns an error and the caller falls through to tract-onnx or
//! skips the model.
//!
//! Both MIGraphX and TensorRT cache compiled plans on disk so
//! subsequent starts skip the slow compilation step.

use std::path::Path;

use anyhow::{Context, Result};
use ort::session::builder::GraphOptimizationLevel;
use tracing::info;

/// Read the `ORT_INTRA_THREADS` env var, falling back to `default`.
///
/// In constrained environments (Docker build, CI) set this to `1` to
/// avoid thread-pool deadlocks during `CreateSession` for models with
/// complex DFT/STFT subgraphs (Perch, BirdNET V3).
fn ort_intra_threads(default: usize) -> usize {
    std::env::var("ORT_INTRA_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

// ── Runtime check ────────────────────────────────────────────────────────

/// Detected acceleration backend from the `GAIA_ACCEL` env var.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccelKind {
    /// AMD ROCm — MIGraphX → ROCm → CPU.
    Rocm,
    /// NVIDIA CUDA — TensorRT → CUDA → CPU.
    Cuda,
    /// No GPU acceleration requested.
    None,
}

/// Parse the `GAIA_ACCEL` environment variable.
pub fn accel_kind() -> AccelKind {
    match std::env::var("GAIA_ACCEL").as_deref() {
        Ok(v) if v.eq_ignore_ascii_case("rocm") => AccelKind::Rocm,
        Ok(v) if v.eq_ignore_ascii_case("cuda") => AccelKind::Cuda,
        _ => AccelKind::None,
    }
}

/// Returns `true` when the operator has requested ROCm acceleration
/// via the `GAIA_ACCEL` environment variable.
///
/// This is set by gaia-core's `inject_rocm_args()` when AMD GPUs with
/// `/dev/kfd` are detected on the host.
pub fn is_rocm_requested() -> bool {
    accel_kind() == AccelKind::Rocm
}

/// Returns `true` when CUDA/TensorRT acceleration is requested.
pub fn is_cuda_requested() -> bool {
    accel_kind() == AccelKind::Cuda
}

/// Returns `true` when any GPU acceleration is requested.
pub fn is_gpu_requested() -> bool {
    accel_kind() != AccelKind::None
}

// ── Adaptive ORT session ─────────────────────────────────────────────────

/// An ONNX Runtime session wrapping the `ort` crate.
///
/// ### GPU mode (`new`)
///
/// When `GAIA_ACCEL=rocm` is set, the session is configured with:
///   1. MIGraphX EP (compiles ONNX → optimised HIP kernels)
///   2. ROCm EP (MIOpen-based, broader op coverage)
///   3. CPU fallback
///
/// When `GAIA_ACCEL=cuda`:
///   1. TensorRT EP
///   2. CUDA EP
///   3. CPU fallback
///
/// ### CPU mode (`new_cpu`)
///
/// CPU-only with explicit thread limits.  Critical for models with
/// DFT/STFT ops (BirdNET V3, Perch): without explicit
/// `intra_threads` / `inter_threads`, ORT's default thread pool
/// deadlocks inside `CreateSession`.
pub struct OrtSession {
    session: ort::session::Session,
    /// Whether we have logged the output shape info (once per session).
    shapes_logged: bool,
    /// Whether we have logged the output data shape (once per session).
    shape_data_logged: bool,
}

impl OrtSession {
    /// Create an ORT session for the given ONNX file with GPU
    /// acceleration (when `GAIA_ACCEL` is set).
    ///
    /// `cache_dir` is used by MIGraphX / TensorRT to store compiled
    /// plans.  Ignored when running on CPU only.
    ///
    /// **Not currently used**: `prefer_ort` models use `new_cpu()`.
    /// Kept for future GPU-accelerated inference.
    #[allow(dead_code)]
    pub fn new(onnx_path: &Path, cache_dir: &Path) -> Result<Self> {
        let kind = accel_kind();

        match kind {
            AccelKind::Rocm => info!(
                "Creating ONNX Runtime session (MIGraphX → ROCm → CPU) for {}",
                onnx_path.display()
            ),
            AccelKind::Cuda => info!(
                "Creating ONNX Runtime session (TensorRT → CUDA → CPU) for {}",
                onnx_path.display()
            ),
            AccelKind::None => info!(
                "Creating ONNX Runtime session (CPU) for {}",
                onnx_path.display()
            ),
        }

        let intra = ort_intra_threads(4);
        info!("ORT thread config: intra={intra}, inter=1");

        let mut builder = ort::session::Session::builder()
            .context("Failed to create ORT session builder (is libonnxruntime.so installed?)")?
            // Thread limits prevent deadlocks in CreateSession for models
            // with DFT/STFT ops.  Applied to all EP chains — the CPU
            // fallback within a GPU chain can trigger the same hang.
            //
            // Configurable via ORT_INTRA_THREADS (default 4).  Set to 1
            // in Docker builds / CI where CPU resources are constrained.
            .with_intra_threads(intra)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_inter_threads(1)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            // Disable memory-pattern optimisation — it does a full graph
            // walk to plan memory reuse and can hang on complex DFT/STFT
            // subgraphs (Perch V2, BirdNET V3).
            .with_memory_pattern(false)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            // Disable thread spinning — in constrained environments
            // (Docker builds / CI) spinning threads cause contention
            // that can escalate into deadlocks during session init.
            .with_intra_op_spinning(false)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_inter_op_spinning(false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        // GPU EPs (MIGraphX, TensorRT) do their own compilation/optimisation,
        // so full graph optimisation is fine.  For CPU-only, disable all
        // graph optimisation — even Level1 (Constant Folding) can hang
        // for 10+ minutes on models with complex DFT/STFT subgraphs.
        if kind == AccelKind::None {
            builder = builder
                .with_optimization_level(GraphOptimizationLevel::Disable)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        }

        builder = match kind {
            AccelKind::Rocm => {
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
            }
            AccelKind::Cuda => {
                // Ensure the TensorRT engine cache directory exists.
                std::fs::create_dir_all(cache_dir).ok();
                let cache_str = cache_dir.to_string_lossy().to_string();

                // TensorRT achieves ~2× speedup over CUDA by compiling
                // the ONNX graph into optimised GPU kernels.  The compiled
                // engine is cached on disk so subsequent starts are fast.
                //
                // FP16 is enabled for models that support it (most bird
                // detection models do) — this halves VRAM usage and
                // further increases throughput on Tensor Core GPUs.
                let tensorrt = ort::ep::TensorRT::default()
                    .with_device_id(0)
                    .with_fp16(true)
                    .with_engine_cache(true)
                    .with_engine_cache_path(&cache_str)
                    .with_timing_cache(true)
                    .with_timing_cache_path(&cache_str);

                let cuda = ort::ep::CUDA::default()
                    .with_device_id(0);

                builder
                    .with_execution_providers([
                        tensorrt.build(),
                        cuda.build(),
                        ort::ep::CPU::default().build(),
                    ])
                    .map_err(|e| anyhow::anyhow!("{e}"))?
            }
            AccelKind::None => {
                builder
                    .with_execution_providers([ort::ep::CPU::default().build()])
                    .map_err(|e| anyhow::anyhow!("{e}"))?
            }
        };

        let session = builder
            .commit_from_file(onnx_path)
            .with_context(|| format!("ORT: failed to load {}", onnx_path.display()))?;

        info!(
            "ORT session created for {} ({} inputs, {} outputs, accel={:?})",
            onnx_path.display(),
            session.inputs().len(),
            session.outputs().len(),
            kind,
        );

        Ok(Self {
            session,
            shapes_logged: false,
            shape_data_logged: false,
        })
    }

    /// Create an ORT session with **CPU-only** execution, ignoring
    /// `GAIA_ACCEL`.
    ///
    /// Used when ORT is a fallback for models that tract cannot load
    /// (e.g. DFT/STFT ops in Perch, BirdNET V3).  These models have
    /// exotic ops that MIGraphX / TensorRT may not support —
    /// attempting GPU compilation hangs for 20+ minutes and produces
    /// no benefit.  CPU inference is fast enough (~66 ms per chunk).
    ///
    /// `cache_dir` is provided for future use but currently ignored
    /// (CPU EP has no caching).
    pub fn new_cpu(onnx_path: &Path, _cache_dir: &Path) -> Result<Self> {
        info!(
            "Creating ONNX Runtime session (CPU-only fallback) for {}",
            onnx_path.display()
        );

        // Thread configuration is critical for models with DFT/STFT ops
        // (BirdNET V3, Perch).  Without explicit limits the ORT default
        // (0 = "let ORT decide") creates a huge thread pool that deadlocks
        // inside CreateSession for complex signal-processing subgraphs.
        //
        // Configurable via ORT_INTRA_THREADS (default 4).  Set to 1
        // in Docker builds / CI to avoid deadlocks when CPU resources
        // are constrained (BuildKit containers often have limited cores).
        let intra = ort_intra_threads(4);
        info!("ORT thread config: intra={intra}, inter=1");

        let session = ort::session::Session::builder()
            .context("Failed to create ORT session builder (is libonnxruntime.so installed?)")?
            .with_intra_threads(intra)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_inter_threads(1)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_optimization_level(GraphOptimizationLevel::Disable)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            // Disable memory-pattern optimisation — it does a full graph
            // walk to plan memory reuse and can hang on complex DFT/STFT
            // subgraphs (Perch V2, BirdNET V3).  Adds negligible runtime
            // cost since these models are loaded once.
            .with_memory_pattern(false)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            // Disable thread spinning to prevent contention deadlocks
            // during session init in constrained environments.
            .with_intra_op_spinning(false)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_inter_op_spinning(false)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_execution_providers([ort::ep::CPU::default().build()])
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .commit_from_file(onnx_path)
            .with_context(|| format!("ORT CPU: failed to load {}", onnx_path.display()))?;

        info!(
            "ORT session created for {} ({} inputs, {} outputs, accel=CPU-only)",
            onnx_path.display(),
            session.inputs().len(),
            session.outputs().len(),
        );

        Ok(Self {
            session,
            shapes_logged: false,
            shape_data_logged: false,
        })
    }

    /// Run inference on a batch of f32 input data.
    ///
    /// `input_data` is the flattened tensor; `input_shape` is its
    /// dimensions.  `output_index` selects which output tensor to read
    /// (multi-output models place predictions at a non-zero index).
    /// Returns the raw output logits/scores.
    pub fn predict(
        &mut self,
        input_data: Vec<f32>,
        input_shape: Vec<usize>,
        output_index: usize,
    ) -> Result<Vec<f32>> {
        let session = &mut self.session;
        let input_name = session.inputs()[0].name().to_string();

        // Log all output names on the first call.
        if !self.shapes_logged {
            self.shapes_logged = true;
            for (i, out_meta) in session.outputs().iter().enumerate() {
                info!("ORT output[{i}]: name={:?}", out_meta.name());
            }
        }

        let shape_i64: Vec<i64> = input_shape.iter().map(|&d| d as i64).collect();
        let input_tensor =
            ort::value::Tensor::from_array((shape_i64, input_data.into_boxed_slice()))
                .context("Failed to create input tensor")?;

        let outputs = session
            .run(ort::inputs![input_name => input_tensor])
            .context("ORT inference failed")?;

        anyhow::ensure!(
            output_index < outputs.len(),
            "Model has {} outputs but prediction_output_index = {output_index}",
            outputs.len()
        );

        let (shape, data) = outputs[output_index]
            .try_extract_tensor::<f32>()
            .context("Cannot extract f32 output tensor")?;

        if !self.shape_data_logged {
            self.shape_data_logged = true;
            info!(
                "ORT predict: output[{output_index}] shape={shape:?}, len={}",
                data.len()
            );
        }

        Ok(data.to_vec())
    }
}
