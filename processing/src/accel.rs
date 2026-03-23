//! ONNX Runtime integration – adaptive GPU / CPU inference.
//!
//! [`OrtSession`] wraps either a native `ort::session::Session` or a
//! persistent Python `onnxruntime` subprocess, and presents a uniform
//! `predict()` API to the rest of the codebase.
//!
//! ## Native ORT (`new`, `new_cpu`)
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
//! ## Python ORT subprocess (`new_python`)
//!
//! For models with DFT/STFT ops (BirdNET V3, Perch), the Rust `ort`
//! crate hangs indefinitely in `CreateSession` regardless of EP or
//! optimization level.  Python's `onnxruntime` pip package handles
//! them without issue.  `new_python()` spawns a persistent
//! `ort_worker.py` child process and communicates via a JSON-lines +
//! raw-binary protocol over stdin/stdout.
//!
//! Both MIGraphX and TensorRT cache compiled plans on disk so
//! subsequent starts skip the slow compilation step.

use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use anyhow::{bail, Context, Result};
use ort::session::builder::GraphOptimizationLevel;
use tracing::info;

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

/// Default path to the Python ORT worker script (baked into the container).
const DEFAULT_ORT_WORKER: &str = "/usr/local/share/gaia/ort_worker.py";

/// An ONNX Runtime session that either uses the native `ort` crate or
/// a persistent Python `onnxruntime` subprocess.
///
/// ### Native mode (`new`, `new_cpu`)
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
/// Otherwise only the CPU EP is registered.
///
/// ### Python mode (`new_python`)
///
/// Spawns `ort_worker.py` and communicates over stdin/stdout.  Used for
/// models with DFT/STFT ops that hang the Rust `ort` crate.
pub struct OrtSession {
    inner: SessionKind,
    /// Whether we have logged the output shape info (once per session).
    shapes_logged: bool,
    /// Whether we have logged the output data shape (once per session).
    shape_data_logged: bool,
}

/// Either a native ORT session or a Python subprocess.
enum SessionKind {
    Native(ort::session::Session),
    Python {
        child: Child,
        stdin: std::io::BufWriter<ChildStdin>,
        stdout: BufReader<ChildStdout>,
        model_id: String,
    },
}

impl OrtSession {
    /// Create an ORT session for the given ONNX file with GPU
    /// acceleration (when `GAIA_ACCEL` is set).
    ///
    /// `cache_dir` is used by MIGraphX / TensorRT to store compiled
    /// plans.  Ignored when running on CPU only.
    ///
    /// **Not currently used**: `prefer_ort` models use `new_cpu()`
    /// because the pip-bundled `libonnxruntime.so` (which handles
    /// DFT/STFT models without hanging) is CPU-only.  Kept for
    /// future use when GPU-capable ORT builds that handle DFT ops
    /// become available.
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

        let mut builder = ort::session::Session::builder()
            .context("Failed to create ORT session builder (is libonnxruntime.so installed?)")?;

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
            inner: SessionKind::Native(session),
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

        let mut builder = ort::session::Session::builder()
            .context("Failed to create ORT session builder (is libonnxruntime.so installed?)")?
            .with_optimization_level(GraphOptimizationLevel::Disable)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_execution_providers([ort::ep::CPU::default().build()])
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        let session = builder
            .commit_from_file(onnx_path)
            .with_context(|| format!("ORT CPU: failed to load {}", onnx_path.display()))?;

        info!(
            "ORT session created for {} ({} inputs, {} outputs, accel=CPU-only)",
            onnx_path.display(),
            session.inputs().len(),
            session.outputs().len(),
        );

        Ok(Self {
            inner: SessionKind::Native(session),
            shapes_logged: false,
            shape_data_logged: false,
        })
    }

    /// Create an ORT session backed by a **persistent Python subprocess**.
    ///
    /// The Rust `ort` crate hangs in `CreateSession` for models with
    /// DFT/STFT ops, regardless of EP or optimisation level.
    /// Python's `onnxruntime` pip package handles them without issue.
    ///
    /// `model_id` is a short unique slug (e.g. `"birdnet3"`) used to
    /// identify the model in the worker protocol.
    pub fn new_python(onnx_path: &Path, model_id: &str) -> Result<Self> {
        let worker_script = std::env::var("GAIA_ORT_WORKER")
            .unwrap_or_else(|_| DEFAULT_ORT_WORKER.to_string());

        info!(
            "Spawning Python ORT worker for {} (script={})",
            onnx_path.display(),
            worker_script,
        );

        let mut child = Command::new("python3")
            .arg(&worker_script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit()) // so Python tracebacks appear in container logs
            .spawn()
            .with_context(|| format!(
                "Failed to spawn Python ORT worker (is python3 installed?). \
                 Script: {worker_script}"
            ))?;

        let child_stdin = child.stdin.take().context("No stdin on Python child")?;
        let child_stdout = child.stdout.take().context("No stdout on Python child")?;

        let mut stdin_writer = std::io::BufWriter::new(child_stdin);
        let mut stdout_reader = BufReader::new(child_stdout);

        // Send load command
        let load_cmd = serde_json::json!({
            "cmd": "load",
            "id": model_id,
            "path": onnx_path.to_string_lossy(),
        });
        writeln!(stdin_writer, "{}", load_cmd).context("Failed to write load command")?;
        stdin_writer.flush()?;

        // Read load response
        let mut resp_line = String::new();
        stdout_reader
            .read_line(&mut resp_line)
            .context("Failed to read load response from Python ORT worker")?;

        let resp: serde_json::Value = serde_json::from_str(resp_line.trim())
            .with_context(|| format!("Invalid JSON from worker: {resp_line:?}"))?;

        if resp.get("ok").and_then(|v| v.as_bool()) != Some(true) {
            let err_msg = resp
                .get("error")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown error");
            bail!("Python ORT worker failed to load {}: {err_msg}", onnx_path.display());
        }

        let n_inputs = resp.get("inputs").and_then(|v| v.as_u64()).unwrap_or(0);
        let n_outputs = resp.get("outputs").and_then(|v| v.as_u64()).unwrap_or(0);

        info!(
            "Python ORT session ready for {} ({} inputs, {} outputs)",
            onnx_path.display(),
            n_inputs,
            n_outputs,
        );

        Ok(Self {
            inner: SessionKind::Python {
                child,
                stdin: stdin_writer,
                stdout: stdout_reader,
                model_id: model_id.to_string(),
            },
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
        match &mut self.inner {
            SessionKind::Native(session) => {
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
            SessionKind::Python {
                stdin,
                stdout,
                model_id,
                ..
            } => {
                let n_floats = input_data.len();

                // 1. Send JSON header
                let cmd = serde_json::json!({
                    "cmd": "predict",
                    "id": model_id,
                    "shape": input_shape,
                    "output_index": output_index,
                    "n_floats": n_floats,
                });
                writeln!(stdin, "{}", cmd).context("Failed to write predict command")?;
                stdin.flush()?;

                // 2. Send raw f32 bytes (little-endian)
                let raw_bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        input_data.as_ptr() as *const u8,
                        n_floats * std::mem::size_of::<f32>(),
                    )
                };
                stdin.write_all(raw_bytes).context("Failed to write input tensor bytes")?;
                stdin.flush()?;

                // 3. Read JSON response header
                let mut resp_line = String::new();
                stdout
                    .read_line(&mut resp_line)
                    .context("Failed to read predict response")?;

                let resp: serde_json::Value = serde_json::from_str(resp_line.trim())
                    .with_context(|| format!("Invalid JSON from worker: {resp_line:?}"))?;

                if resp.get("ok").and_then(|v| v.as_bool()) != Some(true) {
                    let err_msg = resp
                        .get("error")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown error");
                    bail!("Python ORT predict failed: {err_msg}");
                }

                let out_n = resp
                    .get("n_floats")
                    .and_then(|v| v.as_u64())
                    .context("Missing n_floats in response")? as usize;

                if !self.shapes_logged {
                    self.shapes_logged = true;
                    info!("Python ORT predict: output[{output_index}] len={out_n}");
                }

                // 4. Read raw f32 output bytes
                let mut out_bytes = vec![0u8; out_n * std::mem::size_of::<f32>()];
                stdout
                    .read_exact(&mut out_bytes)
                    .context("Failed to read output tensor bytes")?;

                let out_floats: Vec<f32> = out_bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();

                Ok(out_floats)
            }
        }
    }
}

impl Drop for OrtSession {
    fn drop(&mut self) {
        if let SessionKind::Python { stdin, child, .. } = &mut self.inner {
            // Try to send quit command; ignore errors (process may already be dead)
            let _ = writeln!(stdin, r#"{{"cmd":"quit"}}"#);
            let _ = stdin.flush();
            let _ = child.wait();
        }
    }
}
