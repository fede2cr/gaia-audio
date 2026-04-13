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

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Once;

use anyhow::{Context, Result};
use tracing::{info, warn};

/// `true` once ORT has been successfully initialised.
static ORT_AVAILABLE: AtomicBool = AtomicBool::new(false);
/// When set, force ORT initialisation to prefer the CPU-only runtime
/// library even if `GAIA_ACCEL=rocm|cuda` is present.
static ORT_FORCE_CPU_ONLY: AtomicBool = AtomicBool::new(false);

/// Returns `true` when ORT init completed successfully.
///
/// After `init_ort_environment()` returns, callers should check
/// this before attempting to create sessions.
pub fn ort_is_available() -> bool {
    ORT_AVAILABLE.load(Ordering::Acquire)
}

/// Initialise the ORT global environment exactly once.
///
/// **Why subprocess probing?**  The ROCm-enabled `libonnxruntime.so`
/// triggers HSA runtime initialisation during `dlopen`.  If HSA
/// hangs, it holds the **glibc dynamic-linker global lock**, which
/// blocks every subsequent `dlopen` in the same process — including
/// attempts to load the CPU-only library.  We therefore `fork()` a
/// short-lived child that tests each candidate library.  Only the
/// library that loads cleanly is then loaded in the parent process.
fn init_ort_environment() {
    static INIT: Once = Once::new();

    INIT.call_once(|| {
        apply_hsa_mitigations();

        let timeout_secs: u64 = std::env::var("ORT_INIT_TIMEOUT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(30);

        // Probe timeout: how long the fork() probe waits before
        // declaring a library hung.  Short enough to fail fast,
        // long enough for legitimate slow GPU enumeration.
        let probe_secs: u64 = std::env::var("ORT_PROBE_TIMEOUT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10);

        // ── Build candidate list ──────────────────────────────────
        let mut candidates: Vec<PathBuf> = Vec::new();
        let force_cpu_only = ORT_FORCE_CPU_ONLY.load(Ordering::Acquire)
            || std::env::var("GAIA_FORCE_CPU_ORT")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);

        if force_cpu_only {
            info!("ORT: CPU-only failover requested — preferring /usr/lib/ort-cpu runtime");
            if let Some(cpu_lib) = find_ort_in_dir("/usr/lib/ort-cpu") {
                candidates.push(cpu_lib);
            }
        }

        // Primary candidate from resolve_ort_dylib_path()
        if let Some(primary) = resolve_ort_dylib_path() {
            if !candidates.contains(&primary) {
                candidates.push(primary);
            }
        }

        // CPU-only fallback (always a candidate, even when GPU is
        // requested — it's the safety net when the GPU library hangs).
        if let Some(cpu_lib) = find_ort_in_dir("/usr/lib/ort-cpu") {
            if !candidates.contains(&cpu_lib) {
                candidates.push(cpu_lib);
            }
        }

        if candidates.is_empty() {
            tracing::error!(
                "No ORT library found — models requiring ORT will be skipped."
            );
            return;
        }

        // ── Probe each candidate via fork() ───────────────────────
        let mut good_path: Option<PathBuf> = None;
        for candidate in &candidates {
            info!("ORT: probing {} (timeout {probe_secs}s) …", candidate.display());
            if probe_ort_library(candidate, probe_secs) {
                info!("ORT: probe OK — {}", candidate.display());
                good_path = Some(candidate.clone());
                break;
            } else {
                warn!(
                    "ORT: probe failed/timed out for {} — skipping",
                    candidate.display()
                );
            }
        }

        let lib_path = match good_path {
            Some(p) => p,
            None => {
                tracing::error!(
                    "All ORT library candidates failed the dlopen probe — \
                     models requiring ORT will be skipped.  \
                     Candidates tested: {:?}",
                    candidates
                );
                return;
            }
        };

        // ── Load the probed library for real ──────────────────────
        let (tx, rx) = std::sync::mpsc::sync_channel::<bool>(1);
        let path = lib_path.clone();

        let _init_thread = std::thread::Builder::new()
            .name("ort-init".into())
            .spawn(move || {
                let ok = do_ort_init_from(&path);
                let _ = tx.send(ok);
            });

        let timeout = std::time::Duration::from_secs(timeout_secs);
        match rx.recv_timeout(timeout) {
            Ok(true) => {
                ORT_AVAILABLE.store(true, Ordering::Release);
                info!("ORT environment ready ({})", lib_path.display());
            }
            Ok(false) => {
                tracing::error!(
                    "ORT init failed for {} (probe passed but init failed)",
                    lib_path.display()
                );
            }
            Err(_) => {
                tracing::error!(
                    "ORT init timed out after {timeout_secs}s for {} \
                     (probe passed but init hung — unexpected)",
                    lib_path.display()
                );
            }
        }
    });
}

/// Test whether a shared library can be `dlopen`'d without hanging.
///
/// Forks a child process that calls `dlopen` on the given path.
/// If the child exits successfully within `timeout_secs`, the library
/// loads cleanly.  If the child hangs (e.g. HSA global constructor
/// stall), it is killed after the timeout.
///
/// Using `fork()` isolates the test completely: a hung `dlopen` in
/// the child does not affect the parent's dynamic-linker state.
fn probe_ort_library(path: &Path, timeout_secs: u64) -> bool {
    let c_path = match std::ffi::CString::new(path.to_string_lossy().as_bytes()) {
        Ok(p) => p,
        Err(_) => return false,
    };

    unsafe {
        let pid = libc::fork();

        if pid < 0 {
            warn!("fork() failed — cannot probe ORT library");
            return false;
        }

        if pid == 0 {
            // ── Child process ─────────────────────────────────────
            // Only call async-signal-safe / C functions here.
            // No Rust allocator, no stdlib, no logging.
            let handle = libc::dlopen(c_path.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL);
            if handle.is_null() {
                libc::_exit(1);
            }
            libc::dlclose(handle);
            libc::_exit(0);
        }

        // ── Parent process ────────────────────────────────────────
        let deadline = std::time::Instant::now()
            + std::time::Duration::from_secs(timeout_secs);

        loop {
            let mut status: i32 = 0;
            let ret = libc::waitpid(pid, &mut status, libc::WNOHANG);

            if ret == pid {
                // Child exited.
                return libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0;
            }
            if ret < 0 {
                // waitpid error — assume failure.
                return false;
            }

            if std::time::Instant::now() >= deadline {
                // Timeout — library is hanging.  Kill the child.
                info!(
                    "ORT probe: child {} still running after {timeout_secs}s — killing",
                    pid
                );
                libc::kill(pid, libc::SIGKILL);
                libc::waitpid(pid, std::ptr::null_mut(), 0); // reap
                return false;
            }

            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }
}

/// Apply HSA / ROCm environment mitigations before loading ORT.
///
/// The ROCm-enabled `libonnxruntime.so` triggers HSA runtime
/// initialisation during `dlopen`.  On some AMD GPU configurations
/// (particularly when the GPU is passed through to a container or
/// the SDMA engine is unresponsive), HSA init hangs indefinitely.
///
/// Setting `HSA_ENABLE_SDMA=0` is the standard workaround.
/// These variables **must** be set before the library is loaded.
fn apply_hsa_mitigations() {
    // Safety: called from a dedicated init thread before any other
    // thread reads these variables.
    unsafe {
        if std::env::var_os("HSA_ENABLE_SDMA").is_none() {
            info!("ORT: setting HSA_ENABLE_SDMA=0 to mitigate HSA init hangs");
            std::env::set_var("HSA_ENABLE_SDMA", "0");
        }
        // Suppress noisy AMD log spam during init.
        if std::env::var_os("AMD_LOG_LEVEL").is_none() {
            std::env::set_var("AMD_LOG_LEVEL", "0");
        }
    }
}

/// Perform the actual ORT init from a specific library path.
fn do_ort_init_from(lib_path: &Path) -> bool {
    info!("ORT: loading library from {} …", lib_path.display());
    match ort::init_from(lib_path) {
        Ok(builder) => {
            info!("ORT: library loaded, committing environment …");
            builder.commit();
            if let Ok(env) = ort::environment::Environment::current() {
                env.set_log_level(ort::logging::LogLevel::Warning);
            }
            info!("ORT environment initialised from {}", lib_path.display());
            true
        }
        Err(e) => {
            warn!(
                "Failed to init ORT from {}: {e}",
                lib_path.display()
            );
            false
        }
    }
}

/// Resolve the ONNX Runtime shared library path for `ort::init_from`.
///
/// Priority:
/// 1. `ORT_DYLIB_PATH` environment variable (if it points to a file)
/// 2. CPU-only library at `/usr/lib/ort-cpu/` (when GPU not requested)
/// 3. Common system locations used by the container image
///
/// When GPU acceleration is not requested, we prefer the CPU-only
/// build to completely avoid loading ROCm/HSA runtime libraries that
/// may hang during `dlopen` global constructors.
fn resolve_ort_dylib_path() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("ORT_DYLIB_PATH") {
        let p = PathBuf::from(path);
        if p.is_file() {
            return Some(p);
        }
    }

    // When GPU is NOT requested, or CPU-only ORT is explicitly forced,
    // prefer a CPU-only build to completely avoid ROCm/HSA runtime
    // initialisation.
    if accel_kind() == AccelKind::None || ORT_FORCE_CPU_ONLY.load(Ordering::Acquire) {
        if let Some(cpu_lib) = find_ort_in_dir("/usr/lib/ort-cpu") {
            info!(
                "ORT: using CPU-only library: {}",
                cpu_lib.display()
            );
            return Some(cpu_lib);
        }
    }

    const CANDIDATES: &[&str] = &[
        "/usr/lib/libonnxruntime.so",
        "/usr/local/lib/libonnxruntime.so",
    ];
    for candidate in CANDIDATES {
        let p = PathBuf::from(candidate);
        if p.is_file() {
            return Some(p);
        }
    }

    find_ort_in_dir("/usr/lib")
}

/// Find the first `libonnxruntime.so*` file in `dir`.
fn find_ort_in_dir(dir: &str) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let p = entry.path();
        if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
            if name.starts_with("libonnxruntime.so") && p.is_file() {
                return Some(p);
            }
        }
    }
    None
}

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

/// Returns true when an ORT provider shared library appears to be present.
///
/// Example stems: `rocm`, `migraphx`, `cuda`, `tensorrt`.
fn has_provider_library(stem: &str) -> bool {
    let prefix = format!("libonnxruntime_providers_{stem}.so");
    for dir in ["/usr/lib", "/usr/local/lib"] {
        let direct = PathBuf::from(dir).join(&prefix);
        if direct.is_file() {
            return true;
        }
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                    if name.starts_with(&prefix) && p.is_file() {
                        return true;
                    }
                }
            }
        }
    }
    false
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
/// CPU-only with ORT defaults.  Used for DFT/STFT models that tract
/// cannot handle (BirdNET V3, Perch).
///
/// Container validation exercises this path for BirdNET V3/Perch.
pub struct OrtSession {
    session: ort::session::Session,
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
        init_ort_environment();
        anyhow::ensure!(
            ort_is_available(),
            "ORT environment is not available (init timed out or failed)"
        );
        let requested_kind = accel_kind();

        let has_migraphx = has_provider_library("migraphx");
        let has_rocm = has_provider_library("rocm");
        let has_tensorrt = has_provider_library("tensorrt");
        let has_cuda = has_provider_library("cuda");

        let kind = match requested_kind {
            AccelKind::Rocm if !has_migraphx && !has_rocm => {
                warn!(
                    "GAIA_ACCEL=rocm requested but ORT ROCm/MIGraphX provider libraries are not present; using CPU EP"
                );
                AccelKind::None
            }
            AccelKind::Cuda if !has_tensorrt && !has_cuda => {
                warn!(
                    "GAIA_ACCEL=cuda requested but ORT CUDA/TensorRT provider libraries are not present; using CPU EP"
                );
                AccelKind::None
            }
            k => k,
        };

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
            .context("Failed to create ORT session builder (is libonnxruntime.so installed?)")?;

        // For GPU paths, explicit thread config helps overlap EP compilation.
        // For CPU-only, use ORT defaults (matching birdnet-onnx) to avoid
        // deadlocks during graph optimisation of DFT/STFT subgraphs.
        if kind != AccelKind::None {
            builder = builder
                .with_intra_threads(intra)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .with_inter_threads(1)
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

                if has_migraphx && has_rocm {
                    builder
                        .with_execution_providers([
                            migraphx.build(),
                            ort::ep::ROCm::default().with_device_id(0).build(),
                            ort::ep::CPU::default().build(),
                        ])
                        .map_err(|e| anyhow::anyhow!("{e}"))?
                } else if has_migraphx {
                    warn!("ROCm EP library not found; trying MIGraphX + CPU");
                    builder
                        .with_execution_providers([
                            migraphx.build(),
                            ort::ep::CPU::default().build(),
                        ])
                        .map_err(|e| anyhow::anyhow!("{e}"))?
                } else {
                    warn!("MIGraphX EP library not found; trying ROCm + CPU");
                    builder
                        .with_execution_providers([
                            ort::ep::ROCm::default().with_device_id(0).build(),
                            ort::ep::CPU::default().build(),
                        ])
                        .map_err(|e| anyhow::anyhow!("{e}"))?
                }
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

                let cuda = ort::ep::CUDA::default().with_device_id(0);

                if has_tensorrt && has_cuda {
                    builder
                        .with_execution_providers([
                            tensorrt.build(),
                            cuda.build(),
                            ort::ep::CPU::default().build(),
                        ])
                        .map_err(|e| anyhow::anyhow!("{e}"))?
                } else if has_tensorrt {
                    warn!("CUDA EP library not found; trying TensorRT + CPU");
                    builder
                        .with_execution_providers([
                            tensorrt.build(),
                            ort::ep::CPU::default().build(),
                        ])
                        .map_err(|e| anyhow::anyhow!("{e}"))?
                } else {
                    warn!("TensorRT EP library not found; trying CUDA + CPU");
                    builder
                        .with_execution_providers([
                            cuda.build(),
                            ort::ep::CPU::default().build(),
                        ])
                        .map_err(|e| anyhow::anyhow!("{e}"))?
                }
            }
            AccelKind::None => {
                // No GPU EPs requested — ORT defaults to CPU automatically.
                builder
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
        ORT_FORCE_CPU_ONLY.store(true, Ordering::Release);
        info!(
            "Creating ONNX Runtime session (CPU-only fallback) for {}",
            onnx_path.display()
        );

        // Log the model file size — useful for diagnosing issues
        if let Ok(meta) = std::fs::metadata(onnx_path) {
            info!(
                "Model file: {} ({:.1} MB)",
                onnx_path.display(),
                meta.len() as f64 / 1_048_576.0
            );
        }

        // Ensure ORT environment is initialized before creating sessions.
        init_ort_environment();
        anyhow::ensure!(
            ort_is_available(),
            "ORT environment is not available (init timed out or failed)"
        );

        let intra = ort_intra_threads(4);
        info!("  ORT CPU session: intra_threads={intra}, inter_threads=1, opt_level=Level1");

        let start = std::time::Instant::now();
        info!("  Calling CreateSession for {}...", onnx_path.display());

        // Thread limits prevent deadlocks during CreateSession for
        // DFT/STFT models (Perch, BirdNET V3).  Level1 optimization
        // avoids the expensive Level2+ fusions that can hang on
        // complex subgraphs in large models (500+ MB).
        use ort::session::builder::GraphOptimizationLevel;
        let session = ort::session::Session::builder()
            .context("Failed to create ORT session builder (is libonnxruntime.so installed?)")?
            .with_intra_threads(intra)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_inter_threads(1)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level1)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .commit_from_file(onnx_path)
            .with_context(|| format!("ORT: failed to load {}", onnx_path.display()))?;

        let elapsed = start.elapsed();
        info!(
            "ORT session created for {} ({} inputs, {} outputs, accel=CPU-only) in {:.1}s",
            onnx_path.display(),
            session.inputs().len(),
            session.outputs().len(),
            elapsed.as_secs_f64(),
        );

        Ok(Self {
            session,
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

        let (_shape, data) = outputs[output_index]
            .try_extract_tensor::<f32>()
            .context("Cannot extract f32 output tensor")?;

        Ok(data.to_vec())
    }
}
