//! Gaia Processing Server – loads models, polls the capture server,
//! runs TFLite inference, writes detections to SQLite.

mod accel;
mod analysis;
mod client;
mod compress;
mod db;
mod download;
mod live_status;
mod manifest;
mod mel;
mod model;
mod parquet_store;
mod reporting;
mod spectrogram;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;

use anyhow::{Context, Result};
use tracing::info;

use gaia_common::detection::{Detection, ParsedFileName};

static SHUTDOWN: AtomicBool = AtomicBool::new(false);

/// Payload sent from the analysis thread to the reporting thread.
pub struct ReportPayload {
    pub file: ParsedFileName,
    pub detections: Vec<Detection>,
    pub source_node: String,
}

/// A downloaded file ready for analysis by a worker thread.
pub struct WorkItem {
    pub local_path: PathBuf,
    pub filename: String,
    pub base_url: String,
    pub config_snapshot: gaia_common::config::Config,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    // ── validate-model subcommand (build-time dry-run) ───────────────
    // Usage: gaia-processing validate-model <path.onnx> [<path2.onnx> …]
    //
    // Loads each model with tract-onnx (load → optimise → runnable),
    // identical to the runtime code path.  Exits 0 on success, 1 on
    // failure.  This is invoked during `docker build` to catch
    // tract-incompatible ONNX files before the container is published.
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(|s| s.as_str()) == Some("validate-model") {
        let paths = &args[2..];
        if paths.is_empty() {
            eprintln!("Usage: gaia-processing validate-model <model.onnx> [<model2.onnx> …]");
            std::process::exit(2);
        }
        let mut failed = false;
        for path in paths {
            let p = std::path::Path::new(path);
            info!("Validating with tract-onnx: {}", p.display());
            match model::validate_onnx_with_tract(p) {
                Ok(()) => info!("  PASS ✓  {}", p.display()),
                Err(e) => {
                    tracing::error!("  FAIL ✗  {}: {e:#}", p.display());
                    failed = true;
                }
            }
        }
        if failed {
            std::process::exit(1);
        }
        return Ok(());
    }

    if std::env::var("RUST_LOG").map_or(false, |v| v.contains("debug")) {
        info!("🔍 Debug logging ENABLED (RUST_LOG={})", std::env::var("RUST_LOG").unwrap_or_default());
    }

    // ── load config ──────────────────────────────────────────────────
    let config_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| gaia_common::config::Config::default_path().to_string());
    let mut config =
        gaia_common::config::load(&PathBuf::from(&config_path)).context("Config load failed")?;

    info!(
        "Gaia Processing Server starting (capture_url={})",
        config.capture_server_url
    );

    // ── GPU acceleration check ───────────────────────────────────────
    let accel_var = std::env::var("GAIA_ACCEL").unwrap_or_default();
    match accel::accel_kind() {
        accel::AccelKind::Rocm => {
            info!(
                "Acceleration env: GAIA_ACCEL={:?} ROCM_VISIBLE_DEVICES={:?}",
                accel_var,
                std::env::var("ROCM_VISIBLE_DEVICES").unwrap_or_default()
            );
            info!("ROCm acceleration requested — ORT will try MIGraphX → ROCm → CPU");
        }
        accel::AccelKind::Cuda => {
            info!(
                "Acceleration env: GAIA_ACCEL={:?} CUDA_VISIBLE_DEVICES={:?}",
                accel_var,
                std::env::var("CUDA_VISIBLE_DEVICES").unwrap_or_default()
            );
            info!("CUDA acceleration requested — ORT will try TensorRT → CUDA → CPU");
        }
        accel::AccelKind::None => {
            info!(
                "GPU acceleration not requested (GAIA_ACCEL={:?}) — using CPU inference (tract-onnx)",
                accel_var
            );
        }
    }

    // ── initialize database ──────────────────────────────────────────
    db::initialize(&config.db_path)?;

    // ── initialize Parquet detection store ────────────────────────────
    {
        let det_dir = config.db_path.parent().unwrap_or(Path::new("/data")).join("detections");
        let instance = if config.processing_instance.is_empty() {
            "default"
        } else {
            &config.processing_instance
        };
        parquet_store::initialize(&det_dir, instance)?;
    };

    // Register this processing instance so multi-instance deletion
    // coordination knows how many instances exist.
    {
        let instance_id = if config.processing_instance.is_empty() {
            "default"
        } else {
            &config.processing_instance
        };
        db::register_instance(&config.db_path, instance_id)?;
        info!("Registered processing instance: {instance_id:?}");
    }

    // ── discover and load models ─────────────────────────────────────
    let mut manifests = manifest::discover_manifests(&config.model_dir)?;

    // Filter to only the model slugs requested via MODEL_SLUGS (if set).
    if !config.model_slugs.is_empty() {
        let before = manifests.len();
        manifests.retain(|m| config.model_slugs.contains(&m.slug()));
        info!(
            "MODEL_SLUGS filter: {before} manifests discovered, {} retained ({:?})",
            manifests.len(),
            config.model_slugs,
        );
    }

    // ── auto-download models from Zenodo if needed ───────────────────
    for m in &mut manifests {
        // Download individual files (e.g. ONNX from HuggingFace)
        if let Err(e) = download::ensure_direct_files(m) {
            tracing::warn!("Direct file download failed for {}: {e:#}", m.manifest.model.name);
        }
        // Download variant-based files from Zenodo
        if let Some(variant) = m.effective_variant(config.model_variant.as_deref()) {
            download::ensure_model_files(m, &variant)?;
        }
        // Convert TFLite → ONNX if needed (best-effort, non-fatal).
        if let Err(e) = download::ensure_onnx_file(m) {
            tracing::warn!("ONNX conversion failed for {}: {e:#}", m.manifest.model.name);
        }
        // Convert metadata TFLite → ONNX if needed (best-effort, non-fatal).
        if let Err(e) = download::ensure_meta_onnx_file(m) {
            tracing::warn!("Metadata ONNX conversion failed for {}: {e:#}", m.manifest.model.name);
        }
    }

    let mut models = Vec::with_capacity(manifests.len());
    for m in &manifests {
        // Wrap in catch_unwind because tract-tflite can panic on
        // unsupported tensor types (e.g. float16 in the fp16 variant).
        let load_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            model::load_model(m, &config)
        }));
        match load_result {
            Ok(Ok(loaded)) => {
                info!(
                    "Model ready: {} (domain={}, sr={}, chunk={}s)",
                    m.manifest.model.name,
                    m.manifest.model.domain,
                    m.manifest.model.sample_rate,
                    m.manifest.model.chunk_duration,
                );
                models.push(loaded);
            }
            Ok(Err(e)) => {
                tracing::warn!("Cannot load model {}: {e:#}", m.manifest.model.name);
            }
            Err(_) => {
                tracing::error!(
                    "Model {} panicked during loading – this usually means the \
                     TFLite file uses an unsupported tensor type (e.g. float16). \
                     Set MODEL_VARIANT=fp32 or MODEL_VARIANT=int8 in gaia.conf.",
                    m.manifest.model.name,
                );
            }
        }
    }

    if models.is_empty() {
        tracing::error!(
            "No models loaded — the processing server cannot analyse audio \
             without at least one working model. Exiting.\n\
             Check that model files (tflite/onnx) are present in {model_dir} \
             and compatible with the current runtime.",
            model_dir = config.model_dir.display(),
        );
        std::process::exit(1);
    } else {
        let names: Vec<&str> = models.iter().map(|m| m.manifest.manifest.model.name.as_str()).collect();
        info!(
            "{} model(s) loaded: [{}]  – each audio file will be analysed by all models",
            models.len(),
            names.join(", "),
        );
    }

    // ── backfill per-species taxonomic class ─────────────────────────
    // Models whose labels CSV contains a `class` column (e.g. BirdNET+
    // V3.0) now set Domain per-species.  Migrate existing detections
    // that still carry the old model-wide domain.
    for m in &models {
        let class_map = m.csv_classes();
        if !class_map.is_empty() {
            db::migrate_domain_classes(
                &config.db_path,
                &m.manifest.slug(),
                m.domain(),
                class_map,
            );
        }
    }

    let num_workers = config.processing_threads;
    info!(
        "Processing threads: {} (set PROCESSING_THREADS to change)",
        num_workers
    );

    // ── mDNS registration + capture discovery ──────────────────────
    // With network_mode: host, mDNS multicast reaches the physical
    // network and containers discover each other automatically —
    // even across different machines.
    //
    // Setting GAIA_DISABLE_MDNS=1 skips mDNS for environments where
    // multicast is not available (e.g. bridge networking, CI).
    let discovery = if std::env::var("GAIA_DISABLE_MDNS").is_ok() {
        info!(
            "GAIA_DISABLE_MDNS set – using {} (mDNS skipped)",
            config.capture_server_url
        );
        None
    } else {
        match gaia_common::discovery::register(
            gaia_common::discovery::ServiceRole::Processing,
            0, // processing doesn't expose an HTTP port
        ) {
            Ok(h) => {
                info!("mDNS: registered as {}", h.instance_name());
                Some(h)
            }
            Err(e) => {
                tracing::warn!("mDNS registration failed (non-fatal): {e:#}");
                None
            }
        }
    };

    // ── ctrl-c ───────────────────────────────────────────────────────
    ctrlc::set_handler(move || {
        SHUTDOWN.store(true, Ordering::Relaxed);
        info!("Shutdown signal received");
    })
    .context("Cannot set Ctrl-C handler")?;

    // ── compression thread (fallback sweep every 30 min) ──────────
    // Clips are now converted to Opus inline during extraction, but
    // the background sweep catches any files that were missed (e.g.
    // ffmpeg was temporarily unavailable, or legacy WAV files).
    let compress_extracted = config.extracted_dir.clone();
    let compress_db = config.db_path.clone();
    let compress_thread = std::thread::Builder::new()
        .name("compression".into())
        .spawn(move || {
            compress::compress_loop(
                compress_extracted,
                compress_db,
                std::time::Duration::from_secs(30 * 60), // every 30 min
                &SHUTDOWN,
            );
        })
        .context("Cannot spawn compression thread")?;

    // ── reporting thread ─────────────────────────────────────────────
    let (report_tx, report_rx) = mpsc::sync_channel::<ReportPayload>(16);
    let report_config = config.clone();
    let report_db = config.db_path.clone();
    let report_thread = std::thread::Builder::new()
        .name("reporting".into())
        .spawn(move || {
            reporting::handle_queue(report_rx, &report_config, &report_db);
        })
        .context("Cannot spawn reporting thread")?;

    // ── work channel: poll thread → worker threads ───────────────────
    let (work_tx, work_rx) = mpsc::sync_channel::<WorkItem>(num_workers * 2);
    let work_rx = std::sync::Arc::new(std::sync::Mutex::new(work_rx));

    // ── spawn worker threads ─────────────────────────────────────────
    // Worker 0 takes the already-loaded models; workers 1..N each load
    // their own copy from the same manifests.
    let mut worker_handles = Vec::with_capacity(num_workers);

    for worker_id in 0..num_workers {
        let report_tx = report_tx.clone();
        let work_rx = work_rx.clone();
        let db_path = config.db_path.clone();
        let instance_id_owned = if config.processing_instance.is_empty() {
            "default".to_string()
        } else {
            config.processing_instance.clone()
        };

        let mut worker_models = if worker_id == 0 {
            // First worker reuses the models already loaded above.
            std::mem::take(&mut models)
        } else {
            // Additional workers load their own model copies.
            let mut m = Vec::with_capacity(manifests.len());
            for manifest in &manifests {
                let load_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    model::load_model(manifest, &config)
                }));
                match load_result {
                    Ok(Ok(loaded)) => m.push(loaded),
                    Ok(Err(e)) => {
                        tracing::warn!(
                            "Worker {worker_id}: cannot load model {}: {e:#}",
                            manifest.manifest.model.name
                        );
                    }
                    Err(_) => {
                        tracing::error!(
                            "Worker {worker_id}: model {} panicked during loading",
                            manifest.manifest.model.name
                        );
                    }
                }
            }
            m
        };

        let handle = std::thread::Builder::new()
            .name(format!("worker-{worker_id}"))
            .spawn(move || {
                info!("Worker {worker_id} started ({} model(s))", worker_models.len());

                // Build a per-worker HTTP client for deletion requests.
                let client = reqwest::blocking::Client::builder()
                    .timeout(std::time::Duration::from_secs(30))
                    .build()
                    .expect("Cannot create HTTP client");

                loop {
                    // Receive work items from the shared channel.
                    let item = {
                        let rx = work_rx.lock().unwrap();
                        rx.recv()
                    };
                    let item = match item {
                        Ok(item) => item,
                        Err(_) => break, // channel closed → shutdown
                    };

                    tracing::debug!("Worker {worker_id}: analysing {}", item.filename);

                    // ── run analysis ──────────────────────────────────
                    if let Err(e) = analysis::process_file(
                        &item.local_path,
                        &mut worker_models,
                        &item.config_snapshot,
                        &report_tx,
                        &item.base_url,
                    ) {
                        tracing::error!(
                            "Worker {worker_id}: error processing {}: {e:#}",
                            item.filename
                        );
                    }

                    // ── coordinate multi-instance deletion ───────────
                    if let Err(e) = crate::db::mark_file_processed(
                        &db_path,
                        &item.filename,
                        &instance_id_owned,
                    ) {
                        tracing::warn!(
                            "Worker {worker_id}: cannot mark {} as processed: {e}",
                            item.filename
                        );
                    }

                    if crate::db::all_instances_done(&db_path, &item.filename) {
                        tracing::debug!(
                            "Worker {worker_id}: all instances done with {} — deleting",
                            item.filename
                        );
                        let url = format!(
                            "{}/api/recordings/{}",
                            item.base_url, item.filename
                        );
                        match client.delete(&url).send() {
                            Ok(resp) if resp.status().is_success()
                                || resp.status() == reqwest::StatusCode::NOT_FOUND =>
                            {
                                info!(
                                    "Worker {worker_id}: deleted {} from capture server",
                                    item.filename
                                );
                            }
                            Ok(resp) => {
                                tracing::warn!(
                                    "Worker {worker_id}: DELETE {} returned {}",
                                    item.filename,
                                    resp.status()
                                );
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "Worker {worker_id}: failed to delete {}: {e}",
                                    item.filename
                                );
                            }
                        }
                    }
                }

                info!("Worker {worker_id} stopped");
            })
            .with_context(|| format!("Cannot spawn worker-{worker_id}"))?;

        worker_handles.push(handle);
    }

    // ── poll capture server(s) and dispatch to workers ───────────────
    if let Err(e) = client::poll_and_dispatch(
        &mut config,
        discovery.as_ref(),
        &work_tx,
        &SHUTDOWN,
    ) {
        tracing::error!("Processing loop error: {e:#}");
    }

    // Signal workers to finish, then wait.
    drop(work_tx);
    for h in worker_handles {
        h.join().ok();
    }

    // Signal reporting thread to finish
    drop(report_tx);
    report_thread.join().ok();
    compress_thread.join().ok();

    // Clean up mDNS
    if let Some(dh) = discovery {
        dh.shutdown();
    }

    info!("Gaia Processing Server stopped");
    Ok(())
}
