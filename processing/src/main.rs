//! Gaia Processing Server – loads models, polls the capture server,
//! runs TFLite inference, writes detections to SQLite.

mod analysis;
mod client;
mod compress;
mod db;
mod download;
mod live_status;
mod manifest;
mod mel;
mod model;
mod reporting;
mod spectrogram;

use std::path::PathBuf;
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

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

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

    // ── initialize database ──────────────────────────────────────────
    db::initialize(&config.db_path)?;

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
        tracing::warn!(
            "No models loaded. The processing server will run but cannot \
             analyse audio until model files (tflite/onnx) are present."
        );
    } else {
        let names: Vec<&str> = models.iter().map(|m| m.manifest.manifest.model.name.as_str()).collect();
        info!(
            "{} model(s) loaded: [{}]  – each audio file will be analysed by all models",
            models.len(),
            names.join(", "),
        );
    }

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

    // ── poll capture server(s) and process files ─────────────────────
    if let Err(e) = client::poll_and_process(
        &mut models,
        &mut config,
        discovery.as_ref(),
        &report_tx,
        &SHUTDOWN,
    ) {
        tracing::error!("Processing loop error: {e:#}");
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
