//! Gaia Processing Server – loads models, polls the capture server,
//! runs TFLite inference, writes detections to SQLite.

mod analysis;
mod client;
mod db;
mod download;
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
    let config =
        gaia_common::config::load(&PathBuf::from(&config_path)).context("Config load failed")?;

    info!(
        "Gaia Processing Server starting (capture_url={})",
        config.capture_server_url
    );

    // ── initialize database ──────────────────────────────────────────
    db::initialize(&config.db_path)?;

    // ── discover and load models ─────────────────────────────────────
    let mut manifests = manifest::discover_manifests(&config.model_dir)?;

    // ── auto-download models from Zenodo if needed ───────────────────
    for m in &mut manifests {
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
             analyse audio until model files (tflite) are present."
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
        &models,
        &config,
        discovery.as_ref(),
        &report_tx,
        &SHUTDOWN,
    ) {
        tracing::error!("Processing loop error: {e:#}");
    }

    // Signal reporting thread to finish
    drop(report_tx);
    report_thread.join().ok();

    // Clean up mDNS
    if let Some(dh) = discovery {
        dh.shutdown();
    }

    info!("Gaia Processing Server stopped");
    Ok(())
}
