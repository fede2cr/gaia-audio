//! Gaia Capture Server – records audio and serves WAV files over HTTP.
//!
//! This binary:
//! 1. Reads configuration from `gaia.conf`
//! 2. Starts audio capture (arecord / ffmpeg)
//! 3. Runs an axum HTTP server that exposes the recordings to the
//!    processing server over the network.

mod capture;
mod server;

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{Context, Result};
use tracing::info;

static SHUTDOWN: AtomicBool = AtomicBool::new(false);

#[tokio::main]
async fn main() -> Result<()> {
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
        "Gaia Capture Server starting (listen={})",
        config.capture_listen_addr
    );

    // Ensure StreamData directory exists
    std::fs::create_dir_all(config.stream_data_dir())
        .context("Cannot create StreamData directory")?;

    // ── ctrl-c ───────────────────────────────────────────────────────
    ctrlc::set_handler(move || {
        SHUTDOWN.store(true, Ordering::Relaxed);
        info!("Shutdown signal received");
        std::process::exit(0);
    })
    .context("Cannot set Ctrl-C handler")?;

    // ── start capture ────────────────────────────────────────────────
    let mut capture_handle = capture::start(&config)?;

    // ── start HTTP server ────────────────────────────────────────────
    let stream_dir = config.stream_data_dir();
    let listen_addr = config.capture_listen_addr.clone();
    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_clone = shutdown.clone();

    let server_handle = tokio::spawn(async move {
        if let Err(e) = server::run(stream_dir, &listen_addr, shutdown_clone).await {
            tracing::error!("HTTP server error: {e:#}");
        }
    });

    // Wait for the server task (runs until shutdown)
    let _ = server_handle.await;

    // Clean up capture processes
    capture_handle.kill().ok();
    info!("Gaia Capture Server stopped");

    Ok(())
}
