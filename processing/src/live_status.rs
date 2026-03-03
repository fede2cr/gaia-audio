//! Live analysis status – writes a small JSON + spectrogram PNG to the
//! shared `/data` volume so the web dashboard can display the "currently
//! analysing" state even when there are no confident detections.

use std::path::PathBuf;

use anyhow::Result;
use serde::Serialize;
use tracing::{debug, warn};

use crate::spectrogram::{self, SpectrogramParams};

/// JSON written to `<data_dir>/live_status.json`.
#[derive(Debug, Serialize)]
pub struct LiveStatus {
    /// ISO-8601 timestamp of the last update.
    pub timestamp: String,
    /// The filename of the recording being analysed.
    pub filename: String,
    /// Top N predictions for the current chunk (species + confidence).
    pub predictions: Vec<LivePrediction>,
    /// Whether any prediction passed the configured confidence threshold.
    pub has_detections: bool,
}

/// One prediction entry in the live status.
#[derive(Debug, Serialize)]
pub struct LivePrediction {
    pub scientific_name: String,
    pub common_name: String,
    pub confidence: f64,
}

/// Directory where the live files are written.
fn live_dir() -> PathBuf {
    PathBuf::from(
        std::env::var("GAIA_DATA_DIR").unwrap_or_else(|_| "/data".to_string()),
    )
}

/// Write the live status JSON to disk.
pub fn write_status(status: &LiveStatus) -> Result<()> {
    let dir = live_dir();
    std::fs::create_dir_all(&dir)?;

    let path = dir.join("live_status.json");
    let json = serde_json::to_string(status)?;

    // Atomic write: write to tmp first, then rename.
    let tmp = dir.join("live_status.json.tmp");
    std::fs::write(&tmp, &json)?;
    std::fs::rename(&tmp, &path)?;

    debug!("Live status updated: {}", path.display());
    Ok(())
}

/// Generate and write the live spectrogram PNG for the current audio chunk.
///
/// `samples` should be mono f32 audio at `sample_rate` Hz.
pub fn write_spectrogram(samples: &[f32], sample_rate: u32) -> Result<()> {
    let dir = live_dir();
    std::fs::create_dir_all(&dir)?;

    let params = SpectrogramParams {
        fft_size: 1024,
        hop_size: 512,
        max_freq: 12_000.0,
        width: 800,
        height: 256,
    };

    let png_bytes = spectrogram::generate_to_png_buffer(samples, sample_rate, &params)?;

    // Atomic write
    let tmp = dir.join("live_spectrogram.png.tmp");
    let path = dir.join("live_spectrogram.png");
    std::fs::write(&tmp, &png_bytes)?;
    std::fs::rename(&tmp, &path)?;

    debug!("Live spectrogram updated: {}", path.display());
    Ok(())
}

/// Convenience: update both the spectrogram and the status JSON in one call.
pub fn update(
    filename: &str,
    samples: &[f32],
    sample_rate: u32,
    predictions: Vec<LivePrediction>,
    confidence_threshold: f64,
) {
    let has_detections = predictions.iter().any(|p| p.confidence >= confidence_threshold);

    let status = LiveStatus {
        timestamp: chrono::Utc::now().to_rfc3339(),
        filename: filename.to_string(),
        predictions,
        has_detections,
    };

    if let Err(e) = write_spectrogram(samples, sample_rate) {
        warn!("Failed to write live spectrogram: {e:#}");
    }
    if let Err(e) = write_status(&status) {
        warn!("Failed to write live status: {e:#}");
    }
}
