//! Reporting: write detections to DB, extract audio clips, generate
//! spectrograms, send notifications.
//!
//! Evolved from `birdnet-server/src/reporting.rs`.

use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;

use anyhow::Result;
use tracing::{error, info, warn};

use gaia_common::audio;
use gaia_common::config::Config;
use gaia_common::detection::{Detection, ParsedFileName};

use crate::db;
use crate::spectrogram::{self, SpectrogramParams};
use crate::ReportPayload;

/// Run the reporting loop on its own thread.
pub fn handle_queue(rx: Receiver<ReportPayload>, config: &Config, db_path: &Path) {
    while let Ok(payload) = rx.recv() {
        if let Err(e) = process_report(&payload, config, db_path) {
            error!("Reporting error: {e:#}");
        }

        // Notify capture server to delete the source file (if local)
        // or delete it ourselves if running mono-node.
        let src = &payload.file.file_path;
        if src.exists() {
            if let Err(e) = std::fs::remove_file(src) {
                warn!("Cannot remove source file {}: {e}", src.display());
            }
        } else {
            // File was fetched from capture server and already cleaned up
            // by the client after processing. Attempt to ask capture server
            // to delete it.
            delete_from_capture(config, src);
        }
    }
    info!("Reporting thread finished");
}

fn process_report(payload: &ReportPayload, config: &Config, db_path: &Path) -> Result<()> {
    let file = &payload.file;

    write_json_file(file, &payload.detections, config)?;

    for detection in &payload.detections {
        let extracted_path = extract_detection(file, detection, config)?;

        let spec_path = format!("{}.png", extracted_path.display());
        if let Err(e) = spectrogram::generate_from_wav(
            &extracted_path,
            Path::new(&spec_path),
            &SpectrogramParams::default(),
        ) {
            warn!("Spectrogram failed for {}: {e}", extracted_path.display());
        }

        let summary = format_summary(detection, config);
        let basename = extracted_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy();
        info!("{summary};{basename}");

        write_to_log(&summary);

        if let Err(e) = db::insert_detection(
            db_path,
            detection,
            config.latitude,
            config.longitude,
            config.confidence,
            config.sensitivity,
            config.overlap,
            &basename,
            &payload.source_node,
        ) {
            error!("DB insert failed: {e}");
        }
    }

    if config.birdweather_id.is_some() {
        if let Err(e) = bird_weather(file, &payload.detections, config) {
            error!("BirdWeather error: {e}");
        }
    }

    heartbeat(config);
    Ok(())
}

// ── audio clip extraction ────────────────────────────────────────────────

fn extract_detection(
    file: &ParsedFileName,
    detection: &Detection,
    config: &Config,
) -> Result<PathBuf> {
    let spacer = (config.extraction_length as f64 - 3.0).max(0.0) / 2.0;
    let safe_start = (detection.start - spacer).max(0.0);
    let safe_stop = (detection.stop + spacer).min(config.recording_length as f64);

    let new_name = format!(
        "{}-{}-{}-{}-birdnet-{}{}.wav",
        detection.domain,
        detection.common_name_safe,
        detection.confidence_pct(),
        detection.date,
        file.rtsp_id,
        detection.time,
    );
    let new_dir = config
        .extracted_dir
        .join("By_Date")
        .join(&detection.date)
        .join(&detection.common_name_safe);
    let new_path = new_dir.join(&new_name);

    if new_path.exists() {
        warn!("Extraction already exists, skipping: {}", new_path.display());
        return Ok(new_path);
    }

    audio::extract_clip(&file.file_path, &new_path, safe_start, safe_stop)?;
    Ok(new_path)
}

// ── summary / logging ────────────────────────────────────────────────────

fn format_summary(d: &Detection, config: &Config) -> String {
    format!(
        "{};{};{};{};{};{};{};{};{};{};{};{}",
        d.domain,
        d.date,
        d.time,
        d.scientific_name,
        d.common_name,
        d.confidence,
        config.latitude,
        config.longitude,
        config.confidence,
        d.week,
        config.sensitivity,
        config.overlap,
    )
}

fn write_to_log(summary: &str) {
    let log_path = std::env::var("GAIA_DIR")
        .map(|d| PathBuf::from(d).join("GaiaDB.txt"))
        .unwrap_or_else(|_| PathBuf::from("/app/data/GaiaDB.txt"));

    if let Err(e) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .and_then(|mut f| {
            use std::io::Write;
            writeln!(f, "{summary}")
        })
    {
        warn!("Cannot write to log {}: {e}", log_path.display());
    }
}

// ── JSON output ──────────────────────────────────────────────────────────

fn write_json_file(
    file: &ParsedFileName,
    detections: &[Detection],
    config: &Config,
) -> Result<()> {
    let dir = file.file_path.parent().unwrap_or(Path::new("."));
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.ends_with(".json") {
                if file.rtsp_id.is_empty() || name.contains(&file.rtsp_id) {
                    std::fs::remove_file(entry.path()).ok();
                }
            }
        }
    }

    let json_path = format!("{}.json", file.file_path.display());
    let dets: Vec<serde_json::Value> = detections
        .iter()
        .map(|d| {
            serde_json::json!({
                "domain": d.domain,
                "start": d.start,
                "common_name": d.common_name,
                "scientific_name": d.scientific_name,
                "confidence": d.confidence,
            })
        })
        .collect();

    let payload = serde_json::json!({
        "file_name": Path::new(&json_path).file_name().unwrap_or_default().to_string_lossy(),
        "timestamp": file.iso8601(),
        "delay": config.recording_length,
        "detections": dets,
    });

    std::fs::write(&json_path, serde_json::to_string(&payload)?)?;
    Ok(())
}

// ── BirdWeather integration ──────────────────────────────────────────────

fn bird_weather(
    file: &ParsedFileName,
    detections: &[Detection],
    config: &Config,
) -> Result<()> {
    let bw_id = match &config.birdweather_id {
        Some(id) if !id.is_empty() => id,
        _ => return Ok(()),
    };

    // Only POST bird detections to BirdWeather
    let bird_dets: Vec<&Detection> = detections
        .iter()
        .filter(|d| d.domain == "birds")
        .collect();
    if bird_dets.is_empty() {
        return Ok(());
    }

    let wav_bytes = std::fs::read(&file.file_path)?;
    let client = reqwest::blocking::Client::new();

    let soundscape_url = format!(
        "https://app.birdweather.com/api/v1/stations/{bw_id}/soundscapes?timestamp={}",
        file.iso8601(),
    );

    let resp = client
        .post(&soundscape_url)
        .header("Content-Type", "audio/wav")
        .body(wav_bytes)
        .timeout(std::time::Duration::from_secs(30))
        .send()?;

    let sdata: serde_json::Value = resp.json()?;
    if sdata.get("success").and_then(|v| v.as_bool()) != Some(true) {
        let msg = sdata
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown error");
        anyhow::bail!("BirdWeather soundscape POST failed: {msg}");
    }

    let soundscape_id = sdata
        .pointer("/soundscape/id")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);

    let detection_url = format!(
        "https://app.birdweather.com/api/v1/stations/{bw_id}/detections"
    );

    for d in bird_dets {
        let body = serde_json::json!({
            "timestamp": d.iso8601,
            "lat": config.latitude,
            "lon": config.longitude,
            "soundscapeId": soundscape_id,
            "soundscapeStartTime": d.start,
            "soundscapeEndTime": d.stop,
            "commonName": d.common_name,
            "scientificName": d.scientific_name,
            "algorithm": "2p4",
            "confidence": d.confidence,
        });

        match client
            .post(&detection_url)
            .json(&body)
            .timeout(std::time::Duration::from_secs(20))
            .send()
        {
            Ok(r) => info!("BirdWeather detection POST: {}", r.status()),
            Err(e) => error!("BirdWeather detection POST failed: {e}"),
        }
    }

    Ok(())
}

// ── heartbeat ────────────────────────────────────────────────────────────

fn heartbeat(config: &Config) {
    if let Some(url) = &config.heartbeat_url {
        match reqwest::blocking::get(url) {
            Ok(r) => info!("Heartbeat: {}", r.status()),
            Err(e) => error!("Heartbeat failed: {e}"),
        }
    }
}

// ── capture server cleanup ───────────────────────────────────────────────

fn delete_from_capture(config: &Config, file_path: &Path) {
    let filename = file_path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy();
    let url = format!(
        "{}/api/recordings/{}",
        config.capture_server_url, filename
    );
    match reqwest::blocking::Client::new()
        .delete(&url)
        .timeout(std::time::Duration::from_secs(10))
        .send()
    {
        Ok(r) if r.status().is_success() => {
            info!("Deleted {filename} from capture server");
        }
        Ok(r) => {
            warn!("Capture server DELETE {filename}: {}", r.status());
        }
        Err(e) => {
            warn!("Cannot reach capture server for DELETE {filename}: {e}");
        }
    }
}
