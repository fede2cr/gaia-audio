//! Reporting: write detections to DB, extract audio clips, generate
//! spectrograms, send notifications.
//!
//! Evolved from `birdnet-server/src/reporting.rs`.

use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;

use anyhow::Result;
use tracing::{debug, error, info, warn};

use gaia_common::audio;
use gaia_common::config::Config;
use gaia_common::detection::{Detection, ParsedFileName};

use crate::db;
use crate::spectrogram::{self, Colormap, SpectrogramParams};
use crate::ReportPayload;

/// Run the reporting loop on its own thread.
pub fn handle_queue(rx: Receiver<ReportPayload>, config: &Config, db_path: &Path) {
    let mut config = config.clone();
    while let Ok(payload) = rx.recv() {
        // Refresh settings (colormap, thresholds) from the DB so web UI
        // changes are picked up without restarting the container.
        db::apply_settings_overrides(&mut config);

        if let Err(e) = process_report(&payload, &config, db_path) {
            error!("Reporting error: {e:#}");
        }

        // Notify capture server to delete the source file (if local)
        // or delete it ourselves if running mono-node.
        // When running in split mode (capture ↔ processing), the client
        // has already deleted the recording from the remote capture
        // server after processing, so we only clean up local files here.
        let src = &payload.file.file_path;
        if src.exists() {
            if let Err(e) = std::fs::remove_file(src) {
                warn!("Cannot remove source file {}: {e}", src.display());
            } else {
                // Count remaining temp files in the same directory.
                let remaining = src.parent()
                    .and_then(|d| std::fs::read_dir(d).ok())
                    .map(|rd| rd.filter_map(|e| e.ok()).filter(|e| {
                        e.path().extension().map(|x| x == "wav" || x == "mp3").unwrap_or(false)
                    }).count())
                    .unwrap_or(0);
                info!(
                    "Removed local temp {} ({remaining} audio files remaining in tmp)",
                    src.display()
                );
            }
        }
    }
    info!("Reporting thread finished");
}

fn process_report(payload: &ReportPayload, config: &Config, db_path: &Path) -> Result<()> {
    let file = &payload.file;

    // Separate urban-noise detections (Engine, Dog, Human, …) from real
    // species.  Noise detections are counted but NOT stored in the main
    // detections table.
    let (species_dets, noise_dets): (Vec<&Detection>, Vec<&Detection>) = payload
        .detections
        .iter()
        .partition(|d| !db::is_urban_noise(&d.scientific_name));

    debug!(
        "Report for {}: {} total detection(s) ({} species, {} noise)",
        file.file_path.display(),
        payload.detections.len(),
        species_dets.len(),
        noise_dets.len()
    );

    write_json_file(file, &payload.detections, config)?;

    // ── real species detections ──────────────────────────────────────
    for detection in &species_dets {
        // Attempt audio clip extraction.  Extraction failure MUST NOT
        // prevent the detection from being recorded in the database.
        let extracted = match extract_detection(file, detection, config) {
            Ok(path) => {
                let spec_path = format!("{}.png", path.display());
                let spec_params = SpectrogramParams {
                    colormap: config.colormap.parse::<Colormap>().unwrap_or_default(),
                    ..SpectrogramParams::default()
                };
                if let Err(e) = spectrogram::generate_from_wav(
                    &path,
                    Path::new(&spec_path),
                    &spec_params,
                ) {
                    warn!("Spectrogram failed for {}: {e}", path.display());
                }
                // Convert WAV clip → Opus immediately.  Falls back to
                // the original WAV path if ffmpeg is unavailable.
                let final_path = crate::compress::compress_inline(&path)
                    .unwrap_or(path);
                Some(final_path)
            }
            Err(e) => {
                warn!("Clip extraction failed (detection will still be recorded): {e:#}");
                None
            }
        };

        let summary = format_summary(detection, config);
        let basename = extracted
            .as_ref()
            .and_then(|p| p.file_name())
            .unwrap_or_default()
            .to_string_lossy();
        let model_tag = if !detection.model_name.is_empty() {
            &detection.model_name
        } else if !detection.model_slug.is_empty() {
            &detection.model_slug
        } else {
            "unknown"
        };
        info!(
            "[{model_tag}] {} {} ({:.1}%) @ {};{basename}",
            detection.common_name,
            detection.scientific_name,
            detection.confidence * 100.0,
            detection.time,
        );

        write_to_log(&summary, &config.recs_dir);

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

    // ── urban noise detections ───────────────────────────────────────
    for detection in &noise_dets {
        // For non-Human noise (Engine, Dog, …) we still extract the clip
        // so the operator can review.  Human recordings are skipped for
        // privacy reasons.
        let is_human = detection.scientific_name.contains("Human");

        if !is_human {
            match extract_detection(file, detection, config) {
                Ok(path) => {
                    // Convert noise clip to Opus inline as well.
                    crate::compress::compress_inline(&path);
                }
                Err(e) => {
                    warn!("Noise clip extraction failed: {e}");
                }
            }
        }

        // Normalise the category: "Human vocal" / "Human whistle" → "Human"
        let category = if is_human {
            "Human"
        } else {
            &detection.scientific_name
        };

        let hour: u32 = detection
            .time
            .split(':')
            .next()
            .and_then(|h| h.parse().ok())
            .unwrap_or(0);

        if let Err(e) = db::increment_urban_noise(db_path, &detection.date, hour, category) {
            error!("Urban noise DB update failed: {e}");
        }

        info!(
            "[urban-noise] {}: {} ({:.1}%) at {}",
            detection.date,
            category,
            detection.confidence * 100.0,
            detection.time,
        );
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

    let model_tag = if detection.model_slug.is_empty() {
        "unknown"
    } else {
        &detection.model_slug
    };
    let new_name = format!(
        "{}-{}-{}-{}-{}-{}{}.wav",
        detection.domain,
        detection.common_name_safe,
        detection.confidence_pct(),
        detection.date,
        model_tag,
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
        debug!("Extraction already exists (WAV): {}", new_path.display());
        return Ok(new_path);
    }

    // Check whether an Opus-compressed version already exists (from a
    // previous inline compression).  If so, return the Opus path
    // directly — no need to re-extract and re-compress.
    let opus_name = format!("{}.opus", &new_name[..new_name.len() - 4]);
    let opus_path = new_dir.join(&opus_name);
    if opus_path.exists() {
        debug!("Extraction already exists (Opus): {}", opus_path.display());
        return Ok(opus_path);
    }

    audio::extract_clip(&file.file_path, &new_path, safe_start, safe_stop)?;
    debug!(
        "Extracted clip {:.1}s–{:.1}s from {} → {}",
        safe_start, safe_stop,
        file.file_path.display(),
        new_path.display()
    );
    Ok(new_path)
}

// ── summary / logging ────────────────────────────────────────────────────

fn format_summary(d: &Detection, config: &Config) -> String {
    let model = if !d.model_name.is_empty() {
        &d.model_name
    } else if !d.model_slug.is_empty() {
        &d.model_slug
    } else {
        "unknown"
    };
    format!(
        "{};{};{};{};{};{};{};{};{};{};{};{};{}",
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
        model,
    )
}

fn write_to_log(summary: &str, data_dir: &Path) {
    let log_path = data_dir.join("GaiaDB.txt");

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

    // Only POST non-excluded bird detections to BirdWeather
    let bird_dets: Vec<&Detection> = detections
        .iter()
        .filter(|d| d.domain == "birds" && !d.excluded)
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
