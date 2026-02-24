//! Audio capture – spawns `arecord` or `ffmpeg` as child processes.
//!
//! Reused from `birdnet-server/src/capture.rs`.

use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};

use anyhow::{Context, Result};
use tracing::{debug, info, warn};

use gaia_common::config::Config;

/// Opaque handle that owns the recording child process(es).
pub struct CaptureHandle {
    children: Vec<Child>,
}

impl CaptureHandle {
    #[allow(dead_code)]
    pub fn kill(&mut self) -> Result<()> {
        for child in &mut self.children {
            let _ = child.kill();
        }
        Ok(())
    }

    /// Check whether any child has exited.  Returns `Some(status_msg)` if
    /// a child died, `None` if all are still running.
    pub fn check_alive(&mut self) -> Option<String> {
        for (i, child) in self.children.iter_mut().enumerate() {
            match child.try_wait() {
                Ok(Some(status)) => {
                    return Some(format!(
                        "Capture child {} exited with {}",
                        i, status
                    ));
                }
                Ok(None) => {} // still running
                Err(e) => {
                    return Some(format!(
                        "Cannot check capture child {}: {e}",
                        i
                    ));
                }
            }
        }
        None
    }
}

/// Start the audio capture pipeline according to the config.
pub fn start(config: &Config) -> Result<CaptureHandle> {
    std::fs::create_dir_all(config.stream_data_dir())
        .context("Cannot create StreamData directory")?;

    if !config.rtsp_streams.is_empty() {
        start_rtsp(config)
    } else {
        start_microphone(config)
    }
}

// ── RTSP via ffmpeg ──────────────────────────────────────────────────────

fn start_rtsp(config: &Config) -> Result<CaptureHandle> {
    let mut children = Vec::new();

    for (i, url) in config.rtsp_streams.iter().enumerate() {
        let stream_idx = i + 1;
        let output_pattern = config
            .stream_data_dir()
            .join(format!("%F-birdnet-RTSP_{stream_idx}-%H:%M:%S.wav"));

        let timeout_args = if url.starts_with("rtsp://") || url.starts_with("rtsps://") {
            vec!["-timeout".to_string(), "10000000".to_string()]
        } else if url.contains("://") {
            vec!["-rw_timeout".to_string(), "10000000".to_string()]
        } else {
            vec![]
        };

        let mut cmd = Command::new("ffmpeg");
        cmd.args(["-hide_banner", "-loglevel", "error", "-nostdin"]);
        for arg in &timeout_args {
            cmd.arg(arg);
        }
        cmd.args([
            "-i",
            url,
            "-vn",
            "-map",
            "a:0",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "2",
            "-ar",
            "48000",
            "-f",
            "segment",
            "-segment_format",
            "wav",
            "-segment_time",
            &config.recording_length.to_string(),
            "-strftime",
            "1",
        ]);
        cmd.arg(output_pattern.to_str().unwrap());
        cmd.stdout(Stdio::null()).stderr(Stdio::piped());

        let child = cmd
            .spawn()
            .with_context(|| format!("Failed to spawn ffmpeg for stream {stream_idx}: {url}"))?;

        info!("ffmpeg started for RTSP stream {stream_idx}: {url}");
        children.push(child);
    }

    Ok(CaptureHandle { children })
}

// ── Local microphone via arecord ─────────────────────────────────────────

fn start_microphone(config: &Config) -> Result<CaptureHandle> {
    let output_pattern = config
        .stream_data_dir()
        .join("%F-birdnet-%H:%M:%S.wav");

    let mut cmd = Command::new("arecord");
    cmd.args([
        "-f",
        "S16_LE",
        &format!("-c{}", config.channels),
        "-r48000",
        "-t",
        "wav",
        "--max-file-time",
        &config.recording_length.to_string(),
        "--use-strftime",
    ]);

    if let Some(card) = &config.rec_card {
        cmd.args(["-D", card]);
    }

    cmd.arg(output_pattern.to_str().unwrap());
    cmd.stdout(Stdio::null()).stderr(Stdio::piped());

    info!(
        "Spawning: arecord -f S16_LE -c{} -r48000 -t wav --max-file-time {} --use-strftime {} → {}",
        config.channels,
        config.recording_length,
        config.rec_card.as_deref().unwrap_or("(default)"),
        output_pattern.display(),
    );

    let mut child = cmd.spawn().context("Failed to spawn arecord")?;

    // Drain stderr in a background thread so we see any ALSA errors
    // and the pipe buffer doesn't fill up and block arecord.
    if let Some(stderr) = child.stderr.take() {
        std::thread::Builder::new()
            .name("arecord-stderr".into())
            .spawn(move || {
                let reader = BufReader::new(stderr);
                for line in reader.lines() {
                    match line {
                        Ok(l) if l.is_empty() => {}
                        Ok(l) => warn!("[arecord] {l}"),
                        Err(_) => break,
                    }
                }
                debug!("arecord stderr stream ended");
            })
            .ok();
    }

    // Give arecord a moment to fail on bad config before declaring success.
    std::thread::sleep(std::time::Duration::from_millis(500));
    match child.try_wait() {
        Ok(Some(status)) => {
            anyhow::bail!(
                "arecord exited immediately with {status} — check REC_CARD in gaia.conf \
                 (run 'arecord -l' on the host to list capture devices)"
            );
        }
        Ok(None) => {} // still running – good
        Err(e) => warn!("Cannot check arecord status: {e}"),
    }

    info!(
        "arecord started (pid={}, channels={}, card={:?})",
        child.id(),
        config.channels,
        config.rec_card
    );

    Ok(CaptureHandle {
        children: vec![child],
    })
}
