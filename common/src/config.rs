//! Configuration parsing – reads a KEY=VALUE file (compatible with
//! `birdnet.conf` and the new `gaia.conf` format).
//!
//! Reused and extended from `birdnet-server/src/config.rs`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use tracing::info;

/// Application configuration, shared between capture and processing servers.
///
/// Both servers load the same file; each ignores fields it does not need.
#[derive(Debug, Clone)]
pub struct Config {
    // ── location ─────────────────────────────────────────────────────
    pub latitude: f64,
    pub longitude: f64,

    // ── detection thresholds (processing) ────────────────────────────
    pub confidence: f64,
    pub sensitivity: f64,
    pub overlap: f64,

    // ── recording (capture) ──────────────────────────────────────────
    pub recording_length: u32,
    pub channels: u16,
    pub rec_card: Option<String>,
    pub recs_dir: PathBuf,
    pub extracted_dir: PathBuf,
    pub audio_fmt: String,
    pub rtsp_streams: Vec<String>,

    // ── model (processing) ───────────────────────────────────────────
    /// Root directory containing model subdirectories (each with a manifest.toml).
    pub model_dir: PathBuf,
    pub database_lang: String,
    pub sf_thresh: f64,
    pub data_model_version: u32,
    /// Model variant to use (e.g. "fp16", "fp32", "int8").
    /// When set and the manifest has a [download] section, the processing
    /// server will auto-download the corresponding model from Zenodo.
    pub model_variant: Option<String>,

    // ── privacy / extraction (processing) ────────────────────────────
    pub raw_spectrogram: bool,
    pub privacy_threshold: f64,
    pub extraction_length: u32,

    // ── integrations (processing) ────────────────────────────────────
    pub birdweather_id: Option<String>,
    pub heartbeat_url: Option<String>,

    // ── database (processing) ────────────────────────────────────────
    pub db_path: PathBuf,

    // ── network (capture ↔ processing) ───────────────────────────────
    /// Address the capture HTTP server listens on.
    pub capture_listen_addr: String,
    /// URL the processing server uses to reach the capture server.
    pub capture_server_url: String,
    /// Polling interval for the processing server (seconds).
    pub poll_interval_secs: u64,
}

impl Config {
    /// Default config path.
    pub fn default_path() -> &'static str {
        "/etc/gaia/gaia.conf"
    }

    /// Convenience: the StreamData subdirectory under `recs_dir`.
    pub fn stream_data_dir(&self) -> PathBuf {
        self.recs_dir.join("StreamData")
    }
}

/// Parse a `KEY=VALUE` configuration file.
///
/// Lines starting with `#` are comments.  Values may be optionally
/// double-quoted.  Unknown keys are silently ignored.
pub fn load(path: &Path) -> Result<Config> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("Cannot read config: {}", path.display()))?;

    let map = parse_conf(&text);
    info!("Loaded config from {}", path.display());

    let get = |key: &str| -> Option<String> { map.get(key).cloned() };
    let get_f64 = |key: &str, default: f64| -> f64 {
        get(key).and_then(|v| v.parse().ok()).unwrap_or(default)
    };
    let get_u32 = |key: &str, default: u32| -> u32 {
        get(key).and_then(|v| v.parse().ok()).unwrap_or(default)
    };

    let recs_dir = PathBuf::from(get("RECS_DIR").unwrap_or_else(|| "/data".into()));
    let extracted_dir = get("EXTRACTED")
        .map(PathBuf::from)
        .unwrap_or_else(|| recs_dir.join("Extracted"));

    let rtsp_streams: Vec<String> = get("RTSP_STREAMS")
        .map(|s| {
            s.split(',')
                .map(|u| u.trim().to_string())
                .filter(|u| !u.is_empty())
                .collect()
        })
        .unwrap_or_default();

    Ok(Config {
        latitude: get_f64("LATITUDE", -1.0),
        longitude: get_f64("LONGITUDE", -1.0),
        confidence: get_f64("CONFIDENCE", 0.7),
        sensitivity: get_f64("SENSITIVITY", 1.25),
        overlap: get_f64("OVERLAP", 0.0),
        recording_length: get_u32("RECORDING_LENGTH", 15),
        channels: get("CHANNELS").and_then(|v| v.parse().ok()).unwrap_or(1),
        rec_card: get("REC_CARD").filter(|s| !s.is_empty()),
        recs_dir,
        extracted_dir,
        audio_fmt: get("AUDIOFMT").unwrap_or_else(|| "wav".into()),
        rtsp_streams,

        model_dir: PathBuf::from(get("MODEL_DIR").unwrap_or_else(|| "/models".into())),
        database_lang: get("DATABASE_LANG").unwrap_or_else(|| "en".into()),
        sf_thresh: get_f64("SF_THRESH", 0.03),
        data_model_version: get_u32("DATA_MODEL_VERSION", 2),
        model_variant: get("MODEL_VARIANT").filter(|s| !s.is_empty()),
        raw_spectrogram: get("RAW_SPECTROGRAM")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false),
        privacy_threshold: get_f64("PRIVACY_THRESHOLD", 0.0),
        extraction_length: get_u32("EXTRACTION_LENGTH", 6),

        birdweather_id: get("BIRDWEATHER_ID").filter(|s| !s.is_empty()),
        heartbeat_url: get("HEARTBEAT_URL").filter(|s| !s.is_empty()),

        db_path: PathBuf::from(get("DB_PATH").unwrap_or_else(|| "/data/birds.db".into())),

        capture_listen_addr: get("CAPTURE_LISTEN_ADDR")
            .unwrap_or_else(|| "0.0.0.0:8089".into()),
        capture_server_url: get("CAPTURE_SERVER_URL")
            .unwrap_or_else(|| "http://capture:8089".into()),
        poll_interval_secs: get("POLL_INTERVAL_SECS")
            .and_then(|v| v.parse().ok())
            .unwrap_or(5),
    })
}

/// Parse `KEY=VALUE` lines into a map, stripping optional double-quotes.
fn parse_conf(text: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((key, val)) = line.split_once('=') {
            let key = key.trim();
            let val = val.trim().trim_matches('"');
            map.insert(key.to_string(), val.to_string());
        }
    }
    map
}

// ─── tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_conf() {
        let text = r#"
# comment
LATITUDE=42.36
LONGITUDE="-72.52"
CONFIDENCE=0.75
RECS_DIR=/data
RTSP_STREAMS="rtsp://cam1,rtsp://cam2"
CAPTURE_LISTEN_ADDR=0.0.0.0:9090
"#;
        let map = parse_conf(text);
        assert_eq!(map["LATITUDE"], "42.36");
        assert_eq!(map["LONGITUDE"], "-72.52");
        assert_eq!(map["CAPTURE_LISTEN_ADDR"], "0.0.0.0:9090");
    }

    #[test]
    fn test_config_stream_data_dir() {
        let text = "RECS_DIR=/tmp/test\n";
        let tmp = tempfile(text);
        let config = load(tmp.as_path()).unwrap();
        assert_eq!(config.stream_data_dir(), PathBuf::from("/tmp/test/StreamData"));
    }

    fn tempfile(content: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("gaia_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.conf");
        std::fs::write(&path, content).unwrap();
        path
    }
}
