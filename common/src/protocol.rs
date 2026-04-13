//! Shared HTTP protocol types for communication between capture and
//! processing servers.

use serde::{Deserialize, Serialize};

/// Information about a single recording available on the capture server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordingInfo {
    pub filename: String,
    pub size: u64,
    /// ISO-8601 creation timestamp.
    pub created: String,
}

/// Health-check response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub uptime_secs: u64,
    /// Current disk usage of the recording volume (0–100).
    #[serde(default)]
    pub disk_usage_pct: f64,
    /// `true` when capture is paused because disk usage exceeds the
    /// configured threshold.
    #[serde(default)]
    pub capture_paused: bool,
}

/// Server-Sent Event payload for new-recording notifications.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewRecordingEvent {
    pub filename: String,
    pub size: u64,
}
