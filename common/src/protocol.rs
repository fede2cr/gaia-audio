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
}

/// Server-Sent Event payload for new-recording notifications.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewRecordingEvent {
    pub filename: String,
    pub size: u64,
}
