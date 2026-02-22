//! Shared data-transfer objects used by both server and client.

use serde::{Deserialize, Serialize};

// ─── Detection ───────────────────────────────────────────────────────────────

/// A single detection row, fully serialisable (no DateTime).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebDetection {
    pub id: i64,
    pub domain: String,
    pub scientific_name: String,
    pub common_name: String,
    pub confidence: f64,
    pub date: String,
    pub time: String,
    pub file_name: String,
}

// ─── Species ─────────────────────────────────────────────────────────────────

/// Aggregated species information (with optional iNaturalist data).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeciesInfo {
    pub scientific_name: String,
    pub common_name: String,
    pub domain: String,
    pub image_url: Option<String>,
    pub wikipedia_url: Option<String>,
    pub total_detections: u64,
    pub first_seen: Option<String>,
    pub last_seen: Option<String>,
}

// ─── Calendar ────────────────────────────────────────────────────────────────

/// One cell in the monthly calendar view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalendarDay {
    pub date: String,
    pub total_detections: u32,
    pub unique_species: u32,
}

/// All detections for a single species within one day, grouped.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DayDetectionGroup {
    pub scientific_name: String,
    pub common_name: String,
    pub domain: String,
    pub image_url: Option<String>,
    pub detections: Vec<WebDetection>,
    pub max_confidence: f64,
}

// ─── iNaturalist API response subset ─────────────────────────────────────────

/// A cached photo record from iNaturalist.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeciesPhoto {
    pub medium_url: String,
    pub attribution: String,
    pub wikipedia_url: Option<String>,
}

// ─── Species summary (for species list) ──────────────────────────────────────

/// Compact species row shown on the home or calendar page.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeciesSummary {
    pub scientific_name: String,
    pub common_name: String,
    pub domain: String,
    pub detection_count: u32,
    pub last_seen: Option<String>,
    pub image_url: Option<String>,
}

// ─── Import (BirdNET-Pi backup) ──────────────────────────────────────────────

/// Pre-import analysis of a BirdNET-Pi backup archive.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportReport {
    pub tar_path: String,
    pub tar_size_bytes: u64,
    pub total_detections: u64,
    pub today_detections: u64,
    pub total_species: u32,
    pub today_species: u32,
    pub date_min: Option<String>,
    pub date_max: Option<String>,
    pub audio_file_count: u64,
    pub spectrogram_count: u64,
    pub latitude: Option<f64>,
    pub longitude: Option<f64>,
    pub top_species: Vec<(String, u64)>,
}

/// Completed import summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportResult {
    pub detections_imported: u64,
    pub files_extracted: u64,
    pub skipped_existing: u64,
    pub errors: Vec<String>,
}
