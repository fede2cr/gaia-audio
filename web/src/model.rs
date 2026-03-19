//! Shared data-transfer objects used by both server and client.

use serde::{Deserialize, Serialize};

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Return the `NODE_NAME` env var if set, otherwise `"local"`.
///
/// This gives the operator's chosen friendly name for the local station.
fn node_name_or_local() -> String {
    std::env::var("NODE_NAME")
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "local".into())
}

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
    pub source_node: String,
    /// `true` when the detection was excluded by the species-range model.
    #[serde(default)]
    pub excluded: bool,
    /// Species photo URL from iNaturalist (populated server-side).
    #[serde(default)]
    pub image_url: Option<String>,
    /// Short identifier for the model that produced this detection.
    #[serde(default)]
    pub model_slug: String,
    /// Human-readable model name (e.g. `"BirdNET V2.4"`).
    #[serde(default)]
    pub model_name: String,
    /// Timezone-adjusted date for display (YYYY-MM-DD).
    /// Same as `date` when tz_offset is 0 or unset.
    /// Kept separate so `clip_url()` always uses the UTC `date` for file paths.
    #[serde(default)]
    pub display_date: String,
    /// Timezone-adjusted time for display (HH:MM:SS).
    #[serde(default)]
    pub display_time: String,
}

impl WebDetection {
    /// Build the URL to the extracted audio clip served by `/extracted/`.
    ///
    /// Clips are stored as:
    ///   `{extracted_dir}/By_Date/{date}/{common_name_safe}/{file_name}`
    ///
    /// Returns `None` if `file_name` is empty.
    pub fn clip_url(&self) -> Option<String> {
        if self.file_name.is_empty() {
            return None;
        }
        let safe_name = self.common_name.replace('\'', "").replace(' ', "_");
        Some(format!(
            "/extracted/By_Date/{}/{}/{}",
            self.date, safe_name, self.file_name
        ))
    }

    /// URL to the spectrogram PNG (generated alongside the audio clip).
    ///
    /// Spectrograms are named `{clip_file}.png`.  After Opus compression
    /// the clip extension changes from `.wav`/`.mp3` to `.opus`, and the
    /// spectrogram is renamed accordingly, so this always works.
    pub fn spectrogram_url(&self) -> Option<String> {
        self.clip_url().map(|url| format!("{url}.png"))
    }

    /// Human-friendly label for the capture node.
    ///
    /// Uses the `NODE_NAME` environment variable when the source is local
    /// (localhost / 127.x), otherwise extracts hostname from the URL.
    /// Returns `"local"` when no node was recorded and no name is set.
    pub fn source_label(&self) -> String {
        if self.source_node.is_empty() {
            return node_name_or_local();
        }
        let stripped = self.source_node
            .trim_start_matches("http://")
            .trim_start_matches("https://")
            .trim_end_matches('/');
        let host = stripped.split(':').next().unwrap_or(stripped);
        if host == "localhost" || host.starts_with("127.") {
            return node_name_or_local();
        }
        // Remote node — show hostname portion (strip port)
        host.trim_end_matches('.').to_string()
    }

    /// Display label for the model that produced this detection.
    ///
    /// Prefers `model_name` (human-readable) but falls back to
    /// `model_slug` or `"Unknown model"`.
    pub fn model_label(&self) -> String {
        if !self.model_name.is_empty() {
            self.model_name.clone()
        } else if !self.model_slug.is_empty() {
            self.model_slug.clone()
        } else {
            "Unknown model".to_string()
        }
    }
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
    /// Male specimen photo URL (from iNaturalist sex-annotated observations).
    #[serde(default)]
    pub male_image_url: Option<String>,
    /// Female specimen photo URL (from iNaturalist sex-annotated observations).
    #[serde(default)]
    pub female_image_url: Option<String>,
    /// Verification state (loaded separately).
    #[serde(default)]
    pub verification: Option<SpeciesVerification>,
}

/// Verification record for a species.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeciesVerification {
    /// `"ornithologist"` or `"inaturalist"`.
    pub method: String,
    /// iNaturalist observation URL/ID (only when method == "inaturalist").
    #[serde(default)]
    pub inaturalist_obs: String,
    /// When the verification was recorded.
    #[serde(default)]
    pub verified_at: String,
}

// ─── Calendar ────────────────────────────────────────────────────────────────

/// One cell in the monthly calendar view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalendarDay {
    pub date: String,
    pub total_detections: u32,
    pub unique_species: u32,
}

// ─── Hourly histogram ────────────────────────────────────────────────────────

/// Detection count for a single hour (0–23).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HourlyCount {
    pub hour: u32,
    pub count: u32,
}

/// Per-species hourly breakdown (used in day and species views).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeciesHourlyCounts {
    pub scientific_name: String,
    pub common_name: String,
    pub total: u32,
    pub hours: Vec<HourlyCount>,
}

// ─── Top recordings (cached per species) ─────────────────────────────────────

/// A high-confidence recording cached in `species_top_recordings`.
///
/// Displayed on the species detail page so users can listen to the best
/// examples without running an expensive live query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopRecording {
    pub scientific_name: String,
    pub common_name: String,
    pub date: String,
    pub time: String,
    pub confidence: f64,
    pub file_name: String,
    pub source_node: String,
    pub model_name: String,
}

impl TopRecording {
    /// URL to the extracted audio clip.
    pub fn clip_url(&self) -> Option<String> {
        if self.file_name.is_empty() {
            return None;
        }
        let safe_name = self.common_name.replace('\'', "").replace(' ', "_");
        Some(format!(
            "/extracted/By_Date/{}/{}/{}",
            self.date, safe_name, self.file_name
        ))
    }

    /// URL to the spectrogram PNG.
    pub fn spectrogram_url(&self) -> Option<String> {
        self.clip_url().map(|url| format!("{url}.png"))
    }
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

// ─── Conservation status ─────────────────────────────────────────────────────

/// IUCN Red List conservation status codes, ordered from most to least
/// threatened.  The numeric values match the iNaturalist `iucn` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConservationStatus {
    /// Extinct
    EX = 70,
    /// Extinct in the Wild
    EW = 60,
    /// Critically Endangered
    CR = 50,
    /// Endangered
    EN = 40,
    /// Vulnerable
    VU = 30,
    /// Near Threatened
    NT = 20,
    /// Least Concern
    LC = 10,
    /// Data Deficient
    DD = 5,
    /// Not Evaluated
    NE = 0,
}

impl ConservationStatus {
    /// Parse from the iNaturalist numeric `iucn` field.
    pub fn from_iucn(code: u8) -> Option<Self> {
        match code {
            70 => Some(Self::EX),
            60 => Some(Self::EW),
            50 => Some(Self::CR),
            40 => Some(Self::EN),
            30 => Some(Self::VU),
            20 => Some(Self::NT),
            10 => Some(Self::LC),
            5  => Some(Self::DD),
            0  => Some(Self::NE),
            _  => None,
        }
    }

    /// Parse from a short IUCN code string (e.g. `"VU"`, `"EN"`).
    pub fn from_code(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "EX" => Some(Self::EX),
            "EW" => Some(Self::EW),
            "CR" => Some(Self::CR),
            "EN" => Some(Self::EN),
            "VU" => Some(Self::VU),
            "NT" => Some(Self::NT),
            "LC" => Some(Self::LC),
            "DD" => Some(Self::DD),
            "NE" => Some(Self::NE),
            _    => None,
        }
    }

    /// Short IUCN code (e.g. `"CR"`, `"LC"`).
    pub fn code(self) -> &'static str {
        match self {
            Self::EX => "EX",
            Self::EW => "EW",
            Self::CR => "CR",
            Self::EN => "EN",
            Self::VU => "VU",
            Self::NT => "NT",
            Self::LC => "LC",
            Self::DD => "DD",
            Self::NE => "NE",
        }
    }

    /// Human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            Self::EX => "Extinct",
            Self::EW => "Extinct in the Wild",
            Self::CR => "Critically Endangered",
            Self::EN => "Endangered",
            Self::VU => "Vulnerable",
            Self::NT => "Near Threatened",
            Self::LC => "Least Concern",
            Self::DD => "Data Deficient",
            Self::NE => "Not Evaluated",
        }
    }

    /// Numeric sort key — higher = more threatened.
    pub fn threat_level(self) -> u8 {
        self as u8
    }

    /// CSS modifier class for colour-coding the badge.
    pub fn css_class(self) -> &'static str {
        match self {
            Self::EX | Self::EW => "status-extinct",
            Self::CR => "status-critical",
            Self::EN => "status-endangered",
            Self::VU => "status-vulnerable",
            Self::NT => "status-near-threatened",
            Self::LC => "status-least-concern",
            Self::DD | Self::NE => "status-unknown",
        }
    }
}

impl std::fmt::Display for ConservationStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.label(), self.code())
    }
}

// ─── iNaturalist API response subset ─────────────────────────────────────────

/// A cached photo record from iNaturalist.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeciesPhoto {
    pub medium_url: String,
    pub attribution: String,
    pub wikipedia_url: Option<String>,
    /// IUCN conservation status (from iNaturalist `conservation_status`).
    #[serde(default)]
    pub conservation_status: Option<ConservationStatus>,
    /// Male specimen photo URL (from sex-annotated observations).
    #[serde(default)]
    pub male_image_url: Option<String>,
    /// Female specimen photo URL (from sex-annotated observations).
    #[serde(default)]
    pub female_image_url: Option<String>,
}

// ─── Species summary (for species list) ──────────────────────────────────────

/// Compact species row shown on the home or calendar page.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeciesSummary {
    pub scientific_name: String,
    pub common_name: String,
    pub domain: String,
    pub detection_count: u32,
    /// Human-friendly count for display (e.g. `"~230"`, `"42"`).
    #[serde(default)]
    pub display_count: String,
    pub last_seen: Option<String>,
    pub image_url: Option<String>,
    /// IUCN conservation status (populated from iNaturalist).
    #[serde(default)]
    pub conservation_status: Option<ConservationStatus>,
    /// Male specimen photo URL (from iNaturalist sex-annotated observations).
    #[serde(default)]
    pub male_image_url: Option<String>,
    /// Female specimen photo URL (from iNaturalist sex-annotated observations).
    #[serde(default)]
    pub female_image_url: Option<String>,
}

// ─── Model info (for filter dropdowns) ───────────────────────────────────────

/// A detection model available in the database.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelInfo {
    /// Machine slug (e.g. `"birdnet"`, `"birdnet3"`).
    pub slug: String,
    /// Human-readable name (e.g. `"BirdNET V2.4"`, `"BirdNET+ V3.0"`).
    pub name: String,
}

impl ModelInfo {
    /// Display label: prefer name, fall back to slug.
    pub fn label(&self) -> &str {
        if !self.name.is_empty() { &self.name } else { &self.slug }
    }
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

/// A BirdNET-Pi node discovered on the local network via mDNS.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BirdnetNode {
    /// mDNS instance name (e.g. "birdnet").
    pub name: String,
    /// IPv4 address.
    pub address: String,
    /// HTTP port (typically 80).
    pub port: u16,
}

/// A discovered backup archive in the /backups volume.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupFile {
    /// Full path inside the container (e.g. `/backups/my-backup.tar`).
    pub path: String,
    /// Filename only.
    pub name: String,
    /// Size in bytes.
    pub size_bytes: u64,
}

// ─── Live analysis status ────────────────────────────────────────────────────

/// Snapshot of what the processing server is currently analysing.
/// Read from `/data/live_status.json` written by the processing container.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveStatus {
    pub timestamp: String,
    pub filename: String,
    pub predictions: Vec<LivePrediction>,
    pub has_detections: bool,
    /// URL of the capture node this recording came from.
    #[serde(default)]
    pub source_node: String,
    /// ISO-8601 capture timestamp parsed from the recording filename.
    #[serde(default)]
    pub captured_at: String,
}

/// One prediction entry in the live status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LivePrediction {
    pub scientific_name: String,
    pub common_name: String,
    pub confidence: f64,
    /// Short identifier for the model that produced this prediction.
    #[serde(default)]
    pub model_slug: String,
    /// Human-readable model name.
    #[serde(default)]
    pub model_name: String,
}

// ─── Urban noise ─────────────────────────────────────────────────────────────

/// Aggregated urban-noise counts for a single category.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UrbanNoiseSummary {
    pub category: String,
    pub today_count: u32,
    pub week_count: u32,
    pub total_count: u32,
}

// ─── Excluded species ────────────────────────────────────────────────────────

/// A species that has been excluded by the occurrence-threshold filter,
/// aggregated for the "Excluded" page.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExcludedSpecies {
    pub scientific_name: String,
    pub common_name: String,
    pub domain: String,
    pub detection_count: u32,
    pub last_seen: Option<String>,
    pub max_confidence: f64,
    pub image_url: Option<String>,
    /// `true` if an ornithologist has overridden the exclusion.
    pub overridden: bool,
}

// ─── Settings ────────────────────────────────────────────────────────────────

/// Detection settings editable from the web UI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionSettings {
    pub sensitivity: f64,
    pub confidence: f64,
    pub sf_thresh: f64,
    pub overlap: f64,
    /// Spectrogram colour palette name ("default", "coolwarm", "magma", "viridis", "grayscale").
    #[serde(default = "default_colormap")]
    pub colormap: String,
    /// Timezone offset from UTC in hours (e.g. -6 for UTC-6).
    /// Used to convert UTC timestamps for display in the web UI.
    #[serde(default)]
    pub tz_offset: i32,
}

fn default_colormap() -> String {
    "default".to_string()
}

// ─── Learning quiz ───────────────────────────────────────────────────────────

/// A single quiz question: one audio clip the user must identify.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuizItem {
    /// Scientific name (the correct answer).
    pub scientific_name: String,
    /// Common name (the correct answer).
    pub common_name: String,
    /// URL to the audio clip.
    pub clip_url: String,
    /// URL to the spectrogram PNG.
    pub spectrogram_url: String,
    /// Species photo from iNaturalist (if available).
    #[serde(default)]
    pub image_url: Option<String>,
}
