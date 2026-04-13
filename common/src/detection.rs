//! Detection and file-name parsing types.
//!
//! Reused from `birdnet-server/src/detection.rs`, extended with a `domain`
//! field so that a single database / pipeline can hold birds, bats, insects, etc.

use chrono::{Datelike, Local, NaiveDate, NaiveDateTime, NaiveTime, TimeZone};

/// A single species detection within an audio recording.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Detection {
    pub domain: String,
    pub start: f64,
    pub stop: f64,
    #[serde(skip)]
    pub datetime: chrono::DateTime<Local>,
    pub date: String,
    pub time: String,
    pub iso8601: String,
    pub week: u32,
    pub confidence: f64,
    pub scientific_name: String,
    pub common_name: String,
    pub common_name_safe: String,
    /// Populated after extraction.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_name_extr: Option<String>,
}

impl Detection {
    pub fn new(
        domain: &str,
        file_date: NaiveDateTime,
        start: f64,
        stop: f64,
        scientific_name: &str,
        common_name: &str,
        confidence: f64,
    ) -> Self {
        let dt = file_date + chrono::Duration::milliseconds((start * 1000.0) as i64);
        let local_dt = Local
            .from_local_datetime(&dt)
            .single()
            .unwrap_or_else(|| Local::now());

        let common_name_safe = common_name.replace('\'', "").replace(' ', "_");

        Detection {
            domain: domain.to_string(),
            start,
            stop,
            datetime: local_dt,
            date: local_dt.format("%Y-%m-%d").to_string(),
            time: local_dt.format("%H:%M:%S").to_string(),
            iso8601: local_dt.to_rfc3339(),
            week: local_dt.iso_week().week(),
            confidence: (confidence * 10000.0).round() / 10000.0,
            scientific_name: scientific_name.to_string(),
            common_name: common_name.to_string(),
            common_name_safe,
            file_name_extr: None,
        }
    }

    /// Confidence as integer percentage (0..100).
    pub fn confidence_pct(&self) -> u32 {
        (self.confidence * 100.0).round() as u32
    }
}

impl std::fmt::Display for Detection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Detection({}/{}, {}, {:.4}, {})",
            self.domain, self.scientific_name, self.common_name, self.confidence, self.iso8601
        )
    }
}

/// Parsed metadata from a recording filename.
///
/// Filenames follow the pattern:
///   `2024-02-24-birdnet-RTSP_1-16:19:37.wav`
///   `2024-02-24-birdnet-16:19:37.wav`
#[derive(Debug, Clone)]
pub struct ParsedFileName {
    pub file_path: std::path::PathBuf,
    pub file_date: NaiveDateTime,
    pub rtsp_id: String,
}

impl ParsedFileName {
    pub fn parse(path: &std::path::Path) -> anyhow::Result<Self> {
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| anyhow::anyhow!("Invalid filename: {}", path.display()))?;

        // Extract date: leading YYYY-MM-DD
        if stem.len() < 10 {
            anyhow::bail!("Filename too short: {stem}");
        }
        let date_str = &stem[..10];
        let date = NaiveDate::parse_from_str(date_str, "%Y-%m-%d")
            .map_err(|e| anyhow::anyhow!("Bad date in filename {stem}: {e}"))?;

        // Extract time: trailing HH:MM:SS
        if stem.len() < 8 {
            anyhow::bail!("Filename too short for time: {stem}");
        }
        let time_str = &stem[stem.len() - 8..];
        let time = NaiveTime::parse_from_str(time_str, "%H:%M:%S")
            .map_err(|e| anyhow::anyhow!("Bad time in filename {stem}: {e}"))?;

        // Extract RTSP id if present
        let rtsp_id = if let Some(start) = stem.find("RTSP_") {
            let rest = &stem[start..];
            if let Some(end) = rest.find('-') {
                format!("{}-", &rest[..end])
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        Ok(ParsedFileName {
            file_path: path.to_path_buf(),
            file_date: NaiveDateTime::new(date, time),
            rtsp_id,
        })
    }

    /// ISO-8601 representation in the local timezone.
    pub fn iso8601(&self) -> String {
        Local
            .from_local_datetime(&self.file_date)
            .single()
            .map(|dt| dt.to_rfc3339())
            .unwrap_or_default()
    }

    /// ISO week number (1..=53).
    pub fn week(&self) -> u32 {
        self.file_date.date().iso_week().week()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_parse_filename_plain() {
        let p = Path::new("/data/StreamData/2024-02-24-birdnet-16:19:37.wav");
        let pf = ParsedFileName::parse(p).unwrap();
        assert_eq!(pf.file_date.format("%Y-%m-%d").to_string(), "2024-02-24");
        assert_eq!(pf.rtsp_id, "");
    }

    #[test]
    fn test_parse_filename_rtsp() {
        let p = Path::new("/data/StreamData/2024-02-24-birdnet-RTSP_1-16:19:37.wav");
        let pf = ParsedFileName::parse(p).unwrap();
        assert_eq!(pf.rtsp_id, "RTSP_1-");
    }

    #[test]
    fn test_detection_display() {
        let d = Detection::new(
            "birds",
            NaiveDateTime::new(
                NaiveDate::from_ymd_opt(2024, 6, 15).unwrap(),
                NaiveTime::from_hms_opt(10, 30, 0).unwrap(),
            ),
            0.0,
            3.0,
            "Turdus merula",
            "Eurasian Blackbird",
            0.92,
        );
        let s = format!("{d}");
        assert!(s.contains("birds"));
        assert!(s.contains("Turdus merula"));
    }
}
