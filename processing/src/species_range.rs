//! Static species-range lookup for geographic filtering.
//!
//! Loads a CSV file (`species_ranges.csv`) that maps scientific names
//! to bounding-box ranges (lat/lon) so that models without a trained
//! metadata model (e.g. Google Perch for insects, frogs, bats) can
//! still be filtered geographically.
//!
//! ## CSV format
//!
//! ```csv
//! sci_name,lat_min,lat_max,lon_min,lon_max,months
//! Phaneroptera nana,30.0,55.0,-10.0,45.0,4-10
//! Conocephalus fuscus,35.0,60.0,-10.0,50.0,5-9
//! Rana temporaria,35.0,72.0,-25.0,60.0,
//! ```
//!
//! - `sci_name`: canonical scientific name (normalised on load).
//! - `lat_min`, `lat_max`, `lon_min`, `lon_max`: bounding box.
//! - `months` (optional): comma-separated or range of active months
//!   (e.g. `4-10` = April–October, `1,2,11,12` = winter).
//!   Empty = year-round.
//!
//! The file is loaded once at startup and queried per-chunk.

use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;

use gaia_common::detection::normalize_sci_name;
use tracing::info;

/// Global singleton — loaded once at first access.
static RANGE_MAP: OnceLock<SpeciesRangeMap> = OnceLock::new();

/// Get a reference to the global species-range map.
///
/// Loads `$GAIA_DIR/species_ranges.csv` (or `/app/species_ranges.csv`)
/// on the first call; subsequent calls return the cached map.
pub fn global() -> &'static SpeciesRangeMap {
    RANGE_MAP.get_or_init(|| {
        let base = std::env::var("GAIA_DIR").unwrap_or_else(|_| "/app".to_string());
        SpeciesRangeMap::load(Path::new(&base).join("species_ranges.csv").as_path())
    })
}

/// A geographic bounding box with optional seasonal activity window.
#[derive(Debug, Clone)]
pub struct SpeciesRange {
    pub lat_min: f64,
    pub lat_max: f64,
    pub lon_min: f64,
    pub lon_max: f64,
    /// Active months (1-12).  Empty = year-round.
    pub months: Vec<u32>,
}

impl SpeciesRange {
    /// Check if the given location and month falls within this range.
    pub fn contains(&self, lat: f64, lon: f64, month: u32) -> bool {
        let in_box = lat >= self.lat_min
            && lat <= self.lat_max
            && lon >= self.lon_min
            && lon <= self.lon_max;
        let in_season = self.months.is_empty() || self.months.contains(&month);
        in_box && in_season
    }
}

/// In-memory lookup: normalised scientific name → bounding-box range.
pub struct SpeciesRangeMap {
    ranges: HashMap<String, SpeciesRange>,
}

impl SpeciesRangeMap {
    /// Load from a CSV file.  Returns an empty map if the file doesn't exist.
    pub fn load(path: &Path) -> Self {
        let ranges = match std::fs::read_to_string(path) {
            Ok(text) => parse_csv(&text),
            Err(_) => {
                info!(
                    "No static species range file at {} — skipping",
                    path.display()
                );
                HashMap::new()
            }
        };
        if !ranges.is_empty() {
            info!(
                "Loaded {} static species ranges from {}",
                ranges.len(),
                path.display()
            );
        }
        Self { ranges }
    }

    /// Number of species in the range map.
    pub fn len(&self) -> usize {
        self.ranges.len()
    }

    /// Whether the map is empty.
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// Check if a species is expected at the given location and month.
    ///
    /// Returns `None` if the species is not in the range file (= no
    /// opinion, should not be excluded).
    /// Returns `Some(true)` if the species is in range.
    /// Returns `Some(false)` if the species is out of range.
    pub fn check(&self, sci_name: &str, lat: f64, lon: f64, month: u32) -> Option<bool> {
        let norm = normalize_sci_name(sci_name);
        self.ranges.get(&norm).map(|r| r.contains(lat, lon, month))
    }

    /// Return the list of species expected at the given location and month.
    pub fn species_at(&self, lat: f64, lon: f64, month: u32) -> Vec<String> {
        self.ranges
            .iter()
            .filter(|(_, r)| r.contains(lat, lon, month))
            .map(|(name, _)| name.clone())
            .collect()
    }
}

/// Parse the months field: supports ranges like "4-10" and lists like "1,2,11,12".
fn parse_months(s: &str) -> Vec<u32> {
    let s = s.trim();
    if s.is_empty() {
        return vec![];
    }
    let mut result = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if let Some((a, b)) = part.split_once('-') {
            if let (Ok(start), Ok(end)) = (a.trim().parse::<u32>(), b.trim().parse::<u32>()) {
                for m in start..=end {
                    if (1..=12).contains(&m) {
                        result.push(m);
                    }
                }
            }
        } else if let Ok(m) = part.parse::<u32>() {
            if (1..=12).contains(&m) {
                result.push(m);
            }
        }
    }
    result
}

fn parse_csv(text: &str) -> HashMap<String, SpeciesRange> {
    let mut map = HashMap::new();
    let mut lines = text.lines();

    // Skip header.
    let first = match lines.next() {
        Some(l) => l,
        None => return map,
    };
    // Auto-detect delimiter.
    let delim = if first.contains(';') { ';' } else { ',' };

    for line in lines {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split(delim).collect();
        if cols.len() < 5 {
            continue;
        }
        let sci = normalize_sci_name(cols[0]);
        if sci.is_empty() {
            continue;
        }
        let lat_min = cols[1].trim().parse::<f64>().unwrap_or(f64::NEG_INFINITY);
        let lat_max = cols[2].trim().parse::<f64>().unwrap_or(f64::INFINITY);
        let lon_min = cols[3].trim().parse::<f64>().unwrap_or(f64::NEG_INFINITY);
        let lon_max = cols[4].trim().parse::<f64>().unwrap_or(f64::INFINITY);
        let months = if cols.len() > 5 {
            parse_months(cols[5])
        } else {
            vec![]
        };

        map.insert(
            sci,
            SpeciesRange {
                lat_min,
                lat_max,
                lon_min,
                lon_max,
                months,
            },
        );
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_csv() {
        let csv = "\
sci_name,lat_min,lat_max,lon_min,lon_max,months
Phaneroptera nana,30.0,55.0,-10.0,45.0,4-10
Conocephalus_fuscus,35.0,60.0,-10.0,50.0,5-9
Rana temporaria,35.0,72.0,-25.0,60.0,
";
        let map = parse_csv(csv);
        assert_eq!(map.len(), 3);

        let pn = map.get("Phaneroptera nana").unwrap();
        assert!((pn.lat_min - 30.0).abs() < 0.001);
        assert_eq!(pn.months, vec![4, 5, 6, 7, 8, 9, 10]);

        // Underscore in name normalised to space.
        assert!(map.contains_key("Conocephalus fuscus"));

        // Empty months = year-round.
        let rt = map.get("Rana temporaria").unwrap();
        assert!(rt.months.is_empty());
    }

    #[test]
    fn test_contains() {
        let r = SpeciesRange {
            lat_min: 30.0,
            lat_max: 55.0,
            lon_min: -10.0,
            lon_max: 45.0,
            months: vec![4, 5, 6, 7, 8, 9, 10],
        };
        // In range and in season.
        assert!(r.contains(45.0, 10.0, 6));
        // In range but out of season.
        assert!(!r.contains(45.0, 10.0, 1));
        // Out of range.
        assert!(!r.contains(60.0, 10.0, 6));
    }

    #[test]
    fn test_check_unknown_species() {
        let map = SpeciesRangeMap {
            ranges: HashMap::new(),
        };
        // Unknown species → None (no opinion).
        assert!(map.check("Unknown species", 0.0, 0.0, 1).is_none());
    }
}
