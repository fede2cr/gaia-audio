//! SQLite read-only queries for the web dashboard.
//!
//! Uses the same `detections` table written by the processing server.

use std::path::Path;

use rusqlite::{params, Connection};

use crate::model::{CalendarDay, DayDetectionGroup, SpeciesInfo, SpeciesSummary, WebDetection};

/// Open a read-only connection with a busy timeout.
///
/// WAL journal mode is set once by [`ensure_gaia_schema`] at startup (which
/// opens the database read-write).  The mode persists in the file, so
/// read-only connections inherit it automatically without needing a write.
fn open(db_path: &Path) -> Result<Connection, rusqlite::Error> {
    let conn = Connection::open_with_flags(db_path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    conn.execute_batch("PRAGMA busy_timeout=3000;")?;
    Ok(conn)
}

// ─── Recent detections (live feed) ───────────────────────────────────────────

/// Return the most recent `limit` detections.  
/// If `after_rowid` is provided only rows with `rowid > after_rowid` are returned
/// (used for incremental polling).
pub fn recent_detections(
    db_path: &Path,
    limit: u32,
    after_rowid: Option<i64>,
) -> Result<Vec<WebDetection>, rusqlite::Error> {
    let conn = open(db_path)?;
    let (sql, row_params): (String, Vec<Box<dyn rusqlite::types::ToSql>>) = match after_rowid {
        Some(rid) => (
            "SELECT rowid, Domain, Sci_Name, Com_Name, Confidence, Date, Time, File_Name \
             FROM detections WHERE rowid > ?1 ORDER BY rowid DESC LIMIT ?2"
                .into(),
            vec![Box::new(rid), Box::new(limit)],
        ),
        None => (
            "SELECT rowid, Domain, Sci_Name, Com_Name, Confidence, Date, Time, File_Name \
             FROM detections ORDER BY rowid DESC LIMIT ?1"
                .into(),
            vec![Box::new(limit)],
        ),
    };

    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(rusqlite::params_from_iter(row_params.iter()), |row| {
        Ok(WebDetection {
            id: row.get(0)?,
            domain: row.get(1)?,
            scientific_name: row.get(2)?,
            common_name: row.get(3)?,
            confidence: row.get(4)?,
            date: row.get(5)?,
            time: row.get(6)?,
            file_name: row.get(7)?,
        })
    })?;

    rows.collect()
}

// ─── Calendar data ───────────────────────────────────────────────────────────

/// For a given year-month, return per-day aggregates.
pub fn calendar_data(
    db_path: &Path,
    year: i32,
    month: u32,
) -> Result<Vec<CalendarDay>, rusqlite::Error> {
    let conn = open(db_path)?;
    let start = format!("{year:04}-{month:02}-01");
    let end = if month == 12 {
        format!("{:04}-01-01", year + 1)
    } else {
        format!("{year:04}-{:02}-01", month + 1)
    };

    let mut stmt = conn.prepare(
        "SELECT Date, COUNT(*) AS cnt, COUNT(DISTINCT Sci_Name) AS spp \
         FROM detections WHERE Date >= ?1 AND Date < ?2 GROUP BY Date ORDER BY Date",
    )?;

    let rows = stmt.query_map(params![start, end], |row| {
        Ok(CalendarDay {
            date: row.get(0)?,
            total_detections: row.get(1)?,
            unique_species: row.get(2)?,
        })
    })?;

    rows.collect()
}

// ─── Day detail ──────────────────────────────────────────────────────────────

/// Return all detections for a specific date, grouped by species.
pub fn day_detections(
    db_path: &Path,
    date: &str,
) -> Result<Vec<DayDetectionGroup>, rusqlite::Error> {
    let conn = open(db_path)?;
    let mut stmt = conn.prepare(
        "SELECT rowid, Domain, Sci_Name, Com_Name, Confidence, Date, Time, File_Name \
         FROM detections WHERE Date = ?1 ORDER BY Sci_Name, Time DESC",
    )?;

    let rows: Vec<WebDetection> = stmt
        .query_map(params![date], |row| {
            Ok(WebDetection {
                id: row.get(0)?,
                domain: row.get(1)?,
                scientific_name: row.get(2)?,
                common_name: row.get(3)?,
                confidence: row.get(4)?,
                date: row.get(5)?,
                time: row.get(6)?,
                file_name: row.get(7)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;

    // Group by (scientific_name, domain)
    let mut groups: Vec<DayDetectionGroup> = Vec::new();
    for det in rows {
        if let Some(group) = groups
            .iter_mut()
            .find(|g| g.scientific_name == det.scientific_name && g.domain == det.domain)
        {
            if det.confidence > group.max_confidence {
                group.max_confidence = det.confidence;
            }
            group.detections.push(det);
        } else {
            groups.push(DayDetectionGroup {
                scientific_name: det.scientific_name.clone(),
                common_name: det.common_name.clone(),
                domain: det.domain.clone(),
                image_url: None, // filled in later by iNaturalist lookup
                max_confidence: det.confidence,
                detections: vec![det],
            });
        }
    }
    Ok(groups)
}

// ─── Species detail ──────────────────────────────────────────────────────────

/// Aggregate species statistics.
pub fn species_info(
    db_path: &Path,
    scientific_name: &str,
) -> Result<Option<SpeciesInfo>, rusqlite::Error> {
    let conn = open(db_path)?;
    let mut stmt = conn.prepare(
        "SELECT Domain, Com_Name, COUNT(*) AS cnt, \
         MIN(Date) AS first_seen, MAX(Date) AS last_seen \
         FROM detections WHERE Sci_Name = ?1 GROUP BY Domain, Com_Name LIMIT 1",
    )?;

    let mut rows = stmt.query_map(params![scientific_name], |row| {
        Ok(SpeciesInfo {
            scientific_name: scientific_name.to_string(),
            domain: row.get(0)?,
            common_name: row.get(1)?,
            total_detections: row.get(2)?,
            first_seen: row.get(3)?,
            last_seen: row.get(4)?,
            image_url: None,
            wikipedia_url: None,
        })
    })?;

    match rows.next() {
        Some(Ok(info)) => Ok(Some(info)),
        Some(Err(e)) => Err(e),
        None => Ok(None),
    }
}

/// Dates on which a species was detected (for calendar highlighting).
pub fn species_active_dates(
    db_path: &Path,
    scientific_name: &str,
    year: i32,
) -> Result<Vec<String>, rusqlite::Error> {
    let conn = open(db_path)?;
    let start = format!("{year:04}-01-01");
    let end = format!("{:04}-01-01", year + 1);

    let mut stmt = conn.prepare(
        "SELECT DISTINCT Date FROM detections \
         WHERE Sci_Name = ?1 AND Date >= ?2 AND Date < ?3 ORDER BY Date",
    )?;

    let rows = stmt.query_map(params![scientific_name, start, end], |row| row.get(0))?;
    rows.collect()
}

/// Top species (for species list on home page).
pub fn top_species(
    db_path: &Path,
    limit: u32,
) -> Result<Vec<SpeciesSummary>, rusqlite::Error> {
    let conn = open(db_path)?;
    let mut stmt = conn.prepare(
        "SELECT Sci_Name, Com_Name, Domain, COUNT(*) AS cnt, MAX(Date || ' ' || Time) AS last \
         FROM detections GROUP BY Sci_Name, Domain ORDER BY cnt DESC LIMIT ?1",
    )?;

    let rows = stmt.query_map(params![limit], |row| {
        Ok(SpeciesSummary {
            scientific_name: row.get(0)?,
            common_name: row.get(1)?,
            domain: row.get(2)?,
            detection_count: row.get(3)?,
            last_seen: row.get(4)?,
            image_url: None,
        })
    })?;

    rows.collect()
}
