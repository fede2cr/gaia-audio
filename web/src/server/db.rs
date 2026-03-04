//! SQLite read-only queries for the web dashboard.
//!
//! Uses the same `detections` table written by the processing server.

use std::path::Path;

use rusqlite::{params, Connection};

use crate::model::{CalendarDay, DayDetectionGroup, ExcludedSpecies, SpeciesInfo, SpeciesSummary, UrbanNoiseSummary, WebDetection};

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
            "SELECT rowid, Domain, Sci_Name, Com_Name, Confidence, Date, Time, File_Name, \
             COALESCE(Source_Node, ''), COALESCE(Excluded, 0), \
             COALESCE(Model_Slug, ''), COALESCE(Model_Name, '') \
             FROM detections WHERE rowid > ?1 ORDER BY rowid DESC LIMIT ?2"
                .into(),
            vec![Box::new(rid), Box::new(limit)],
        ),
        None => (
            "SELECT rowid, Domain, Sci_Name, Com_Name, Confidence, Date, Time, File_Name, \
             COALESCE(Source_Node, ''), COALESCE(Excluded, 0), \
             COALESCE(Model_Slug, ''), COALESCE(Model_Name, '') \
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
            source_node: row.get(8)?,
            excluded: row.get::<_, i32>(9)? != 0,
            image_url: None,
            model_slug: row.get(10)?,
            model_name: row.get(11)?,
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
         FROM detections \
         WHERE Date >= ?1 AND Date < ?2 \
           AND (COALESCE(Excluded, 0) = 0 \
                OR Sci_Name IN (SELECT Sci_Name FROM exclusion_overrides)) \
         GROUP BY Date ORDER BY Date",
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
        "SELECT rowid, Domain, Sci_Name, Com_Name, Confidence, Date, Time, File_Name, \
         COALESCE(Source_Node, ''), COALESCE(Excluded, 0), \
         COALESCE(Model_Slug, ''), COALESCE(Model_Name, '') \
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
                source_node: row.get(8)?,
                excluded: row.get::<_, i32>(9)? != 0,
                image_url: None,
                model_slug: row.get(10)?,
                model_name: row.get(11)?,
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
///
/// Includes excluded detections only if the species has been overridden.
pub fn species_info(
    db_path: &Path,
    scientific_name: &str,
) -> Result<Option<SpeciesInfo>, rusqlite::Error> {
    let conn = open(db_path)?;
    let mut stmt = conn.prepare(
        "SELECT Domain, Com_Name, COUNT(*) AS cnt, \
         MIN(Date) AS first_seen, MAX(Date) AS last_seen \
         FROM detections \
         WHERE Sci_Name = ?1 \
           AND (COALESCE(Excluded, 0) = 0 \
                OR Sci_Name IN (SELECT Sci_Name FROM exclusion_overrides)) \
         GROUP BY Domain, Com_Name LIMIT 1",
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
///
/// Excluded detections are omitted unless the species has been overridden
/// in the `exclusion_overrides` table.
pub fn top_species(
    db_path: &Path,
    limit: u32,
) -> Result<Vec<SpeciesSummary>, rusqlite::Error> {
    let conn = open(db_path)?;
    let mut stmt = conn.prepare(
        "SELECT d.Sci_Name, d.Com_Name, d.Domain, COUNT(*) AS cnt, \
         MAX(d.Date || ' ' || d.Time) AS last \
         FROM detections d \
         WHERE (COALESCE(d.Excluded, 0) = 0 \
                OR d.Sci_Name IN (SELECT Sci_Name FROM exclusion_overrides)) \
         GROUP BY d.Sci_Name, d.Domain ORDER BY cnt DESC LIMIT ?1",
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

// ─── Settings ────────────────────────────────────────────────────────────────

use std::collections::HashMap;

/// Read all rows from the `settings` table as a key-value map.
pub fn get_all_settings(db_path: &Path) -> Result<HashMap<String, String>, rusqlite::Error> {
    let conn = open(db_path)?;
    // The table may not exist in older databases.
    let _ = conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
    );
    let mut stmt = conn.prepare("SELECT key, value FROM settings")?;
    let rows = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    })?;
    let mut map = HashMap::new();
    for r in rows {
        let (k, v) = r?;
        map.insert(k, v);
    }
    Ok(map)
}

/// Open a read-write connection with a busy timeout.
fn open_rw(db_path: &Path) -> Result<Connection, rusqlite::Error> {
    let conn = Connection::open(db_path)?;
    conn.execute_batch("PRAGMA busy_timeout=3000;")?;
    Ok(conn)
}

/// Save multiple settings in one transaction.
pub fn save_settings(db_path: &Path, entries: &[(&str, &str)]) -> Result<(), rusqlite::Error> {
    let conn = open_rw(db_path)?;
    let _ = conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
    );
    let tx = conn.unchecked_transaction()?;
    {
        let mut stmt = tx.prepare(
            "INSERT INTO settings (key, value) VALUES (?1, ?2) \
             ON CONFLICT(key) DO UPDATE SET value = excluded.value"
        )?;
        for (k, v) in entries {
            stmt.execute(params![k, v])?;
        }
    }
    tx.commit()?;
    Ok(())
}

// ─── Urban noise ─────────────────────────────────────────────────────────────

/// Aggregated urban-noise counts per category (today, last 7 days, all time).
pub fn urban_noise_summary(
    db_path: &Path,
) -> Result<Vec<UrbanNoiseSummary>, rusqlite::Error> {
    let conn = open(db_path)?;
    let today = chrono::Local::now().format("%Y-%m-%d").to_string();

    let mut stmt = conn.prepare(
        "SELECT Category,
                COALESCE(SUM(Count), 0) AS total,
                COALESCE(SUM(CASE WHEN Date = ?1 THEN Count ELSE 0 END), 0) AS today,
                COALESCE(SUM(CASE WHEN Date >= DATE(?1, '-7 day') THEN Count ELSE 0 END), 0) AS week
         FROM urban_noise
         GROUP BY Category
         ORDER BY total DESC",
    )?;

    let rows = stmt.query_map(params![today], |row| {
        Ok(UrbanNoiseSummary {
            category: row.get(0)?,
            total_count: row.get(1)?,
            today_count: row.get(2)?,
            week_count: row.get(3)?,
        })
    })?;

    rows.collect()
}

// ─── Excluded species ────────────────────────────────────────────────────────

/// List species that have at least one excluded detection.
pub fn excluded_species(db_path: &Path) -> Result<Vec<ExcludedSpecies>, rusqlite::Error> {
    let conn = open(db_path)?;

    let mut stmt = conn.prepare(
        "SELECT d.Sci_Name, d.Com_Name, d.Domain, \
                COUNT(*) AS cnt, \
                MAX(d.Date || ' ' || d.Time) AS last, \
                MAX(d.Confidence) AS max_conf, \
                CASE WHEN eo.Sci_Name IS NOT NULL THEN 1 ELSE 0 END AS overridden \
         FROM detections d \
         LEFT JOIN exclusion_overrides eo ON d.Sci_Name = eo.Sci_Name \
         WHERE COALESCE(d.Excluded, 0) = 1 \
         GROUP BY d.Sci_Name, d.Domain \
         ORDER BY overridden ASC, cnt DESC",
    )?;

    let rows = stmt.query_map([], |row| {
        Ok(ExcludedSpecies {
            scientific_name: row.get(0)?,
            common_name: row.get(1)?,
            domain: row.get(2)?,
            detection_count: row.get(3)?,
            last_seen: row.get(4)?,
            max_confidence: row.get(5)?,
            image_url: None,
            overridden: row.get::<_, i32>(6)? != 0,
        })
    })?;

    rows.collect()
}

/// Return individual excluded detections for a given species, newest first.
///
/// Used by the Excluded page so the user can listen to / inspect the
/// recordings before confirming an override.
pub fn excluded_detections_for_species(
    db_path: &Path,
    scientific_name: &str,
    limit: u32,
) -> Result<Vec<WebDetection>, rusqlite::Error> {
    let conn = open(db_path)?;
    let mut stmt = conn.prepare(
        "SELECT rowid, Domain, Sci_Name, Com_Name, Confidence, Date, Time, File_Name, \
         COALESCE(Source_Node, ''), COALESCE(Excluded, 0), \
         COALESCE(Model_Slug, ''), COALESCE(Model_Name, '') \
         FROM detections \
         WHERE Sci_Name = ?1 AND COALESCE(Excluded, 0) = 1 \
         ORDER BY Date DESC, Time DESC LIMIT ?2",
    )?;

    let rows = stmt.query_map(params![scientific_name, limit], |row| {
        Ok(WebDetection {
            id: row.get(0)?,
            domain: row.get(1)?,
            scientific_name: row.get(2)?,
            common_name: row.get(3)?,
            confidence: row.get(4)?,
            date: row.get(5)?,
            time: row.get(6)?,
            file_name: row.get(7)?,
            source_node: row.get(8)?,
            excluded: row.get::<_, i32>(9)? != 0,
            image_url: None,
            model_slug: row.get(10)?,
            model_name: row.get(11)?,
        })
    })?;

    rows.collect()
}

/// Add an exclusion override (ornithologist confirms the species is real).
pub fn add_exclusion_override(
    db_path: &Path,
    scientific_name: &str,
    notes: &str,
) -> Result<(), rusqlite::Error> {
    let conn = open_rw(db_path)?;
    conn.execute(
        "INSERT INTO exclusion_overrides (Sci_Name, overridden_at, notes) \
         VALUES (?1, datetime('now'), ?2) \
         ON CONFLICT(Sci_Name) DO UPDATE SET overridden_at = datetime('now'), notes = excluded.notes",
        params![scientific_name, notes],
    )?;
    Ok(())
}

/// Remove an exclusion override (undo confirmation).
pub fn remove_exclusion_override(
    db_path: &Path,
    scientific_name: &str,
) -> Result<(), rusqlite::Error> {
    let conn = open_rw(db_path)?;
    conn.execute(
        "DELETE FROM exclusion_overrides WHERE Sci_Name = ?1",
        params![scientific_name],
    )?;
    Ok(())
}
