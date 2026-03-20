//! SQLite queries for the web dashboard (libsql / Turso edition).
//!
//! Uses the same `detections` table written by the processing server.
//!
//! All timestamps in the database are stored in UTC.  The `tz_offset`
//! setting (hours from UTC, e.g. -6) is applied at read time to populate
//! `display_date` / `display_time` on [`WebDetection`] for the UI.

use std::path::Path;

use chrono::{NaiveDate, NaiveDateTime, NaiveTime, Duration, Utc};
use libsql::params;

use crate::model::{CalendarDay, DayDetectionGroup, ExcludedSpecies, QuizItem, SpeciesInfo, SpeciesSummary, TopRecording, UrbanNoiseSummary, WebDetection,
                    HourlyCount, SpeciesHourlyCounts};

// ── Turso / libsql connection helpers ───────────────────────────────────────

/// Busy-timeout in milliseconds.
const BUSY_TIMEOUT_MS: u32 = 30_000;

/// Resolve the database path: `TURSO_DATABASE_URL` overrides `db_path`.
fn effective_db_url(db_path: &Path) -> Result<String, libsql::Error> {
    if let Ok(url) = std::env::var("TURSO_DATABASE_URL") {
        if !url.is_empty() {
            return Ok(url);
        }
    }
    db_path
        .to_str()
        .map(|s| s.to_string())
        .ok_or_else(|| libsql::Error::SqliteFailure(0, "Non-UTF-8 path".into()))
}

/// Open a `libsql::Database` from the effective URL.
async fn build_db(db_path: &Path) -> Result<libsql::Database, libsql::Error> {
    let url = effective_db_url(db_path)?;
    libsql::Builder::new_local(&url)
        .build()
        .await
}

// ─── Timezone helpers ────────────────────────────────────────────────────────

/// Read the `tz_offset` value (hours) from the settings table.
/// Returns 0 (UTC) if unset or on any error.
async fn read_tz_offset(conn: &libsql::Connection) -> i32 {
    let mut rows = match conn.query("SELECT value FROM settings WHERE key = 'tz_offset'", ()).await {
        Ok(r) => r,
        Err(_) => return 0,
    };
    match rows.next().await {
        Ok(Some(row)) => row.get::<String>(0).ok().and_then(|v| v.parse::<i32>().ok()).unwrap_or(0),
        _ => 0,
    }
}

/// Apply a timezone offset (hours) to UTC `date` (YYYY-MM-DD) and `time`
/// (HH:MM:SS) strings, returning `(display_date, display_time)`.
///
/// If parsing fails the original strings are returned unchanged.
fn apply_tz(date: &str, time: &str, offset_hours: i32) -> (String, String) {
    if offset_hours == 0 {
        return (date.to_string(), time.to_string());
    }
    let Ok(d) = NaiveDate::parse_from_str(date, "%Y-%m-%d") else {
        return (date.to_string(), time.to_string());
    };
    let Ok(t) = NaiveTime::parse_from_str(time, "%H:%M:%S") else {
        return (date.to_string(), time.to_string());
    };
    let dt = NaiveDateTime::new(d, t) + Duration::hours(offset_hours as i64);
    (
        dt.format("%Y-%m-%d").to_string(),
        dt.format("%H:%M:%S").to_string(),
    )
}

/// Fill `display_date` / `display_time` on a `WebDetection` in-place.
fn stamp(det: &mut WebDetection, offset: i32) {
    let (dd, dt) = apply_tz(&det.date, &det.time, offset);
    det.display_date = dd;
    det.display_time = dt;
}

/// Fill `display_date` / `display_time` on a `TopRecording` in-place.
fn stamp_recording(rec: &mut TopRecording, offset: i32) {
    let (dd, dt) = apply_tz(&rec.date, &rec.time, offset);
    rec.display_date = dd;
    rec.display_time = dt;
}

/// Return today's date string (YYYY-MM-DD) according to the user's TZ offset.
///
/// The database stores everything in UTC.  When `tz_offset = -6`, midnight
/// in the user's timezone corresponds to `06:00 UTC`, so "today" starts
/// 6 hours later than UTC midnight.  We apply the offset to `Utc::now()`
/// to derive the correct calendar date for the user.
fn today_for_tz(offset: i32) -> String {
    let utc_now = Utc::now().naive_utc();
    let local_now = utc_now + Duration::hours(offset as i64);
    local_now.format("%Y-%m-%d").to_string()
}

/// Public version — reads the TZ offset from the database and returns today's
/// date string.  Used by server functions that don't already hold a connection.
pub async fn today_for_tz_pub(db_path: &Path) -> Result<String, libsql::Error> {
    let conn = open(db_path).await?;
    let tz = read_tz_offset(&conn).await;
    Ok(today_for_tz(tz))
}

/// Open a read-only connection with a busy timeout.
///
/// WAL mode is also set once at schema creation, but re-asserting it here
/// is cheap (no-op when already active) and ensures every connection is
/// consistent even if the database was reset externally.
///
/// Reads `TURSO_DATABASE_URL` to resolve the database location,
/// falling back to `db_path`.
async fn open(db_path: &Path) -> Result<libsql::Connection, libsql::Error> {
    let db = build_db(db_path).await?;
    let conn = db.connect()?;
    conn.execute_batch(&format!("PRAGMA busy_timeout={BUSY_TIMEOUT_MS};")).await?;
    Ok(conn)
}

/// Open a connection usable from outside this module (e.g. `species.rs`).
///
/// Reads `TURSO_DATABASE_URL` to resolve the database, falling back to
/// `db_path`.  Sets `busy_timeout` immediately.
pub async fn open_conn(db_path: &Path) -> Result<libsql::Connection, libsql::Error> {
    open(db_path).await
}

// ─── Recent detections (live feed) ───────────────────────────────────────────

/// Return the most recent `limit` detections.  
/// If `after_rowid` is provided only rows with `rowid > after_rowid` are returned
/// (used for incremental polling).
pub async fn recent_detections(
    db_path: &Path,
    limit: u32,
    after_rowid: Option<i64>,
) -> Result<Vec<WebDetection>, libsql::Error> {
    let conn = open(db_path).await?;
    let (sql, row_params): (String, Vec<libsql::Value>) = match after_rowid {
        Some(rid) => (
            "SELECT rowid, Domain, Sci_Name, Com_Name, Confidence, Date, Time, File_Name, \
             COALESCE(Source_Node, ''), COALESCE(Excluded, 0), \
             COALESCE(Model_Slug, ''), COALESCE(Model_Name, '') \
             FROM detections WHERE rowid > ?1 ORDER BY rowid DESC LIMIT ?2"
                .into(),
            vec![libsql::Value::from(rid), libsql::Value::from(limit as i64)],
        ),
        None => (
            "SELECT rowid, Domain, Sci_Name, Com_Name, Confidence, Date, Time, File_Name, \
             COALESCE(Source_Node, ''), COALESCE(Excluded, 0), \
             COALESCE(Model_Slug, ''), COALESCE(Model_Name, '') \
             FROM detections ORDER BY rowid DESC LIMIT ?1"
                .into(),
            vec![libsql::Value::from(limit as i64)],
        ),
    };

    let tz = read_tz_offset(&conn).await;
    let mut rows = conn.query(&sql, row_params).await?;
    let mut dets = Vec::new();
    while let Some(row) = rows.next().await? {
        dets.push(WebDetection {
            id: row.get::<i64>(0)?,
            domain: row.get::<String>(1)?,
            scientific_name: row.get::<String>(2)?,
            common_name: row.get::<String>(3)?,
            confidence: row.get::<f64>(4)?,
            date: row.get::<String>(5)?,
            time: row.get::<String>(6)?,
            file_name: row.get::<String>(7)?,
            source_node: row.get::<String>(8)?,
            excluded: row.get::<i32>(9)? != 0,
            image_url: None,
            model_slug: row.get::<String>(10)?,
            model_name: row.get::<String>(11)?,
            display_date: String::new(),
            display_time: String::new(),
        });
    }

    for d in &mut dets { stamp(d, tz); }
    Ok(dets)
}

// ─── Calendar data ───────────────────────────────────────────────────────────

/// For a given year-month, return per-day aggregates.
pub async fn calendar_data(
    db_path: &Path,
    year: i32,
    month: u32,
) -> Result<Vec<CalendarDay>, libsql::Error> {
    let conn = open(db_path).await?;
    let start = format!("{year:04}-{month:02}-01");
    let end = if month == 12 {
        format!("{:04}-01-01", year + 1)
    } else {
        format!("{year:04}-{:02}-01", month + 1)
    };

    let mut rows = conn.query(
        "SELECT Date, COUNT(*) AS cnt, COUNT(DISTINCT Sci_Name) AS spp \
         FROM detections \
         WHERE Date >= ?1 AND Date < ?2 \
           AND (COALESCE(Excluded, 0) = 0 \
                OR Sci_Name IN (SELECT Sci_Name FROM exclusion_overrides)) \
         GROUP BY Date ORDER BY Date",
        params![start, end],
    ).await?;

    let mut results = Vec::new();
    while let Some(row) = rows.next().await? {
        results.push(CalendarDay {
            date: row.get::<String>(0)?,
            total_detections: row.get::<u32>(1)?,
            unique_species: row.get::<u32>(2)?,
        });
    }
    Ok(results)
}

// ─── Day detail ──────────────────────────────────────────────────────────────

/// Return all detections for a specific date, grouped by species.
pub async fn day_detections(
    db_path: &Path,
    date: &str,
) -> Result<Vec<DayDetectionGroup>, libsql::Error> {
    let conn = open(db_path).await?;
    let tz = read_tz_offset(&conn).await;
    let mut rows = conn.query(
        "SELECT rowid, Domain, Sci_Name, Com_Name, Confidence, Date, Time, File_Name, \
         COALESCE(Source_Node, ''), COALESCE(Excluded, 0), \
         COALESCE(Model_Slug, ''), COALESCE(Model_Name, '') \
         FROM detections WHERE Date = ?1 ORDER BY Sci_Name, Time DESC",
        params![date.to_string()],
    ).await?;

    let mut all_dets: Vec<WebDetection> = Vec::new();
    while let Some(row) = rows.next().await? {
        all_dets.push(WebDetection {
            id: row.get::<i64>(0)?,
            domain: row.get::<String>(1)?,
            scientific_name: row.get::<String>(2)?,
            common_name: row.get::<String>(3)?,
            confidence: row.get::<f64>(4)?,
            date: row.get::<String>(5)?,
            time: row.get::<String>(6)?,
            file_name: row.get::<String>(7)?,
            source_node: row.get::<String>(8)?,
            excluded: row.get::<i32>(9)? != 0,
            image_url: None,
            model_slug: row.get::<String>(10)?,
            model_name: row.get::<String>(11)?,
            display_date: String::new(),
            display_time: String::new(),
        });
    }
    let all_dets: Vec<WebDetection> = all_dets.into_iter().map(|mut d| { stamp(&mut d, tz); d }).collect();

    // Group by (scientific_name, domain)
    let mut groups: Vec<DayDetectionGroup> = Vec::new();
    for det in all_dets {
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
pub async fn species_info(
    db_path: &Path,
    scientific_name: &str,
) -> Result<Option<SpeciesInfo>, libsql::Error> {
    let conn = open(db_path).await?;
    let mut rows = conn.query(
        "SELECT Domain, Com_Name, COUNT(*) AS cnt, \
         MIN(Date) AS first_seen, MAX(Date) AS last_seen \
         FROM detections \
         WHERE Sci_Name = ?1 \
           AND (COALESCE(Excluded, 0) = 0 \
                OR Sci_Name IN (SELECT Sci_Name FROM exclusion_overrides)) \
         GROUP BY Domain, Com_Name LIMIT 1",
        params![scientific_name.to_string()],
    ).await?;

    match rows.next().await? {
        Some(row) => Ok(Some(SpeciesInfo {
            scientific_name: scientific_name.to_string(),
            domain: row.get::<String>(0)?,
            common_name: row.get::<String>(1)?,
            total_detections: row.get::<i64>(2)? as u64,
            first_seen: row.get::<String>(3).ok(),
            last_seen: row.get::<String>(4).ok(),
            image_url: None,
            wikipedia_url: None,
            male_image_url: None,
            female_image_url: None,
            verification: None,
        })),
        None => Ok(None),
    }
}

/// Dates on which a species was detected (for calendar highlighting).
pub async fn species_active_dates(
    db_path: &Path,
    scientific_name: &str,
    year: i32,
) -> Result<Vec<String>, libsql::Error> {
    let conn = open(db_path).await?;
    let start = format!("{year:04}-01-01");
    let end = format!("{:04}-01-01", year + 1);

    let mut rows = conn.query(
        "SELECT DISTINCT Date FROM detections \
         WHERE Sci_Name = ?1 AND Date >= ?2 AND Date < ?3 ORDER BY Date",
        params![scientific_name.to_string(), start, end],
    ).await?;

    let mut dates = Vec::new();
    while let Some(row) = rows.next().await? {
        dates.push(row.get::<String>(0)?);
    }
    Ok(dates)
}

/// Hourly detection histogram for a single species (all-time).
///
/// Hours are shifted by the user's `tz_offset` so the histogram
/// reflects local time, not UTC.
pub async fn species_hourly_histogram(
    db_path: &Path,
    scientific_name: &str,
) -> Result<Vec<HourlyCount>, libsql::Error> {
    let conn = open(db_path).await?;
    let tz = read_tz_offset(&conn).await;
    let mut rows = conn.query(
        "SELECT CAST(SUBSTR(Time, 1, 2) AS INTEGER) AS hour, COUNT(*) AS cnt \
         FROM detections \
         WHERE Sci_Name = ?1 \
           AND (COALESCE(Excluded, 0) = 0 \
                OR Sci_Name IN (SELECT Sci_Name FROM exclusion_overrides)) \
         GROUP BY hour ORDER BY hour",
        params![scientific_name.to_string()],
    ).await?;

    let mut raw: Vec<HourlyCount> = Vec::new();
    while let Some(row) = rows.next().await? {
        raw.push(HourlyCount {
            hour: row.get::<u32>(0)?,
            count: row.get::<u32>(1)?,
        });
    }

    if tz == 0 {
        return Ok(raw);
    }

    // Re-bucket hours into local time.
    let mut buckets = [0u32; 24];
    for hc in &raw {
        let local_hour = ((hc.hour as i32 + tz).rem_euclid(24)) as u32;
        buckets[local_hour as usize] += hc.count;
    }
    Ok(buckets.iter().enumerate()
        .filter(|(_, &c)| c > 0)
        .map(|(h, &c)| HourlyCount { hour: h as u32, count: c })
        .collect())
}

/// Per-species hourly breakdown for a specific date (for the day view chart).
///
/// Hours are shifted by the user's `tz_offset`.
pub async fn daily_species_hourly(
    db_path: &Path,
    date: &str,
) -> Result<Vec<SpeciesHourlyCounts>, libsql::Error> {
    let conn = open(db_path).await?;
    let tz = read_tz_offset(&conn).await;

    // First get distinct species for the day, ordered by total count.
    let mut sp_rows = conn.query(
        "SELECT Sci_Name, Com_Name, COUNT(*) AS cnt \
         FROM detections \
         WHERE Date = ?1 \
           AND (COALESCE(Excluded, 0) = 0 \
                OR Sci_Name IN (SELECT Sci_Name FROM exclusion_overrides)) \
         GROUP BY Sci_Name ORDER BY cnt DESC",
        params![date.to_string()],
    ).await?;

    let mut species: Vec<(String, String, u32)> = Vec::new();
    while let Some(row) = sp_rows.next().await? {
        species.push((
            row.get::<String>(0)?,
            row.get::<String>(1)?,
            row.get::<u32>(2)?,
        ));
    }

    // Then get hourly breakdown for each species.
    let mut result = Vec::new();
    for (sci, com, total) in species {
        let mut hour_rows = conn.query(
            "SELECT CAST(SUBSTR(Time, 1, 2) AS INTEGER) AS hour, COUNT(*) AS cnt \
             FROM detections \
             WHERE Date = ?1 AND Sci_Name = ?2 \
             GROUP BY hour ORDER BY hour",
            params![date.to_string(), sci.clone()],
        ).await?;

        let mut raw_hours: Vec<HourlyCount> = Vec::new();
        while let Some(row) = hour_rows.next().await? {
            raw_hours.push(HourlyCount {
                hour: row.get::<u32>(0)?,
                count: row.get::<u32>(1)?,
            });
        }

        let hours = if tz == 0 {
            raw_hours
        } else {
            let mut buckets = [0u32; 24];
            for hc in &raw_hours {
                let local_hour = ((hc.hour as i32 + tz).rem_euclid(24)) as u32;
                buckets[local_hour as usize] += hc.count;
            }
            buckets.iter().enumerate()
                .filter(|(_, &c)| c > 0)
                .map(|(h, &c)| HourlyCount { hour: h as u32, count: c })
                .collect()
        };

        result.push(SpeciesHourlyCounts {
            scientific_name: sci,
            common_name: com,
            total,
            hours,
        });
    }

    Ok(result)
}

/// Top species for a specific date (for daily top species on home page).
pub async fn top_species_for_date(
    db_path: &Path,
    date: &str,
    limit: u32,
) -> Result<Vec<SpeciesSummary>, libsql::Error> {
    let conn = open(db_path).await?;
    let mut rows = conn.query(
        "SELECT d.Sci_Name, d.Com_Name, d.Domain, COUNT(*) AS cnt, \
         MAX(d.Date || ' ' || d.Time) AS last \
         FROM detections d \
         WHERE d.Date = ?1 \
           AND (COALESCE(d.Excluded, 0) = 0 \
                OR d.Sci_Name IN (SELECT Sci_Name FROM exclusion_overrides)) \
         GROUP BY d.Sci_Name, d.Domain ORDER BY cnt DESC LIMIT ?2",
        params![date.to_string(), limit],
    ).await?;

    let mut results = Vec::new();
    while let Some(row) = rows.next().await? {
        let count: u32 = row.get::<u32>(3)?;
        results.push(SpeciesSummary {
            scientific_name: row.get::<String>(0)?,
            common_name: row.get::<String>(1)?,
            domain: row.get::<String>(2)?,
            detection_count: count,
            display_count: round_count(count),
            last_seen: row.get::<String>(4).ok(),
            image_url: None,
            conservation_status: None,
            male_image_url: None,
            female_image_url: None,
        });
    }
    Ok(results)
}

/// Top species (for species list on home page).
///
/// Reads from the `species_stats` cache table for fast loading.
/// Falls back to a live COUNT(*) query if the cache is empty.
pub async fn top_species(
    db_path: &Path,
    limit: u32,
) -> Result<Vec<SpeciesSummary>, libsql::Error> {
    let conn = open(db_path).await?;

    // Check if the cache table exists and has data.
    let cache_count: i64 = match conn.query(
        "SELECT COUNT(*) FROM sqlite_master \
         WHERE type='table' AND name='species_stats'",
        (),
    ).await {
        Ok(mut r) => match r.next().await {
            Ok(Some(row)) => row.get::<i64>(0).unwrap_or(0),
            _ => 0,
        },
        Err(_) => 0,
    };

    if cache_count > 0 {
        let has_rows: i64 = match conn.query(
            "SELECT COUNT(*) FROM species_stats", (),
        ).await {
            Ok(mut r) => match r.next().await {
                Ok(Some(row)) => row.get::<i64>(0).unwrap_or(0),
                _ => 0,
            },
            Err(_) => 0,
        };

        if has_rows > 0 {
            return top_species_from_cache(&conn, limit).await;
        }
    }

    // Fallback: live query (slow on large databases).
    top_species_live(&conn, limit).await
}

/// Read species from the cached `species_stats` table.
async fn top_species_from_cache(
    conn: &libsql::Connection,
    limit: u32,
) -> Result<Vec<SpeciesSummary>, libsql::Error> {
    let mut rows = conn.query(
        "SELECT Sci_Name, Com_Name, Domain, detection_count, last_seen \
         FROM species_stats \
         ORDER BY detection_count DESC LIMIT ?1",
        params![limit],
    ).await?;

    let mut results = Vec::new();
    while let Some(row) = rows.next().await? {
        let count: u32 = row.get::<u32>(3)?;
        results.push(SpeciesSummary {
            scientific_name: row.get::<String>(0)?,
            common_name: row.get::<String>(1)?,
            domain: row.get::<String>(2)?,
            detection_count: count,
            display_count: round_count(count),
            last_seen: row.get::<String>(4).ok(),
            image_url: None,
            conservation_status: None,
            male_image_url: None,
            female_image_url: None,
        });
    }
    Ok(results)
}

/// Live COUNT(*) query — used as fallback when the cache is empty.
async fn top_species_live(
    conn: &libsql::Connection,
    limit: u32,
) -> Result<Vec<SpeciesSummary>, libsql::Error> {
    let mut rows = conn.query(
        "SELECT d.Sci_Name, d.Com_Name, d.Domain, COUNT(*) AS cnt, \
         MAX(d.Date || ' ' || d.Time) AS last \
         FROM detections d \
         WHERE (COALESCE(d.Excluded, 0) = 0 \
                OR d.Sci_Name IN (SELECT Sci_Name FROM exclusion_overrides)) \
         GROUP BY d.Sci_Name, d.Domain ORDER BY cnt DESC LIMIT ?1",
        params![limit],
    ).await?;

    let mut results = Vec::new();
    while let Some(row) = rows.next().await? {
        let count: u32 = row.get::<u32>(3)?;
        results.push(SpeciesSummary {
            scientific_name: row.get::<String>(0)?,
            common_name: row.get::<String>(1)?,
            domain: row.get::<String>(2)?,
            detection_count: count,
            display_count: round_count(count),
            last_seen: row.get::<String>(4).ok(),
            image_url: None,
            conservation_status: None,
            male_image_url: None,
            female_image_url: None,
        });
    }
    Ok(results)
}

/// Round a detection count for display:
///   - 0–100: exact
///   - 101–1000: round to nearest 10
///   - 1001+: round to nearest 100
///
/// Returns a display string like `"42"`, `"~230"`, `"~4,200"`.
pub fn round_count(n: u32) -> String {
    if n <= 100 {
        format_with_commas(n)
    } else if n <= 1000 {
        let rounded = ((n + 5) / 10) * 10;
        format!("~{}", format_with_commas(rounded))
    } else {
        let rounded = ((n + 50) / 100) * 100;
        format!("~{}", format_with_commas(rounded))
    }
}

/// Format an integer with comma separators (e.g. 1234 → "1,234").
fn format_with_commas(n: u32) -> String {
    let s = n.to_string();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

/// Refresh the `species_stats` cache table.
///
/// This does a full recount from the `detections` table.  Designed to be
/// called once at startup and then nightly.
pub async fn refresh_species_stats(db_path: &Path) -> Result<(), libsql::Error> {
    let conn = open_rw(db_path).await?;

    // Ensure the table exists (in case this runs before ensure_gaia_schema).
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS species_stats (
            Sci_Name       VARCHAR(100) NOT NULL,
            Com_Name       VARCHAR(100) NOT NULL,
            Domain         VARCHAR(50)  NOT NULL DEFAULT 'birds',
            detection_count INTEGER     NOT NULL DEFAULT 0,
            last_seen      TEXT,
            updated_at     TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (Sci_Name, Domain)
        );",
    ).await?;

    conn.execute_batch(
        "DELETE FROM species_stats;
         INSERT INTO species_stats (Sci_Name, Com_Name, Domain, detection_count, last_seen)
         SELECT d.Sci_Name, d.Com_Name, d.Domain, COUNT(*), MAX(d.Date || ' ' || d.Time)
         FROM detections d
         WHERE COALESCE(d.Excluded, 0) = 0
            OR d.Sci_Name IN (SELECT Sci_Name FROM exclusion_overrides)
         GROUP BY d.Sci_Name, d.Domain;",
    ).await?;

    // ── Refresh top recordings per species ───────────────────────────
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS species_top_recordings (
            Sci_Name       VARCHAR(100) NOT NULL,
            Com_Name       VARCHAR(100) NOT NULL,
            Date           DATE         NOT NULL,
            Time           TIME         NOT NULL,
            Confidence     FLOAT        NOT NULL,
            File_Name      VARCHAR(100) NOT NULL,
            Source_Node    VARCHAR(200) NOT NULL DEFAULT '',
            Model_Name     VARCHAR(200) NOT NULL DEFAULT '',
            rank           INTEGER      NOT NULL DEFAULT 0,
            PRIMARY KEY (Sci_Name, rank)
        );",
    ).await?;

    // Keep the 10 highest-confidence recordings per species.
    conn.execute_batch("DELETE FROM species_top_recordings;").await?;
    conn.execute_batch(
        "INSERT INTO species_top_recordings
            (Sci_Name, Com_Name, Date, Time, Confidence, File_Name, Source_Node, Model_Name, rank)
         SELECT Sci_Name, Com_Name, Date, Time, Confidence, File_Name,
                COALESCE(Source_Node, ''), COALESCE(Model_Name, ''), rn
         FROM (
             SELECT d.Sci_Name, d.Com_Name, d.Date, d.Time, d.Confidence,
                    d.File_Name, d.Source_Node, d.Model_Name,
                    ROW_NUMBER() OVER (
                        PARTITION BY d.Sci_Name
                        ORDER BY d.Confidence DESC, d.Date DESC, d.Time DESC
                    ) AS rn
             FROM detections d
             WHERE COALESCE(d.Excluded, 0) = 0
               AND d.File_Name != ''
               AND d.Confidence >= 0.5
         ) ranked
         WHERE rn <= 10;",
    ).await?;

    Ok(())
}

/// Fetch the top recordings for a species from the cache table.
///
/// Returns up to `limit` recordings sorted by confidence descending.
/// Falls back to a live query if the cache table is empty or missing.
pub async fn get_top_recordings(
    db_path: &Path,
    scientific_name: &str,
    limit: u32,
) -> Result<Vec<TopRecording>, libsql::Error> {
    let conn = open(db_path).await?;
    let tz = read_tz_offset(&conn).await;

    // Try the cached table first.
    let has_table: bool = match conn.query(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='species_top_recordings'",
        (),
    ).await {
        Ok(mut r) => match r.next().await {
            Ok(Some(row)) => row.get::<i64>(0).unwrap_or(0) > 0,
            _ => false,
        },
        Err(_) => false,
    };

    if has_table {
        let has_rows: i64 = match conn.query(
            "SELECT COUNT(*) FROM species_top_recordings WHERE Sci_Name = ?1",
            params![scientific_name.to_string()],
        ).await {
            Ok(mut r) => match r.next().await {
                Ok(Some(row)) => row.get::<i64>(0).unwrap_or(0),
                _ => 0,
            },
            Err(_) => 0,
        };

        if has_rows > 0 {
            let mut recs = get_top_recordings_cached(&conn, scientific_name, limit).await?;
            for r in &mut recs { stamp_recording(r, tz); }
            return Ok(recs);
        }
    }

    // Fallback: live query.
    let mut recs = get_top_recordings_live(&conn, scientific_name, limit).await?;
    for r in &mut recs { stamp_recording(r, tz); }
    Ok(recs)
}

async fn get_top_recordings_cached(
    conn: &libsql::Connection,
    scientific_name: &str,
    limit: u32,
) -> Result<Vec<TopRecording>, libsql::Error> {
    let mut rows = conn.query(
        "SELECT Sci_Name, Com_Name, Date, Time, Confidence, File_Name, \
         Source_Node, Model_Name \
         FROM species_top_recordings \
         WHERE Sci_Name = ?1 \
         ORDER BY Confidence DESC, Date DESC, Time DESC \
         LIMIT ?2",
        params![scientific_name.to_string(), limit],
    ).await?;

    let mut results = Vec::new();
    while let Some(row) = rows.next().await? {
        let date: String = row.get::<String>(2)?;
        let time: String = row.get::<String>(3)?;
        results.push(TopRecording {
            scientific_name: row.get::<String>(0)?,
            common_name: row.get::<String>(1)?,
            display_date: date.clone(),
            display_time: time.clone(),
            date,
            time,
            confidence: row.get::<f64>(4)?,
            file_name: row.get::<String>(5)?,
            source_node: row.get::<String>(6)?,
            model_name: row.get::<String>(7)?,
        });
    }
    Ok(results)
}

async fn get_top_recordings_live(
    conn: &libsql::Connection,
    scientific_name: &str,
    limit: u32,
) -> Result<Vec<TopRecording>, libsql::Error> {
    let mut rows = conn.query(
        "SELECT d.Sci_Name, d.Com_Name, d.Date, d.Time, d.Confidence, d.File_Name, \
         COALESCE(d.Source_Node, ''), COALESCE(d.Model_Name, '') \
         FROM detections d \
         WHERE d.Sci_Name = ?1 \
           AND COALESCE(d.Excluded, 0) = 0 \
           AND d.File_Name != '' \
           AND d.Confidence >= 0.5 \
         ORDER BY d.Confidence DESC, d.Date DESC, d.Time DESC \
         LIMIT ?2",
        params![scientific_name.to_string(), limit],
    ).await?;

    let mut results = Vec::new();
    while let Some(row) = rows.next().await? {
        let date: String = row.get::<String>(2)?;
        let time: String = row.get::<String>(3)?;
        results.push(TopRecording {
            scientific_name: row.get::<String>(0)?,
            common_name: row.get::<String>(1)?,
            display_date: date.clone(),
            display_time: time.clone(),
            date,
            time,
            confidence: row.get::<f64>(4)?,
            file_name: row.get::<String>(5)?,
            source_node: row.get::<String>(6)?,
            model_name: row.get::<String>(7)?,
        });
    }
    Ok(results)
}

// ─── Model filter helpers ────────────────────────────────────────────────────

/// A model that has been used for at least one detection.
#[derive(Debug, Clone)]
pub struct AvailableModel {
    pub slug: String,
    pub name: String,
}

/// Return distinct `(Model_Slug, Model_Name)` pairs recorded in the DB.
///
/// Only non-empty slugs are returned, ordered by the most recent use.
pub async fn available_models(db_path: &Path) -> Result<Vec<AvailableModel>, libsql::Error> {
    let conn = open(db_path).await?;
    let mut rows = conn.query(
        "SELECT COALESCE(Model_Slug, ''), COALESCE(Model_Name, ''), MAX(rowid) AS last \
         FROM detections \
         WHERE COALESCE(Model_Slug, '') != '' \
         GROUP BY Model_Slug \
         ORDER BY last DESC",
        (),
    ).await?;

    let mut results = Vec::new();
    while let Some(row) = rows.next().await? {
        results.push(AvailableModel {
            slug: row.get::<String>(0)?,
            name: row.get::<String>(1)?,
        });
    }
    Ok(results)
}

/// Recent detections optionally filtered by model slug.
pub async fn recent_detections_filtered(
    db_path: &Path,
    limit: u32,
    after_rowid: Option<i64>,
    model_slug: Option<&str>,
) -> Result<Vec<WebDetection>, libsql::Error> {
    // No filter → delegate to existing function.
    if model_slug.is_none() || model_slug == Some("") {
        return recent_detections(db_path, limit, after_rowid).await;
    }
    let slug = model_slug.unwrap();

    let conn = open(db_path).await?;
    let tz = read_tz_offset(&conn).await;

    let (sql, row_params): (String, Vec<libsql::Value>) = match after_rowid {
        Some(rid) => (
            "SELECT rowid, Domain, Sci_Name, Com_Name, Confidence, Date, Time, File_Name, \
             COALESCE(Source_Node, ''), COALESCE(Excluded, 0), \
             COALESCE(Model_Slug, ''), COALESCE(Model_Name, '') \
             FROM detections \
             WHERE rowid > ?1 AND COALESCE(Model_Slug, '') = ?2 \
             ORDER BY rowid DESC LIMIT ?3"
                .into(),
            vec![
                libsql::Value::from(rid),
                libsql::Value::from(slug.to_string()),
                libsql::Value::from(limit as i64),
            ],
        ),
        None => (
            "SELECT rowid, Domain, Sci_Name, Com_Name, Confidence, Date, Time, File_Name, \
             COALESCE(Source_Node, ''), COALESCE(Excluded, 0), \
             COALESCE(Model_Slug, ''), COALESCE(Model_Name, '') \
             FROM detections \
             WHERE COALESCE(Model_Slug, '') = ?1 \
             ORDER BY rowid DESC LIMIT ?2"
                .into(),
            vec![
                libsql::Value::from(slug.to_string()),
                libsql::Value::from(limit as i64),
            ],
        ),
    };

    let mut rows = conn.query(&sql, row_params).await?;
    let mut dets = Vec::new();
    while let Some(row) = rows.next().await? {
        dets.push(WebDetection {
            id: row.get::<i64>(0)?,
            domain: row.get::<String>(1)?,
            scientific_name: row.get::<String>(2)?,
            common_name: row.get::<String>(3)?,
            confidence: row.get::<f64>(4)?,
            date: row.get::<String>(5)?,
            time: row.get::<String>(6)?,
            file_name: row.get::<String>(7)?,
            source_node: row.get::<String>(8)?,
            excluded: row.get::<i32>(9)? != 0,
            image_url: None,
            model_slug: row.get::<String>(10)?,
            model_name: row.get::<String>(11)?,
            display_date: String::new(),
            display_time: String::new(),
        });
    }

    for d in &mut dets { stamp(d, tz); }
    Ok(dets)
}

/// Top species (all-time) optionally filtered by model slug.
pub async fn top_species_filtered(
    db_path: &Path,
    limit: u32,
    model_slug: Option<&str>,
) -> Result<Vec<crate::model::SpeciesSummary>, libsql::Error> {
    if model_slug.is_none() || model_slug == Some("") {
        return top_species(db_path, limit).await;
    }
    let slug = model_slug.unwrap();
    let conn = open(db_path).await?;
    let mut rows = conn.query(
        "SELECT d.Sci_Name, d.Com_Name, d.Domain, COUNT(*) AS cnt, \
         MAX(d.Date || ' ' || d.Time) AS last \
         FROM detections d \
         WHERE (COALESCE(d.Excluded, 0) = 0 \
                OR d.Sci_Name IN (SELECT Sci_Name FROM exclusion_overrides)) \
           AND COALESCE(d.Model_Slug, '') = ?1 \
         GROUP BY d.Sci_Name, d.Domain ORDER BY cnt DESC LIMIT ?2",
        params![slug.to_string(), limit],
    ).await?;

    let mut results = Vec::new();
    while let Some(row) = rows.next().await? {
        let count: u32 = row.get::<u32>(3)?;
        results.push(crate::model::SpeciesSummary {
            scientific_name: row.get::<String>(0)?,
            common_name: row.get::<String>(1)?,
            domain: row.get::<String>(2)?,
            detection_count: count,
            display_count: round_count(count),
            last_seen: row.get::<String>(4).ok(),
            image_url: None,
            conservation_status: None,
            male_image_url: None,
            female_image_url: None,
        });
    }
    Ok(results)
}

/// Top species for a specific date, optionally filtered by model slug.
pub async fn top_species_for_date_filtered(
    db_path: &Path,
    date: &str,
    limit: u32,
    model_slug: Option<&str>,
) -> Result<Vec<crate::model::SpeciesSummary>, libsql::Error> {
    if model_slug.is_none() || model_slug == Some("") {
        return top_species_for_date(db_path, date, limit).await;
    }
    let slug = model_slug.unwrap();
    let conn = open(db_path).await?;
    let mut rows = conn.query(
        "SELECT d.Sci_Name, d.Com_Name, d.Domain, COUNT(*) AS cnt, \
         MAX(d.Date || ' ' || d.Time) AS last \
         FROM detections d \
         WHERE d.Date = ?1 \
           AND (COALESCE(d.Excluded, 0) = 0 \
                OR d.Sci_Name IN (SELECT Sci_Name FROM exclusion_overrides)) \
           AND COALESCE(d.Model_Slug, '') = ?2 \
         GROUP BY d.Sci_Name, d.Domain ORDER BY cnt DESC LIMIT ?3",
        params![date.to_string(), slug.to_string(), limit],
    ).await?;

    let mut results = Vec::new();
    while let Some(row) = rows.next().await? {
        let count: u32 = row.get::<u32>(3)?;
        results.push(crate::model::SpeciesSummary {
            scientific_name: row.get::<String>(0)?,
            common_name: row.get::<String>(1)?,
            domain: row.get::<String>(2)?,
            detection_count: count,
            display_count: round_count(count),
            last_seen: row.get::<String>(4).ok(),
            image_url: None,
            conservation_status: None,
            male_image_url: None,
            female_image_url: None,
        });
    }
    Ok(results)
}

/// Day detections optionally filtered by model slug.
pub async fn day_detections_filtered(
    db_path: &Path,
    date: &str,
    model_slug: Option<&str>,
) -> Result<Vec<DayDetectionGroup>, libsql::Error> {
    if model_slug.is_none() || model_slug == Some("") {
        return day_detections(db_path, date).await;
    }
    let slug = model_slug.unwrap();
    let conn = open(db_path).await?;
    let tz = read_tz_offset(&conn).await;
    let mut rows = conn.query(
        "SELECT rowid, Domain, Sci_Name, Com_Name, Confidence, Date, Time, File_Name, \
         COALESCE(Source_Node, ''), COALESCE(Excluded, 0), \
         COALESCE(Model_Slug, ''), COALESCE(Model_Name, '') \
         FROM detections \
         WHERE Date = ?1 AND COALESCE(Model_Slug, '') = ?2 \
         ORDER BY Sci_Name, Time DESC",
        params![date.to_string(), slug.to_string()],
    ).await?;

    let mut all_dets: Vec<WebDetection> = Vec::new();
    while let Some(row) = rows.next().await? {
        all_dets.push(WebDetection {
            id: row.get::<i64>(0)?,
            domain: row.get::<String>(1)?,
            scientific_name: row.get::<String>(2)?,
            common_name: row.get::<String>(3)?,
            confidence: row.get::<f64>(4)?,
            date: row.get::<String>(5)?,
            time: row.get::<String>(6)?,
            file_name: row.get::<String>(7)?,
            source_node: row.get::<String>(8)?,
            excluded: row.get::<i32>(9)? != 0,
            image_url: None,
            model_slug: row.get::<String>(10)?,
            model_name: row.get::<String>(11)?,
            display_date: String::new(),
            display_time: String::new(),
        });
    }
    let all_dets: Vec<WebDetection> = all_dets.into_iter().map(|mut d| { stamp(&mut d, tz); d }).collect();

    let mut groups: Vec<DayDetectionGroup> = Vec::new();
    for det in all_dets {
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
                image_url: None,
                max_confidence: det.confidence,
                detections: vec![det],
            });
        }
    }
    Ok(groups)
}

/// Species info optionally filtered by model slug.
pub async fn species_detections_by_model(
    db_path: &Path,
    scientific_name: &str,
    limit: u32,
    model_slug: Option<&str>,
) -> Result<Vec<WebDetection>, libsql::Error> {
    let conn = open(db_path).await?;
    let tz = read_tz_offset(&conn).await;

    let (sql, row_params): (String, Vec<libsql::Value>) =
        if let Some(slug) = model_slug.filter(|s| !s.is_empty()) {
            (
                "SELECT rowid, Domain, Sci_Name, Com_Name, Confidence, Date, Time, File_Name, \
                 COALESCE(Source_Node, ''), COALESCE(Excluded, 0), \
                 COALESCE(Model_Slug, ''), COALESCE(Model_Name, '') \
                 FROM detections \
                 WHERE Sci_Name = ?1 AND COALESCE(Model_Slug, '') = ?2 \
                 ORDER BY Date DESC, Time DESC LIMIT ?3"
                    .into(),
                vec![
                    libsql::Value::from(scientific_name.to_string()),
                    libsql::Value::from(slug.to_string()),
                    libsql::Value::from(limit as i64),
                ],
            )
        } else {
            (
                "SELECT rowid, Domain, Sci_Name, Com_Name, Confidence, Date, Time, File_Name, \
                 COALESCE(Source_Node, ''), COALESCE(Excluded, 0), \
                 COALESCE(Model_Slug, ''), COALESCE(Model_Name, '') \
                 FROM detections \
                 WHERE Sci_Name = ?1 \
                 ORDER BY Date DESC, Time DESC LIMIT ?2"
                    .into(),
                vec![
                    libsql::Value::from(scientific_name.to_string()),
                    libsql::Value::from(limit as i64),
                ],
            )
        };

    let mut rows = conn.query(&sql, row_params).await?;
    let mut dets = Vec::new();
    while let Some(row) = rows.next().await? {
        dets.push(WebDetection {
            id: row.get::<i64>(0)?,
            domain: row.get::<String>(1)?,
            scientific_name: row.get::<String>(2)?,
            common_name: row.get::<String>(3)?,
            confidence: row.get::<f64>(4)?,
            date: row.get::<String>(5)?,
            time: row.get::<String>(6)?,
            file_name: row.get::<String>(7)?,
            source_node: row.get::<String>(8)?,
            excluded: row.get::<i32>(9)? != 0,
            image_url: None,
            model_slug: row.get::<String>(10)?,
            model_name: row.get::<String>(11)?,
            display_date: String::new(),
            display_time: String::new(),
        });
    }

    for d in &mut dets { stamp(d, tz); }
    Ok(dets)
}

// ─── Settings ────────────────────────────────────────────────────────────────

use std::collections::HashMap;

/// Read all rows from the `settings` table as a key-value map.
pub async fn get_all_settings(db_path: &Path) -> Result<HashMap<String, String>, libsql::Error> {
    let conn = open(db_path).await?;
    // The table may not exist in older databases.
    let _ = conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
    ).await;
    let mut rows = conn.query("SELECT key, value FROM settings", ()).await?;
    let mut map = HashMap::new();
    while let Some(row) = rows.next().await? {
        let k: String = row.get::<String>(0)?;
        let v: String = row.get::<String>(1)?;
        map.insert(k, v);
    }
    Ok(map)
}

/// Open a read-write connection with WAL and a busy timeout.
async fn open_rw(db_path: &Path) -> Result<libsql::Connection, libsql::Error> {
    let db = build_db(db_path).await?;
    let conn = db.connect()?;
    conn.execute_batch(&format!(
        "PRAGMA busy_timeout={BUSY_TIMEOUT_MS}; PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;"
    )).await?;
    Ok(conn)
}

/// Save multiple settings in one transaction.
pub async fn save_settings(db_path: &Path, entries: &[(&str, &str)]) -> Result<(), libsql::Error> {
    let conn = open_rw(db_path).await?;
    let _ = conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
    ).await;
    conn.execute_batch("BEGIN").await?;
    for (k, v) in entries {
        conn.execute(
            "INSERT INTO settings (key, value) VALUES (?1, ?2) \
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![k.to_string(), v.to_string()],
        ).await?;
    }
    conn.execute_batch("COMMIT").await?;
    Ok(())
}

// ─── Urban noise ─────────────────────────────────────────────────────────────

/// Aggregated urban-noise counts per category (today, last 7 days, all time).
pub async fn urban_noise_summary(
    db_path: &Path,
) -> Result<Vec<UrbanNoiseSummary>, libsql::Error> {
    let conn = open(db_path).await?;
    let tz = read_tz_offset(&conn).await;
    let today = today_for_tz(tz);

    let mut rows = conn.query(
        "SELECT Category,
                COALESCE(SUM(Count), 0) AS total,
                COALESCE(SUM(CASE WHEN Date = ?1 THEN Count ELSE 0 END), 0) AS today,
                COALESCE(SUM(CASE WHEN Date >= DATE(?1, '-7 day') THEN Count ELSE 0 END), 0) AS week
         FROM urban_noise
         GROUP BY Category
         ORDER BY total DESC",
        params![today],
    ).await?;

    let mut results = Vec::new();
    while let Some(row) = rows.next().await? {
        results.push(UrbanNoiseSummary {
            category: row.get::<String>(0)?,
            total_count: row.get::<u32>(1)?,
            today_count: row.get::<u32>(2)?,
            week_count: row.get::<u32>(3)?,
        });
    }
    Ok(results)
}

// ─── Excluded species ────────────────────────────────────────────────────────

/// List species that have at least one excluded detection.
pub async fn excluded_species(db_path: &Path) -> Result<Vec<ExcludedSpecies>, libsql::Error> {
    let conn = open(db_path).await?;

    let mut rows = conn.query(
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
        (),
    ).await?;

    let mut results = Vec::new();
    while let Some(row) = rows.next().await? {
        results.push(ExcludedSpecies {
            scientific_name: row.get::<String>(0)?,
            common_name: row.get::<String>(1)?,
            domain: row.get::<String>(2)?,
            detection_count: row.get::<u32>(3)?,
            last_seen: row.get::<String>(4).ok(),
            max_confidence: row.get::<f64>(5)?,
            image_url: None,
            overridden: row.get::<i32>(6)? != 0,
        });
    }
    Ok(results)
}

/// Return individual excluded detections for a given species, newest first.
///
/// Used by the Excluded page so the user can listen to / inspect the
/// recordings before confirming an override.
pub async fn excluded_detections_for_species(
    db_path: &Path,
    scientific_name: &str,
    limit: u32,
) -> Result<Vec<WebDetection>, libsql::Error> {
    let conn = open(db_path).await?;
    let tz = read_tz_offset(&conn).await;
    let mut rows = conn.query(
        "SELECT rowid, Domain, Sci_Name, Com_Name, Confidence, Date, Time, File_Name, \
         COALESCE(Source_Node, ''), COALESCE(Excluded, 0), \
         COALESCE(Model_Slug, ''), COALESCE(Model_Name, '') \
         FROM detections \
         WHERE Sci_Name = ?1 AND COALESCE(Excluded, 0) = 1 \
         ORDER BY Date DESC, Time DESC LIMIT ?2",
        params![scientific_name.to_string(), limit],
    ).await?;

    let mut dets = Vec::new();
    while let Some(row) = rows.next().await? {
        dets.push(WebDetection {
            id: row.get::<i64>(0)?,
            domain: row.get::<String>(1)?,
            scientific_name: row.get::<String>(2)?,
            common_name: row.get::<String>(3)?,
            confidence: row.get::<f64>(4)?,
            date: row.get::<String>(5)?,
            time: row.get::<String>(6)?,
            file_name: row.get::<String>(7)?,
            source_node: row.get::<String>(8)?,
            excluded: row.get::<i32>(9)? != 0,
            image_url: None,
            model_slug: row.get::<String>(10)?,
            model_name: row.get::<String>(11)?,
            display_date: String::new(),
            display_time: String::new(),
        });
    }

    for d in &mut dets { stamp(d, tz); }
    Ok(dets)
}

/// Add an exclusion override (ornithologist confirms the species is real).
pub async fn add_exclusion_override(
    db_path: &Path,
    scientific_name: &str,
    notes: &str,
) -> Result<(), libsql::Error> {
    let conn = open_rw(db_path).await?;
    conn.execute(
        "INSERT INTO exclusion_overrides (Sci_Name, overridden_at, notes) \
         VALUES (?1, datetime('now'), ?2) \
         ON CONFLICT(Sci_Name) DO UPDATE SET overridden_at = datetime('now'), notes = excluded.notes",
        params![scientific_name.to_string(), notes.to_string()],
    ).await?;
    Ok(())
}

/// Remove an exclusion override (undo confirmation).
pub async fn remove_exclusion_override(
    db_path: &Path,
    scientific_name: &str,
) -> Result<(), libsql::Error> {
    let conn = open_rw(db_path).await?;
    conn.execute(
        "DELETE FROM exclusion_overrides WHERE Sci_Name = ?1",
        params![scientific_name.to_string()],
    ).await?;
    Ok(())
}

// ─── Species verification ────────────────────────────────────────────────────

use crate::model::SpeciesVerification;

/// Get verification record for a species (if any).
pub async fn get_species_verification(
    db_path: &Path,
    scientific_name: &str,
) -> Result<Option<SpeciesVerification>, libsql::Error> {
    let conn = open(db_path).await?;
    // Table may not exist in older DBs.
    let mut rows = match conn.query(
        "SELECT method, COALESCE(inaturalist_obs, ''), verified_at \
         FROM species_verifications WHERE Sci_Name = ?1",
        params![scientific_name.to_string()],
    ).await {
        Ok(r) => r,
        Err(e) => {
            // Table doesn't exist yet – treat as no verification.
            if e.to_string().contains("no such table") {
                return Ok(None);
            }
            return Err(e);
        }
    };
    match rows.next().await? {
        Some(row) => Ok(Some(SpeciesVerification {
            method: row.get::<String>(0)?,
            inaturalist_obs: row.get::<String>(1)?,
            verified_at: row.get::<String>(2)?,
        })),
        None => Ok(None),
    }
}

/// Save a verification record for a species.
pub async fn set_species_verification(
    db_path: &Path,
    scientific_name: &str,
    method: &str,
    inaturalist_obs: &str,
) -> Result<(), libsql::Error> {
    let conn = open_rw(db_path).await?;
    let _ = conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS species_verifications (
            Sci_Name         VARCHAR(100) PRIMARY KEY,
            method           VARCHAR(50) NOT NULL DEFAULT 'ornithologist',
            inaturalist_obs  TEXT NOT NULL DEFAULT '',
            verified_at      TEXT NOT NULL DEFAULT (datetime('now'))
        );"
    ).await;
    conn.execute(
        "INSERT INTO species_verifications (Sci_Name, method, inaturalist_obs, verified_at) \
         VALUES (?1, ?2, ?3, datetime('now')) \
         ON CONFLICT(Sci_Name) DO UPDATE SET method = excluded.method, \
         inaturalist_obs = excluded.inaturalist_obs, verified_at = datetime('now')",
        params![scientific_name.to_string(), method.to_string(), inaturalist_obs.to_string()],
    ).await?;
    Ok(())
}

/// Remove verification for a species.
pub async fn remove_species_verification(
    db_path: &Path,
    scientific_name: &str,
) -> Result<(), libsql::Error> {
    let conn = open_rw(db_path).await?;
    let _ = conn.execute(
        "DELETE FROM species_verifications WHERE Sci_Name = ?1",
        params![scientific_name.to_string()],
    ).await;
    Ok(())
}

// ─── Learning quiz ───────────────────────────────────────────────────────────

/// Select 4 random quiz-worthy detections (one per species).
///
/// A detection qualifies when:
///   * confidence ≥ 0.75
///   * not excluded
///   * has an extracted audio file
///   * every detection for that same `File_Name` shares the same `Sci_Name`
///     (i.e. the clip contains only one species)
///
/// When `today_only` is `true` only detections from today (TZ-aware) are returned.
pub async fn quiz_candidates(
    db_path: &Path,
    today_only: bool,
) -> Result<Vec<QuizItem>, libsql::Error> {
    let conn = open(db_path).await?;
    let tz = read_tz_offset(&conn).await;
    let today_str = today_for_tz(tz);

    // Use separate date filters: the CTE has no table alias while the
    // outer query uses `d`.
    let cte_date_filter = if today_only {
        "AND Date = ?1"
    } else {
        ""
    };
    let outer_date_filter = if today_only {
        "AND d.Date = ?1"
    } else {
        ""
    };

    let sql = format!(
        "WITH clean_files AS (
             SELECT File_Name
             FROM detections
             WHERE File_Name != ''
               AND COALESCE(Excluded, 0) = 0
               {cte_date_filter}
             GROUP BY File_Name
             HAVING COUNT(DISTINCT Sci_Name) = 1
                AND MAX(Confidence) >= 0.75
         )
         SELECT d.Sci_Name, d.Com_Name, d.Date, d.File_Name, d.Confidence
         FROM detections d
         INNER JOIN clean_files cf ON d.File_Name = cf.File_Name
         WHERE d.Confidence >= 0.75
           AND COALESCE(d.Excluded, 0) = 0
           AND d.File_Name != ''
           {outer_date_filter}
         ORDER BY RANDOM()"
    );

    // When today_only we bind the TZ-aware date; otherwise no params.
    let mut rows = if today_only {
        conn.query(&sql, params![today_str.clone()]).await?
    } else {
        conn.query(&sql, ()).await?
    };

    // Pick one clip per species, up to 4 distinct species.
    let mut seen_species = std::collections::HashSet::new();
    let mut items = Vec::new();

    while let Some(row) = rows.next().await? {
        let sci_name: String = row.get::<String>(0)?;
        let com_name: String = row.get::<String>(1)?;
        let date: String = row.get::<String>(2)?;
        let file_name: String = row.get::<String>(3)?;

        if seen_species.contains(&sci_name) {
            continue;
        }
        seen_species.insert(sci_name.clone());

        let safe_name = com_name.replace('\'', "").replace(' ', "_");
        let clip_url = format!("/extracted/By_Date/{date}/{safe_name}/{file_name}");
        let spectrogram_url = format!("{clip_url}.png");

        items.push(QuizItem {
            scientific_name: sci_name,
            common_name: com_name,
            clip_url,
            spectrogram_url,
            image_url: None,
        });

        if items.len() >= 4 {
            break;
        }
    }

    Ok(items)
}
