//! DuckDB-backed detection queries (columnar reads over Parquet files).
//!
//! The processing containers write one Parquet file per batch into
//! `/data/detections/`.  This module opens a **read-only, in-memory**
//! DuckDB instance that creates a `detections` view over those files:
//!
//! ```sql
//! CREATE OR REPLACE VIEW detections AS
//!     SELECT * FROM read_parquet('/data/detections/*.parquet', union_by_name=true);
//! ```
//!
//! All analytical queries (species counts, calendar, histograms, etc.)
//! go through DuckDB, which reads only the columns it needs — orders of
//! magnitude faster than row-oriented SQLite at scale.
//!
//! The small OLTP tables (`settings`, `exclusion_overrides`,
//! `urban_noise`, `species_verifications`, etc.) live in Redis / Valkey.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::OnceLock;

#[allow(unused_imports)]
use duckdb::params;
use tracing::info;

use crate::model::{
    CacheSummaryStatus, CalendarDay, DayDetectionGroup, ExcludedSpecies, HourlyCount, ModelInfo,
    QuizItem, SpeciesHourlyCounts, SpeciesInfo, SpeciesSummary, TopRecording,
    WebDetection,
};

// Re-export AvailableModel used by model_filter component.
pub use super::db::AvailableModel;

// ─── Connection management ───────────────────────────────────────────────────

/// Cached DuckDB connection (in-memory).
static DUCK: OnceLock<Mutex<duckdb::Connection>> = OnceLock::new();

/// Directory where Parquet files live.
static DET_DIR: OnceLock<PathBuf> = OnceLock::new();

/// Epoch‐second when the view was last refreshed (throttled to avoid
/// re-creating the Parquet glob view on every single HTTP request).
static VIEW_REFRESHED_AT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Minimum seconds between view refreshes.
const VIEW_REFRESH_INTERVAL_SECS: u64 = 30;

/// Whether the in-memory stats cache tables have been populated.
static STATS_POPULATED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Epoch-second of the latest successful stats cache refresh.
static STATS_REFRESHED_AT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Number of `.parquet` files at the last stats cache build.
/// Used to auto-detect new detections and invalidate the cache.
static PARQUET_FILE_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Initialise the DuckDB read layer.
///
/// * `detections_dir` — path to the directory containing `*.parquet` files
///   (typically `/data/detections`).
///
/// Creates an in-memory DuckDB instance and a `detections` view over the
/// Parquet files.  Must be called once at startup.
pub fn initialize(detections_dir: &Path) -> Result<(), duckdb::Error> {
    std::fs::create_dir_all(detections_dir).ok();

    // Run the one-time Sci_Name normalisation migration so that existing
    // Parquet files have consistent species names before we create the view.
    if let Err(e) = normalize_parquet_sci_names(detections_dir) {
        tracing::warn!("Parquet Sci_Name migration failed (non-fatal): {e:#}");
    }

    let conn = duckdb::Connection::open_in_memory()?;

    // Create the view.  If the glob matches nothing DuckDB returns an
    // error, so we first check whether any Parquet files exist.
    refresh_view_inner(&conn, detections_dir)?;

    let _ = DET_DIR.set(detections_dir.to_path_buf());
    let _ = DUCK.set(Mutex::new(conn));
    info!(
        "DuckDB detections layer ready (dir={})",
        detections_dir.display()
    );
    Ok(())
}

/// Re-create the `detections` view so newly-written Parquet files are
/// visible.  Throttled: skips the refresh if the last one was less than
/// `VIEW_REFRESH_INTERVAL_SECS` ago to avoid repeating the directory scan
/// and `CREATE OR REPLACE VIEW` on every HTTP request.
fn refresh_view(conn: &duckdb::Connection) {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let prev = VIEW_REFRESHED_AT.load(std::sync::atomic::Ordering::Relaxed);
    if now.saturating_sub(prev) < VIEW_REFRESH_INTERVAL_SECS {
        return; // recently refreshed — skip
    }
    if let Some(dir) = DET_DIR.get() {
        if refresh_view_inner(conn, dir).is_ok() {
            VIEW_REFRESHED_AT.store(now, std::sync::atomic::Ordering::Relaxed);

            // Auto‐detect new Parquet files and invalidate the stats cache
            // so the species / excluded pages pick up fresh data.
            let file_count = count_parquet_files(dir);
            let prev_count = PARQUET_FILE_COUNT.load(std::sync::atomic::Ordering::Relaxed);
            if file_count != prev_count && prev_count != 0 {
                // New files appeared (or old ones removed); stats are stale.
                STATS_POPULATED.store(false, std::sync::atomic::Ordering::Relaxed);
                tracing::debug!(
                    "Parquet file count changed ({prev_count} → {file_count}), stats cache invalidated"
                );
            }
            PARQUET_FILE_COUNT.store(file_count, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

/// Count `.parquet` files in a directory (non-recursive, fast).
fn count_parquet_files(dir: &Path) -> u64 {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .filter(|e| {
                    e.path().extension().map(|x| x == "parquet").unwrap_or(false)
                })
                .count() as u64
        })
        .unwrap_or(0)
}

/// Force a view refresh on the next `conn()` call, regardless of throttle.
pub fn invalidate_view() {
    VIEW_REFRESHED_AT.store(0, std::sync::atomic::Ordering::Relaxed);
}

/// Check whether the in-memory stats cache is currently populated.
pub fn stats_populated() -> bool {
    STATS_POPULATED.load(std::sync::atomic::Ordering::Relaxed)
}

fn table_row_count(duck: &duckdb::Connection, table: &str) -> u64 {
    let sql = format!("SELECT COUNT(*) FROM {table}");
    duck.query_row(&sql, [], |row| row.get::<_, i64>(0))
        .ok()
        .map(|n| n.max(0) as u64)
        .unwrap_or(0)
}

/// Snapshot of the summary cache health/size for UI diagnostics.
pub async fn stats_cache_status(_db_path: &Path) -> Res<CacheSummaryStatus> {
    let duck = conn_raw()?;
    let refreshed_epoch = STATS_REFRESHED_AT.load(std::sync::atomic::Ordering::Relaxed);
    let refreshed_at_utc = chrono::DateTime::from_timestamp(refreshed_epoch as i64, 0)
        .map(|dt| dt.to_rfc3339())
        .unwrap_or_default();

    Ok(CacheSummaryStatus {
        populated: STATS_POPULATED.load(std::sync::atomic::Ordering::Relaxed),
        species_rows: table_row_count(&duck, "species_stats"),
        excluded_rows: table_row_count(&duck, "excluded_species_stats"),
        model_species_rows: table_row_count(&duck, "model_species_stats"),
        parquet_files: DET_DIR.get().map(|d| count_parquet_files(d)).unwrap_or(0),
        refreshed_at_utc,
    })
}

fn refresh_view_inner(conn: &duckdb::Connection, dir: &Path) -> Result<(), duckdb::Error> {
    let glob = format!("{}/*.parquet", dir.display());
    // Check if any Parquet files exist — read_parquet errors on empty glob.
    let has_files = std::fs::read_dir(dir)
        .map(|rd| rd.flatten().any(|e| {
            e.path().extension().map(|x| x == "parquet").unwrap_or(false)
        }))
        .unwrap_or(false);

    if has_files {
        conn.execute_batch(&format!(
            "CREATE OR REPLACE VIEW detections AS \
             SELECT * FROM read_parquet('{glob}', union_by_name=true)"
        ))?;
    } else {
        // Empty placeholder with the correct schema so queries don't fail.
        conn.execute_batch(
            "CREATE OR REPLACE VIEW detections AS SELECT \
             0::BIGINT AS id, \
             ''::VARCHAR AS Date, ''::VARCHAR AS Time, \
             ''::VARCHAR AS Domain, ''::VARCHAR AS Sci_Name, \
             ''::VARCHAR AS Com_Name, 0.0::DOUBLE AS Confidence, \
             0.0::DOUBLE AS Lat, 0.0::DOUBLE AS Lon, \
             0.0::DOUBLE AS Cutoff, 0::INTEGER AS Week, \
             0.0::DOUBLE AS Sens, 0.0::DOUBLE AS Overlap, \
             ''::VARCHAR AS File_Name, ''::VARCHAR AS Source_Node, \
             0::INTEGER AS Excluded, ''::VARCHAR AS Model_Slug, \
             ''::VARCHAR AS Model_Name, \
             0::INTEGER AS Model_Beta, \
             0.0::DOUBLE AS Agreement_Score, \
             ''::VARCHAR AS Agreement_Models \
             WHERE false",
        )?;
    }
    Ok(())
}

/// Get a lock on the DuckDB connection, refreshing the view first.
fn conn() -> Result<std::sync::MutexGuard<'static, duckdb::Connection>, String> {
    let guard = DUCK
        .get()
        .ok_or("DuckDB not initialised")?
        .lock()
        .map_err(|e| format!("DuckDB lock poisoned: {e}"))?;
    refresh_view(&guard);
    Ok(guard)
}

/// Get a lock on the DuckDB connection **without** refreshing the view.
///
/// Used by functions that only read from the in-memory cache tables
/// (`species_stats`, `excluded_species_stats`, etc.) and don't need the
/// Parquet view to be up-to-date.
fn conn_raw() -> Result<std::sync::MutexGuard<'static, duckdb::Connection>, String> {
    DUCK.get()
        .ok_or("DuckDB not initialised")?
        .lock()
        .map_err(|e| format!("DuckDB lock poisoned: {e}"))
}

/// Shorthand error type used in this module.
type Res<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Build the exclusion-override filter clause.
///
/// Many queries need:
///   `(COALESCE(Excluded, 0) = 0 OR Sci_Name IN (…overrides…))`
///
/// The override list comes from the **SQLite** `exclusion_overrides` table.
/// We read it once per query and inline it.
fn exclusion_clause(overrides: &[String]) -> String {
    if overrides.is_empty() {
        "COALESCE(Excluded, 0) = 0".to_string()
    } else {
        let list = overrides
            .iter()
            .map(|s| format!("'{}'", s.replace('\'', "''")))
            .collect::<Vec<_>>()
            .join(",");
        format!("(COALESCE(Excluded, 0) = 0 OR Sci_Name IN ({list}))")
    }
}

/// Read the exclusion-override species list from Redis.
pub async fn read_overrides(_db_path: &Path) -> Vec<String> {
    super::kv::read_overrides().await
}

/// Read tz_offset from Redis settings.
pub async fn read_tz_offset(_db_path: &Path) -> i32 {
    super::kv::read_tz_offset().await
}

/// Apply TZ offset to date/time strings.
fn apply_tz(date: &str, time: &str, offset_hours: i32) -> (String, String) {
    if offset_hours == 0 {
        return (date.to_string(), time.to_string());
    }
    use chrono::{Duration, NaiveDate, NaiveDateTime, NaiveTime};
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

fn stamp(det: &mut WebDetection, offset: i32) {
    let (dd, dt) = apply_tz(&det.date, &det.time, offset);
    det.display_date = dd;
    det.display_time = dt;
}

fn stamp_recording(rec: &mut TopRecording, offset: i32) {
    let (dd, dt) = apply_tz(&rec.date, &rec.time, offset);
    rec.display_date = dd;
    rec.display_time = dt;
}

/// Parse a WebDetection from a DuckDB row (standard 12-column SELECT).
fn parse_detection(row: &duckdb::Row<'_>) -> Result<WebDetection, duckdb::Error> {
    Ok(WebDetection {
        id: row.get::<_, i64>(0)?,
        domain: row.get::<_, String>(1)?,
        scientific_name: row.get::<_, String>(2)?,
        common_name: row.get::<_, String>(3)?,
        confidence: row.get::<_, f64>(4)?,
        date: row.get::<_, String>(5)?,
        time: row.get::<_, String>(6)?,
        file_name: row.get::<_, String>(7)?,
        source_node: row.get::<_, String>(8)?,
        excluded: row.get::<_, i32>(9)? != 0,
        image_url: None,
        model_slug: row.get::<_, String>(10)?,
        model_name: row.get::<_, String>(11)?,
        model_beta: row.get::<_, i32>(12).unwrap_or(0) != 0,
        agreement_score: row.get::<_, f64>(13).unwrap_or(0.0),
        agreement_models: row.get::<_, String>(14).unwrap_or_default(),
        display_date: String::new(),
        display_time: String::new(),
    })
}

/// Approximate display count (e.g. "~2.3K").
fn round_count(n: u32) -> String {
    if n < 1000 {
        n.to_string()
    } else {
        let k = n as f64 / 1000.0;
        if k < 10.0 {
            format!("~{k:.1}K")
        } else {
            format!("~{k:.0}K")
        }
    }
}

// ─── Detection queries ───────────────────────────────────────────────────────

/// Recent detections, optionally filtered by model slug and after a cursor ID.
pub async fn recent_detections_filtered(
    db_path: &Path,
    limit: u32,
    after_id: Option<i64>,
    model_slug: Option<&str>,
) -> Res<Vec<WebDetection>> {
    let tz = read_tz_offset(db_path).await;
    let duck = conn()?;

    let slug_filter = match model_slug {
        Some(s) if !s.is_empty() => format!("AND COALESCE(Model_Slug, '') = '{}'", s.replace('\'', "''")),
        _ => String::new(),
    };
    let id_filter = match after_id {
        Some(rid) => format!("AND id > {rid}"),
        None => String::new(),
    };

    let sql = format!(
        "SELECT id, Domain, Sci_Name, Com_Name, Confidence, Date, Time, File_Name, \
         COALESCE(Source_Node, ''), COALESCE(Excluded, 0), \
         COALESCE(Model_Slug, ''), COALESCE(Model_Name, ''), \
         COALESCE(Model_Beta, 0), \
         COALESCE(Agreement_Score, 0.0), COALESCE(Agreement_Models, '') \
         FROM detections \
         WHERE true {id_filter} {slug_filter} \
         ORDER BY id DESC LIMIT {limit}"
    );

    let mut stmt = duck.prepare(&sql)?;
    let rows = stmt.query_map([], |row| parse_detection(row))?;
    let mut dets: Vec<WebDetection> = rows.filter_map(|r| r.ok()).collect();
    for d in &mut dets {
        stamp(d, tz);
    }
    Ok(dets)
}

/// Unfiltered recent detections (backward-compat wrapper).
pub async fn recent_detections(
    db_path: &Path,
    limit: u32,
    after_id: Option<i64>,
) -> Res<Vec<WebDetection>> {
    recent_detections_filtered(db_path, limit, after_id, None).await
}

/// Calendar aggregates for a year/month.
pub async fn calendar_data(
    db_path: &Path,
    year: i32,
    month: u32,
) -> Res<Vec<CalendarDay>> {
    let overrides = read_overrides(db_path).await;
    let excl = exclusion_clause(&overrides);
    let start = format!("{year:04}-{month:02}-01");
    let end = if month == 12 {
        format!("{:04}-01-01", year + 1)
    } else {
        format!("{year:04}-{:02}-01", month + 1)
    };

    let duck = conn()?;
    let sql = format!(
        "SELECT Date, COUNT(*) AS cnt, COUNT(DISTINCT Sci_Name) AS spp \
         FROM detections \
         WHERE Date >= '{start}' AND Date < '{end}' AND {excl} \
         GROUP BY Date ORDER BY Date"
    );
    let mut stmt = duck.prepare(&sql)?;
    let rows = stmt.query_map([], |row| {
        Ok(CalendarDay {
            date: row.get(0)?,
            total_detections: row.get(1)?,
            unique_species: row.get(2)?,
        })
    })?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

/// All detections for a specific date, grouped by species.
pub async fn day_detections(
    db_path: &Path,
    date: &str,
) -> Res<Vec<DayDetectionGroup>> {
    day_detections_filtered(db_path, date, None).await
}

/// Day detections filtered by model slug.
pub async fn day_detections_filtered(
    db_path: &Path,
    date: &str,
    model_slug: Option<&str>,
) -> Res<Vec<DayDetectionGroup>> {
    let tz = read_tz_offset(db_path).await;
    let duck = conn()?;

    let slug_filter = match model_slug {
        Some(s) if !s.is_empty() => format!("AND COALESCE(Model_Slug, '') = '{}'", s.replace('\'', "''")),
        _ => String::new(),
    };
    let safe_date = date.replace('\'', "''");
    let sql = format!(
        "SELECT id, Domain, Sci_Name, Com_Name, Confidence, Date, Time, File_Name, \
         COALESCE(Source_Node, ''), COALESCE(Excluded, 0), \
         COALESCE(Model_Slug, ''), COALESCE(Model_Name, ''), \
         COALESCE(Model_Beta, 0), \
         COALESCE(Agreement_Score, 0.0), COALESCE(Agreement_Models, '') \
         FROM detections WHERE Date = '{safe_date}' {slug_filter} \
         ORDER BY Sci_Name, Time DESC"
    );

    let mut stmt = duck.prepare(&sql)?;
    let rows = stmt.query_map([], |row| parse_detection(row))?;
    let dets: Vec<WebDetection> = rows.filter_map(|r| r.ok()).collect();

    // Group by species, preserving insertion order via Vec<(key, group)>.
    let mut groups: Vec<(String, DayDetectionGroup)> = Vec::new();
    let mut key_idx: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for mut d in dets {
        stamp(&mut d, tz);
        let key = d.scientific_name.clone();
        let conf = d.confidence;
        if let Some(&idx) = key_idx.get(&key) {
            let g = &mut groups[idx].1;
            if conf > g.max_confidence {
                g.max_confidence = conf;
            }
            g.detections.push(d);
        } else {
            let idx = groups.len();
            key_idx.insert(key.clone(), idx);
            groups.push((key, DayDetectionGroup {
                scientific_name: d.scientific_name.clone(),
                common_name: d.common_name.clone(),
                domain: d.domain.clone(),
                image_url: None,
                detections: vec![d],
                max_confidence: conf,
            }));
        }
    }

    Ok(groups.into_iter().map(|(_, g)| g).collect())
}

/// Aggregated info for a single species.
pub async fn species_info(db_path: &Path, scientific_name: &str) -> Res<Option<SpeciesInfo>> {
    let overrides = read_overrides(db_path).await;
    let excl = exclusion_clause(&overrides);
    let safe = scientific_name.replace('\'', "''");
    let duck = conn()?;

    let sql = format!(
        "SELECT Domain, \
         COALESCE( \
             MAX(CASE WHEN Com_Name != Sci_Name THEN Com_Name ELSE NULL END), \
             MAX(Com_Name) \
         ) AS Com_Name, \
         COUNT(*) AS cnt, \
         MIN(Date) AS first_seen, MAX(Date) AS last_seen \
         FROM detections WHERE Sci_Name = '{safe}' AND {excl} \
         GROUP BY Domain LIMIT 1"
    );
    let mut stmt = duck.prepare(&sql)?;
    let mut rows = stmt.query_map([], |row| {
        Ok(SpeciesInfo {
            scientific_name: scientific_name.to_string(),
            common_name: row.get(1)?,
            domain: row.get(0)?,
            image_url: None,
            wikipedia_url: None,
            total_detections: row.get::<_, u64>(2)?,
            first_seen: row.get(3)?,
            last_seen: row.get(4)?,
            male_image_url: None,
            female_image_url: None,
            verification: None,
        })
    })?;
    Ok(rows.next().and_then(|r| r.ok()))
}

/// Distinct dates a species was detected in a given year.
pub async fn species_active_dates(
    _db_path: &Path,
    scientific_name: &str,
    year: i32,
) -> Res<Vec<String>> {
    let safe = scientific_name.replace('\'', "''");
    let start = format!("{year:04}-01-01");
    let end = format!("{:04}-01-01", year + 1);
    let duck = conn()?;
    let sql = format!(
        "SELECT DISTINCT Date FROM detections \
         WHERE Sci_Name = '{safe}' AND Date >= '{start}' AND Date < '{end}' \
         ORDER BY Date"
    );
    let mut stmt = duck.prepare(&sql)?;
    let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

/// Hourly histogram for a species (all-time).
pub async fn species_hourly_histogram(
    db_path: &Path,
    scientific_name: &str,
) -> Res<Vec<HourlyCount>> {
    let overrides = read_overrides(db_path).await;
    let excl = exclusion_clause(&overrides);
    let safe = scientific_name.replace('\'', "''");
    let duck = conn()?;
    let sql = format!(
        "SELECT CAST(SUBSTR(Time, 1, 2) AS INTEGER) AS hour, COUNT(*) AS cnt \
         FROM detections WHERE Sci_Name = '{safe}' AND {excl} \
         GROUP BY hour ORDER BY hour"
    );
    let mut stmt = duck.prepare(&sql)?;
    let rows = stmt.query_map([], |row| {
        Ok(HourlyCount {
            hour: row.get(0)?,
            count: row.get(1)?,
        })
    })?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

/// Per-species hourly breakdown for a given date.
pub async fn daily_species_hourly(
    db_path: &Path,
    date: &str,
) -> Res<Vec<SpeciesHourlyCounts>> {
    let overrides = read_overrides(db_path).await;
    let excl = exclusion_clause(&overrides);
    let safe_date = date.replace('\'', "''");
    let duck = conn()?;

    // 1) Species list for the day
    let list_sql = format!(
        "SELECT Sci_Name, Com_Name, COUNT(*) AS cnt \
         FROM detections WHERE Date = '{safe_date}' AND {excl} \
         GROUP BY Sci_Name, Com_Name ORDER BY cnt DESC"
    );
    let mut list_stmt = duck.prepare(&list_sql)?;
    let species: Vec<(String, String, u32)> = list_stmt
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?
        .filter_map(|r| r.ok())
        .collect();

    // 2) Per-species hourly counts
    let mut results = Vec::with_capacity(species.len());
    for (sci, com, total) in species {
        let safe_sci = sci.replace('\'', "''");
        let hour_sql = format!(
            "SELECT CAST(SUBSTR(Time, 1, 2) AS INTEGER) AS hour, COUNT(*) AS cnt \
             FROM detections WHERE Date = '{safe_date}' AND Sci_Name = '{safe_sci}' \
             GROUP BY hour ORDER BY hour"
        );
        let mut hour_stmt = duck.prepare(&hour_sql)?;
        let hours: Vec<HourlyCount> = hour_stmt
            .query_map([], |row| {
                Ok(HourlyCount {
                    hour: row.get(0)?,
                    count: row.get(1)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();

        results.push(SpeciesHourlyCounts {
            scientific_name: sci,
            common_name: com,
            total,
            hours,
        });
    }
    Ok(results)
}

/// Top species for a specific date, optionally filtered by model.
pub async fn top_species_for_date_filtered(
    db_path: &Path,
    date: &str,
    limit: u32,
    model_slug: Option<&str>,
) -> Res<Vec<SpeciesSummary>> {
    let overrides = read_overrides(db_path).await;
    let excl = exclusion_clause(&overrides);
    let safe_date = date.replace('\'', "''");
    let slug_filter = match model_slug {
        Some(s) if !s.is_empty() => format!("AND COALESCE(d.Model_Slug, '') = '{}'", s.replace('\'', "''")),
        _ => String::new(),
    };
    let duck = conn()?;
    let sql = format!(
        "SELECT d.Sci_Name, \
         COALESCE( \
             MAX(CASE WHEN d.Com_Name != d.Sci_Name THEN d.Com_Name ELSE NULL END), \
             MAX(d.Com_Name) \
         ) AS Com_Name, \
         d.Domain, COUNT(*) AS cnt, \
         MAX(d.Date || ' ' || d.Time) AS last \
         FROM detections d \
         WHERE d.Date = '{safe_date}' AND {excl} {slug_filter} \
         GROUP BY d.Sci_Name, d.Domain ORDER BY cnt DESC LIMIT {limit}"
    );
    let mut stmt = duck.prepare(&sql)?;
    let rows = stmt.query_map([], |row| {
        let count: u32 = row.get(3)?;
        Ok(SpeciesSummary {
            scientific_name: row.get(0)?,
            common_name: row.get(1)?,
            domain: row.get(2)?,
            detection_count: count,
            display_count: round_count(count),
            last_seen: row.get(4)?,
            image_url: None,
            conservation_status: None,
            male_image_url: None,
            female_image_url: None,
        })
    })?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

/// Top species for a date (unfiltered).
pub async fn top_species_for_date(
    db_path: &Path,
    date: &str,
    limit: u32,
) -> Res<Vec<SpeciesSummary>> {
    top_species_for_date_filtered(db_path, date, limit, None).await
}

/// Top species (all-time), optionally filtered by model slug.
pub async fn top_species_filtered(
    db_path: &Path,
    limit: u32,
    model_slug: Option<&str>,
) -> Res<Vec<SpeciesSummary>> {
    if !STATS_POPULATED.load(std::sync::atomic::Ordering::Relaxed) {
        refresh_species_stats(db_path).await?;
    }

    let has_model_filter = matches!(model_slug, Some(s) if !s.is_empty());

    // ── Fast path: read from in-memory cache tables ──────────────────
    if STATS_POPULATED.load(std::sync::atomic::Ordering::Relaxed) {
        if has_model_filter {
            let safe = model_slug.unwrap().replace('\'', "''");
            let duck = conn()?;
            let sql = format!(
                "SELECT Sci_Name, Com_Name, Domain, detection_count, last_seen \
                 FROM model_species_stats \
                 WHERE Model_Slug = '{safe}' \
                 ORDER BY detection_count DESC LIMIT {limit}"
            );
            return read_species_summaries(&duck, &sql);
        }
        let duck = conn()?;
        let sql = format!(
            "SELECT Sci_Name, Com_Name, Domain, detection_count, last_seen \
             FROM species_stats ORDER BY detection_count DESC LIMIT {limit}"
        );
        return read_species_summaries(&duck, &sql);
    }

    Ok(vec![])
}

/// Helper: execute a query returning (Sci_Name, Com_Name, Domain, count, last_seen)
/// and map to `SpeciesSummary`.
fn read_species_summaries(
    duck: &duckdb::Connection,
    sql: &str,
) -> Res<Vec<SpeciesSummary>> {
    let mut stmt = duck.prepare(sql)?;
    let rows = stmt.query_map([], |row| {
        let count: u32 = row.get(3)?;
        Ok(SpeciesSummary {
            scientific_name: row.get(0)?,
            common_name: row.get(1)?,
            domain: row.get(2)?,
            detection_count: count,
            display_count: round_count(count),
            last_seen: row.get(4)?,
            image_url: None,
            conservation_status: None,
            male_image_url: None,
            female_image_url: None,
        })
    })?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

/// Top species (all-time, unfiltered).
pub async fn top_species(db_path: &Path, limit: u32) -> Res<Vec<SpeciesSummary>> {
    top_species_filtered(db_path, limit, None).await
}

/// Top recordings for a species (by confidence).
pub async fn get_top_recordings(
    db_path: &Path,
    scientific_name: &str,
    limit: u32,
) -> Res<Vec<TopRecording>> {
    let tz = read_tz_offset(db_path).await;
    let safe = scientific_name.replace('\'', "''");
    let duck = conn()?;
    let sql = format!(
        "SELECT Sci_Name, Com_Name, Date, Time, Confidence, File_Name, \
         COALESCE(Source_Node, ''), COALESCE(Model_Name, '') \
         FROM detections \
         WHERE Sci_Name = '{safe}' AND COALESCE(Excluded, 0) = 0 \
           AND File_Name != '' AND Confidence >= 0.5 \
         ORDER BY Confidence DESC, Date DESC, Time DESC \
         LIMIT {limit}"
    );
    let mut stmt = duck.prepare(&sql)?;
    let rows = stmt.query_map([], |row| {
        Ok(TopRecording {
            scientific_name: row.get(0)?,
            common_name: row.get(1)?,
            date: row.get(2)?,
            time: row.get(3)?,
            confidence: row.get(4)?,
            file_name: row.get(5)?,
            source_node: row.get(6)?,
            model_name: row.get(7)?,
            display_date: String::new(),
            display_time: String::new(),
        })
    })?;
    let mut recs: Vec<TopRecording> = rows.filter_map(|r| r.ok()).collect();
    for r in &mut recs {
        stamp_recording(r, tz);
    }
    Ok(recs)
}

/// Distinct models that have produced detections.
pub async fn available_models(_db_path: &Path) -> Res<Vec<ModelInfo>> {
    let duck = conn()?;
    let sql = "SELECT COALESCE(Model_Slug, ''), COALESCE(Model_Name, '') \
               FROM detections \
               WHERE COALESCE(Model_Slug, '') != '' \
               GROUP BY Model_Slug, Model_Name \
               ORDER BY COUNT(*) DESC";
    let mut stmt = duck.prepare(sql)?;
    let rows = stmt.query_map([], |row| {
        Ok(ModelInfo {
            slug: row.get(0)?,
            name: row.get(1)?,
        })
    })?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

/// Species detections by model (or all models).
pub async fn species_detections_by_model(
    db_path: &Path,
    scientific_name: &str,
    limit: u32,
    model_slug: Option<&str>,
) -> Res<Vec<WebDetection>> {
    let tz = read_tz_offset(db_path).await;
    let safe = scientific_name.replace('\'', "''");
    let slug_filter = match model_slug {
        Some(s) if !s.is_empty() => format!("AND COALESCE(Model_Slug, '') = '{}'", s.replace('\'', "''")),
        _ => String::new(),
    };
    let duck = conn()?;
    let sql = format!(
        "SELECT id, Domain, Sci_Name, Com_Name, Confidence, Date, Time, File_Name, \
         COALESCE(Source_Node, ''), COALESCE(Excluded, 0), \
         COALESCE(Model_Slug, ''), COALESCE(Model_Name, ''), \
         COALESCE(Model_Beta, 0), \
         COALESCE(Agreement_Score, 0.0), COALESCE(Agreement_Models, '') \
         FROM detections WHERE Sci_Name = '{safe}' {slug_filter} \
         ORDER BY Date DESC, Time DESC LIMIT {limit}"
    );
    let mut stmt = duck.prepare(&sql)?;
    let rows = stmt.query_map([], |row| parse_detection(row))?;
    let mut dets: Vec<WebDetection> = rows.filter_map(|r| r.ok()).collect();
    for d in &mut dets {
        stamp(d, tz);
    }
    Ok(dets)
}

/// Excluded species summary.
pub async fn excluded_species(db_path: &Path) -> Res<Vec<ExcludedSpecies>> {
    if !STATS_POPULATED.load(std::sync::atomic::Ordering::Relaxed) {
        refresh_species_stats(db_path).await?;
    }

    let overrides = read_overrides(db_path).await;

    // ── Fast path: use in-memory cache table ─────────────────────────
    if STATS_POPULATED.load(std::sync::atomic::Ordering::Relaxed) {
        let duck = conn()?;
        let sql = "SELECT Sci_Name, Com_Name, Domain, detection_count, \
                   last_seen, max_confidence \
                   FROM excluded_species_stats \
                   ORDER BY detection_count DESC";
        let mut stmt = duck.prepare(sql)?;
        let rows = stmt.query_map([], |row| {
            let sci: String = row.get(0)?;
            Ok(ExcludedSpecies {
                scientific_name: sci.clone(),
                common_name: row.get(1)?,
                domain: row.get(2)?,
                detection_count: row.get(3)?,
                last_seen: row.get(4)?,
                max_confidence: row.get(5)?,
                image_url: None,
                overridden: false,
            })
        })?;
        let mut results: Vec<ExcludedSpecies> = rows.filter_map(|r| r.ok()).collect();
        for es in &mut results {
            es.overridden = overrides.contains(&es.scientific_name);
        }
        return Ok(results);
    }

    Ok(vec![])
}

/// Excluded detections for a specific species.
pub async fn excluded_detections_for_species(
    db_path: &Path,
    scientific_name: &str,
    limit: u32,
) -> Res<Vec<WebDetection>> {
    let tz = read_tz_offset(db_path).await;
    let safe = scientific_name.replace('\'', "''");
    let duck = conn()?;
    let sql = format!(
        "SELECT id, Domain, Sci_Name, Com_Name, Confidence, Date, Time, File_Name, \
         COALESCE(Source_Node, ''), COALESCE(Excluded, 0), \
         COALESCE(Model_Slug, ''), COALESCE(Model_Name, ''), \
         COALESCE(Model_Beta, 0), \
         COALESCE(Agreement_Score, 0.0), COALESCE(Agreement_Models, '') \
         FROM detections WHERE Sci_Name = '{safe}' AND COALESCE(Excluded, 0) = 1 \
         ORDER BY Date DESC, Time DESC LIMIT {limit}"
    );
    let mut stmt = duck.prepare(&sql)?;
    let rows = stmt.query_map([], |row| parse_detection(row))?;
    let mut dets: Vec<WebDetection> = rows.filter_map(|r| r.ok()).collect();
    for d in &mut dets {
        stamp(d, tz);
    }
    Ok(dets)
}

/// Refresh the species stats cache (materialised in DuckDB in-memory).
///
/// Unlike the old SQLite version, this uses DuckDB analytics directly
/// and stores the cache as in-memory tables within the DuckDB instance.
pub async fn refresh_species_stats(db_path: &Path) -> Res<()> {
    let overrides = read_overrides(db_path).await;
    let excl = exclusion_clause(&overrides);

    // Force a fresh view so the cache picks up the latest Parquet files.
    invalidate_view();
    let duck = conn()?;

    let t0 = std::time::Instant::now();

    // species_stats — used by the species list page (all models).
    duck.execute_batch(&format!(
        "CREATE OR REPLACE TABLE species_stats AS \
         SELECT Sci_Name, \
         COALESCE( \
             MAX(CASE WHEN Com_Name != Sci_Name THEN Com_Name ELSE NULL END), \
             MAX(Com_Name) \
         ) AS Com_Name, \
         Domain, COUNT(*) AS detection_count, \
         MAX(Date || ' ' || Time) AS last_seen \
         FROM detections d \
         WHERE {excl} \
         GROUP BY Sci_Name, Domain",
    ))?;

    // model_species_stats — used by the species list page (model filter).
    duck.execute_batch(&format!(
        "CREATE OR REPLACE TABLE model_species_stats AS \
         SELECT COALESCE(d.Model_Slug, '') AS Model_Slug, \
                d.Sci_Name, \
                COALESCE( \
                    MAX(CASE WHEN d.Com_Name != d.Sci_Name THEN d.Com_Name ELSE NULL END), \
                    MAX(d.Com_Name) \
                ) AS Com_Name, \
                d.Domain, \
                COUNT(*) AS detection_count, \
                MAX(d.Date || ' ' || d.Time) AS last_seen \
         FROM detections d \
         WHERE {excl} AND COALESCE(d.Model_Slug, '') != '' \
         GROUP BY d.Model_Slug, d.Sci_Name, d.Domain",
    ))?;

    // excluded_species_stats — used by the excluded species page.
    duck.execute_batch(
        "CREATE OR REPLACE TABLE excluded_species_stats AS \
         SELECT Sci_Name, \
                COALESCE( \
                    MAX(CASE WHEN Com_Name != Sci_Name THEN Com_Name ELSE NULL END), \
                    MAX(Com_Name) \
                ) AS Com_Name, \
                Domain, \
                COUNT(*) AS detection_count, \
                MAX(Date || ' ' || Time) AS last_seen, \
                MAX(Confidence) AS max_confidence \
         FROM detections \
         WHERE COALESCE(Excluded, 0) = 1 \
         GROUP BY Sci_Name, Domain",
    )?;

    // species_top_recordings (top 10 per species by confidence)
    duck.execute_batch(&format!(
        "CREATE OR REPLACE TABLE species_top_recordings AS \
         SELECT Sci_Name, Com_Name, Date, Time, Confidence, File_Name, \
                COALESCE(Source_Node, '') AS Source_Node, \
                COALESCE(Model_Name, '') AS Model_Name, rn AS rank \
         FROM ( \
             SELECT *, ROW_NUMBER() OVER ( \
                 PARTITION BY Sci_Name \
                 ORDER BY Confidence DESC, Date DESC, Time DESC \
             ) AS rn \
             FROM detections \
             WHERE COALESCE(Excluded, 0) = 0 AND File_Name != '' AND Confidence >= 0.5 \
         ) WHERE rn <= 10"
    ))?;

    STATS_POPULATED.store(true, std::sync::atomic::Ordering::Relaxed);
    STATS_REFRESHED_AT.store(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        std::sync::atomic::Ordering::Relaxed,
    );
    // Record current file count so the view-refresh auto-detect
    // doesn't immediately re-invalidate.
    if let Some(dir) = DET_DIR.get() {
        PARQUET_FILE_COUNT.store(count_parquet_files(dir), std::sync::atomic::Ordering::Relaxed);
    }
    info!("Species stats cache refreshed in {:.1}s", t0.elapsed().as_secs_f64());
    Ok(())
}

/// Notify the DuckDB layer that new detections have been written.
///
/// This invalidates the Parquet view throttle and triggers an async
/// cache refresh so the species / excluded pages see the new data
/// within a few seconds rather than waiting for the nightly rebuild.
pub async fn notify_new_detections(db_path: &Path) {
    invalidate_view();
    // Refresh cache tables in the background.  Errors are non-fatal.
    if let Err(e) = refresh_species_stats(db_path).await {
        tracing::warn!("Post-detection stats refresh failed: {e}");
    }
}

/// Quiz candidates — high-confidence, unambiguous detections for learning.
pub async fn quiz_candidates(
    _db_path: &Path,
    today_only: bool,
) -> Res<Vec<QuizItem>> {
    let today = super::kv::today_for_tz()
        .await;
    let duck = conn()?;

    let date_filter = if today_only {
        format!("AND Date = '{}'", today.replace('\'', "''"))
    } else {
        String::new()
    };

    let sql = format!(
        "WITH clean_files AS ( \
             SELECT File_Name FROM detections \
             WHERE File_Name != '' AND COALESCE(Excluded, 0) = 0 {date_filter} \
             GROUP BY File_Name \
             HAVING COUNT(DISTINCT Sci_Name) = 1 AND MAX(Confidence) >= 0.75 \
         ) \
         SELECT d.Sci_Name, d.Com_Name, d.Date, d.File_Name, d.Confidence \
         FROM detections d \
         INNER JOIN clean_files cf ON d.File_Name = cf.File_Name \
         WHERE d.Confidence >= 0.75 AND COALESCE(d.Excluded, 0) = 0 \
           AND d.File_Name != '' {date_filter} \
         ORDER BY RANDOM()"
    );
    let mut stmt = duck.prepare(&sql)?;
    let all_rows: Vec<(String, String, String, String, f64)> = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, f64>(4)?,
            ))
        })?
        .filter_map(|r| r.ok())
        .collect();

    // Pick one clip per species, up to 4 distinct species.
    let mut seen = std::collections::HashSet::new();
    let mut items = Vec::new();
    for (sci, com, date, file_name, _conf) in all_rows {
        if seen.contains(&sci) {
            continue;
        }
        seen.insert(sci.clone());

        let safe_name = com.replace('\'', "").replace(' ', "_");
        let clip_url = format!("/extracted/By_Date/{date}/{safe_name}/{file_name}");
        let spectrogram_url = format!("{clip_url}.png");

        items.push(QuizItem {
            scientific_name: sci,
            common_name: com,
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

/// Get species models (distinct Model_Slug/Model_Name for a species).
pub async fn get_species_models(
    _db_path: &Path,
    scientific_name: &str,
) -> Res<Vec<ModelInfo>> {
    let safe = scientific_name.replace('\'', "''");
    let duck = conn()?;
    let sql = format!(
        "SELECT COALESCE(Model_Slug, ''), COALESCE(Model_Name, ''), COUNT(*) AS cnt \
         FROM detections \
         WHERE Sci_Name = '{safe}' AND COALESCE(Model_Slug, '') != '' \
         GROUP BY Model_Slug, Model_Name ORDER BY cnt DESC"
    );
    let mut stmt = duck.prepare(&sql)?;
    let rows = stmt.query_map([], |row| {
        Ok(ModelInfo {
            slug: row.get(0)?,
            name: row.get(1)?,
        })
    })?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

/// Get all existing filenames in detections (used by import deduplication).
pub async fn get_existing_filenames() -> Res<Vec<String>> {
    let duck = conn()?;
    let sql = "SELECT DISTINCT File_Name FROM detections WHERE File_Name != ''";
    let mut stmt = duck.prepare(sql)?;
    let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

/// Total detection count.
pub async fn total_detections() -> Res<u64> {
    let duck = conn()?;
    let count: u64 = duck.query_row(
        "SELECT COUNT(*) FROM detections",
        [],
        |row| row.get(0),
    )?;
    Ok(count)
}

/// Return the detections directory path (for import Parquet writes).
pub fn get_detections_dir() -> Option<PathBuf> {
    DET_DIR.get().cloned()
}

// ─── One-time SQLite → Parquet migration ─────────────────────────────────────

/// Migrate existing SQLite detections to Parquet files.
///
/// Called once at startup.  If the detections directory already contains a
/// `.migrated` marker file the function is a no-op.  Otherwise it reads
/// every row from the SQLite `detections` table and writes a single
/// ZSTD-compressed Parquet file, then drops the marker so subsequent
/// starts skip the migration.
///
/// Columns that may be absent in older databases (`Model_Slug`,
/// `Model_Name`, `Source_Node`, `Excluded`) are handled with `COALESCE`
/// defaults.
pub async fn migrate_sqlite_to_parquet(db_path: &Path) -> Result<(), String> {
    let det_dir = match DET_DIR.get() {
        Some(d) => d.clone(),
        None => return Err("DuckDB not initialised yet".into()),
    };

    let marker = det_dir.join(".migrated");
    if marker.exists() {
        info!("SQLite→Parquet migration already done (marker present)");
        return Ok(());
    }

    // Check if the SQLite detections table exists and has rows.
    let conn = super::db::open_conn(db_path).await
        .map_err(|e| format!("Cannot open SQLite for migration: {e}"))?;

    let count: i64 = {
        let mut rows = conn
            .query(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='detections'",
                (),
            )
            .await
            .map_err(|e| format!("Schema check error: {e}"))?;
        match rows.next().await {
            Ok(Some(row)) => row.get::<i64>(0).unwrap_or(0),
            _ => 0,
        }
    };
    if count == 0 {
        info!("No SQLite detections table found — nothing to migrate");
        std::fs::write(&marker, b"no-table").ok();
        return Ok(());
    }

    let total: i64 = {
        let mut rows = conn
            .query("SELECT COUNT(*) FROM detections", ())
            .await
            .map_err(|e| format!("Count error: {e}"))?;
        match rows.next().await {
            Ok(Some(row)) => row.get::<i64>(0).unwrap_or(0),
            _ => 0,
        }
    };
    if total == 0 {
        info!("SQLite detections table is empty — nothing to migrate");
        std::fs::write(&marker, b"empty").ok();
        return Ok(());
    }

    info!("Migrating {total} detections from SQLite → Parquet…");

    // Read all rows from SQLite.
    // Use COALESCE for columns that may not exist in older schemas.
    // We try the full column set first; if that fails we fall back to
    // the minimal column set without model columns.
    let full_sql = "SELECT \
        rowid, Date, Time, \
        COALESCE(Domain, 'birds'), Sci_Name, Com_Name, Confidence, \
        COALESCE(Lat, 0.0), COALESCE(Lon, 0.0), COALESCE(Cutoff, 0.0), \
        COALESCE(Week, 0), COALESCE(Sens, 1.0), COALESCE(Overlap, 0.0), \
        File_Name, \
        COALESCE(Source_Node, ''), COALESCE(Excluded, 0), \
        COALESCE(Model_Slug, ''), COALESCE(Model_Name, '') \
        FROM detections ORDER BY Date, Time";

    let mut rows_result = conn.query(full_sql, ()).await;

    // If the full query fails (missing columns), try without model columns.
    let has_model_cols = rows_result.is_ok();
    if !has_model_cols {
        let fallback_sql = "SELECT \
            rowid, Date, Time, \
            COALESCE(Domain, 'birds'), Sci_Name, Com_Name, Confidence, \
            COALESCE(Lat, 0.0), COALESCE(Lon, 0.0), COALESCE(Cutoff, 0.0), \
            COALESCE(Week, 0), COALESCE(Sens, 1.0), COALESCE(Overlap, 0.0), \
            File_Name, \
            COALESCE(Source_Node, ''), COALESCE(Excluded, 0) \
            FROM detections ORDER BY Date, Time";
        rows_result = conn.query(fallback_sql, ()).await;
    }

    let mut src_rows = rows_result
        .map_err(|e| format!("SQLite query error: {e}"))?;

    // Buffer into an in-memory DuckDB table.
    let duck = duckdb::Connection::open_in_memory()
        .map_err(|e| format!("DuckDB migration open: {e}"))?;
    duck.execute_batch(
        "CREATE TABLE buffer (
            id          BIGINT   NOT NULL,
            Date        VARCHAR  NOT NULL,
            Time        VARCHAR  NOT NULL,
            Domain      VARCHAR  NOT NULL,
            Sci_Name    VARCHAR  NOT NULL,
            Com_Name    VARCHAR  NOT NULL,
            Confidence  DOUBLE   NOT NULL,
            Lat         DOUBLE   NOT NULL,
            Lon         DOUBLE   NOT NULL,
            Cutoff      DOUBLE   NOT NULL,
            Week        INTEGER  NOT NULL,
            Sens        DOUBLE   NOT NULL,
            Overlap     DOUBLE   NOT NULL,
            File_Name   VARCHAR  NOT NULL,
            Source_Node VARCHAR  NOT NULL,
            Excluded    INTEGER  NOT NULL,
            Model_Slug  VARCHAR  NOT NULL,
            Model_Name  VARCHAR  NOT NULL,
            Model_Beta  INTEGER  NOT NULL,
            Agreement_Score  DOUBLE  NOT NULL,
            Agreement_Models VARCHAR NOT NULL
        )",
    )
    .map_err(|e| format!("DuckDB buffer schema: {e}"))?;

    let base_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let mut seq: u64 = 0;
    let mut migrated: u64 = 0;

    while let Some(row) = src_rows
        .next()
        .await
        .map_err(|e| format!("Row iteration: {e}"))?
    {
        let date: String = row.get(1).unwrap_or_default();
        let time: String = row.get(2).unwrap_or_default();
        let domain: String = row.get(3).unwrap_or_else(|_| "birds".into());
        let sci: String = row.get(4).unwrap_or_default();
        let com: String = row.get(5).unwrap_or_default();
        let conf: f64 = row.get(6).unwrap_or(0.0);
        let lat: f64 = row.get(7).unwrap_or(0.0);
        let lon: f64 = row.get(8).unwrap_or(0.0);
        let cutoff: f64 = row.get(9).unwrap_or(0.0);
        let week: i32 = row.get::<i64>(10).unwrap_or(0) as i32;
        let sens: f64 = row.get(11).unwrap_or(1.0);
        let overlap: f64 = row.get(12).unwrap_or(0.0);
        let fname: String = row.get(13).unwrap_or_default();
        let source: String = row.get(14).unwrap_or_default();
        let excluded: i32 = row.get::<i64>(15).unwrap_or(0) as i32;
        let model_slug: String = if has_model_cols {
            row.get(16).unwrap_or_default()
        } else {
            String::new()
        };
        let model_name: String = if has_model_cols {
            row.get(17).unwrap_or_default()
        } else {
            String::new()
        };

        seq += 1;
        let id = ((base_ms & 0xFFFF_FFFF_FFFF) << 16) | (seq & 0xFFFF);

        if let Err(e) = duck.execute(
            "INSERT INTO buffer VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            duckdb::params![
                id as i64, date, time, domain, sci, com, conf,
                lat, lon, cutoff, week, sens, overlap, fname,
                source, excluded, model_slug, model_name, 0i32,
                0.0f64, String::new()
            ],
        ) {
            tracing::warn!("Migration row insert error: {e}");
            continue;
        }
        migrated += 1;

        if migrated % 50_000 == 0 {
            info!("Migration progress: {migrated}/{total} rows buffered…");
        }
    }

    if migrated == 0 {
        info!("No rows migrated (all empty or errored)");
        std::fs::write(&marker, b"0-rows").ok();
        return Ok(());
    }

    // Flush to Parquet.
    let ts = chrono::Utc::now().format("%Y%m%d-%H%M%S");
    let filename = format!("migration-{ts}.parquet");
    let final_path = det_dir.join(&filename);
    let tmp_path = det_dir.join(format!(".{filename}.tmp"));

    duck.execute(
        &format!(
            "COPY buffer TO '{}' (FORMAT PARQUET, COMPRESSION ZSTD)",
            tmp_path.display()
        ),
        [],
    )
    .map_err(|e| format!("Parquet write: {e}"))?;

    std::fs::rename(&tmp_path, &final_path)
        .map_err(|e| format!("Rename: {e}"))?;

    info!(
        "Migration complete: {migrated} detections → {}",
        final_path.display()
    );

    // Refresh the view so the migrated data is immediately queryable.
    if let Ok(guard) = DUCK
        .get()
        .ok_or("DuckDB not init")
        .and_then(|m| m.lock().map_err(|_| "lock poisoned"))
    {
        refresh_view(&guard);
    }

    // Write marker so we don't re-migrate next startup.
    std::fs::write(&marker, format!("{migrated}").as_bytes()).ok();

    Ok(())
}

/// Get all existing filenames as a `HashSet`, expanded with Opus/mp3/wav
/// variants for deduplication during import.
///
/// This is a **sync** function so it can be called from `spawn_blocking` or
/// any non-async import context.
pub fn get_existing_filenames_set() -> Result<std::collections::HashSet<String>, String> {
    let duck = conn()?;
    let sql = "SELECT DISTINCT File_Name FROM detections WHERE File_Name != ''";
    let mut stmt = duck.prepare(sql).map_err(|e| format!("DuckDB prepare: {e}"))?;
    let rows = stmt
        .query_map([], |row| row.get::<_, String>(0))
        .map_err(|e| format!("DuckDB query: {e}"))?;

    let mut names: std::collections::HashSet<String> = rows.filter_map(|r| r.ok()).collect();

    // For every .opus filename, also add .mp3 and .wav so the source
    // BirdNET-Pi filenames (which are always .mp3) still match.
    let opus_variants: Vec<String> = names
        .iter()
        .filter(|n| n.ends_with(".opus"))
        .flat_map(|n| {
            let stem = &n[..n.len() - 5];
            [format!("{stem}.mp3"), format!("{stem}.wav")]
        })
        .collect();
    names.extend(opus_variants);

    Ok(names)
}

// ─── Parquet Sci_Name normalisation migration ────────────────────────────────

/// One-time migration: normalise `Sci_Name` in existing Parquet files and
/// pick the best `Com_Name` per species (preferring a real common name
/// over one that just repeats the scientific name).
///
/// Writes a marker file (`.sci_name_normalised`) so it only runs once.
fn normalize_parquet_sci_names(detections_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let marker = detections_dir.join(".sci_name_normalised");
    if marker.exists() {
        return Ok(());
    }

    let has_files = std::fs::read_dir(detections_dir)
        .map(|rd| {
            rd.flatten()
                .any(|e| e.path().extension().map(|x| x == "parquet").unwrap_or(false))
        })
        .unwrap_or(false);

    if !has_files {
        std::fs::write(&marker, "no files to migrate\n").ok();
        return Ok(());
    }

    info!(
        "Running Sci_Name normalisation migration on {}",
        detections_dir.display()
    );

    let t0 = std::time::Instant::now();
    let glob = format!("{}/*.parquet", detections_dir.display());

    let conn = duckdb::Connection::open_in_memory()
        .map_err(|e| format!("migration DuckDB: {e}"))?;

    conn.execute_batch(&format!(
        "CREATE TABLE migration AS SELECT * FROM read_parquet('{glob}', union_by_name=true)"
    ))
    .map_err(|e| format!("read parquet: {e}"))?;

    let count: u64 = conn
        .query_row("SELECT COUNT(*) FROM migration", [], |row| row.get(0))
        .unwrap_or(0);

    if count == 0 {
        std::fs::write(&marker, "0 rows\n").ok();
        return Ok(());
    }

    // Normalise: replace underscores → spaces, capitalise only genus.
    conn.execute_batch(
        "ALTER TABLE migration ADD COLUMN Sci_Norm VARCHAR; \
         UPDATE migration SET Sci_Norm = \
             CONCAT( \
                 UPPER(LEFT(TRIM(REPLACE(Sci_Name, '_', ' ')), 1)), \
                 LOWER(SUBSTR(TRIM(REPLACE(Sci_Name, '_', ' ')), 2)) \
             );"
    ).map_err(|e| format!("normalise: {e}"))?;

    // Best common name per species: prefer real names over sci_name copies.
    conn.execute_batch(
        "CREATE TABLE best_names AS \
         SELECT Sci_Norm, \
             COALESCE( \
                 MAX(CASE WHEN TRIM(Com_Name) != '' \
                          AND TRIM(Com_Name) != Sci_Name \
                          AND TRIM(Com_Name) != Sci_Norm \
                     THEN Com_Name ELSE NULL END), \
                 MAX(Com_Name) \
             ) AS Best_Com_Name \
         FROM migration \
         GROUP BY Sci_Norm"
    ).map_err(|e| format!("best names: {e}"))?;

    conn.execute_batch(
        "UPDATE migration SET \
             Sci_Name = Sci_Norm, \
             Com_Name = COALESCE( \
                 (SELECT b.Best_Com_Name FROM best_names b WHERE b.Sci_Norm = migration.Sci_Norm), \
                 Com_Name \
             )"
    ).map_err(|e| format!("update: {e}"))?;

    conn.execute_batch(
        "ALTER TABLE migration DROP COLUMN Sci_Norm; \
         DROP TABLE best_names;"
    ).ok();

    // Write corrected data.
    let out_path = detections_dir.join("_migrated.parquet");
    let out_tmp = detections_dir.join("._migrated.parquet.tmp");
    conn.execute(
        &format!(
            "COPY migration TO '{}' (FORMAT PARQUET, COMPRESSION ZSTD)",
            out_tmp.display()
        ),
        [],
    )
    .map_err(|e| format!("write: {e}"))?;

    std::fs::rename(&out_tmp, &out_path)
        .map_err(|e| format!("rename: {e}"))?;

    // Remove old files.
    let mut removed = 0u32;
    for entry in std::fs::read_dir(detections_dir)?.flatten() {
        let path = entry.path();
        if path.extension().map(|x| x == "parquet").unwrap_or(false)
            && path.file_name() != Some(std::ffi::OsStr::new("_migrated.parquet"))
        {
            std::fs::remove_file(&path).ok();
            removed += 1;
        }
    }

    info!(
        "Migration complete: {count} rows, removed {removed} old file(s), \
         wrote _migrated.parquet ({:.1}s)",
        t0.elapsed().as_secs_f64()
    );

    std::fs::write(&marker, format!("{count} rows migrated\n")).ok();
    Ok(())
}
