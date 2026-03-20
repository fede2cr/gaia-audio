//! Key-value layer backed by Valkey / Redis (async, SSR-only).
//!
//! Mirrors the processing crate's `kv.rs` but uses the async Redis API
//! so it integrates naturally with the Axum / Leptos SSR runtime.
//!
//! ## Data model
//!
//! See `processing/src/kv.rs` for the canonical key-pattern reference.

use std::collections::HashMap;
use std::sync::OnceLock;

use redis::AsyncCommands;
use tracing::info;

use crate::model::{SpeciesVerification, UrbanNoiseSummary};

// ── Connection management ────────────────────────────────────────────────────

static REDIS: OnceLock<redis::aio::MultiplexedConnection> = OnceLock::new();

fn redis_url() -> String {
    std::env::var("REDIS_URL")
        .unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string())
}

/// Open the Redis connection.  Must be called once at startup.
///
/// Retries for up to 30 seconds so the web container can start
/// before Valkey is fully ready.
pub async fn initialize() -> Result<(), String> {
    let url = redis_url();
    info!("Connecting to Redis: {url}");
    let client = redis::Client::open(url.as_str())
        .map_err(|e| format!("Cannot parse Redis URL: {e}"))?;

    let mut attempt = 0u32;
    let conn = loop {
        attempt += 1;
        match client.get_multiplexed_tokio_connection().await {
            Ok(c) => break c,
            Err(_e) if attempt < 30 => {
                if attempt == 1 {
                    info!("Waiting for Redis…");
                }
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            }
            Err(e) => {
                return Err(format!(
                    "Cannot connect to Redis after {attempt} attempts: {e}"
                ))
            }
        }
    };

    let _ = REDIS.set(conn);
    info!("Redis connection ready");
    Ok(())
}

/// Get a clone of the multiplexed connection (cheap, lock-free).
fn conn() -> redis::aio::MultiplexedConnection {
    REDIS
        .get()
        .expect("kv: Redis not initialized — call kv::initialize() first")
        .clone()
}

// ── Settings ─────────────────────────────────────────────────────────────────

/// Read all settings as a key-value map.
pub async fn get_all_settings() -> Result<HashMap<String, String>, String> {
    let mut c = conn();
    let map: HashMap<String, String> = c
        .hgetall("settings")
        .await
        .map_err(|e| format!("Redis error: {e}"))?;
    Ok(map)
}

/// Save multiple settings atomically.
pub async fn save_settings(entries: &[(&str, &str)]) -> Result<(), String> {
    let mut c = conn();
    let mut pipe = redis::pipe();
    for (k, v) in entries {
        pipe.hset("settings", *k, *v);
    }
    pipe.query_async::<()>(&mut c)
        .await
        .map_err(|e| format!("Redis error: {e}"))?;
    Ok(())
}

// ── TZ & today ───────────────────────────────────────────────────────────────

/// Read the TZ offset (hours) from the settings hash.
pub async fn read_tz_offset() -> i32 {
    let mut c = conn();
    let val: Option<String> = c.hget("settings", "tz_offset").await.ok();
    val.and_then(|v| v.parse::<i32>().ok()).unwrap_or(0)
}

/// Return today's date string (YYYY-MM-DD) adjusted by the user's TZ offset.
pub async fn today_for_tz() -> String {
    let offset = read_tz_offset().await;
    let utc_now = chrono::Utc::now().naive_utc();
    let local_now = utc_now + chrono::Duration::hours(offset as i64);
    local_now.format("%Y-%m-%d").to_string()
}

// ── Exclusion overrides ──────────────────────────────────────────────────────

/// Add an exclusion override (ornithologist confirms the species).
pub async fn add_exclusion_override(sci_name: &str, notes: &str) -> Result<(), String> {
    let mut c = conn();
    let now = chrono::Utc::now()
        .format("%Y-%m-%d %H:%M:%S")
        .to_string();
    let value = format!("{now}|{notes}");
    c.hset::<_, _, _, ()>("exclusion_overrides", sci_name, &value)
        .await
        .map_err(|e| format!("Redis error: {e}"))?;
    Ok(())
}

/// Remove an exclusion override.
pub async fn remove_exclusion_override(sci_name: &str) -> Result<(), String> {
    let mut c = conn();
    c.hdel::<_, _, ()>("exclusion_overrides", sci_name)
        .await
        .map_err(|e| format!("Redis error: {e}"))?;
    Ok(())
}

/// Read all excluded species names.
pub async fn read_overrides() -> Vec<String> {
    let mut c = conn();
    c.hkeys("exclusion_overrides").await.unwrap_or_default()
}

// ── Urban noise ──────────────────────────────────────────────────────────────

/// Aggregated urban-noise counts per category (today, last 7 days, all-time).
pub async fn urban_noise_summary() -> Result<Vec<UrbanNoiseSummary>, String> {
    let mut c = conn();
    let today = today_for_tz().await;
    let offset = read_tz_offset().await;

    // All-time totals
    let totals: HashMap<String, i64> = c
        .hgetall("urban_noise:total")
        .await
        .unwrap_or_default();

    // Today
    let today_key = format!("urban_noise:day:{today}");
    let today_counts: HashMap<String, i64> = c
        .hgetall(&today_key)
        .await
        .unwrap_or_default();

    // Last 7 days
    let mut week_counts: HashMap<String, i64> = HashMap::new();
    for i in 0..7i64 {
        let date = {
            let utc_now = chrono::Utc::now().naive_utc();
            let local_now =
                utc_now + chrono::Duration::hours(offset as i64) - chrono::Duration::days(i);
            local_now.format("%Y-%m-%d").to_string()
        };
        let day_key = format!("urban_noise:day:{date}");
        let day: HashMap<String, i64> = c.hgetall(&day_key).await.unwrap_or_default();
        for (cat, cnt) in day {
            *week_counts.entry(cat).or_insert(0) += cnt;
        }
    }

    let mut results: Vec<UrbanNoiseSummary> = totals
        .iter()
        .map(|(cat, &total)| UrbanNoiseSummary {
            category: cat.clone(),
            total_count: total as u32,
            today_count: *today_counts.get(cat).unwrap_or(&0) as u32,
            week_count: *week_counts.get(cat).unwrap_or(&0) as u32,
        })
        .collect();
    results.sort_by(|a, b| b.total_count.cmp(&a.total_count));
    Ok(results)
}

// ── Species verification ─────────────────────────────────────────────────────

/// Get verification record for a species (if any).
pub async fn get_species_verification(
    sci_name: &str,
) -> Result<Option<SpeciesVerification>, String> {
    let mut c = conn();
    let key = format!("verification:{sci_name}");
    let map: HashMap<String, String> = c.hgetall(&key).await.unwrap_or_default();
    if map.is_empty() {
        return Ok(None);
    }
    Ok(Some(SpeciesVerification {
        method: map.get("method").cloned().unwrap_or_default(),
        inaturalist_obs: map.get("inaturalist_obs").cloned().unwrap_or_default(),
        verified_at: map.get("verified_at").cloned().unwrap_or_default(),
    }))
}

/// Save a verification record for a species.
pub async fn set_species_verification(
    sci_name: &str,
    method: &str,
    inaturalist_obs: &str,
) -> Result<(), String> {
    let mut c = conn();
    let key = format!("verification:{sci_name}");
    let now = chrono::Utc::now()
        .format("%Y-%m-%d %H:%M:%S")
        .to_string();
    redis::pipe()
        .hset(&key, "method", method)
        .hset(&key, "inaturalist_obs", inaturalist_obs)
        .hset(&key, "verified_at", &now)
        .query_async::<()>(&mut c)
        .await
        .map_err(|e| format!("Redis error: {e}"))?;
    Ok(())
}

/// Remove verification for a species.
pub async fn remove_species_verification(sci_name: &str) -> Result<(), String> {
    let mut c = conn();
    c.del::<_, ()>(&format!("verification:{sci_name}"))
        .await
        .map_err(|e| format!("Redis error: {e}"))?;
    Ok(())
}

// ── SQLite → Redis migration ─────────────────────────────────────────────────

/// One-time migration: copy settings, overrides, verifications, and
/// urban-noise counters from SQLite to Redis.
///
/// Uses a marker key (`kv:migrated`) in Redis itself so the migration
/// runs exactly once.
pub async fn migrate_from_sqlite(db_path: &std::path::Path) -> Result<(), String> {
    let mut c = conn();

    // Check marker
    let done: bool = c.exists("kv:migrated").await.unwrap_or(false);
    if done {
        info!("SQLite→Redis migration already done (marker present)");
        return Ok(());
    }

    info!("Migrating SQLite OLTP data → Redis…");

    let sqlite_conn = super::db::open_conn(db_path)
        .await
        .map_err(|e| format!("Cannot open SQLite for migration: {e}"))?;

    // ── settings ─────────────────────────────────────────────────────
    if let Ok(mut rows) = sqlite_conn
        .query("SELECT key, value FROM settings", ())
        .await
    {
        let mut pipe = redis::pipe();
        let mut count = 0u32;
        while let Ok(Some(row)) = rows.next().await {
            if let (Ok(k), Ok(v)) = (row.get::<String>(0), row.get::<String>(1)) {
                pipe.hset("settings", k, v);
                count += 1;
            }
        }
        if count > 0 {
            let _ = pipe.query_async::<()>(&mut c).await;
            info!("  Migrated {count} setting(s)");
        }
    }

    // ── exclusion_overrides ──────────────────────────────────────────
    if let Ok(mut rows) = sqlite_conn
        .query(
            "SELECT Sci_Name, COALESCE(overridden_at, ''), COALESCE(notes, '') \
             FROM exclusion_overrides",
            (),
        )
        .await
    {
        let mut pipe = redis::pipe();
        let mut count = 0u32;
        while let Ok(Some(row)) = rows.next().await {
            if let (Ok(sci), Ok(at), Ok(notes)) = (
                row.get::<String>(0),
                row.get::<String>(1),
                row.get::<String>(2),
            ) {
                let val = format!("{at}|{notes}");
                pipe.hset("exclusion_overrides", sci, val);
                count += 1;
            }
        }
        if count > 0 {
            let _ = pipe.query_async::<()>(&mut c).await;
            info!("  Migrated {count} exclusion override(s)");
        }
    }

    // ── species_verifications ────────────────────────────────────────
    if let Ok(mut rows) = sqlite_conn
        .query(
            "SELECT Sci_Name, COALESCE(method, 'ornithologist'), \
             COALESCE(inaturalist_obs, ''), COALESCE(verified_at, '') \
             FROM species_verifications",
            (),
        )
        .await
    {
        let mut count = 0u32;
        while let Ok(Some(row)) = rows.next().await {
            if let (Ok(sci), Ok(method), Ok(obs), Ok(at)) = (
                row.get::<String>(0),
                row.get::<String>(1),
                row.get::<String>(2),
                row.get::<String>(3),
            ) {
                let key = format!("verification:{sci}");
                let _ = redis::pipe()
                    .hset(&key, "method", &method)
                    .hset(&key, "inaturalist_obs", &obs)
                    .hset(&key, "verified_at", &at)
                    .query_async::<()>(&mut c)
                    .await;
                count += 1;
            }
        }
        if count > 0 {
            info!("  Migrated {count} species verification(s)");
        }
    }

    // ── urban_noise ──────────────────────────────────────────────────
    if let Ok(mut rows) = sqlite_conn
        .query("SELECT Date, Hour, Category, Count FROM urban_noise", ())
        .await
    {
        let mut total_pipe = redis::pipe();
        let mut count = 0u32;
        // Accumulate per-day and total
        let mut day_map: HashMap<String, HashMap<String, i64>> = HashMap::new();
        let mut total_map: HashMap<String, i64> = HashMap::new();

        while let Ok(Some(row)) = rows.next().await {
            if let (Ok(date), Ok(_hour), Ok(cat), Ok(cnt)) = (
                row.get::<String>(0),
                row.get::<i32>(1),
                row.get::<String>(2),
                row.get::<i64>(3),
            ) {
                *total_map.entry(cat.clone()).or_insert(0) += cnt;
                *day_map
                    .entry(date)
                    .or_default()
                    .entry(cat)
                    .or_insert(0) += cnt;
                count += 1;
            }
        }

        // Write totals
        for (cat, cnt) in &total_map {
            total_pipe.hset("urban_noise:total", cat.as_str(), *cnt);
        }
        if !total_map.is_empty() {
            let _ = total_pipe.query_async::<()>(&mut c).await;
        }

        // Write day counters (with TTL)
        for (date, cats) in &day_map {
            let day_key = format!("urban_noise:day:{date}");
            let mut pipe = redis::pipe();
            for (cat, cnt) in cats {
                pipe.hset(&day_key, cat.as_str(), *cnt);
            }
            pipe.expire(&day_key, 30 * 24 * 3600);
            let _ = pipe.query_async::<()>(&mut c).await;
        }

        if count > 0 {
            info!("  Migrated {count} urban-noise row(s)");
        }
    }

    // ── set marker ───────────────────────────────────────────────────
    let _: () = c
        .set("kv:migrated", "1")
        .await
        .map_err(|e| format!("Cannot set migration marker: {e}"))?;
    info!("SQLite→Redis migration complete");
    Ok(())
}
