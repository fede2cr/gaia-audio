//! Key-value coordination layer backed by Valkey / Redis.
//!
//! Replaces the SQLite-based coordination in `db.rs` with atomic Redis
//! operations.  Because every Redis command is inherently atomic and
//! lock-free, multiple processing containers can coordinate without
//! "database is locked" errors or retry loops.
//!
//! ## Data model
//!
//! | Key pattern                      | Type | Purpose                          |
//! |----------------------------------|------|----------------------------------|
//! | `settings`                       | HASH | Runtime tuning knobs             |
//! | `exclusion_overrides`            | HASH | Sci_Name → "overridden_at\|notes"|
//! | `instances`                      | HASH | instance_id → unix_timestamp     |
//! | `processed:{filename}`           | SET  | instance IDs (TTL 1 h)           |
//! | `urban_noise:total`              | HASH | category → count (all-time)      |
//! | `urban_noise:day:{YYYY-MM-DD}`   | HASH | category → count (TTL 30 d)      |
//! | `verification:{Sci_Name}`        | HASH | method, inaturalist_obs, …       |

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use redis::Commands;
use tracing::{debug, info, warn};

// ── Connection management ────────────────────────────────────────────────────

static CLIENT: OnceLock<redis::Client> = OnceLock::new();
static CONN: OnceLock<Mutex<redis::Connection>> = OnceLock::new();

fn redis_url() -> String {
    std::env::var("REDIS_URL")
        .unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string())
}

/// Open the Redis connection.  Must be called once at startup.
///
/// Retries indefinitely with exponential back-off (1 s → 30 s cap)
/// so the processing container survives slow Valkey starts or brief
/// outages.
pub fn initialize() -> Result<()> {
    let url = redis_url();
    info!("Connecting to Redis: {url}");
    let client = redis::Client::open(url.as_str())
        .with_context(|| format!("Cannot parse Redis URL: {url}"))?;

    let mut attempt = 0u32;
    let mut backoff = std::time::Duration::from_secs(1);
    let max_backoff = std::time::Duration::from_secs(30);
    let conn = loop {
        attempt += 1;
        match client.get_connection() {
            Ok(c) => break c,
            Err(e) => {
                if attempt == 1 {
                    info!("Waiting for Redis…");
                }
                if attempt % 10 == 0 {
                    warn!("Still waiting for Redis (attempt {attempt}): {e}");
                }
                std::thread::sleep(backoff);
                backoff = (backoff * 2).min(max_backoff);
            }
        }
    };

    let _ = CLIENT.set(client);
    let _ = CONN.set(Mutex::new(conn));
    info!("Redis connection ready");
    Ok(())
}

/// Get the cached connection.
fn conn() -> std::sync::MutexGuard<'static, redis::Connection> {
    CONN.get()
        .expect("kv: Redis not initialized — call kv::initialize() first")
        .lock()
        .unwrap()
}

/// Reconnect if the connection was lost.
fn reconnect() {
    if let Some(client) = CLIENT.get() {
        if let Ok(new_conn) = client.get_connection() {
            if let Some(mutex) = CONN.get() {
                *mutex.lock().unwrap() = new_conn;
                info!("Redis reconnected");
            }
        }
    }
}

/// Execute a closure with one auto-reconnect attempt on failure.
fn with_retry<T, F>(f: F) -> redis::RedisResult<T>
where
    F: Fn(&mut redis::Connection) -> redis::RedisResult<T>,
{
    let result = {
        let mut c = conn();
        f(&mut c)
    };
    match result {
        Ok(v) => Ok(v),
        Err(_) => {
            reconnect();
            let mut c = conn();
            f(&mut c)
        }
    }
}

// ── Constants ────────────────────────────────────────────────────────────────

/// TTL for daily urban-noise counters — 30 days.
const URBAN_NOISE_DAY_TTL_SECS: i64 = 30 * 24 * 3600;

/// Labels that the BirdNET model emits which are not actual bird species.
/// These are counted in the urban-noise counters instead of detections.
pub const URBAN_NOISE_LABELS: &[&str] = &[
    "Engine",
    "Dog",
    "Human",
    "Human vocal",
    "Human whistle",
    "Human_Human",
    "Power tools",
    "Siren",
    "Gun",
    "Fireworks",
    "Noise",
    "Environmental",
];

/// Returns `true` if the given scientific name is a known urban-noise label.
pub fn is_urban_noise(sci_name: &str) -> bool {
    URBAN_NOISE_LABELS
        .iter()
        .any(|&label| sci_name.eq_ignore_ascii_case(label))
}

// ── Instance coordination ────────────────────────────────────────────────────

fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

/// Register this processing instance.
pub fn register_instance(instance: &str) -> Result<()> {
    with_retry(|c| c.hset::<_, _, _, ()>("instances", instance, now_unix()))
        .context("register_instance")?;
    debug!("Registered processing instance {instance:?}");
    Ok(())
}

/// Update the heartbeat timestamp for this instance.
pub fn update_heartbeat(instance: &str) {
    if let Err(e) = with_retry(|c| c.hset::<_, _, _, ()>("instances", instance, now_unix())) {
        warn!("update_heartbeat failed: {e}");
    }
}

/// Remove instances whose heartbeat is older than `stale_minutes`.
pub fn prune_stale_instances(stale_minutes: u32) -> usize {
    let cutoff = now_unix() - (stale_minutes as i64) * 60;
    let instances: HashMap<String, i64> = match with_retry(|c| c.hgetall("instances")) {
        Ok(m) => m,
        Err(_) => return 0,
    };

    let stale: Vec<String> = instances
        .iter()
        .filter(|(_, &ts)| ts < cutoff)
        .map(|(id, _)| id.clone())
        .collect();

    if stale.is_empty() {
        return 0;
    }

    let removed = stale.len();
    let _ = with_retry(|c| c.hdel::<_, _, ()>("instances", stale.clone()));
    info!("Pruned {removed} stale processing instance(s) (no heartbeat in >{stale_minutes} min)");
    removed
}

// ── File processing coordination ─────────────────────────────────────────────

/// Check whether this instance has already processed a specific file.
pub fn is_file_processed(filename: &str, instance: &str) -> bool {
    let key = format!("processed:{filename}");
    with_retry(|c| c.sismember(&key, instance)).unwrap_or(false)
}

/// Clean up old processing log entries.
///
/// With Redis TTL on `processed:*` keys, entries expire automatically.
/// This function is a no-op kept for caller compatibility.
pub fn cleanup_processing_log() {
    // TTL handles cleanup — nothing to do.
}

// ── Urban noise ──────────────────────────────────────────────────────────────

/// Increment the urban-noise counter for a category / date / hour.
///
/// Writes both an all-time total and a per-day counter (with 30-day TTL)
/// so the web dashboard can aggregate by timeframe.
pub fn increment_urban_noise(date: &str, _hour: u32, category: &str) -> Result<()> {
    let day_key = format!("urban_noise:day:{date}");
    redis::pipe()
        .hincr("urban_noise:total", category, 1i64)
        .hincr(&day_key, category, 1i64)
        .expire(&day_key, URBAN_NOISE_DAY_TTL_SECS)
        .exec(&mut *conn())
        .context("increment_urban_noise")?;
    Ok(())
}

// ── Settings ─────────────────────────────────────────────────────────────────
/// Read the set of enabled audio model slugs from Redis.
///
/// Returns:
/// - `None` when the key does not exist (legacy behavior: treat as
///   "all loaded models are enabled")
/// - `Some(vec![])` when the key exists but is empty (explicitly all
///   models disabled)
/// - `Some(slugs)` when one or more models are enabled
pub fn get_enabled_models_state() -> Option<Vec<String>> {
    with_retry(|c| {
        let exists: bool = c.exists("audio:enabled_models")?;
        if !exists {
            return Ok(None);
        }
        let members: Vec<String> = c.smembers("audio:enabled_models")?;
        Ok(Some(members))
    })
    .ok()
    .flatten()
}
/// Read a single setting from the `settings` hash.
pub fn get_setting(key: &str) -> Option<String> {
    with_retry(|c| c.hget("settings", key)).ok()
}

/// Read a setting as `f64`, returning `None` on missing / parse error.
pub fn get_setting_f64(key: &str) -> Option<f64> {
    get_setting(key).and_then(|v| v.parse().ok())
}

/// Refresh a `Config` with any overrides stored in Redis.
///
/// Called on each poll cycle so that changes made via the web UI
/// take effect without restarting the processing container.
pub fn apply_settings_overrides(config: &mut gaia_common::config::Config) {
    if let Some(v) = get_setting_f64("sensitivity") {
        config.sensitivity = v;
    }
    if let Some(v) = get_setting_f64("confidence") {
        config.confidence = v;
    }
    if let Some(v) = get_setting_f64("sf_thresh") {
        config.sf_thresh = v;
    }
    if let Some(v) = get_setting_f64("overlap") {
        config.overlap = v;
    }
    if let Some(v) = get_setting("colormap") {
        config.colormap = v;
    }
}

/// Load all excluded species (scientific names) from the overrides hash.
pub fn load_exclusion_overrides() -> Vec<String> {
    with_retry(|c| c.hkeys("exclusion_overrides")).unwrap_or_default()
}
