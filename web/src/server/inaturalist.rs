//! iNaturalist API client with an in-memory cache.
//!
//! Uses the public `v1/taxa` endpoint to look up species photos, Wikipedia
//! links, conservation status and establishment means by scientific name.
//!
//! When a `preferred_place_id` is supplied, the taxa response includes
//! `establishment_means` so we can tell whether a species is introduced.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::model::SpeciesPhoto;

/// Thread-safe cache shared across requests.
pub type PhotoCache = Arc<Mutex<HashMap<String, Option<SpeciesPhoto>>>>;

/// Cached iNaturalist place ID resolved from lat/lon (resolved once).
pub type PlaceIdCache = Arc<Mutex<Option<Option<u64>>>>;

/// Create an empty photo cache.
pub fn new_cache() -> PhotoCache {
    Arc::new(Mutex::new(HashMap::new()))
}

/// Create an uninitialised place-ID cache.
pub fn new_place_cache() -> PlaceIdCache {
    Arc::new(Mutex::new(None))
}

/// Resolve the iNaturalist `place_id` for the given coordinates.
///
/// The result is cached so only one HTTP call is ever made.  Returns `None`
/// when no place matches or the coordinates are invalid (< -90 / > 90, etc.).
pub async fn resolve_place_id(
    cache: &PlaceIdCache,
    lat: f64,
    lon: f64,
) -> Option<u64> {
    // Fast path: already resolved
    {
        let guard = cache.lock().unwrap();
        if let Some(cached) = *guard {
            return cached;
        }
    }

    let result = fetch_place_id(lat, lon).await;

    {
        let mut guard = cache.lock().unwrap();
        *guard = Some(result);
    }

    result
}

/// Query the iNaturalist `/v1/places/nearby` endpoint to find the most
/// specific standard place (e.g. country or state) for the given coords.
async fn fetch_place_id(lat: f64, lon: f64) -> Option<u64> {
    if !(-90.0..=90.0).contains(&lat) || !(-180.0..=180.0).contains(&lon) {
        return None;
    }

    let url = format!(
        "https://api.inaturalist.org/v1/places/nearby?nelat={lat}&nelng={lon}&swlat={lat}&swlng={lon}",
    );

    let resp = reqwest::get(&url).await.ok()?;
    let body: serde_json::Value = resp.json().await.ok()?;

    // The response has `results.standard` – pick the first (most specific) place.
    let standard = body
        .get("results")?
        .get("standard")?
        .as_array()?;

    standard
        .first()
        .and_then(|p| p.get("id"))
        .and_then(|id| id.as_u64())
}

/// Look up a species photo.  Returns a cached result if available, otherwise
/// queries the iNaturalist API (and caches the answer).
///
/// When `place_id` is `Some`, it is passed as `preferred_place_id` so the
/// response includes establishment means (introduced / native).
pub async fn lookup(
    cache: &PhotoCache,
    scientific_name: &str,
    place_id: Option<u64>,
) -> Option<SpeciesPhoto> {
    // Fast-path: serve from cache
    {
        let guard = cache.lock().unwrap();
        if let Some(cached) = guard.get(scientific_name) {
            return cached.clone();
        }
    }

    // Fetch from iNaturalist
    let result = fetch_from_inaturalist(scientific_name, place_id).await;

    // Store in cache
    {
        let mut guard = cache.lock().unwrap();
        guard.insert(scientific_name.to_string(), result.clone());
    }

    result
}

/// Raw HTTP call to the iNaturalist taxa search API.
async fn fetch_from_inaturalist(
    scientific_name: &str,
    place_id: Option<u64>,
) -> Option<SpeciesPhoto> {
    let mut url = format!(
        "https://api.inaturalist.org/v1/taxa?q={}&rank=species&per_page=1",
        urlencoded(scientific_name),
    );
    if let Some(pid) = place_id {
        url.push_str(&format!("&preferred_place_id={pid}"));
    }

    let resp = reqwest::get(&url).await.ok()?;
    let body: serde_json::Value = resp.json().await.ok()?;

    let result = body.get("results")?.as_array()?.first()?;

    let photo = result.get("default_photo")?;
    let medium_url = photo.get("medium_url")?.as_str()?.to_string();
    let attribution = photo
        .get("attribution")
        .and_then(|a| a.as_str())
        .unwrap_or("iNaturalist")
        .to_string();

    let wikipedia_url = result
        .get("wikipedia_url")
        .and_then(|w| w.as_str())
        .map(String::from);

    // ── Conservation status (IUCN) ──────────────────────────────────────
    let (conservation_status, conservation_status_name) =
        if let Some(cs) = result.get("conservation_status") {
            (
                cs.get("status").and_then(|s| s.as_str()).map(|s| s.to_uppercase()),
                cs.get("status_name").and_then(|s| s.as_str()).map(String::from),
            )
        } else {
            (None, None)
        };

    // ── Establishment means (introduced / native) ───────────────────────
    // Present when the API is queried with `preferred_place_id`.  Also
    // fall back to the `establishment_means` object on the taxon itself.
    let is_introduced = result
        .get("establishment_means")
        .and_then(|em| em.get("id"))
        .and_then(|id| id.as_str())
        .map(|id| id == "introduced")
        .or_else(|| {
            // Some responses carry a flat boolean
            result.get("introduced").and_then(|v| v.as_bool())
        });

    Some(SpeciesPhoto {
        medium_url,
        attribution,
        wikipedia_url,
        conservation_status,
        conservation_status_name,
        is_introduced,
    })
}

/// Minimal URL-encoding for the query parameter.
fn urlencoded(s: &str) -> String {
    s.replace(' ', "+")
        .replace('&', "%26")
        .replace('=', "%3D")
}
