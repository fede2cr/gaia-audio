//! iNaturalist API client with an in-memory cache.
//!
//! Uses the public `v1/taxa` endpoint to look up species photos and Wikipedia
//! links by scientific name.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::model::SpeciesPhoto;

/// Thread-safe cache shared across requests.
pub type PhotoCache = Arc<Mutex<HashMap<String, Option<SpeciesPhoto>>>>;

/// Create an empty cache.
pub fn new_cache() -> PhotoCache {
    Arc::new(Mutex::new(HashMap::new()))
}

/// Look up a species photo.  Returns a cached result if available, otherwise
/// queries the iNaturalist API (and caches the answer).
pub async fn lookup(
    cache: &PhotoCache,
    scientific_name: &str,
) -> Option<SpeciesPhoto> {
    // Fast-path: serve from cache
    {
        let guard = cache.lock().unwrap();
        if let Some(cached) = guard.get(scientific_name) {
            return cached.clone();
        }
    }

    // Fetch from iNaturalist
    let result = fetch_from_inaturalist(scientific_name).await;

    // Store in cache
    {
        let mut guard = cache.lock().unwrap();
        guard.insert(scientific_name.to_string(), result.clone());
    }

    result
}

/// Raw HTTP call to the iNaturalist taxa search API.
async fn fetch_from_inaturalist(scientific_name: &str) -> Option<SpeciesPhoto> {
    let url = format!(
        "https://api.inaturalist.org/v1/taxa?q={}&rank=species&per_page=1",
        urlencoded(scientific_name),
    );

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

    Some(SpeciesPhoto {
        medium_url,
        attribution,
        wikipedia_url,
    })
}

/// Minimal URL-encoding for the query parameter.
fn urlencoded(s: &str) -> String {
    s.replace(' ', "+")
        .replace('&', "%26")
        .replace('=', "%3D")
}
