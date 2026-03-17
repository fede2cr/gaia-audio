//! iNaturalist API client with an in-memory cache.
//!
//! Uses the public `v1/taxa` endpoint to look up species photos, Wikipedia
//! links, and conservation status by scientific name.  Also fetches
//! sex-annotated observation photos (male / female) from the
//! `v1/observations` endpoint so both sexes can be shown on species cards.
//!
//! The cache is versioned: when new fields are added to [`SpeciesPhoto`]
//! the [`CACHE_VERSION`] is bumped, causing stale entries to be re-fetched
//! automatically after an upgrade.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::model::SpeciesPhoto;

/// Bump this whenever [`SpeciesPhoto`] gains new fields that require a
/// fresh iNaturalist fetch.  Stale cache entries with an older version
/// are silently discarded and re-fetched.
const CACHE_VERSION: u16 = 2;

/// Wrapper stored in the cache so we can detect outdated entries.
#[derive(Clone, Debug)]
pub struct CacheEntry {
    version: u16,
    photo: Option<SpeciesPhoto>,
}

/// Thread-safe cache shared across requests.
pub type PhotoCache = Arc<Mutex<HashMap<String, CacheEntry>>>;

/// Create an empty cache.
pub fn new_cache() -> PhotoCache {
    Arc::new(Mutex::new(HashMap::new()))
}

/// Look up a species photo.  Returns a cached result if available and
/// up-to-date, otherwise queries the iNaturalist API (and caches the
/// answer).
pub async fn lookup(
    cache: &PhotoCache,
    scientific_name: &str,
) -> Option<SpeciesPhoto> {
    // Fast-path: serve from cache if version matches
    {
        let guard = cache.lock().unwrap();
        if let Some(entry) = guard.get(scientific_name) {
            if entry.version == CACHE_VERSION {
                return entry.photo.clone();
            }
            // Version mismatch → stale; fall through to re-fetch.
        }
    }

    // Fetch from iNaturalist
    let result = fetch_from_inaturalist(scientific_name).await;

    // Store in cache
    {
        let mut guard = cache.lock().unwrap();
        guard.insert(scientific_name.to_string(), CacheEntry {
            version: CACHE_VERSION,
            photo: result.clone(),
        });
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

    let taxon_id = result.get("id").and_then(|v| v.as_u64());

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

    // Parse conservation status from the search response.
    let mut conservation_status = parse_conservation_status(result);

    // The taxa search endpoint sometimes omits `conservation_status` for
    // Least Concern species.  If missing, try the direct /v1/taxa/{id}
    // endpoint which is more complete.
    if conservation_status.is_none() {
        if let Some(id) = taxon_id {
            conservation_status = fetch_conservation_status(id).await;
        }
    }

    // Fetch male and female observation photos concurrently.
    // iNaturalist annotation term_id 9 = "Sex", value 10 = Female, 11 = Male.
    let (male_image_url, female_image_url) =
        fetch_sex_photos(scientific_name).await;

    Some(SpeciesPhoto {
        medium_url,
        attribution,
        wikipedia_url,
        conservation_status,
        male_image_url,
        female_image_url,
    })
}

/// Extract conservation status from an iNaturalist taxon JSON object.
fn parse_conservation_status(taxon: &serde_json::Value) -> Option<crate::model::ConservationStatus> {
    // Try the singular `conservation_status` object first.
    if let Some(cs) = taxon.get("conservation_status") {
        let from_obj = cs
            .get("iucn")
            .and_then(|v| v.as_u64())
            .and_then(|n| crate::model::ConservationStatus::from_iucn(n as u8))
            .or_else(|| {
                cs.get("status")
                    .and_then(|v| v.as_str())
                    .and_then(crate::model::ConservationStatus::from_code)
            });
        if from_obj.is_some() {
            return from_obj;
        }
    }

    // Some responses use the plural `conservation_statuses` array instead.
    if let Some(arr) = taxon.get("conservation_statuses").and_then(|v| v.as_array()) {
        // Prefer IUCN authority; fall back to the first entry.
        let iucn_entry = arr.iter().find(|e| {
            e.get("authority")
                .and_then(|a| a.as_str())
                .map(|a| a.contains("IUCN"))
                .unwrap_or(false)
        });
        let entry = iucn_entry.or_else(|| arr.first());
        if let Some(e) = entry {
            return e
                .get("iucn")
                .and_then(|v| v.as_u64())
                .and_then(|n| crate::model::ConservationStatus::from_iucn(n as u8))
                .or_else(|| {
                    e.get("status")
                        .and_then(|v| v.as_str())
                        .and_then(crate::model::ConservationStatus::from_code)
                });
        }
    }

    None
}

/// Fetch conservation status via the direct `/v1/taxa/{id}` endpoint.
///
/// This endpoint is more likely to include `conservation_statuses` than the
/// search endpoint.
async fn fetch_conservation_status(taxon_id: u64) -> Option<crate::model::ConservationStatus> {
    let url = format!("https://api.inaturalist.org/v1/taxa/{taxon_id}");
    let resp = reqwest::get(&url).await.ok()?;
    let body: serde_json::Value = resp.json().await.ok()?;
    let result = body.get("results")?.as_array()?.first()?;
    parse_conservation_status(result)
}

// ─── Sex-annotated observation photos ────────────────────────────────────────

/// iNaturalist annotation term IDs.
const SEX_TERM_ID: u8 = 9;
const SEX_MALE_VALUE: u8 = 11;
const SEX_FEMALE_VALUE: u8 = 10;

/// Fetch a single observation photo annotated with the given sex value.
///
/// Queries Research Grade observations sorted by votes (most agreed-upon
/// annotation first) and returns the medium-size photo URL of the first
/// result that has one.
async fn fetch_sex_photo(scientific_name: &str, term_value_id: u8) -> Option<String> {
    let url = format!(
        "https://api.inaturalist.org/v1/observations?\
         taxon_name={name}&term_id={tid}&term_value_id={vid}\
         &quality_grade=research&photos=true&per_page=1\
         &order_by=votes",
        name = urlencoded(scientific_name),
        tid = SEX_TERM_ID,
        vid = term_value_id,
    );

    let resp = reqwest::get(&url).await.ok()?;
    let body: serde_json::Value = resp.json().await.ok()?;

    let obs = body.get("results")?.as_array()?.first()?;
    let photo = obs.get("photos")?.as_array()?.first()?;

    // The `url` field on observation photos is the square thumbnail;
    // replace "square" with "medium" for a higher-res image.
    let photo_url = photo.get("url")?.as_str()?;
    Some(photo_url.replace("/square.", "/medium."))
}

/// Fetch both male and female photos concurrently.
///
/// Returns `(male_url, female_url)`.  Either or both may be `None`.
async fn fetch_sex_photos(scientific_name: &str) -> (Option<String>, Option<String>) {
    let (male, female) = tokio::join!(
        fetch_sex_photo(scientific_name, SEX_MALE_VALUE),
        fetch_sex_photo(scientific_name, SEX_FEMALE_VALUE),
    );
    (male, female)
}

/// Minimal URL-encoding for the query parameter.
fn urlencoded(s: &str) -> String {
    s.replace(' ', "+")
        .replace('&', "%26")
        .replace('=', "%3D")
}
