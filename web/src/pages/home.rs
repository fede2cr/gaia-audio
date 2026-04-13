//! Home page – real-time detection feed.

use leptos::*;

use crate::components::detection_card::DetectionCard;
use crate::components::species_card::SpeciesCard;
use crate::model::{SpeciesSummary, WebDetection};

// ─── Server functions ────────────────────────────────────────────────────────

#[server(GetRecentDetections, "/api")]
pub async fn get_recent_detections(
    limit: u32,
    after_rowid: Option<i64>,
) -> Result<Vec<WebDetection>, ServerFnError> {
    use crate::server::db;
    let state = use_context::<crate::app::AppState>()
        .ok_or_else(|| ServerFnError::new("Missing AppState"))?;
    db::recent_detections(&state.db_path, limit, after_rowid)
        .map_err(|e| ServerFnError::new(format!("DB error: {e}")))
}

#[server(GetTopSpecies, "/api")]
pub async fn get_top_species(limit: u32) -> Result<Vec<SpeciesSummary>, ServerFnError> {
    use crate::server::{db, inaturalist};
    let state = use_context::<crate::app::AppState>()
        .ok_or_else(|| ServerFnError::new("Missing AppState"))?;
    let mut species = db::top_species(&state.db_path, limit)
        .map_err(|e| ServerFnError::new(format!("DB error: {e}")))?;

    // Enrich with iNaturalist images
    for sp in species.iter_mut() {
        if let Some(photo) = inaturalist::lookup(&state.photo_cache, &sp.scientific_name).await {
            sp.image_url = Some(photo.medium_url);
        }
    }
    Ok(species)
}

// ─── Page component ──────────────────────────────────────────────────────────

/// Live detection feed with auto-polling + top species sidebar.
#[component]
pub fn Home() -> impl IntoView {
    // Latest detections resource (initial load)
    let detections = create_resource(|| (), |_| async { get_recent_detections(50, None).await });

    // Top species
    let top_species = create_resource(|| (), |_| async { get_top_species(12).await });

    // Auto-refresh: poll every 4 seconds for new detections
    let (feed, set_feed) = create_signal::<Vec<WebDetection>>(vec![]);
    #[allow(unused_variables)] // read only in the hydrate (WASM) build
    let (max_rowid, set_max_rowid) = create_signal::<Option<i64>>(None);

    // When initial data loads, populate the feed
    create_effect(move |_| {
        if let Some(Ok(initial)) = detections.get() {
            if let Some(first) = initial.first() {
                set_max_rowid.set(Some(first.id));
            }
            set_feed.set(initial);
        }
    });

    // Polling interval
    #[cfg(feature = "hydrate")]
    {
        set_interval_with_handle(
            move || {
                let rid = max_rowid.get();
                spawn_local(async move {
                    if let Ok(new) = get_recent_detections(20, rid).await {
                        if !new.is_empty() {
                            if let Some(first) = new.first() {
                                set_max_rowid.set(Some(first.id));
                            }
                            set_feed.update(|f| {
                                let mut combined = new;
                                combined.extend(f.drain(..));
                                combined.truncate(100); // keep last 100
                                *f = combined;
                            });
                        }
                    }
                });
            },
            std::time::Duration::from_secs(4),
        )
        .ok();
    }

    view! {
        <div class="home-page">
            <section class="live-feed">
                <h1>"Live Detections"</h1>
                <div class="feed-list">
                    <Suspense fallback=move || view! { <p class="loading">"Loading…"</p> }>
                        <For
                            each=move || feed.get()
                            key=|d| d.id
                            children=move |det: WebDetection| {
                                view! { <DetectionCard detection=det /> }
                            }
                        />
                    </Suspense>
                </div>
            </section>

            <aside class="top-species">
                <h2>"Top Species"</h2>
                <Suspense fallback=move || view! { <p class="loading">"Loading…"</p> }>
                    {move || top_species.get().map(|res| match res {
                        Ok(species) => view! {
                            <div class="species-grid">
                                <For
                                    each=move || species.clone()
                                    key=|s| s.scientific_name.clone()
                                    children=move |sp: SpeciesSummary| {
                                        view! { <SpeciesCard species=sp /> }
                                    }
                                />
                            </div>
                        }.into_view(),
                        Err(e) => view! {
                            <p class="error">"Error: " {e.to_string()}</p>
                        }.into_view(),
                    })}
                </Suspense>
            </aside>
        </div>
    }
}
