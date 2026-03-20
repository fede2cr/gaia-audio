//! Home page – real-time detection feed.

use leptos::prelude::*;
use leptos::prelude::{
    signal, Effect, ElementChild, For, IntoView, Resource,
    ServerFnError, Suspense,
};
use leptos::either::Either;

use crate::components::detection_card::DetectionCard;
use crate::components::live_analysis::LiveAnalysis;
use crate::components::model_filter::ModelFilter;
use crate::components::species_card::SpeciesCard;
use crate::components::urban_noise::UrbanNoise;
use crate::model::{SpeciesSummary, WebDetection};

// ─── Server functions ────────────────────────────────────────────────────────

#[server(prefix = "/api")]
pub async fn get_recent_detections(
    limit: u32,
    after_rowid: Option<i64>,
    model_slug: String,
) -> Result<Vec<WebDetection>, ServerFnError> {
    use crate::server::{detections_duckdb as ddb, inaturalist};
    let state = use_context::<crate::app::AppState>()
        .ok_or_else(|| ServerFnError::new("Missing AppState"))?;
    let slug_opt = if model_slug.is_empty() { None } else { Some(model_slug.as_str()) };
    let mut detections = ddb::recent_detections_filtered(&state.db_path, limit, after_rowid, slug_opt)
        .await
        .map_err(|e| ServerFnError::new(format!("DB error: {e}")))?;

    // Enrich with iNaturalist species photos
    for det in detections.iter_mut() {
        if let Some(photo) = inaturalist::lookup(&state.photo_cache, &det.scientific_name).await {
            det.image_url = Some(photo.medium_url);
        }
    }
    Ok(detections)
}

#[server(prefix = "/api")]
pub async fn get_top_species(
    limit: u32,
    model_slug: String,
) -> Result<Vec<SpeciesSummary>, ServerFnError> {
    use crate::server::{db, detections_duckdb as ddb, inaturalist};
    let state = use_context::<crate::app::AppState>()
        .ok_or_else(|| ServerFnError::new("Missing AppState"))?;

    // Show today's top species instead of all-time.
    // Use the user's TZ offset so "today" matches their local midnight.
    let today = db::today_for_tz_pub(&state.db_path)
        .await
        .map_err(|e| ServerFnError::new(format!("DB error: {e}")))?;
    let slug_opt = if model_slug.is_empty() { None } else { Some(model_slug.as_str()) };
    let mut species = ddb::top_species_for_date_filtered(&state.db_path, &today, limit, slug_opt)
        .await
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
    // Model filter
    let (model_slug, set_model_slug) = signal(String::new());

    // Latest detections resource (initial load) – re-fetches when model changes
    let detections = Resource::new(
        move || model_slug.get(),
        |slug| async move { get_recent_detections(50, None, slug).await },
    );

    // Top species – also re-fetches when model changes
    let top_species = Resource::new(
        move || model_slug.get(),
        |slug| async move { get_top_species(12, slug).await },
    );

    // Auto-refresh: poll every 4 seconds for new detections
    let (feed, set_feed) = signal::<Vec<WebDetection>>(vec![]);
    #[allow(unused_variables)] // read only in the hydrate (WASM) build
    let (max_rowid, set_max_rowid) = signal::<Option<i64>>(None);

    // When initial data loads, populate the feed
    Effect::new(move || {
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
        use wasm_bindgen::prelude::*;
        use wasm_bindgen::JsCast;
        let cb = Closure::wrap(Box::new(move || {
            let rid = max_rowid.get();
            let slug = model_slug.get();
            leptos::task::spawn_local(async move {
                if let Ok(new) = get_recent_detections(20, rid, slug).await {
                    if !new.is_empty() {
                        if let Some(first) = new.first() {
                            set_max_rowid.set(Some(first.id));
                        }
                        set_feed.update(|f| {
                            let mut combined = new;
                            combined.extend(f.drain(..));
                            combined.truncate(100);
                            *f = combined;
                        });
                    }
                }
            });
        }) as Box<dyn Fn()>);
        let _ = web_sys::window()
            .unwrap()
            .set_interval_with_callback_and_timeout_and_arguments_0(
                cb.as_ref().unchecked_ref(),
                4000,
            );
        cb.forget();
    }

    view! {
        <div class="home-page">
            <section class="live-feed">
                <LiveAnalysis/>
                <h1>"Live Detections"</h1>
                <ModelFilter selected=model_slug set_selected=set_model_slug />
                <div class="feed-list">
                    <Suspense fallback=|| view! { <p class="loading">"Loading…"</p> }>
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
                <h2>"Today's Top Species"</h2>
                <Suspense fallback=|| view! { <p class="loading">"Loading\u{2026}"</p> }>
                    {move || top_species.get().map(|res| match res {
                        Ok(species) => Either::Left(view! {
                            <div class="species-grid">
                                <For
                                    each=move || species.clone()
                                    key=|s| s.scientific_name.clone()
                                    children=move |sp: SpeciesSummary| {
                                        view! { <SpeciesCard species=sp /> }
                                    }
                                />
                            </div>
                        }),
                        Err(e) => Either::Right(view! {
                            <p class="error">"Error: " {e.to_string()}</p>
                        }),
                    })}
                </Suspense>
                <UrbanNoise/>
            </aside>
        </div>
    }
}
