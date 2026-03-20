//! Day detail page – all detections for a specific date, grouped by species.

use leptos::prelude::*;
use leptos::prelude::{
    ElementChild, For, IntoView, Resource, ServerFnError,
    Suspense,
};
use leptos_router::hooks::use_params_map;

use crate::components::detection_card::DetectionCard;
use crate::components::hourly_chart::SpeciesHourlyGrid;
use crate::components::model_filter::ModelFilter;
use crate::model::{DayDetectionGroup, SpeciesHourlyCounts, WebDetection};

// ─── Server functions ────────────────────────────────────────────────────────

#[server(prefix = "/api")]
pub async fn get_day_detections(
    date: String,
    model_slug: String,
) -> Result<Vec<DayDetectionGroup>, ServerFnError> {
    use crate::server::{db, inaturalist};
    let state = use_context::<crate::app::AppState>()
        .ok_or_else(|| ServerFnError::new("Missing AppState"))?;
    let slug_opt = if model_slug.is_empty() { None } else { Some(model_slug.as_str()) };
    let mut groups = db::day_detections_filtered(&state.db_path, &date, slug_opt)
        .await
        .map_err(|e| ServerFnError::new(format!("DB error: {e}")))?;

    // Enrich with images
    for g in groups.iter_mut() {
        if let Some(photo) =
            inaturalist::lookup(&state.photo_cache, &g.scientific_name).await
        {
            g.image_url = Some(photo.medium_url);
        }
    }
    Ok(groups)
}

/// Hourly species×hour grid for a specific date.
#[server(prefix = "/api")]
pub async fn get_day_hourly(
    date: String,
) -> Result<Vec<SpeciesHourlyCounts>, ServerFnError> {
    use crate::server::db;
    let state = use_context::<crate::app::AppState>()
        .ok_or_else(|| ServerFnError::new("Missing AppState"))?;
    db::daily_species_hourly(&state.db_path, &date)
        .await
        .map_err(|e| ServerFnError::new(format!("DB error: {e}")))
}

// ─── Page component ──────────────────────────────────────────────────────────

/// Detail view for a single day, showing every species detected.
#[component]
pub fn DayView() -> impl IntoView {
    let params = use_params_map();
    let date = move || {
        params.with(|p| p.get("date").unwrap_or_default())
    };
    let (model_slug, set_model_slug) = signal(String::new());

    let data = Resource::new(
        move || (date(), model_slug.get()),
        |(d, slug)| async move { get_day_detections(d.clone(), slug).await },
    );
    let hourly = Resource::new(date, |d| async move { get_day_hourly(d).await });

    view! {
        <div class="day-page">
            <h1>"Detections for " {date}</h1>
            <a href="/calendar" class="back-link">"← Back to Calendar"</a>

            <ModelFilter selected=model_slug set_selected=set_model_slug />

            // ── Species × Hour heatmap ───────────────────────────────
            <Suspense fallback=|| view! { <p class="loading">"Loading chart\u{2026}"</p> }>
                {move || hourly.get().map(|res| match res {
                    Ok(grid) if !grid.is_empty() => Some(view! {
                        <section class="day-hourly-section">
                            <h2>"Activity by Hour"</h2>
                            <SpeciesHourlyGrid data=grid />
                        </section>
                    }.into_any()),
                    _ => None,
                })}
            </Suspense>

            <Suspense fallback=|| view! { <p class="loading">"Loading…"</p> }>
                {move || data.get().map(|res| match res {
                    Ok(groups) => view! {
                        <div class="day-groups">
                            <p class="day-summary">
                                {groups.len()} " species detected"
                            </p>
                            <For
                                each=move || groups.clone()
                                key=|g| g.scientific_name.clone()
                                children=move |group: DayDetectionGroup| {
                                    view! { <DayGroup group=group /> }
                                }
                            />
                        </div>
                    }.into_any(),
                    Err(e) => view! {
                        <p class="error">"Error: " {e.to_string()}</p>
                    }.into_any(),
                })}
            </Suspense>
        </div>
    }
}

/// A single species group within the day view.
#[component]
fn DayGroup(group: DayDetectionGroup) -> impl IntoView {
    let img_src = group
        .image_url
        .clone()
        .unwrap_or_else(|| "/pkg/placeholder.svg".to_string());
    let conf = format!("{:.0}%", group.max_confidence * 100.0);
    let species_href = format!(
        "/species/{}",
        group.scientific_name.replace(' ', "%20")
    );

    view! {
        <div class="day-group">
            <div class="day-group-header">
                <img src={img_src} alt={group.common_name.clone()} class="day-group-img" loading="lazy" />
                <div class="day-group-info">
                    <a href={species_href} class="day-group-name">
                        <strong>{group.common_name.clone()}</strong>
                        " ("{group.scientific_name.clone()}")"
                    </a>
                    <span class="domain-badge">{group.domain.clone()}</span>
                    <span class="confidence">"Best: " {conf}</span>
                    <span class="detection-count">{group.detections.len()}" detections"</span>
                </div>
            </div>
            <div class="day-group-detections">
                <For
                    each=move || group.detections.clone()
                    key=|d| d.id
                    children=move |det: WebDetection| {
                        view! { <DetectionCard detection=det /> }
                    }
                />
            </div>
        </div>
    }.into_any()
}
