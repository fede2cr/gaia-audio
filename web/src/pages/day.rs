//! Day detail page – all detections for a specific date, grouped by species.

use leptos::*;
use leptos_router::*;

use crate::components::detection_card::DetectionCard;
use crate::model::{DayDetectionGroup, WebDetection};

// ─── Server function ─────────────────────────────────────────────────────────

#[server(GetDayDetections, "/api")]
pub async fn get_day_detections(
    date: String,
) -> Result<Vec<DayDetectionGroup>, ServerFnError> {
    use crate::server::{db, inaturalist};
    let state = use_context::<crate::app::AppState>()
        .ok_or_else(|| ServerFnError::new("Missing AppState"))?;
    let mut groups = db::day_detections(&state.db_path, &date)
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

// ─── Page component ──────────────────────────────────────────────────────────

/// Detail view for a single day, showing every species detected.
#[component]
pub fn DayView() -> impl IntoView {
    let params = use_params_map();
    let date = move || {
        params.with(|p| p.get("date").cloned().unwrap_or_default())
    };

    let data = create_resource(date, |d| async move { get_day_detections(d).await });

    view! {
        <div class="day-page">
            <h1>"Detections for " {date}</h1>
            <a href="/calendar" class="back-link">"← Back to Calendar"</a>

            <Suspense fallback=move || view! { <p class="loading">"Loading…"</p> }>
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
                    }.into_view(),
                    Err(e) => view! {
                        <p class="error">"Error: " {e.to_string()}</p>
                    }.into_view(),
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
                        <strong>{&group.common_name}</strong>
                        " ("{&group.scientific_name}")"
                    </a>
                    <span class="domain-badge">{&group.domain}</span>
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
    }
}
