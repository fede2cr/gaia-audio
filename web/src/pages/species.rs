//! Species detail page – iNaturalist photo, detection history, calendar overlay.

use leptos::*;
use leptos_router::*;

use crate::components::calendar_grid::CalendarGrid;
use crate::model::{CalendarDay, SpeciesInfo};

// ─── Server functions ────────────────────────────────────────────────────────

#[server(GetSpeciesInfo, "/api")]
pub async fn get_species_info(
    scientific_name: String,
) -> Result<Option<SpeciesInfo>, ServerFnError> {
    use crate::server::{db, inaturalist};
    let state = use_context::<crate::app::AppState>()
        .ok_or_else(|| ServerFnError::new("Missing AppState"))?;
    let mut info = db::species_info(&state.db_path, &scientific_name)
        .map_err(|e| ServerFnError::new(format!("DB error: {e}")))?;

    if let Some(ref mut sp) = info {
        if let Some(photo) = inaturalist::lookup(&state.photo_cache, &scientific_name).await {
            sp.image_url = Some(photo.medium_url);
            sp.wikipedia_url = photo.wikipedia_url;
        }
    }
    Ok(info)
}

#[server(GetSpeciesCalendar, "/api")]
pub async fn get_species_calendar(
    scientific_name: String,
    year: i32,
) -> Result<(Vec<CalendarDay>, Vec<String>), ServerFnError> {
    use crate::server::db;
    let state = use_context::<crate::app::AppState>()
        .ok_or_else(|| ServerFnError::new("Missing AppState"))?;

    // Full-year calendar data
    let mut all_days = Vec::new();
    for m in 1..=12 {
        let mut month_days = db::calendar_data(&state.db_path, year, m)
            .map_err(|e| ServerFnError::new(format!("DB error: {e}")))?;
        all_days.append(&mut month_days);
    }

    // Dates this species was active
    let active = db::species_active_dates(&state.db_path, &scientific_name, year)
        .map_err(|e| ServerFnError::new(format!("DB error: {e}")))?;

    Ok((all_days, active))
}

// ─── Page component ──────────────────────────────────────────────────────────

/// Detail page for a single species.
#[component]
pub fn SpeciesPage() -> impl IntoView {
    let params = use_params_map();
    let sci_name = move || {
        params.with(|p| {
            p.get("name")
                .cloned()
                .unwrap_or_default()
                .replace("%20", " ")
        })
    };

    let info = create_resource(sci_name, |name| async move {
        get_species_info(name).await
    });

    view! {
        <div class="species-page">
            <a href="/species" class="back-link">"← All Species"</a>

            <Suspense fallback=move || view! { <p class="loading">"Loading…"</p> }>
                {move || info.get().map(|res| match res {
                    Ok(Some(sp)) => view! { <SpeciesDetail species=sp /> }.into_view(),
                    Ok(None) => view! {
                        <p class="error">"Species not found."</p>
                    }.into_view(),
                    Err(e) => view! {
                        <p class="error">"Error: " {e.to_string()}</p>
                    }.into_view(),
                })}
            </Suspense>
        </div>
    }
}

/// Species detail content (factored out for clarity).
#[component]
fn SpeciesDetail(species: SpeciesInfo) -> impl IntoView {
    let img_src = species
        .image_url
        .clone()
        .unwrap_or_else(|| "/pkg/placeholder.svg".to_string());

    let wiki_link = species.wikipedia_url.clone();

    // Current year for the calendar overlay
    let year = {
        #[cfg(feature = "ssr")]
        { chrono::Local::now().format("%Y").to_string().parse::<i32>().unwrap_or(2025) }
        #[cfg(not(feature = "ssr"))]
        { 2025i32 }
    };

    let sci_name_for_cal = species.scientific_name.clone();
    let calendar_data = create_resource(
        move || (sci_name_for_cal.clone(), year),
        |(name, y)| async move { get_species_calendar(name, y).await },
    );

    view! {
        <div class="species-detail">
            <div class="species-hero">
                <img src={img_src} alt={species.common_name.clone()} class="species-hero-img" />
                <div class="species-hero-info">
                    <h1>{&species.common_name}</h1>
                    <p class="species-sci-name">{&species.scientific_name}</p>
                    <span class="domain-badge">{&species.domain}</span>
                    <div class="species-stats-bar">
                        <div class="stat">
                            <span class="stat-value">{species.total_detections}</span>
                            <span class="stat-label">"Total Detections"</span>
                        </div>
                        <div class="stat">
                            <span class="stat-value">{species.first_seen.clone().unwrap_or_default()}</span>
                            <span class="stat-label">"First Seen"</span>
                        </div>
                        <div class="stat">
                            <span class="stat-value">{species.last_seen.clone().unwrap_or_default()}</span>
                            <span class="stat-label">"Last Seen"</span>
                        </div>
                    </div>
                    {wiki_link.map(|url| view! {
                        <a href={url} target="_blank" rel="noopener" class="wiki-link">
                            "Wikipedia →"
                        </a>
                    })}
                </div>
            </div>

            // Activity calendar
            <section class="species-calendar">
                <h2>"Activity in " {year}</h2>
                <Suspense fallback=move || view! { <p class="loading">"Loading calendar…"</p> }>
                    {move || calendar_data.get().map(|res| match res {
                        Ok((_all_days, active_dates)) => {
                            // Show current month calendar with species dates highlighted
                            let month = {
                                #[cfg(feature = "ssr")]
                                { chrono::Local::now().format("%m").to_string().parse::<u32>().unwrap_or(1) }
                                #[cfg(not(feature = "ssr"))]
                                { 1u32 }
                            };
                            view! {
                                <CalendarGrid
                                    year=year
                                    month=month
                                    days=vec![]
                                    highlight_dates=active_dates
                                />
                            }.into_view()
                        },
                        Err(e) => view! {
                            <p class="error">"Error: " {e.to_string()}</p>
                        }.into_view(),
                    })}
                </Suspense>
            </section>
        </div>
    }
}
