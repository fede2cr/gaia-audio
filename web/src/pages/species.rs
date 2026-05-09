//! Species detail page – iNaturalist photo, detection history, calendar overlay.

use leptos::prelude::*;
use leptos::prelude::{
    signal, Action, Effect, ElementChild, IntoView, Resource,
    ServerFnError, StoredValue, Suspense,
};
use leptos::either::Either;
use leptos_router::hooks::use_params_map;

use crate::components::calendar_grid::CalendarGrid;
use crate::components::detection_card::DetectionCard;
use crate::components::hourly_chart::HourlyChart;
use crate::components::model_filter::ModelFilter;
use crate::model::{CalendarDay, HourlyCount, ModelInfo, SpeciesInfo, TopRecording, WebDetection};

// ─── Server functions ────────────────────────────────────────────────────────

#[server(prefix = "/api")]
pub async fn get_species_info(
    scientific_name: String,
) -> Result<Option<SpeciesInfo>, ServerFnError> {
    use crate::server::{detections_duckdb as ddb, inaturalist};
    let state = use_context::<crate::app::AppState>()
        .ok_or_else(|| ServerFnError::new("Missing AppState"))?;
    let mut info = ddb::species_info(&state.db_path, &scientific_name)
        .await
        .map_err(|e| ServerFnError::new(format!("DB error: {e}")))?;

    if let Some(ref mut sp) = info {
        if let Some(photo) = inaturalist::lookup(&state.photo_cache, &scientific_name).await {
            sp.image_url = Some(photo.medium_url);
            sp.wikipedia_url = photo.wikipedia_url;
            sp.male_image_url = photo.male_image_url;
            sp.female_image_url = photo.female_image_url;
        }
        // Load verification state.
        sp.verification = crate::server::kv::get_species_verification(&scientific_name)
            .await
            .map_err(|e| ServerFnError::new(format!("KV error: {e}")))?;
    }
    Ok(info)
}

#[server(prefix = "/api")]
pub async fn get_species_calendar(
    scientific_name: String,
    year: i32,
) -> Result<(Vec<CalendarDay>, Vec<String>), ServerFnError> {
    use crate::server::detections_duckdb as ddb;
    let state = use_context::<crate::app::AppState>()
        .ok_or_else(|| ServerFnError::new("Missing AppState"))?;

    // Full-year calendar data
    let mut all_days = Vec::new();
    for m in 1..=12 {
        let mut month_days = ddb::calendar_data(&state.db_path, year, m)
            .await
            .map_err(|e| ServerFnError::new(format!("DB error: {e}")))?;
        all_days.append(&mut month_days);
    }

    // Dates this species was active
    let active = ddb::species_active_dates(&state.db_path, &scientific_name, year)
        .await
        .map_err(|e| ServerFnError::new(format!("DB error: {e}")))?;

    Ok((all_days, active))
}

/// Save or update a verification for a species.
#[server(prefix = "/api")]
pub async fn set_species_verification(
    scientific_name: String,
    method: String,
    inaturalist_obs: String,
) -> Result<(), ServerFnError> {
    use crate::server::kv;
    kv::set_species_verification(&scientific_name, &method, &inaturalist_obs)
        .await
        .map_err(|e| ServerFnError::new(format!("KV error: {e}")))?;
    Ok(())
}

/// Remove verification for a species.
#[server(prefix = "/api")]
pub async fn remove_species_verification(
    scientific_name: String,
) -> Result<(), ServerFnError> {
    use crate::server::kv;
    kv::remove_species_verification(&scientific_name)
        .await
        .map_err(|e| ServerFnError::new(format!("KV error: {e}")))?;
    Ok(())
}

/// Hourly detection histogram for a species (all-time).
#[server(prefix = "/api")]
pub async fn get_species_hourly(
    scientific_name: String,
) -> Result<Vec<HourlyCount>, ServerFnError> {
    use crate::server::detections_duckdb as ddb;
    let state = use_context::<crate::app::AppState>()
        .ok_or_else(|| ServerFnError::new("Missing AppState"))?;
    ddb::species_hourly_histogram(&state.db_path, &scientific_name)
        .await
        .map_err(|e| ServerFnError::new(format!("DB error: {e}")))
}

/// Top recordings for a species (from the nightly cache).
#[server(prefix = "/api")]
pub async fn get_species_top_recordings(
    scientific_name: String,
) -> Result<Vec<TopRecording>, ServerFnError> {
    use crate::server::detections_duckdb as ddb;
    let state = use_context::<crate::app::AppState>()
        .ok_or_else(|| ServerFnError::new("Missing AppState"))?;
    ddb::get_top_recordings(&state.db_path, &scientific_name, 10)
        .await
        .map_err(|e| ServerFnError::new(format!("DB error: {e}")))
}

/// Recent detections for a species, optionally filtered by model slug.
#[server(prefix = "/api")]
pub async fn get_species_detections(
    scientific_name: String,
    model_slug: String,
    limit: u32,
) -> Result<Vec<WebDetection>, ServerFnError> {
    use crate::server::{detections_duckdb as ddb, inaturalist};
    let state = use_context::<crate::app::AppState>()
        .ok_or_else(|| ServerFnError::new("Missing AppState"))?;
    let slug_opt = if model_slug.is_empty() { None } else { Some(model_slug.as_str()) };
    let mut dets = ddb::species_detections_by_model(&state.db_path, &scientific_name, limit, slug_opt)
        .await
        .map_err(|e| ServerFnError::new(format!("DB error: {e}")))?;
    for det in dets.iter_mut() {
        if let Some(photo) = inaturalist::lookup(&state.photo_cache, &det.scientific_name).await {
            det.image_url = Some(photo.medium_url);
        }
    }
    Ok(dets)
}

/// Which models have detected this species (for the model filter bar).
#[server(prefix = "/api")]
pub async fn get_species_models(
    scientific_name: String,
) -> Result<Vec<ModelInfo>, ServerFnError> {
    use crate::server::detections_duckdb as ddb;
    let state = use_context::<crate::app::AppState>()
        .ok_or_else(|| ServerFnError::new("Missing AppState"))?;
    ddb::get_species_models(&state.db_path, &scientific_name)
        .await
        .map_err(|e| ServerFnError::new(format!("DB error: {e}")))
}

// ─── Page component ──────────────────────────────────────────────────────────

/// Detail page for a single species.
#[component]
pub fn SpeciesPage() -> impl IntoView {
    let params = use_params_map();
    let sci_name = move || {
        params.with(|p| {
            p.get("name")
                .unwrap_or_default()
                .replace("%20", " ")
        })
    };

    let info = Resource::new(sci_name, |name| async move {
        get_species_info(name).await
    });

    view! {
        <div class="species-page">
            <a href="/species" class="back-link">"← All Species"</a>

            <Suspense fallback=|| view! { <p class="loading">"Loading\u{2026}"</p> }>
                {move || info.get().map(|res| match res {
                    Ok(Some(sp)) => view! { <SpeciesDetail species=sp /> }.into_any(),
                    Ok(None) => view! {
                        <p class="error">"Species not found."</p>
                    }.into_any(),
                    Err(e) => view! {
                        <p class="error">"Error: " {e.to_string()}</p>
                    }.into_any(),
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

    let male_img = species.male_image_url.clone();
    let female_img = species.female_image_url.clone();
    let has_dimorphism = male_img.is_some() || female_img.is_some();

    let wiki_link = species.wikipedia_url.clone();
    let sci_name = species.scientific_name.clone();

    // ── Calendar state: default to current year/month ────────────────────
    let (now_y, now_m) = js_sys_now();
    let (year, set_year) = signal(now_y);
    let (month, set_month) = signal(now_m);

    let sci_name_for_cal = sci_name.clone();
    let calendar_data = Resource::new(
        move || (sci_name_for_cal.clone(), year.get()),
        |(name, y)| async move { get_species_calendar(name, y).await },
    );

    // ── Hourly histogram (all-time) ─────────────────────────────────────
    let sci_name_for_hourly = sci_name.clone();
    let hourly_data = Resource::new(
        move || sci_name_for_hourly.clone(),
        |name| async move { get_species_hourly(name).await },
    );
    let common_name_for_chart = StoredValue::new(species.common_name.clone());
    let total_dets = species.total_detections as u32;

    // ── Top recordings ──────────────────────────────────────────────────
    let sci_name_for_recs = sci_name.clone();
    let recordings = Resource::new(
        move || sci_name_for_recs.clone(),
        |name| async move { get_species_top_recordings(name).await },
    );

    // ── Model-filtered detection list ───────────────────────────────────
    let (model_slug, set_model_slug) = signal(String::new());
    let sci_name_for_dets = sci_name.clone();
    let model_detections = Resource::new(
        move || (sci_name_for_dets.clone(), model_slug.get()),
        |(name, slug)| async move { get_species_detections(name, slug, 50).await },
    );

    // ── Verification state ───────────────────────────────────────────────
    let initial_verification = species.verification.clone();
    let (verified, set_verified) = signal(initial_verification.is_some());
    let (verify_method, set_verify_method) = signal(
        initial_verification
            .as_ref()
            .map(|v| v.method.clone())
            .unwrap_or_else(|| "ornithologist".to_string()),
    );
    let (inat_obs, set_inat_obs) = signal(
        initial_verification
            .as_ref()
            .map(|v| v.inaturalist_obs.clone())
            .unwrap_or_default(),
    );
    let (verify_status, set_verify_status) = signal(Option::<String>::None);

    let sci_name_save = sci_name.clone();
    let save_verification = Action::new(move |(method, obs): &(String, String)| {
        let name = sci_name_save.clone();
        let method = method.clone();
        let obs = obs.clone();
        async move { set_species_verification(name, method, obs).await }
    });

    let sci_name_remove = sci_name.clone();
    let remove_verification = Action::new(move |_: &()| {
        let name = sci_name_remove.clone();
        async move { remove_species_verification(name).await }
    });

    // Feedback effects.
    Effect::new(move || {
        if let Some(result) = save_verification.value().get() {
            match result {
                Ok(()) => set_verify_status.set(Some("Verification saved.".into())),
                Err(e) => set_verify_status.set(Some(format!("Error: {e}"))),
            }
        }
    });
    Effect::new(move || {
        if let Some(result) = remove_verification.value().get() {
            match result {
                Ok(()) => set_verify_status.set(Some("Verification removed.".into())),
                Err(e) => set_verify_status.set(Some(format!("Error: {e}"))),
            }
        }
    });

    let on_verify_toggle = move |_| {
        let new_state = !verified.get();
        set_verified.set(new_state);
        set_verify_status.set(None);
        if new_state {
            save_verification.dispatch((verify_method.get(), inat_obs.get()));
        } else {
            remove_verification.dispatch(());
        }
    };

    let on_method_change = move |ev: leptos::ev::Event| {
        let val = event_target_value(&ev);
        set_verify_method.set(val.clone());
        set_verify_status.set(None);
        if verified.get() {
            save_verification.dispatch((val, inat_obs.get()));
        }
    };

    let on_inat_save = move |ev: leptos::ev::SubmitEvent| {
        ev.prevent_default();
        set_verify_status.set(None);
        if verified.get() {
            save_verification.dispatch((verify_method.get(), inat_obs.get()));
        }
    };

    // Month names for the dropdown.
    let month_names = vec![
        (1, "January"), (2, "February"), (3, "March"), (4, "April"),
        (5, "May"), (6, "June"), (7, "July"), (8, "August"),
        (9, "September"), (10, "October"), (11, "November"), (12, "December"),
    ];

    view! {
        <div class="species-detail">
            <div class="species-hero">
                <img src={img_src} alt={species.common_name.clone()} class="species-hero-img" />
                <div class="species-hero-info">
                    <h1>{species.common_name.clone()}</h1>
                    <p class="species-sci-name">{species.scientific_name.clone()}</p>
                    <span class="domain-badge">{species.domain.clone()}</span>
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

            // ── Sexual dimorphism photos ─────────────────────────────
            {has_dimorphism.then(|| view! {
                <section class="species-dimorphism">
                    <h2>"Male & Female"</h2>
                    <div class="dimorphism-pair">
                        {male_img.clone().map(|url| view! {
                            <div class="dimorphism-card">
                                <img src={url} alt="Male" class="dimorphism-img" loading="lazy" />
                                <span class="dimorphism-label">"♂ Male"</span>
                            </div>
                        })}
                        {female_img.clone().map(|url| view! {
                            <div class="dimorphism-card">
                                <img src={url} alt="Female" class="dimorphism-img" loading="lazy" />
                                <span class="dimorphism-label">"♀ Female"</span>
                            </div>
                        })}
                    </div>
                </section>
            })}

            // ── Verification section ─────────────────────────────────
            <section class="species-verification">
                <h2>"Verification"</h2>
                <div class="verification-row">
                    <label class="verification-check">
                        <input
                            type="checkbox"
                            prop:checked=verified
                            on:change=on_verify_toggle
                        />
                        " Verified by"
                    </label>
                    <select
                        class="verification-method"
                        prop:value=verify_method
                        on:change=on_method_change
                        disabled=move || !verified.get()
                    >
                        <option value="ornithologist">"Ornithologist"</option>
                        <option value="inaturalist">"iNaturalist observation"</option>
                    </select>
                </div>

                // iNaturalist observation field (shown when method is inaturalist)
                {move || {
                    if verify_method.get() == "inaturalist" && verified.get() {
                        Some(view! {
                            <form class="inat-obs-form" on:submit=on_inat_save>
                                <label class="inat-obs-label">
                                    "Observation URL or ID"
                                    <input
                                        type="text"
                                        class="inat-obs-input"
                                        placeholder="e.g. https://www.inaturalist.org/observations/12345"
                                        prop:value=inat_obs
                                        on:input=move |ev| set_inat_obs.set(event_target_value(&ev))
                                    />
                                </label>
                                <button type="submit" class="inat-obs-save">"Save"</button>
                            </form>
                        })
                    } else {
                        None
                    }
                }}

                {move || verify_status.get().map(|msg| {
                    let cls = if msg.starts_with("Error") {
                        "verification-status verification-error"
                    } else {
                        "verification-status verification-ok"
                    };
                    view! { <p class=cls>{msg}</p> }
                })}
            </section>

            // ── Hourly activity chart ────────────────────────────────
            <section class="species-hourly">
                <h2>"Activity by Hour"</h2>
                <Suspense fallback=|| view! { <p class="loading">"Loading\u{2026}"</p> }>
                    {move || {
                        let chart_name = common_name_for_chart.get_value();
                        hourly_data.get().map(|res| match res {
                            Ok(hours) => {
                                Either::Left(view! {
                                    <HourlyChart
                                        title=chart_name
                                        hours=hours
                                        total=total_dets
                                    />
                                })
                            },
                            Err(e) => Either::Right(view! {
                                <p class="error">"Error: " {e.to_string()}</p>
                            }),
                        })
                    }}
                </Suspense>
            </section>

            // ── Top recordings ────────────────────────────────────────
            <section class="species-recordings">
                <h2>"Top Recordings"</h2>
                <Suspense fallback=|| view! { <p class="loading">"Loading recordings\u{2026}"</p> }>
                    {move || recordings.get().map(|res| match res {
                        Ok(recs) if recs.is_empty() => Either::Left(view! {
                            <p class="no-data">"No recordings cached yet."</p>
                        }),
                        Ok(recs) => Either::Right(Either::Left(view! {
                            <div class="recordings-grid">
                                {recs.into_iter().map(|r| {
                                    let spec_url = r.spectrogram_url();
                                    let clip     = r.clip_url();
                                    let conf_pct = format!("{:.0}%", r.confidence * 100.0);
                                    let date     = if r.display_date.is_empty() { r.date.clone() } else { r.display_date.clone() };
                                    let time     = if r.display_time.is_empty() { r.time.clone() } else { r.display_time.clone() };
                                    view! {
                                        <div class="recording-card">
                                            <img
                                                src={spec_url}
                                                alt="spectrogram"
                                                class="recording-spectrogram"
                                                loading="lazy"
                                            />
                                            <div class="recording-info">
                                                <span class="recording-confidence">{conf_pct}</span>
                                                <span class="recording-datetime">{date} " " {time}</span>
                                            </div>
                                            <audio controls preload="none" class="recording-audio">
                                                <source src={clip} type="audio/mp3" />
                                            </audio>
                                        </div>
                                    }
                                }).collect::<Vec<_>>()}
                            </div>
                        })),
                        Err(e) => Either::Right(Either::Right(view! {
                            <p class="error">"Error: " {e.to_string()}</p>
                        })),
                    })}
                </Suspense>
            </section>

            // ── Recent detections (model filtered) ───────────────────
            <section class="species-detections-section">
                <h2>"Recent Detections"</h2>
                <ModelFilter selected=model_slug set_selected=set_model_slug />
                <Suspense fallback=|| view! { <p class="loading">"Loading detections\u{2026}"</p> }>
                    {move || model_detections.get().map(|res| match res {
                        Ok(dets) if dets.is_empty() => view! {
                            <p class="no-data">"No detections"
                                {move || if model_slug.get().is_empty() { "" } else { " for this model" }}
                                "."
                            </p>
                        }.into_any(),
                        Ok(dets) => view! {
                            <div class="feed-list">
                                {dets.into_iter().map(|det| view! {
                                    <DetectionCard detection=det />
                                }).collect::<Vec<_>>()}
                            </div>
                        }.into_any(),
                        Err(e) => view! {
                            <p class="error">"Error: " {e.to_string()}</p>
                        }.into_any(),
                    })}
                </Suspense>
            </section>

            <section class="species-calendar">
                <h2>"Calendar"</h2>
                <div class="species-cal-nav">
                    <button
                        class="cal-nav-btn"
                        on:click=move |_| set_year.update(|y| *y -= 1)
                    >"−"</button>
                    <span class="cal-nav-year">{move || year.get()}</span>
                    <button
                        class="cal-nav-btn"
                        on:click=move |_| set_year.update(|y| *y += 1)
                    >"+"</button>

                    <select
                        class="cal-nav-month-select"
                        on:change=move |ev| {
                            if let Ok(m) = event_target_value(&ev).parse::<u32>() {
                                set_month.set(m);
                            }
                        }
                    >
                        {month_names.into_iter().map(|(num, name)| {
                            let selected = num == now_m;
                            view! {
                                <option value={num.to_string()} selected=selected>{name}</option>
                            }
                        }).collect::<Vec<_>>()}
                    </select>
                </div>

                <Suspense fallback=|| view! { <p class="loading">"Loading calendar\u{2026}"</p> }>
                    {move || calendar_data.get().map(|res| match res {
                        Ok((_all_days, active_dates)) => {
                            Either::Left(view! {
                                <CalendarGrid
                                    year=year.get()
                                    month=month.get()
                                    days=vec![]
                                    highlight_dates=active_dates
                                />
                            })
                        },
                        Err(e) => Either::Right(view! {
                            <p class="error">"Error: " {e.to_string()}</p>
                        }),
                    })}
                </Suspense>
            </section>
        </div>
    }
}

/// Returns (year, month) for the current date.
fn js_sys_now() -> (i32, u32) {
    #[cfg(feature = "ssr")]
    {
        let now = chrono::Utc::now();
        return (
            now.format("%Y").to_string().parse().unwrap_or(2025),
            now.format("%m").to_string().parse().unwrap_or(1),
        );
    }

    #[cfg(not(feature = "ssr"))]
    {
        (2025, 1)
    }
}
