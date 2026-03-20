//! Excluded species page – lists species excluded by the occurrence threshold
//! filter, with the ability for an ornithologist to override the exclusion.
//!
//! Each species card can be expanded to show the individual excluded
//! detections with spectrograms and audio playback, so the user can
//! listen before deciding whether to confirm the override.

use leptos::prelude::*;
use leptos::prelude::{
    signal, ElementChild, For, IntoView, Resource,
    ServerFnError, Suspense, WriteSignal,
};

use crate::components::detection_card::DetectionCard;
use crate::model::{ExcludedSpecies, WebDetection};

// ─── Server functions ────────────────────────────────────────────────────────

#[server(prefix = "/api")]
pub async fn get_excluded_species() -> Result<Vec<ExcludedSpecies>, ServerFnError> {
    use crate::server::{db, inaturalist};
    let state = use_context::<crate::app::AppState>()
        .ok_or_else(|| ServerFnError::new("Missing AppState"))?;
    let mut species = db::excluded_species(&state.db_path)
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

#[server(prefix = "/api")]
pub async fn override_exclusion(
    scientific_name: String,
    notes: String,
) -> Result<(), ServerFnError> {
    let state = use_context::<crate::app::AppState>()
        .ok_or_else(|| ServerFnError::new("Missing AppState"))?;
    crate::server::db::add_exclusion_override(&state.db_path, &scientific_name, &notes)
        .await
        .map_err(|e| ServerFnError::new(format!("DB error: {e}")))?;
    Ok(())
}

#[server(prefix = "/api")]
pub async fn remove_override(scientific_name: String) -> Result<(), ServerFnError> {
    let state = use_context::<crate::app::AppState>()
        .ok_or_else(|| ServerFnError::new("Missing AppState"))?;
    crate::server::db::remove_exclusion_override(&state.db_path, &scientific_name)
        .await
        .map_err(|e| ServerFnError::new(format!("DB error: {e}")))?;
    Ok(())
}

#[server(prefix = "/api")]
pub async fn get_excluded_detections(
    scientific_name: String,
) -> Result<Vec<WebDetection>, ServerFnError> {
    use crate::server::{db, inaturalist};
    let state = use_context::<crate::app::AppState>()
        .ok_or_else(|| ServerFnError::new("Missing AppState"))?;
    let mut detections = db::excluded_detections_for_species(&state.db_path, &scientific_name, 20)
        .await
        .map_err(|e| ServerFnError::new(format!("DB error: {e}")))?;

    // Enrich with iNaturalist images
    for det in detections.iter_mut() {
        if let Some(photo) = inaturalist::lookup(&state.photo_cache, &det.scientific_name).await {
            det.image_url = Some(photo.medium_url);
        }
    }
    Ok(detections)
}

// ─── Page component ──────────────────────────────────────────────────────────

/// Excluded species page with override controls.
#[component]
pub fn ExcludedPage() -> impl IntoView {
    let (version, set_version) = signal(0u32);
    let species = Resource::new(
        move || version.get(),
        |_| async { get_excluded_species().await },
    );

    view! {
        <div class="excluded-page">
            <h1>"Excluded Species"</h1>
            <p class="page-desc">
                "These species were detected with sufficient confidence but excluded by the "
                "species-range model (occurrence threshold). Audio and spectrograms are preserved. "
                "If you can confirm the identification, click "
                <strong>"Confirm Override"</strong>
                " to include the species in the main Species list and future detections."
            </p>

            <Suspense fallback=|| view! { <p class="loading">"Loading excluded species…"</p> }>
                {move || species.get().map(|res| match res {
                    Ok(list) if list.is_empty() => view! {
                        <div class="empty-state">
                            <p>"No excluded detections yet."</p>
                        </div>
                    }.into_any(),
                    Ok(list) => {
                        let overridden: Vec<ExcludedSpecies> = list.iter().filter(|s| s.overridden).cloned().collect();
                        let pending: Vec<ExcludedSpecies> = list.iter().filter(|s| !s.overridden).cloned().collect();

                        view! {
                            {if !pending.is_empty() {
                                view! {
                                    <h2 class="section-heading">"Pending Review"</h2>
                                    <div class="excluded-grid">
                                        <For
                                            each=move || pending.clone()
                                            key=|s| s.scientific_name.clone()
                                            children=move |sp: ExcludedSpecies| {
                                                view! { <ExcludedCard species=sp version=set_version/> }
                                            }
                                        />
                                    </div>
                                }.into_any()
                            } else {
                                ().into_any()
                            }}
                            {if !overridden.is_empty() {
                                view! {
                                    <h2 class="section-heading">"Confirmed (Overridden)"</h2>
                                    <div class="excluded-grid">
                                        <For
                                            each=move || overridden.clone()
                                            key=|s| s.scientific_name.clone()
                                            children=move |sp: ExcludedSpecies| {
                                                view! { <ExcludedCard species=sp version=set_version/> }
                                            }
                                        />
                                    </div>
                                }.into_any()
                            } else {
                                ().into_any()
                            }}
                        }.into_any()
                    },
                    Err(e) => view! {
                        <p class="error">"Error: " {e.to_string()}</p>
                    }.into_any(),
                })}
            </Suspense>
        </div>
    }
}

/// Card for a single excluded species.
#[component]
fn ExcludedCard(
    species: ExcludedSpecies,
    version: WriteSignal<u32>,
) -> impl IntoView {
    let sci_name = species.scientific_name.clone();
    let sci_name_action = sci_name.clone();
    let is_overridden = species.overridden;

    let (busy, set_busy) = signal(false);

    let on_override = move |_| {
        set_busy.set(true);
        let name = sci_name_action.clone();
        leptos::task::spawn_local(async move {
            let _ = override_exclusion(name, String::new()).await;
            set_busy.set(false);
            version.update(|v| *v += 1);
        });
    };

    let sci_name_undo = sci_name.clone();
    let on_undo = move |_| {
        set_busy.set(true);
        let name = sci_name_undo.clone();
        leptos::task::spawn_local(async move {
            let _ = remove_override(name).await;
            set_busy.set(false);
            version.update(|v| *v += 1);
        });
    };

    let confidence_pct = format!("{:.0}%", species.max_confidence * 100.0);
    let count_label = format!(
        "{} detection{}",
        species.detection_count,
        if species.detection_count == 1 { "" } else { "s" }
    );
    let species_href = format!(
        "/species/{}",
        species.scientific_name.replace(' ', "%20")
    );

    let card_class = if is_overridden {
        "excluded-card overridden"
    } else {
        "excluded-card"
    };

    // ── expandable detections drawer ─────────────────────────────────
    let (expanded, set_expanded) = signal(false);
    let sci_name_det = sci_name.clone();
    let detections = Resource::new(
        move || (expanded.get(), sci_name_det.clone()),
        move |(open, name)| async move {
            if open {
                get_excluded_detections(name).await
            } else {
                Ok(vec![])
            }
        },
    );

    view! {
        <div class={card_class}>
            <div class="excluded-thumb">
                {match &species.image_url {
                    Some(url) => view! {
                        <img class="species-thumb" src={url.clone()} alt={species.common_name.clone()} loading="lazy"/>
                    }.into_any(),
                    None => view! {
                        <div class="species-thumb-placeholder">
                            <svg viewBox="0 0 24 24" width="32" height="32" fill="none" stroke="currentColor" stroke-width="1.5">
                                <path d="M12 3c-1.5 2-4 3-6 3 0 4 1.5 8 6 11 4.5-3 6-7 6-11-2 0-4.5-1-6-3z"/>
                            </svg>
                        </div>
                    }.into_any(),
                }}
            </div>

            <div class="excluded-info">
                <a href={species_href} class="excluded-species-name">
                    <span class="common-name">{species.common_name.clone()}</span>
                    <span class="sci-name">{species.scientific_name.clone()}</span>
                </a>
                <div class="excluded-meta">
                    <span class="domain-badge">{species.domain.clone()}</span>
                    <span class="confidence high">{confidence_pct}</span>
                    <span class="count-label">{count_label}</span>
                    {species.last_seen.as_ref().map(|d| view! {
                        <span class="last-seen">"Last: " {d.clone()}</span>
                    })}
                </div>

                <div class="excluded-actions">
                    <button
                        class="btn btn-sm btn-review"
                        on:click=move |_| set_expanded.update(|v| *v = !*v)
                    >
                        {move || if expanded.get() { "Hide Recordings ▲" } else { "Review Recordings ▼" }}
                    </button>
                    {if is_overridden {
                        view! {
                            <span class="override-badge">"✓ Confirmed"</span>
                            <button
                                class="btn btn-sm btn-outline"
                                on:click=on_undo
                                disabled=move || busy.get()
                            >
                                {move || if busy.get() { "…" } else { "Undo" }}
                            </button>
                        }.into_any()
                    } else {
                        view! {
                            <button
                                class="btn btn-sm btn-confirm"
                                on:click=on_override
                                disabled=move || busy.get()
                            >
                                {move || if busy.get() { "Confirming…" } else { "Confirm Override" }}
                            </button>
                        }.into_any()
                    }}
                </div>

                // Expanded detections drawer
                {move || if expanded.get() {
                    view! {
                        <div class="excluded-detections">
                            <Suspense fallback=|| view! { <p class="loading">"Loading recordings…"</p> }>
                                {move || detections.get().map(|res| match res {
                                    Ok(dets) if dets.is_empty() => view! {
                                        <p class="no-detections">"No audio clips available for this species."</p>
                                    }.into_any(),
                                    Ok(dets) => view! {
                                        <div class="excluded-detection-list">
                                            <For
                                                each=move || dets.clone()
                                                key=|d| d.id
                                                children=move |det: WebDetection| {
                                                    view! { <DetectionCard detection=det/> }
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
                    }.into_any()
                } else {
                    ().into_any()
                }}
            </div>
        </div>
    }
}
