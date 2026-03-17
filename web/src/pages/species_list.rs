//! Species list page – browse all detected species.

use leptos::prelude::*;
use leptos::prelude::{
    use_context, ElementChild, For, IntoView, Resource, ServerFnError,
    Suspense,
};
use leptos::either::Either;

use crate::components::species_card::SpeciesCard;
use crate::model::SpeciesSummary;

/// Sort criteria for the species list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeciesSort {
    /// Most detections first (default).
    Detections,
    /// Most threatened first (IUCN Red List order).
    ConservationStatus,
    /// Alphabetical by common name.
    Name,
}

impl SpeciesSort {
    /// Apply this sort to a species list **in place**.
    pub fn apply(self, list: &mut [SpeciesSummary]) {
        match self {
            Self::Detections => list.sort_by(|a, b| b.detection_count.cmp(&a.detection_count)),
            Self::ConservationStatus => list.sort_by(|a, b| {
                let ta = a
                    .conservation_status
                    .map(|s| s.threat_level())
                    .unwrap_or(0);
                let tb = b
                    .conservation_status
                    .map(|s| s.threat_level())
                    .unwrap_or(0);
                // Most threatened first, then by detection count.
                tb.cmp(&ta).then_with(|| b.detection_count.cmp(&a.detection_count))
            }),
            Self::Name => list.sort_by(|a, b| a.common_name.to_lowercase().cmp(&b.common_name.to_lowercase())),
        }
    }
}

/// Fetch **all-time** species from the database (not just today).
///
/// Excluded detections are omitted unless the species has been
/// overridden in the `exclusion_overrides` table.
#[server(prefix = "/api")]
pub async fn get_all_species(limit: u32) -> Result<Vec<SpeciesSummary>, ServerFnError> {
    use crate::server::{db, inaturalist};
    let state = use_context::<crate::app::AppState>()
        .ok_or_else(|| ServerFnError::new("Missing AppState"))?;

    let mut species = db::top_species(&state.db_path, limit)
        .map_err(|e| ServerFnError::new(format!("DB error: {e}")))?;

    // Enrich with iNaturalist images, conservation status, and sex photos
    for sp in species.iter_mut() {
        if let Some(photo) = inaturalist::lookup(&state.photo_cache, &sp.scientific_name).await {
            sp.image_url = Some(photo.medium_url);
            sp.conservation_status = photo.conservation_status;
            sp.male_image_url = photo.male_image_url;
            sp.female_image_url = photo.female_image_url;
        }
    }
    Ok(species)
}

/// Browse all detected species with sort controls.
#[component]
pub fn SpeciesListPage() -> impl IntoView {
    let species = Resource::new(|| (), |_| async { get_all_species(500).await });
    let (sort, set_sort) = signal(SpeciesSort::Detections);

    view! {
        <div class="species-list-page">
            <h1>"All Species"</h1>

            <div class="sort-bar">
                <span class="sort-label">"Sort by:"</span>
                <button
                    class=move || if sort.get() == SpeciesSort::Detections { "sort-btn active" } else { "sort-btn" }
                    on:click=move |_| set_sort.set(SpeciesSort::Detections)
                >
                    "Detections"
                </button>
                <button
                    class=move || if sort.get() == SpeciesSort::ConservationStatus { "sort-btn active" } else { "sort-btn" }
                    on:click=move |_| set_sort.set(SpeciesSort::ConservationStatus)
                >
                    "Conservation Status"
                </button>
                <button
                    class=move || if sort.get() == SpeciesSort::Name { "sort-btn active" } else { "sort-btn" }
                    on:click=move |_| set_sort.set(SpeciesSort::Name)
                >
                    "Name"
                </button>
            </div>

            <Suspense fallback=|| view! { <p class="loading">"Loading species\u{2026}"</p> }>
                {move || species.get().map(|res| match res {
                    Ok(list) => {
                        let sorted = move || {
                            let mut v = list.clone();
                            sort.get().apply(&mut v);
                            v
                        };
                        Either::Left(view! {
                            <div class="species-grid full">
                                <For
                                    each=sorted
                                    key=|s| s.scientific_name.clone()
                                    children=move |sp: SpeciesSummary| {
                                        view! { <SpeciesCard species=sp /> }
                                    }
                                />
                            </div>
                        })
                    }
                    Err(e) => Either::Right(view! {
                        <p class="error">"Error: " {e.to_string()}</p>
                    }),
                })}
            </Suspense>
        </div>
    }
}
