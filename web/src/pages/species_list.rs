//! Species list page – browse all detected species.

use leptos::*;

use crate::components::species_card::SpeciesCard;
use crate::model::SpeciesSummary;

/// Fetch **all-time** species from the database (not just today).
///
/// Excluded detections are omitted unless the species has been
/// overridden in the `exclusion_overrides` table.
#[server(GetAllSpecies, "/api")]
pub async fn get_all_species(limit: u32) -> Result<Vec<SpeciesSummary>, ServerFnError> {
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

/// Browse all detected species (sorted by detection count).
#[component]
pub fn SpeciesListPage() -> impl IntoView {
    let species = create_resource(|| (), |_| async { get_all_species(500).await });

    view! {
        <div class="species-list-page">
            <h1>"All Species"</h1>

            <Suspense fallback=move || view! { <p class="loading">"Loading species…"</p> }>
                {move || species.get().map(|res| match res {
                    Ok(list) => view! {
                        <div class="species-grid full">
                            <For
                                each=move || list.clone()
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
        </div>
    }
}
