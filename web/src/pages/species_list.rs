//! Species list page – browse all detected species.

use leptos::prelude::*;
use leptos::prelude::{
    use_context, ElementChild, For, IntoView, Resource, ServerFnError,
    Suspense,
};
use leptos::either::Either;

use crate::components::species_card::SpeciesCard;
use crate::model::SpeciesSummary;

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
    let species = Resource::new(|| (), |_| async { get_all_species(500).await });

    view! {
        <div class="species-list-page">
            <h1>"All Species"</h1>

            <Suspense fallback=|| view! { <p class="loading">"Loading species\u{2026}"</p> }>
                {move || species.get().map(|res| match res {
                    Ok(list) => Either::Left(view! {
                        <div class="species-grid full">
                            <For
                                each=move || list.clone()
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
        </div>
    }
}
