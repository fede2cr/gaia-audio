//! Species list page – browse all detected species.

use leptos::*;

use crate::components::species_card::SpeciesCard;
use crate::model::SpeciesSummary;

// Re-use the server function from home (or define a dedicated one)
use crate::pages::home::get_top_species;

/// Browse all detected species (paginated by detection count).
#[component]
pub fn SpeciesListPage() -> impl IntoView {
    let species = create_resource(|| (), |_| async { get_top_species(100).await });

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
