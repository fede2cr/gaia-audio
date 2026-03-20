//! Reusable model-filter pill bar.
//!
//! Shows an "All Models" pill plus one pill per model that has at least
//! one detection in the database.  The selected model slug is stored in
//! a signal so parent pages can react to changes.

use leptos::prelude::*;
use leptos::prelude::{
    ElementChild, For, IntoView, Resource, ServerFnError, Suspense,
};

use crate::model::ModelInfo;

// ─── Server function ─────────────────────────────────────────────────────────

/// Fetch the list of models that have recorded detections.
#[server(prefix = "/api")]
pub async fn get_available_models() -> Result<Vec<ModelInfo>, ServerFnError> {
    use crate::server::db;
    let state = use_context::<crate::app::AppState>()
        .ok_or_else(|| ServerFnError::new("Missing AppState"))?;
    let models = db::available_models(&state.db_path)
        .await
        .map_err(|e| ServerFnError::new(format!("DB error: {e}")))?;
    Ok(models
        .into_iter()
        .map(|m| ModelInfo {
            slug: m.slug,
            name: m.name,
        })
        .collect())
}

// ─── Component ───────────────────────────────────────────────────────────────

/// A pill bar that lets the user pick a detection model (or "All").
///
/// * `selected` – read signal with the currently chosen slug (`""` = all).
/// * `set_selected` – write signal to update the selection.
///
/// The component only renders when there are ≥ 2 models in the database
/// (otherwise there is nothing to filter).
#[component]
pub fn ModelFilter(
    selected: ReadSignal<String>,
    set_selected: WriteSignal<String>,
) -> impl IntoView {
    let models = Resource::new(|| (), |_| async { get_available_models().await });

    view! {
        <Suspense fallback=|| ()>
            {move || models.get().map(|res| match res {
                Ok(list) if list.len() >= 2 => {
                    let items = list.clone();
                    view! {
                        <div class="model-filter-bar">
                            <span class="model-filter-label">"Model:"</span>
                            <button
                                class=move || if selected.get().is_empty() { "model-pill active" } else { "model-pill" }
                                on:click=move |_| set_selected.set(String::new())
                            >
                                "All"
                            </button>
                            <For
                                each=move || items.clone()
                                key=|m| m.slug.clone()
                                children=move |m: ModelInfo| {
                                    let slug = m.slug.clone();
                                    let slug2 = slug.clone();
                                    let label = m.label().to_string();
                                    view! {
                                        <button
                                            class=move || if selected.get() == slug { "model-pill active" } else { "model-pill" }
                                            on:click=move |_| set_selected.set(slug2.clone())
                                        >
                                            {label}
                                        </button>
                                    }
                                }
                            />
                        </div>
                    }.into_any()
                }
                _ => ().into_any(),
            })}
        </Suspense>
    }
}
