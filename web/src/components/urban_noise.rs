//! Urban noise panel – shows aggregated counts of non-bird detections
//! (Engine, Dog, etc.) without storing or exposing human audio.

use leptos::prelude::*;
use leptos::prelude::{
    ElementChild, IntoView, Resource, ServerFnError, Suspense,
};

use crate::model::UrbanNoiseSummary;

// ─── Server function ─────────────────────────────────────────────────────────

#[server(prefix = "/api")]
pub async fn get_urban_noise() -> Result<Vec<UrbanNoiseSummary>, ServerFnError> {
    use crate::server::db;
    let state = use_context::<crate::app::AppState>()
        .ok_or_else(|| ServerFnError::new("Missing AppState"))?;
    db::urban_noise_summary(&state.db_path)
        .map_err(|e| ServerFnError::new(format!("DB error: {e}")))
}

// ─── Component ───────────────────────────────────────────────────────────────

/// Compact panel showing urban-noise detection tallies.
#[component]
pub fn UrbanNoise() -> impl IntoView {
    let data = Resource::new(|| (), |_| async { get_urban_noise().await });

    view! {
        <Suspense fallback=|| ()>
            {move || data.get().map(|res| match res {
                Ok(items) if !items.is_empty() => {
                    let total_today: u32 = items.iter().map(|i| i.today_count).sum();
                    let total_week: u32 = items.iter().map(|i| i.week_count).sum();
                    let items_clone = items.clone();

                    view! {
                        <div class="urban-noise">
                            <div class="urban-noise-header">
                                <h3>
                                    <svg viewBox="0 0 16 16" width="14" height="14" class="icon-noise">
                                        <circle cx="8" cy="8" r="6" fill="none" stroke="currentColor" stroke-width="1.5"/>
                                        <line x1="8" y1="5" x2="8" y2="9" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
                                        <circle cx="8" cy="11.5" r="0.8" fill="currentColor"/>
                                    </svg>
                                    "Urban Noise"
                                </h3>
                                <div class="urban-noise-totals">
                                    <span class="noise-stat" title="Today">
                                        {format!("{total_today} today")}
                                    </span>
                                    <span class="noise-stat" title="Last 7 days">
                                        {format!("{total_week} this week")}
                                    </span>
                                </div>
                            </div>
                            <ul class="urban-noise-list">
                                {items_clone.into_iter().map(|item| {
                                    let icon = noise_icon(&item.category);
                                    view! {
                                        <li class="noise-item">
                                            <span class="noise-icon">{icon}</span>
                                            <span class="noise-category">{item.category.clone()}</span>
                                            <span class="noise-count today">{item.today_count}</span>
                                            <span class="noise-count week">{item.week_count}</span>
                                            <span class="noise-count total">{item.total_count}</span>
                                        </li>
                                    }
                                }).collect::<Vec<_>>()}
                            </ul>
                            <div class="noise-legend">
                                <span>"today"</span>
                                <span>"7d"</span>
                                <span>"all"</span>
                            </div>
                        </div>
                    }.into_any()
                }
                _ => view! { <div></div> }.into_any(),
            })}
        </Suspense>
    }
}

/// Pick a small emoji/character to represent each noise category.
fn noise_icon(category: &str) -> &'static str {
    match category {
        "Engine" => "🚗",
        "Dog" => "🐕",
        "Human" => "👤",
        "Power tools" => "🔧",
        "Siren" => "🚨",
        "Gun" => "💥",
        "Fireworks" => "🎆",
        _ => "🔊",
    }
}
