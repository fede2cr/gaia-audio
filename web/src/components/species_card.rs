//! Species card with image (from iNaturalist) and detection stats.

use leptos::*;

use crate::model::SpeciesSummary;

/// A compact card showing species photo, name, and detection count.
#[component]
pub fn SpeciesCard(species: SpeciesSummary) -> impl IntoView {
    let href = format!("/species/{}", urlencoded(&species.scientific_name));
    let img_src = species
        .image_url
        .clone()
        .unwrap_or_else(|| "/pkg/placeholder.svg".to_string());

    view! {
        <a href={href} class="species-card">
            <div class="species-img-wrap">
                <img
                    src={img_src}
                    alt={species.common_name.clone()}
                    class="species-img"
                    loading="lazy"
                />
            </div>
            <div class="species-card-body">
                <h3 class="species-common">{&species.common_name}</h3>
                <p class="species-sci">{&species.scientific_name}</p>
                <div class="species-stats">
                    <span class="domain-badge">{&species.domain}</span>
                    <span class="detection-count">
                        {species.detection_count} " detections"
                    </span>
                </div>
            </div>
        </a>
    }
}

fn urlencoded(s: &str) -> String {
    s.replace(' ', "%20")
}
