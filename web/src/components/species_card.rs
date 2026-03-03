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

    // Conservation status badge – hidden when Least Concern or absent
    let conservation_badge = species
        .conservation_status
        .as_deref()
        .filter(|s| *s != "LC")
        .map(|code| {
            let css_class = conservation_css_class(code);
            let label = conservation_short_label(code);
            view! { <span class={format!("conservation-badge {css_class}")}>{label}</span> }
        });

    let introduced_badge = species
        .is_introduced
        .filter(|&b| b)
        .map(|_| view! { <span class="introduced-badge">"Introduced"</span> });

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
                    {conservation_badge}
                    {introduced_badge}
                    <span class="detection-count">
                        {species.detection_count} " detections"
                    </span>
                </div>
            </div>
        </a>
    }
}

/// Map IUCN status code to a CSS modifier class.
fn conservation_css_class(code: &str) -> &'static str {
    match code {
        "EX" => "extinct",
        "EW" => "extinct",
        "CR" => "critical",
        "EN" => "endangered",
        "VU" => "vulnerable",
        "NT" => "near-threatened",
        "DD" => "data-deficient",
        _ => "unknown",
    }
}

/// Short human-readable label for the IUCN code.
fn conservation_short_label(code: &str) -> &'static str {
    match code {
        "EX" => "Extinct",
        "EW" => "Extinct in Wild",
        "CR" => "Critically Endangered",
        "EN" => "Endangered",
        "VU" => "Vulnerable",
        "NT" => "Near Threatened",
        "DD" => "Data Deficient",
        _ => "At Risk",
    }
}

fn urlencoded(s: &str) -> String {
    s.replace(' ', "%20")
}
