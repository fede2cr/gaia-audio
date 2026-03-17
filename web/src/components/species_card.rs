//! Species card with image (from iNaturalist) and detection stats.
//!
//! When male and female observation photos are available, both are shown
//! side by side so the user can compare sexual dimorphism at a glance.

use leptos::prelude::*;
use leptos::prelude::{ElementChild, IntoView};

use crate::model::SpeciesSummary;

/// A compact card showing species photo, name, and detection count.
///
/// If male **and** female photos are present the image area is split into
/// two labelled halves; otherwise the default species image is shown.
#[component]
pub fn SpeciesCard(species: SpeciesSummary) -> impl IntoView {
    let href = format!("/species/{}", urlencoded(&species.scientific_name));
    let default_img = species
        .image_url
        .clone()
        .unwrap_or_else(|| "/pkg/placeholder.svg".to_string());

    let has_sex_photos =
        species.male_image_url.is_some() || species.female_image_url.is_some();
    let male_src = species
        .male_image_url
        .clone()
        .unwrap_or_else(|| default_img.clone());
    let female_src = species
        .female_image_url
        .clone()
        .unwrap_or_else(|| default_img.clone());

    // Conservation badge (if status is known).
    let badge = species.conservation_status.map(|cs| {
        let class = format!("conservation-badge {}", cs.css_class());
        let title = cs.label().to_string();
        let code = cs.code().to_string();
        (class, title, code)
    });

    view! {
        <a href={href} class="species-card">
            {if has_sex_photos {
                leptos::either::Either::Left(view! {
                    <div class="species-img-wrap sex-split">
                        <div class="sex-half">
                            <img
                                src={male_src}
                                alt=format!("{} male", species.common_name)
                                class="species-img"
                                loading="lazy"
                            />
                            <span class="sex-label male">"♂"</span>
                        </div>
                        <div class="sex-half">
                            <img
                                src={female_src}
                                alt=format!("{} female", species.common_name)
                                class="species-img"
                                loading="lazy"
                            />
                            <span class="sex-label female">"♀"</span>
                        </div>
                    </div>
                })
            } else {
                leptos::either::Either::Right(view! {
                    <div class="species-img-wrap">
                        <img
                            src={default_img}
                            alt={species.common_name.clone()}
                            class="species-img"
                            loading="lazy"
                        />
                    </div>
                })
            }}
            <div class="species-card-body">
                <h3 class="species-common">{species.common_name.clone()}</h3>
                <p class="species-sci">{species.scientific_name.clone()}</p>
                <div class="species-stats">
                    <span class="domain-badge">{species.domain.clone()}</span>
                    {badge.map(|(class, title, code)| view! {
                        <span class={class} title={title}>{code}</span>
                    })}
                    <span class="detection-count">
                        {species.display_count.clone()} " detections"
                    </span>
                </div>
            </div>
        </a>
    }
}

fn urlencoded(s: &str) -> String {
    s.replace(' ', "%20")
}
