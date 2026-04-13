//! Species card with image (from iNaturalist) and detection stats.
//!
//! When male and female observation photos are available, both are shown
//! side by side so the user can compare sexual dimorphism at a glance.

use leptos::prelude::*;
use leptos::prelude::{ElementChild, IntoView};

use crate::model::SpeciesSummary;
#[cfg(target_arch = "wasm32")]
use crate::pages::species_list::get_species_photo;

/// A compact card showing species photo, name, and detection count.
///
/// If male **and** female photos are present the image area is split into
/// two labelled halves; otherwise the default species image is shown.
#[component]
pub fn SpeciesCard(species: SpeciesSummary) -> impl IntoView {
    let href = format!("/species/{}", urlencoded(&species.scientific_name));
    let fallback_img = "/pkg/placeholder.svg".to_string();
    let default_img = species
        .image_url
        .clone()
        .unwrap_or_else(|| fallback_img.clone());

    #[cfg(target_arch = "wasm32")]
    let (image_url, set_image_url) = signal(default_img.clone());
    #[cfg(not(target_arch = "wasm32"))]
    let (image_url, _set_image_url) = signal(default_img.clone());

    #[cfg(target_arch = "wasm32")]
    let (male_image_url, set_male_image_url) = signal(species.male_image_url.clone());
    #[cfg(not(target_arch = "wasm32"))]
    let (male_image_url, _set_male_image_url) = signal(species.male_image_url.clone());

    #[cfg(target_arch = "wasm32")]
    let (female_image_url, set_female_image_url) = signal(species.female_image_url.clone());
    #[cfg(not(target_arch = "wasm32"))]
    let (female_image_url, _set_female_image_url) = signal(species.female_image_url.clone());

    #[cfg(target_arch = "wasm32")]
    let (conservation_status, set_conservation_status) = signal(species.conservation_status);
    #[cfg(not(target_arch = "wasm32"))]
    let (conservation_status, _set_conservation_status) = signal(species.conservation_status);

    #[cfg(target_arch = "wasm32")]
    {
        let scientific_name = species.scientific_name.clone();
        let current_image = image_url;
        Effect::new(move |_| {
            // Skip request when this card already has a non-placeholder image.
            if current_image.get() != "/pkg/placeholder.svg" {
                return;
            }

            let scientific_name = scientific_name.clone();
            let set_image_url = set_image_url;
            let set_male_image_url = set_male_image_url;
            let set_female_image_url = set_female_image_url;
            let set_conservation_status = set_conservation_status;
            leptos::task::spawn_local(async move {
                if let Ok(Some(photo)) = get_species_photo(scientific_name).await {
                    set_image_url.set(photo.medium_url.clone());
                    set_male_image_url.set(photo.male_image_url);
                    set_female_image_url.set(photo.female_image_url);
                    set_conservation_status.set(photo.conservation_status);
                }
            });
        });
    }

    let has_sex_photos = move || {
        male_image_url.get().is_some() || female_image_url.get().is_some()
    };
    let male_src = move || {
        male_image_url
            .get()
            .unwrap_or_else(|| image_url.get())
    };
    let female_src = move || {
        female_image_url
            .get()
            .unwrap_or_else(|| image_url.get())
    };

    // Conservation badge (if status is known).
    let badge = move || {
        conservation_status.get().map(|cs| {
            let class = format!("conservation-badge {}", cs.css_class());
            let title = cs.label().to_string();
            let code = cs.code().to_string();
            (class, title, code)
        })
    };

    let domains: Vec<String> = species
        .domain
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect();

    let verification_badge = species.verification.as_ref().map(|v| {
        let (label, title) = if v.method.eq_ignore_ascii_case("inaturalist") {
            ("iNat Verified", "Verified with iNaturalist observation")
        } else {
            ("Ornithologist", "Verified by ornithologist")
        };
        (label.to_string(), title.to_string())
    });

    view! {
        <a href={href} class="species-card">
            {if has_sex_photos() {
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
                            src={image_url}
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
                    {domains.iter().map(|d| view! {
                        <span class="domain-badge">{d.clone()}</span>
                    }).collect::<Vec<_>>()}
                    {verification_badge.as_ref().map(|(label, title)| view! {
                        <span class="verification-badge" title={title.clone()}>{label.clone()}</span>
                    })}
                    {badge().map(|(class, title, code)| view! {
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
