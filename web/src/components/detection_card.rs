//! Card component for a single detection in the live feed.

use leptos::*;

use crate::model::WebDetection;

/// Renders a detection card with species image, spectrogram, species info, capture node, and audio player.
#[component]
pub fn DetectionCard(detection: WebDetection) -> impl IntoView {
    let confidence_pct = format!("{:.0}%", detection.confidence * 100.0);
    let confidence_class = if detection.confidence >= 0.8 {
        "confidence high"
    } else if detection.confidence >= 0.5 {
        "confidence medium"
    } else {
        "confidence low"
    };

    let datetime = if detection.display_date.is_empty() {
        format!("{} {}", &detection.date, &detection.time)
    } else {
        format!("{} {}", &detection.display_date, &detection.display_time)
    };
    let source_label = detection.source_label();
    let model_label = detection.model_label();
    let is_excluded = detection.excluded;

    // URLs for the extracted audio clip and its spectrogram
    let audio_url = detection.clip_url();
    let spectrogram_url = detection.spectrogram_url();
    let species_image = detection.image_url.clone();
    let common_name_alt = detection.common_name.clone();

    let species_href = format!("/species/{}", urlencoded(&detection.scientific_name));

    let card_class = if is_excluded {
        "detection-card excluded"
    } else {
        "detection-card"
    };

    view! {
        <div class={card_class}>
            // Species photo (left side)
            <div class="detection-thumb">
                {match species_image {
                    Some(url) => view! {
                        <img class="species-thumb" src={url} alt={common_name_alt} loading="lazy"/>
                    }.into_view(),
                    None => view! {
                        <div class="species-thumb-placeholder">
                            <svg viewBox="0 0 24 24" width="32" height="32" fill="none" stroke="currentColor" stroke-width="1.5">
                                <path d="M12 3c-1.5 2-4 3-6 3 0 4 1.5 8 6 11 4.5-3 6-7 6-11-2 0-4.5-1-6-3z"/>
                            </svg>
                        </div>
                    }.into_view(),
                }}
            </div>

            // Detection details (center)
            <div class="detection-info">
                <a href={species_href} class="detection-species">
                    <span class="common-name">{&detection.common_name}</span>
                    <span class="sci-name">{&detection.scientific_name}</span>
                </a>
                <div class="detection-meta">
                    <span class="domain-badge">{&detection.domain}</span>
                    <span class={confidence_class}>{confidence_pct}</span>
                    <span class="model-badge" title="Detection model">"🧠 " {model_label}</span>
                    {if is_excluded {
                        view! { <span class="excluded-badge">"Excluded"</span> }.into_view()
                    } else {
                        view! {}.into_view()
                    }}
                    <span class="source-badge" title="Capture node">{source_label}</span>
                </div>
                <div class="detection-timestamp">
                    <svg class="icon-clock" viewBox="0 0 16 16" width="14" height="14">
                        <circle cx="8" cy="8" r="7" fill="none" stroke="currentColor" stroke-width="1.5"/>
                        <polyline points="8,4 8,8 11,10" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
                    </svg>
                    <time>{datetime}</time>
                </div>

                // Spectrogram inline (below metadata)
                {spectrogram_url.map(|url| view! {
                    <div class="detection-spectrogram">
                        <img src={url} alt="spectrogram" loading="lazy"/>
                    </div>
                })}

                {audio_url.map(|url| view! {
                    <audio class="detection-audio" controls preload="metadata">
                        <source src={url} type="audio/wav"/>
                    </audio>
                })}
            </div>
        </div>
    }
}

fn urlencoded(s: &str) -> String {
    s.replace(' ', "%20")
}
