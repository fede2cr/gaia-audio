//! Card component for a single detection in the live feed.

use leptos::*;

use crate::model::WebDetection;

/// Renders a detection card with spectrogram, species info, capture node, and audio player.
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

    let datetime = format!("{} {}", &detection.date, &detection.time);
    let source_label = detection.source_label();

    // URLs for the extracted audio clip and its spectrogram
    let audio_url = detection.clip_url();
    let spectrogram_url = detection.spectrogram_url();

    let species_href = format!("/species/{}", urlencoded(&detection.scientific_name));

    view! {
        <div class="detection-card">
            // Spectrogram thumbnail (left side)
            {spectrogram_url.map(|url| view! {
                <div class="detection-spectrogram">
                    <img src={url} alt="spectrogram" loading="lazy"/>
                </div>
            })}

            // Detection details (right side)
            <div class="detection-info">
                <a href={species_href} class="detection-species">
                    <span class="common-name">{&detection.common_name}</span>
                    <span class="sci-name">{&detection.scientific_name}</span>
                </a>
                <div class="detection-meta">
                    <span class="domain-badge">{&detection.domain}</span>
                    <span class={confidence_class}>{confidence_pct}</span>
                    <span class="source-badge" title="Capture node">{source_label}</span>
                </div>
                <div class="detection-timestamp">
                    <svg class="icon-clock" viewBox="0 0 16 16" width="14" height="14">
                        <circle cx="8" cy="8" r="7" fill="none" stroke="currentColor" stroke-width="1.5"/>
                        <polyline points="8,4 8,8 11,10" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
                    </svg>
                    <time>{datetime}</time>
                </div>
                {audio_url.map(|url| view! {
                    <audio class="detection-audio" controls preload="none">
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
