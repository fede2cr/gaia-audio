//! Card component for a single detection in the live feed.

use leptos::*;

use crate::model::WebDetection;

/// Renders a single detection row with species name, confidence, date/time, and audio player.
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

    // Build the URL to the extracted audio clip.
    // Clips live at: {extracted_dir}/By_Date/{date}/{common_name_safe}/{file_name}
    // Served via /extracted/By_Date/...
    let audio_url = detection.clip_url();

    view! {
        <div class="detection-card">
            <div class="detection-info">
                <a href={format!("/species/{}", urlencoded(&detection.scientific_name))}
                   class="detection-species">
                    <span class="common-name">{&detection.common_name}</span>
                    <span class="sci-name">{&detection.scientific_name}</span>
                </a>
                <div class="detection-meta">
                    <span class="domain-badge">{&detection.domain}</span>
                    <span class={confidence_class}>{confidence_pct}</span>
                    <span class="detection-time">{datetime}</span>
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
