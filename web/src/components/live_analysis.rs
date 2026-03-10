//! Live analysis panel – shows the spectrogram currently being processed
//! and the top model predictions (even when no detection crosses the
//! confidence threshold).
//!
//! The component polls `/api/GetLiveStatus` and **only updates the DOM when
//! the `timestamp` field changes**, preventing the UI from blinking when
//! there is nothing new to show.  Spectrogram image and prediction bars
//! crossfade smoothly via CSS transitions.

use leptos::*;

use crate::model::{LivePrediction, LiveStatus};

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Extract a short node label from a capture URL.
///
/// Uses `NODE_NAME` env var for local sources, otherwise strips the URL
/// to just the hostname.
fn node_label(url: &str) -> String {
    if url.is_empty() {
        return node_name_or_local();
    }
    let stripped = url
        .strip_prefix("http://")
        .or_else(|| url.strip_prefix("https://"))
        .unwrap_or(url);
    let host = stripped.split(':').next().unwrap_or(stripped).trim_end_matches('.');
    if host == "localhost" || host.starts_with("127.") {
        return node_name_or_local();
    }
    host.to_string()
}

/// Return the `NODE_NAME` env var if set, otherwise `"local"`.
fn node_name_or_local() -> String {
    std::env::var("NODE_NAME")
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "local".into())
}

/// Format an ISO-8601 timestamp with a UTC offset into
/// a human-readable string like "4:21pm (UTC-6)".
///
/// Pure arithmetic — no chrono dependency (WASM-safe).
fn format_capture_time(iso: &str, offset_hours: i32) -> String {
    if iso.len() < 16 {
        return String::new();
    }
    let parse = || -> Option<String> {
        let year: i32 = iso[0..4].parse().ok()?;
        let month: u32 = iso[5..7].parse().ok()?;
        let day: u32 = iso[8..10].parse().ok()?;
        let hour: i32 = iso[11..13].parse().ok()?;
        let min: u32 = iso[14..16].parse().ok()?;

        let mut h = hour + offset_hours;
        let mut d = day as i32;
        let mut m = month as i32;
        let mut y = year;
        if h >= 24 {
            h -= 24;
            d += 1;
            let days_in_month = match m {
                1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
                4 | 6 | 9 | 11 => 30,
                2 => if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) { 29 } else { 28 },
                _ => 31,
            };
            if d > days_in_month {
                d = 1;
                m += 1;
                if m > 12 { m = 1; y += 1; }
            }
        } else if h < 0 {
            h += 24;
            d -= 1;
            if d < 1 {
                m -= 1;
                if m < 1 { m = 12; y -= 1; }
                d = match m {
                    1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
                    4 | 6 | 9 | 11 => 30,
                    2 => if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) { 29 } else { 28 },
                    _ => 31,
                };
            }
        }

        let (h12, ampm) = match h {
            0 => (12, "am"),
            1..=11 => (h, "am"),
            12 => (12, "pm"),
            _ => (h - 12, "pm"),
        };

        let tz_label = if offset_hours >= 0 {
            format!("UTC+{offset_hours}")
        } else {
            format!("UTC{offset_hours}")
        };

        Some(format!("{y}-{m:02}-{d:02} {h12}:{min:02}{ampm} ({tz_label})"))
    };
    parse().unwrap_or_default()
}

// ─── Server function ─────────────────────────────────────────────────────────

#[server(GetLiveStatus, "/api")]
pub async fn get_live_status() -> Result<Option<LiveStatus>, ServerFnError> {
    let data_dir = std::env::var("GAIA_DATA_DIR").unwrap_or_else(|_| "/data".into());
    let path = std::path::PathBuf::from(&data_dir).join("live_status.json");

    if !path.exists() {
        return Ok(None);
    }

    let contents = std::fs::read_to_string(&path)
        .map_err(|e| ServerFnError::new(format!("Cannot read live_status.json: {e}")))?;

    let status: LiveStatus = serde_json::from_str(&contents)
        .map_err(|e| ServerFnError::new(format!("Cannot parse live_status.json: {e}")))?;

    Ok(Some(status))
}

#[server(GetTzOffset, "/api")]
pub async fn get_tz_offset() -> Result<i32, ServerFnError> {
    let state = use_context::<crate::app::AppState>()
        .ok_or_else(|| ServerFnError::new("Missing AppState"))?;
    let map = crate::server::db::get_all_settings(&state.db_path)
        .map_err(|e| ServerFnError::new(format!("DB error: {e}")))?;
    Ok(map.get("tz_offset").and_then(|v| v.parse::<i32>().ok()).unwrap_or(0))
}

// ─── Component ───────────────────────────────────────────────────────────────

/// Live analysis panel that polls the processing server's current state.
///
/// DOM is only updated when `live_status.json` has a new timestamp, so
/// when nothing is being processed the card stays perfectly still.
#[component]
pub fn LiveAnalysis() -> impl IntoView {
    // Reactive signals that drive the view.
    let (status, set_status) = create_signal::<Option<LiveStatus>>(None);
    let (img_url, set_img_url) = create_signal(String::new());
    let (tz_offset, set_tz_offset) = create_signal(0_i32);

    // Track the last-seen timestamp so we can skip no-op updates.
    let last_ts = store_value(String::new());

    // Helper: only push to signals when the timestamp actually changed.
    let apply_update = move |new: Option<LiveStatus>| {
        match &new {
            Some(st) if st.timestamp != last_ts.get_value() => {
                last_ts.set_value(st.timestamp.clone());
                // Cache-bust image URL with the timestamp itself.
                set_img_url.set(format!(
                    "/live/live_spectrogram.png?t={}",
                    st.timestamp.replace([':', ' '], "-")
                ));
                set_status.set(Some(st.clone()));
            }
            Some(_) => { /* same timestamp – do nothing */ }
            None => {
                if !last_ts.get_value().is_empty() {
                    last_ts.set_value(String::new());
                    set_status.set(None);
                }
            }
        }
    };

    // Initial load
    let initial = create_resource(|| (), |_| async { get_live_status().await });
    let tz_res = create_resource(|| (), |_| async { get_tz_offset().await });
    create_effect(move |_| {
        if let Some(Ok(s)) = initial.get() {
            apply_update(s);
        }
    });
    create_effect(move |_| {
        if let Some(Ok(off)) = tz_res.get() {
            set_tz_offset.set(off);
        }
    });

    // Poll every 4 seconds (hydrate/WASM only)
    #[cfg(feature = "hydrate")]
    {
        set_interval_with_handle(
            move || {
                spawn_local(async move {
                    if let Ok(s) = get_live_status().await {
                        apply_update(s);
                    }
                });
            },
            std::time::Duration::from_secs(4),
        )
        .ok();
    }

    view! {
        <div class="live-analysis">
            <h2>"Live Analysis"</h2>
            {move || {
                match status.get() {
                    None => view! {
                        <div class="live-analysis-empty">
                            <p class="text-muted">"Waiting for processing server…"</p>
                        </div>
                    }.into_view(),
                    Some(st) => {
                        let url = img_url.get();
                        let has_det = st.has_detections;
                        let preds = st.predictions.clone();
                        let node = node_label(&st.source_node);
                        let cap_time = format_capture_time(&st.captured_at, tz_offset.get());

                        view! {
                            <div class="live-analysis-card">
                                <div class="live-spectrogram">
                                    <img class="live-spectrogram-img" src={url} alt="Live spectrogram"/>
                                    {if has_det {
                                        view! { <span class="live-badge detection">"Detection!"</span> }.into_view()
                                    } else {
                                        view! { <span class="live-badge listening">"Listening…"</span> }.into_view()
                                    }}
                                </div>
                                <div class="live-details">
                                    <p class="live-filename">
                                        <svg class="icon-mic" viewBox="0 0 16 16" width="14" height="14">
                                            <rect x="5" y="1" width="6" height="9" rx="3" fill="none" stroke="currentColor" stroke-width="1.5"/>
                                            <path d="M3 7.5 a5 5 0 0 0 10 0" fill="none" stroke="currentColor" stroke-width="1.5"/>
                                            <line x1="8" y1="13" x2="8" y2="15" stroke="currentColor" stroke-width="1.5"/>
                                        </svg>
                                        "Now processing from "
                                        <strong>{node}</strong>
                                        {if !cap_time.is_empty() {
                                            view! { <span>", captured at " {cap_time}</span> }.into_view()
                                        } else {
                                            view! {}.into_view()
                                        }}
                                    </p>
                                    <div class="live-predictions">
                                        <LivePredictionList predictions=preds/>
                                    </div>
                                </div>
                            </div>
                        }.into_view()
                    }
                }
            }}
        </div>
    }
}

/// List of predictions with inline confidence bars.
#[component]
fn LivePredictionList(predictions: Vec<LivePrediction>) -> impl IntoView {
    if predictions.is_empty() {
        return view! { <p class="text-muted">"No predictions"</p> }.into_view();
    }

    view! {
        <ul class="prediction-list">
            {predictions.into_iter().map(|p| {
                let pct = format!("{:.0}%", p.confidence * 100.0);
                let width = format!("{}%", (p.confidence * 100.0).min(100.0));
                let bar_class = if p.confidence >= 0.7 {
                    "pred-bar high"
                } else if p.confidence >= 0.4 {
                    "pred-bar medium"
                } else {
                    "pred-bar low"
                };
                let model_tag = if !p.model_name.is_empty() {
                    p.model_name.clone()
                } else if !p.model_slug.is_empty() {
                    p.model_slug.clone()
                } else {
                    String::new()
                };

                view! {
                    <li class="prediction-item">
                        <span class="pred-name" title={p.scientific_name.clone()}>
                            {&p.common_name}
                        </span>
                        {if !model_tag.is_empty() {
                            view! { <span class="pred-model">{model_tag}</span> }.into_view()
                        } else {
                            view! {}.into_view()
                        }}
                        <div class="pred-bar-track">
                            <div class={bar_class} style={format!("width: {width}")}></div>
                        </div>
                        <span class="pred-confidence">{pct}</span>
                    </li>
                }
            }).collect_view()}
        </ul>
    }.into_view()
}
