//! Live analysis panel – shows the spectrogram currently being processed
//! and the top model predictions (even when no detection crosses the
//! confidence threshold).

use leptos::*;

use crate::model::{LivePrediction, LiveStatus};

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

// ─── Component ───────────────────────────────────────────────────────────────

/// Live analysis panel that polls the processing server's current state.
#[component]
pub fn LiveAnalysis() -> impl IntoView {
    let (status, set_status) = create_signal::<Option<LiveStatus>>(None);
    #[allow(unused_variables)]
    let (tick, set_tick) = create_signal(0u64);

    // Initial load
    let initial = create_resource(|| (), |_| async { get_live_status().await });
    create_effect(move |_| {
        if let Some(Ok(s)) = initial.get() {
            set_status.set(s);
        }
    });

    // Poll every 4 seconds (hydrate/WASM only)
    #[cfg(feature = "hydrate")]
    {
        set_interval_with_handle(
            move || {
                set_tick.update(|t| *t += 1);
                spawn_local(async move {
                    if let Ok(s) = get_live_status().await {
                        set_status.set(s);
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
                        // Cache-bust the image URL with a tick counter.
                        let t = tick.get();
                        let img_url = format!("/live/live_spectrogram.png?t={t}");
                        let filename = st.filename.clone();
                        let has_det = st.has_detections;
                        let preds = st.predictions.clone();

                        view! {
                            <div class="live-analysis-card">
                                <div class="live-spectrogram">
                                    <img src={img_url} alt="Live spectrogram"/>
                                    {if has_det {
                                        view! { <span class="live-badge detection">"Detection!"</span> }.into_view()
                                    } else {
                                        view! { <span class="live-badge listening">"Listening…"</span> }.into_view()
                                    }}
                                </div>
                                <div class="live-details">
                                    <p class="live-filename" title={filename.clone()}>
                                        <svg class="icon-mic" viewBox="0 0 16 16" width="14" height="14">
                                            <rect x="5" y="1" width="6" height="9" rx="3" fill="none" stroke="currentColor" stroke-width="1.5"/>
                                            <path d="M3 7.5 a5 5 0 0 0 10 0" fill="none" stroke="currentColor" stroke-width="1.5"/>
                                            <line x1="8" y1="13" x2="8" y2="15" stroke="currentColor" stroke-width="1.5"/>
                                        </svg>
                                        {filename}
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

                view! {
                    <li class="prediction-item">
                        <span class="pred-name" title={p.scientific_name.clone()}>
                            {&p.common_name}
                        </span>
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
