//! Settings page – adjust detection thresholds from the web UI.

use leptos::*;

use crate::model::DetectionSettings;

// ─── Default values (match gaia_common::config defaults) ─────────────────────

const DEFAULT_SENSITIVITY: f64 = 1.25;
const DEFAULT_CONFIDENCE: f64 = 0.7;
const DEFAULT_SF_THRESH: f64 = 0.03;
const DEFAULT_OVERLAP: f64 = 0.0;
const DEFAULT_COLORMAP: &str = "default";

// ─── Server functions ────────────────────────────────────────────────────────

#[server(GetSettings, "/api")]
pub async fn get_settings() -> Result<DetectionSettings, ServerFnError> {
    let state = use_context::<crate::app::AppState>()
        .ok_or_else(|| ServerFnError::new("Missing AppState"))?;

    let map = crate::server::db::get_all_settings(&state.db_path)
        .map_err(|e| ServerFnError::new(format!("DB error: {e}")))?;

    let f = |key: &str, default: f64| -> f64 {
        map.get(key)
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(default)
    };

    Ok(DetectionSettings {
        sensitivity: f("sensitivity", DEFAULT_SENSITIVITY),
        confidence: f("confidence", DEFAULT_CONFIDENCE),
        sf_thresh: f("sf_thresh", DEFAULT_SF_THRESH),
        overlap: f("overlap", DEFAULT_OVERLAP),
        colormap: map.get("colormap").cloned().unwrap_or_else(|| DEFAULT_COLORMAP.to_string()),
    })
}

#[server(SaveSettings, "/api")]
pub async fn save_settings(settings: DetectionSettings) -> Result<(), ServerFnError> {
    let state = use_context::<crate::app::AppState>()
        .ok_or_else(|| ServerFnError::new("Missing AppState"))?;

    let sens = settings.sensitivity.to_string();
    let conf = settings.confidence.to_string();
    let sf = settings.sf_thresh.to_string();
    let ovlp = settings.overlap.to_string();
    let cmap = settings.colormap.clone();

    crate::server::db::save_settings(
        &state.db_path,
        &[
            ("sensitivity", &sens),
            ("confidence", &conf),
            ("sf_thresh", &sf),
            ("overlap", &ovlp),
            ("colormap", &cmap),
        ],
    )
    .map_err(|e| ServerFnError::new(format!("DB error: {e}")))?;

    Ok(())
}

// ─── Page component ──────────────────────────────────────────────────────────

/// Detection settings page.
#[component]
pub fn SettingsPage() -> impl IntoView {
    let (saving, set_saving) = create_signal(false);
    let (saved_msg, set_saved_msg) = create_signal::<Option<String>>(None);
    let (error_msg, set_error_msg) = create_signal::<Option<String>>(None);

    // Fetch current settings on mount.
    let settings_resource = create_resource(|| (), |_| get_settings());

    // Local signals for each field – initialised from the resource.
    let (sensitivity, set_sensitivity) = create_signal(DEFAULT_SENSITIVITY);
    let (confidence, set_confidence) = create_signal(DEFAULT_CONFIDENCE);
    let (sf_thresh, set_sf_thresh) = create_signal(DEFAULT_SF_THRESH);
    let (overlap, set_overlap) = create_signal(DEFAULT_OVERLAP);
    let (colormap, set_colormap) = create_signal(DEFAULT_COLORMAP.to_string());

    // Sync resource → local signals once loaded.
    create_effect(move |_| {
        if let Some(Ok(s)) = settings_resource.get() {
            set_sensitivity.set(s.sensitivity);
            set_confidence.set(s.confidence);
            set_sf_thresh.set(s.sf_thresh);
            set_overlap.set(s.overlap);
            set_colormap.set(s.colormap);
        }
    });

    let on_save = move |_| {
        set_saving.set(true);
        set_saved_msg.set(None);
        set_error_msg.set(None);

        let payload = DetectionSettings {
            sensitivity: sensitivity.get(),
            confidence: confidence.get(),
            sf_thresh: sf_thresh.get(),
            overlap: overlap.get(),
            colormap: colormap.get(),
        };

        spawn_local(async move {
            match save_settings(payload).await {
                Ok(()) => set_saved_msg.set(Some("Settings saved. Changes take effect on the next analysis cycle.".into())),
                Err(e) => set_error_msg.set(Some(format!("Failed to save: {e}"))),
            }
            set_saving.set(false);
        });
    };

    view! {
        <div class="settings-page">
            <h1>"Detection Settings"</h1>
            <p class="settings-desc">
                "Adjust the detection thresholds used by the processing server. "
                "Changes are stored in the shared database and picked up on the next analysis cycle."
            </p>

            <Suspense fallback=move || view! { <p class="text-muted">"Loading settings…"</p> }>
                <div class="settings-form">

                    // ── Sensitivity ──────────────────────────────
                    <div class="setting-group">
                        <label class="setting-label" for="sensitivity">
                            "Sensitivity"
                            <span class="setting-value">{move || format!("{:.2}", sensitivity.get())}</span>
                        </label>
                        <p class="setting-help">
                            "Controls the steepness of the sigmoid applied to model logits. "
                            "Higher values produce higher confidence scores (more detections). "
                            "Range: 0.5 – 1.5. Default: 1.25."
                        </p>
                        <input
                            id="sensitivity"
                            type="range"
                            class="setting-slider"
                            min="0.5"
                            max="1.5"
                            step="0.05"
                            prop:value=move || sensitivity.get().to_string()
                            on:input=move |ev| {
                                if let Ok(v) = event_target_value(&ev).parse::<f64>() {
                                    set_sensitivity.set(v);
                                }
                            }
                        />
                        <div class="setting-range-labels">
                            <span>"0.5 (conservative)"</span>
                            <span>"1.5 (aggressive)"</span>
                        </div>
                    </div>

                    // ── Confidence threshold ─────────────────────
                    <div class="setting-group">
                        <label class="setting-label" for="confidence">
                            "Confidence Threshold"
                            <span class="setting-value">{move || format!("{:.0}%", confidence.get() * 100.0)}</span>
                        </label>
                        <p class="setting-help">
                            "Minimum confidence score to report a detection. "
                            "Lower values catch more species but increase false positives. "
                            "Range: 10% – 99%. Default: 70%."
                        </p>
                        <input
                            id="confidence"
                            type="range"
                            class="setting-slider"
                            min="0.1"
                            max="0.99"
                            step="0.01"
                            prop:value=move || confidence.get().to_string()
                            on:input=move |ev| {
                                if let Ok(v) = event_target_value(&ev).parse::<f64>() {
                                    set_confidence.set(v);
                                }
                            }
                        />
                        <div class="setting-range-labels">
                            <span>"10% (sensitive)"</span>
                            <span>"99% (strict)"</span>
                        </div>
                    </div>

                    // ── Species filter threshold ─────────────────
                    <div class="setting-group">
                        <label class="setting-label" for="sf_thresh">
                            "Species Filter Threshold"
                            <span class="setting-value">{move || format!("{:.3}", sf_thresh.get())}</span>
                        </label>
                        <p class="setting-help">
                            "Minimum score from the species-range model to include a species as expected "
                            "at this location. Lower values allow more species through the location filter. "
                            "Range: 0.001 – 0.1. Default: 0.03."
                        </p>
                        <input
                            id="sf_thresh"
                            type="range"
                            class="setting-slider"
                            min="0.001"
                            max="0.1"
                            step="0.001"
                            prop:value=move || sf_thresh.get().to_string()
                            on:input=move |ev| {
                                if let Ok(v) = event_target_value(&ev).parse::<f64>() {
                                    set_sf_thresh.set(v);
                                }
                            }
                        />
                        <div class="setting-range-labels">
                            <span>"0.001 (permissive)"</span>
                            <span>"0.1 (strict)"</span>
                        </div>
                    </div>

                    // ── Overlap ──────────────────────────────────
                    <div class="setting-group">
                        <label class="setting-label" for="overlap">
                            "Chunk Overlap"
                            <span class="setting-value">{move || format!("{:.1}s", overlap.get())}</span>
                        </label>
                        <p class="setting-help">
                            "Overlap in seconds between consecutive 3-second audio chunks. "
                            "Higher overlap may catch more detections but increases processing time. "
                            "Range: 0 – 2.5s. Default: 0."
                        </p>
                        <input
                            id="overlap"
                            type="range"
                            class="setting-slider"
                            min="0.0"
                            max="2.5"
                            step="0.5"
                            prop:value=move || overlap.get().to_string()
                            on:input=move |ev| {
                                if let Ok(v) = event_target_value(&ev).parse::<f64>() {
                                    set_overlap.set(v);
                                }
                            }
                        />
                        <div class="setting-range-labels">
                            <span>"0s (no overlap)"</span>
                            <span>"2.5s (heavy)"</span>
                        </div>
                    </div>

                    // ── Colormap ─────────────────────────────────
                    <div class="setting-group">
                        <label class="setting-label" for="colormap">
                            "Spectrogram Colormap"
                        </label>
                        <p class="setting-help">
                            "Colour palette used for spectrogram images. "
                            "Changes apply to newly generated spectrograms."
                        </p>
                        <select
                            id="colormap"
                            class="setting-select"
                            on:change=move |ev| {
                                set_colormap.set(event_target_value(&ev));
                            }
                        >
                            <option value="default"  selected=move || colormap.get() == "default">"Default (green → yellow → red)"</option>
                            <option value="coolwarm" selected=move || colormap.get() == "coolwarm">"Coolwarm (blue → red)"</option>
                            <option value="magma"    selected=move || colormap.get() == "magma">"Magma (black → magenta → yellow)"</option>
                            <option value="viridis"  selected=move || colormap.get() == "viridis">"Viridis (purple → green → yellow)"</option>
                            <option value="grayscale" selected=move || colormap.get() == "grayscale">"Grayscale"</option>
                        </select>
                    </div>

                    // ── Save button + messages ───────────────────
                    <div class="settings-actions">
                        <button
                            class="btn btn-primary"
                            on:click=on_save
                            disabled=move || saving.get()
                        >
                            {move || if saving.get() { "Saving…" } else { "Save Settings" }}
                        </button>
                    </div>

                    {move || saved_msg.get().map(|msg| view! {
                        <div class="settings-success">{msg}</div>
                    })}

                    {move || error_msg.get().map(|msg| view! {
                        <div class="settings-error">{msg}</div>
                    })}

                </div>
            </Suspense>
        </div>
    }
}
