//! Settings page – adjust detection thresholds from the web UI.

use leptos::prelude::*;
use leptos::prelude::{
    signal, Effect, ElementChild, IntoView, Resource,
    ServerFnError, Suspense,
};

use crate::model::{DetectionSettings, TaxonomyAdminStatus};

// ─── Default values (match gaia_common::config defaults) ─────────────────────

const DEFAULT_SENSITIVITY: f64 = 1.25;
const DEFAULT_CONFIDENCE: f64 = 0.7;
const DEFAULT_SF_THRESH: f64 = 0.03;
const DEFAULT_OVERLAP: f64 = 0.0;
const DEFAULT_COLORMAP: &str = "default";
const DEFAULT_TZ_OFFSET: i32 = 0;

// ─── Server functions ────────────────────────────────────────────────────────

#[server(prefix = "/api")]
pub async fn get_settings() -> Result<DetectionSettings, ServerFnError> {
    let map = crate::server::kv::get_all_settings()
        .await
        .map_err(|e| ServerFnError::new(format!("KV error: {e}")))?;

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
        tz_offset: map.get("tz_offset").and_then(|v| v.parse::<i32>().ok()).unwrap_or(DEFAULT_TZ_OFFSET),
    })
}

#[server(prefix = "/api")]
pub async fn save_settings(settings: DetectionSettings) -> Result<(), ServerFnError> {
    let sens = settings.sensitivity.to_string();
    let conf = settings.confidence.to_string();
    let sf = settings.sf_thresh.to_string();
    let ovlp = settings.overlap.to_string();
    let cmap = settings.colormap.clone();
    let tz = settings.tz_offset.to_string();

    crate::server::kv::save_settings(
        &[
            ("sensitivity", &sens),
            ("confidence", &conf),
            ("sf_thresh", &sf),
            ("overlap", &ovlp),
            ("colormap", &cmap),
            ("tz_offset", &tz),
        ],
    )
    .await
    .map_err(|e| ServerFnError::new(format!("KV error: {e}")))?;

    Ok(())
}

#[server(prefix = "/api")]
pub async fn get_taxonomy_status() -> Result<TaxonomyAdminStatus, ServerFnError> {
    #[cfg(feature = "ssr")]
    {
        crate::server::taxonomy_admin::taxonomy_status()
            .map_err(ServerFnError::new)
    }
    #[cfg(not(feature = "ssr"))]
    {
        Err(ServerFnError::new("SSR only"))
    }
}

#[server(prefix = "/api")]
pub async fn add_taxonomy_alias(
    canonical: String,
    alias: String,
    class_name: String,
) -> Result<String, ServerFnError> {
    #[cfg(feature = "ssr")]
    {
        let class_opt = if class_name.trim().is_empty() {
            None
        } else {
            Some(class_name.trim())
        };
        crate::server::taxonomy_admin::upsert_species_alias(&canonical, &alias, class_opt)
            .map_err(ServerFnError::new)
    }
    #[cfg(not(feature = "ssr"))]
    {
        Err(ServerFnError::new("SSR only"))
    }
}

#[server(prefix = "/api")]
pub async fn add_class_alias(
    alias: String,
    canonical_class: String,
) -> Result<String, ServerFnError> {
    #[cfg(feature = "ssr")]
    {
        crate::server::taxonomy_admin::upsert_class_alias(&alias, &canonical_class)
            .map_err(ServerFnError::new)
    }
    #[cfg(not(feature = "ssr"))]
    {
        Err(ServerFnError::new("SSR only"))
    }
}

// ─── Page component ──────────────────────────────────────────────────────────

/// Detection settings page.
#[component]
pub fn SettingsPage() -> impl IntoView {
    let (saving, set_saving) = signal(false);
    let (saved_msg, set_saved_msg) = signal::<Option<String>>(None);
    let (error_msg, set_error_msg) = signal::<Option<String>>(None);
    let (tax_msg, set_tax_msg) = signal::<Option<String>>(None);
    let (tax_err, set_tax_err) = signal::<Option<String>>(None);
    let (tax_busy, set_tax_busy) = signal(false);
    let (tax_version, set_tax_version) = signal(0u32);

    let (tax_canonical, set_tax_canonical) = signal(String::new());
    let (tax_alias, set_tax_alias) = signal(String::new());
    let (tax_class, set_tax_class) = signal("birds".to_string());
    let (class_alias_src, set_class_alias_src) = signal(String::new());
    let (class_alias_dst, set_class_alias_dst) = signal("birds".to_string());

    // Fetch current settings on mount.
    let settings_resource = Resource::new(|| (), |_| get_settings());
    let taxonomy_status = Resource::new(
        move || tax_version.get(),
        |_| async { get_taxonomy_status().await },
    );

    // Local signals for each field – initialised from the resource.
    let (sensitivity, set_sensitivity) = signal(DEFAULT_SENSITIVITY);
    let (confidence, set_confidence) = signal(DEFAULT_CONFIDENCE);
    let (sf_thresh, set_sf_thresh) = signal(DEFAULT_SF_THRESH);
    let (overlap, set_overlap) = signal(DEFAULT_OVERLAP);
    let (colormap, set_colormap) = signal(DEFAULT_COLORMAP.to_string());
    let (tz_offset, set_tz_offset) = signal(DEFAULT_TZ_OFFSET);

    // Sync resource → local signals once loaded.
    Effect::new(move || {
        if let Some(Ok(s)) = settings_resource.get() {
            set_sensitivity.set(s.sensitivity);
            set_confidence.set(s.confidence);
            set_sf_thresh.set(s.sf_thresh);
            set_overlap.set(s.overlap);
            set_colormap.set(s.colormap);
            set_tz_offset.set(s.tz_offset);
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
            tz_offset: tz_offset.get(),
        };

        leptos::task::spawn_local(async move {
            match save_settings(payload).await {
                Ok(()) => set_saved_msg.set(Some("Settings saved. Changes take effect on the next analysis cycle.".into())),
                Err(e) => set_error_msg.set(Some(format!("Failed to save: {e}"))),
            }
            set_saving.set(false);
        });
    };

    let on_add_alias = move |_| {
        set_tax_busy.set(true);
        set_tax_msg.set(None);
        set_tax_err.set(None);

        let canonical = tax_canonical.get();
        let alias = tax_alias.get();
        let class_name = tax_class.get();

        leptos::task::spawn_local(async move {
            match add_taxonomy_alias(canonical, alias, class_name).await {
                Ok(msg) => {
                    set_tax_msg.set(Some(msg));
                    set_tax_canonical.set(String::new());
                    set_tax_alias.set(String::new());
                    set_tax_version.update(|v| *v += 1);
                }
                Err(e) => set_tax_err.set(Some(format!("Failed to save alias: {e}"))),
            }
            set_tax_busy.set(false);
        });
    };

    let on_add_class_alias = move |_| {
        set_tax_busy.set(true);
        set_tax_msg.set(None);
        set_tax_err.set(None);

        let alias = class_alias_src.get();
        let canonical_class = class_alias_dst.get();

        leptos::task::spawn_local(async move {
            match add_class_alias(alias, canonical_class).await {
                Ok(msg) => {
                    set_tax_msg.set(Some(msg));
                    set_class_alias_src.set(String::new());
                    set_tax_version.update(|v| *v += 1);
                }
                Err(e) => set_tax_err.set(Some(format!("Failed to save class alias: {e}"))),
            }
            set_tax_busy.set(false);
        });
    };

    view! {
        <div class="settings-page">
            <h1>"Detection Settings"</h1>
            <p class="settings-desc">
                "Adjust the detection thresholds used by the processing server. "
                "Changes are stored in the shared database and picked up on the next analysis cycle."
            </p>

            <Suspense fallback=|| view! { <p class="text-muted">"Loading settings…"</p> }>
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

                    // ── Timezone offset ──────────────────────────
                    <div class="setting-group">
                        <label class="setting-label" for="tz_offset">
                            "Timezone"
                            <span class="setting-value">{move || {
                                let h = tz_offset.get();
                                if h >= 0 { format!("UTC+{h}") } else { format!("UTC{h}") }
                            }}</span>
                        </label>
                        <p class="setting-help">
                            "Offset from UTC for displayed timestamps (no DST). "
                            "Recordings are always stored in UTC; this only affects the web interface."
                        </p>
                        <select
                            id="tz_offset"
                            class="setting-select"
                            on:change=move |ev| {
                                if let Ok(v) = event_target_value(&ev).parse::<i32>() {
                                    set_tz_offset.set(v);
                                }
                            }
                        >
                            <option value="-12" selected=move || tz_offset.get() == -12>"UTC−12"</option>
                            <option value="-11" selected=move || tz_offset.get() == -11>"UTC−11"</option>
                            <option value="-10" selected=move || tz_offset.get() == -10>"UTC−10 (Hawaii)"</option>
                            <option value="-9"  selected=move || tz_offset.get() == -9>"UTC−9 (Alaska)"</option>
                            <option value="-8"  selected=move || tz_offset.get() == -8>"UTC−8 (Pacific)"</option>
                            <option value="-7"  selected=move || tz_offset.get() == -7>"UTC−7 (Mountain)"</option>
                            <option value="-6"  selected=move || tz_offset.get() == -6>"UTC−6 (Central)"</option>
                            <option value="-5"  selected=move || tz_offset.get() == -5>"UTC−5 (Eastern)"</option>
                            <option value="-4"  selected=move || tz_offset.get() == -4>"UTC−4 (Atlantic)"</option>
                            <option value="-3"  selected=move || tz_offset.get() == -3>"UTC−3"</option>
                            <option value="-2"  selected=move || tz_offset.get() == -2>"UTC−2"</option>
                            <option value="-1"  selected=move || tz_offset.get() == -1>"UTC−1"</option>
                            <option value="0"   selected=move || tz_offset.get() == 0>"UTC+0 (UTC)"</option>
                            <option value="1"   selected=move || tz_offset.get() == 1>"UTC+1 (CET)"</option>
                            <option value="2"   selected=move || tz_offset.get() == 2>"UTC+2 (EET)"</option>
                            <option value="3"   selected=move || tz_offset.get() == 3>"UTC+3 (Moscow)"</option>
                            <option value="4"   selected=move || tz_offset.get() == 4>"UTC+4"</option>
                            <option value="5"   selected=move || tz_offset.get() == 5>"UTC+5"</option>
                            <option value="6"   selected=move || tz_offset.get() == 6>"UTC+6"</option>
                            <option value="7"   selected=move || tz_offset.get() == 7>"UTC+7"</option>
                            <option value="8"   selected=move || tz_offset.get() == 8>"UTC+8"</option>
                            <option value="9"   selected=move || tz_offset.get() == 9>"UTC+9 (Japan)"</option>
                            <option value="10"  selected=move || tz_offset.get() == 10>"UTC+10 (AEST)"</option>
                            <option value="11"  selected=move || tz_offset.get() == 11>"UTC+11"</option>
                            <option value="12"  selected=move || tz_offset.get() == 12>"UTC+12 (NZST)"</option>
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

                    // ── Taxonomy Admin ─────────────────────────
                    <div class="setting-group">
                        <label class="setting-label">"Taxonomy Admin"</label>
                        <p class="setting-help">
                            "Add scientific-name aliases and class aliases without editing TOML manually. "
                            "This updates the shared taxonomy table used by processing startup."
                        </p>

                        <Suspense fallback=|| view! { <p class="text-muted">"Loading taxonomy status…"</p> }>
                            {move || taxonomy_status.get().map(|res| match res {
                                Ok(s) => view! {
                                    <p class="setting-help">
                                        "Table: " <code>{s.path.clone()}</code>
                                        " · exists=" {s.exists.to_string()}
                                        " · writable=" {s.writable.to_string()}
                                        " · species=" {s.species_count}
                                        " · aliases=" {s.alias_count}
                                        " · class aliases=" {s.class_alias_count}
                                    </p>
                                }.into_any(),
                                Err(e) => view! {
                                    <p class="settings-error">"Taxonomy status error: " {e.to_string()}</p>
                                }.into_any(),
                            })}
                        </Suspense>

                        <div class="taxonomy-admin-row">
                            <input
                                class="setting-input"
                                placeholder="Canonical scientific name (e.g. Cyanocorax morio)"
                                prop:value=move || tax_canonical.get()
                                on:input=move |ev| set_tax_canonical.set(event_target_value(&ev))
                            />
                            <input
                                class="setting-input"
                                placeholder="Equivalent alias (e.g. Psilorhinus morio)"
                                prop:value=move || tax_alias.get()
                                on:input=move |ev| set_tax_alias.set(event_target_value(&ev))
                            />
                            <input
                                class="setting-input"
                                placeholder="Class (optional, e.g. birds)"
                                prop:value=move || tax_class.get()
                                on:input=move |ev| set_tax_class.set(event_target_value(&ev))
                            />
                            <button
                                class="btn btn-primary"
                                on:click=on_add_alias
                                disabled=move || tax_busy.get()
                            >
                                {move || if tax_busy.get() { "Saving…" } else { "Add Species Alias" }}
                            </button>
                        </div>

                        <div class="taxonomy-admin-row">
                            <input
                                class="setting-input"
                                placeholder="Class alias (e.g. AVES)"
                                prop:value=move || class_alias_src.get()
                                on:input=move |ev| set_class_alias_src.set(event_target_value(&ev))
                            />
                            <input
                                class="setting-input"
                                placeholder="Canonical class (e.g. birds)"
                                prop:value=move || class_alias_dst.get()
                                on:input=move |ev| set_class_alias_dst.set(event_target_value(&ev))
                            />
                            <button
                                class="btn btn-primary"
                                on:click=on_add_class_alias
                                disabled=move || tax_busy.get()
                            >
                                {move || if tax_busy.get() { "Saving…" } else { "Add Class Alias" }}
                            </button>
                        </div>

                        <p class="setting-help">
                            "Note: changes affect new processing runs after processing services reload taxonomy at startup. "
                            "Use the same GAIA_TAXONOMY_TABLE path in web + processing for shared edits."
                        </p>

                        {move || tax_msg.get().map(|msg| view! {
                            <div class="settings-success">{msg}</div>
                        })}

                        {move || tax_err.get().map(|msg| view! {
                            <div class="settings-error">{msg}</div>
                        })}
                    </div>

                </div>
            </Suspense>
        </div>
    }
}
