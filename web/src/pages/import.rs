//! Import page – analyse and import a BirdNET-Pi backup archive.

use leptos::*;

use crate::model::{ImportReport, ImportResult};

// ─── Server functions ────────────────────────────────────────────────────────

/// Analyse a BirdNET-Pi backup tar without importing.
#[server(AnalyseBackup, "/api")]
pub async fn analyse_backup(tar_path: String) -> Result<ImportReport, ServerFnError> {
    use crate::server::import;
    use std::path::Path;

    let path = Path::new(&tar_path);
    let report = import::analyse_backup(path).map_err(|e| ServerFnError::new(e))?;

    // Convert from server::import types to shared model types
    Ok(ImportReport {
        tar_path: report.tar_path,
        tar_size_bytes: report.tar_size_bytes,
        total_detections: report.total_detections,
        today_detections: report.today_detections,
        total_species: report.total_species,
        today_species: report.today_species,
        date_min: report.date_min,
        date_max: report.date_max,
        audio_file_count: report.audio_file_count,
        spectrogram_count: report.spectrogram_count,
        latitude: report.latitude,
        longitude: report.longitude,
        top_species: report.top_species,
    })
}

/// Perform the full import of a BirdNET-Pi backup.
#[server(RunImport, "/api")]
pub async fn run_import(tar_path: String) -> Result<ImportResult, ServerFnError> {
    use crate::server::import;
    use std::path::Path;

    let state = use_context::<crate::app::AppState>()
        .ok_or_else(|| ServerFnError::new("Missing AppState"))?;

    let path = Path::new(&tar_path);
    let result = import::import_backup(path, &state.db_path, &state.extracted_dir)
        .map_err(|e| ServerFnError::new(e))?;

    Ok(ImportResult {
        detections_imported: result.detections_imported,
        files_extracted: result.files_extracted,
        skipped_existing: result.skipped_existing,
        errors: result.errors,
    })
}

// ─── Page component ──────────────────────────────────────────────────────────

/// BirdNET-Pi backup import wizard.
#[component]
pub fn ImportPage() -> impl IntoView {
    let (tar_path, set_tar_path) = create_signal(String::new());
    let (report, set_report) = create_signal::<Option<ImportReport>>(None);
    let (import_result, set_import_result) = create_signal::<Option<ImportResult>>(None);
    let (analysing, set_analysing) = create_signal(false);
    let (importing, set_importing) = create_signal(false);
    let (error_msg, set_error_msg) = create_signal::<Option<String>>(None);

    // Analyse button handler
    let on_analyse = move |_| {
        let path = tar_path.get();
        if path.is_empty() {
            set_error_msg.set(Some("Please enter a backup file path.".into()));
            return;
        }
        set_error_msg.set(None);
        set_report.set(None);
        set_import_result.set(None);
        set_analysing.set(true);

        spawn_local(async move {
            match analyse_backup(path).await {
                Ok(r) => {
                    set_report.set(Some(r));
                    set_error_msg.set(None);
                }
                Err(e) => set_error_msg.set(Some(format!("Analysis failed: {e}"))),
            }
            set_analysing.set(false);
        });
    };

    // Import button handler
    let on_import = move |_| {
        let path = tar_path.get();
        set_importing.set(true);
        set_error_msg.set(None);

        spawn_local(async move {
            match run_import(path).await {
                Ok(r) => {
                    set_import_result.set(Some(r));
                    set_error_msg.set(None);
                }
                Err(e) => set_error_msg.set(Some(format!("Import failed: {e}"))),
            }
            set_importing.set(false);
        });
    };

    view! {
        <div class="import-page">
            <h1>"Import BirdNET-Pi Backup"</h1>
            <p class="import-desc">
                "Enter the path to a BirdNET-Pi backup "
                <code>".tar"</code>
                " file on the server. The backup will be analysed first so you "
                "can review its contents before importing."
            </p>

            // ── Path input + Analyse ─────────────────────────────────────
            <div class="import-input-row">
                <input
                    type="text"
                    class="import-path-input"
                    placeholder="/path/to/backup.tar"
                    prop:value=move || tar_path.get()
                    on:input=move |ev| {
                        set_tar_path.set(event_target_value(&ev));
                    }
                />
                <button
                    class="btn btn-primary"
                    on:click=on_analyse
                    disabled=move || analysing.get()
                >
                    {move || if analysing.get() { "Analysing…" } else { "Analyse Backup" }}
                </button>
            </div>

            // ── Error message ────────────────────────────────────────────
            {move || error_msg.get().map(|msg| view! {
                <div class="import-error">{msg}</div>
            })}

            // ── Analysis report ──────────────────────────────────────────
            {move || report.get().map(|r| {
                let size_mb = r.tar_size_bytes as f64 / (1024.0 * 1024.0);
                let top = r.top_species.clone();

                view! {
                    <div class="import-report">
                        <h2>"Backup Report"</h2>

                        <div class="report-grid">
                            <div class="report-card">
                                <span class="report-label">"Total Detections"</span>
                                <span class="report-value">{format_number(r.total_detections)}</span>
                            </div>
                            <div class="report-card">
                                <span class="report-label">"Today's Detections"</span>
                                <span class="report-value">{format_number(r.today_detections)}</span>
                            </div>
                            <div class="report-card">
                                <span class="report-label">"Total Species"</span>
                                <span class="report-value">{r.total_species.to_string()}</span>
                            </div>
                            <div class="report-card">
                                <span class="report-label">"Today's Species"</span>
                                <span class="report-value">{r.today_species.to_string()}</span>
                            </div>
                            <div class="report-card">
                                <span class="report-label">"Date Range"</span>
                                <span class="report-value">
                                    {r.date_min.clone().unwrap_or_default()}
                                    " → "
                                    {r.date_max.clone().unwrap_or_default()}
                                </span>
                            </div>
                            <div class="report-card">
                                <span class="report-label">"Archive Size"</span>
                                <span class="report-value">{format!("{size_mb:.1} MB")}</span>
                            </div>
                            <div class="report-card">
                                <span class="report-label">"Audio Files"</span>
                                <span class="report-value">{format_number(r.audio_file_count)}</span>
                            </div>
                            <div class="report-card">
                                <span class="report-label">"Spectrograms"</span>
                                <span class="report-value">{format_number(r.spectrogram_count)}</span>
                            </div>
                        </div>

                        {(!top.is_empty()).then(|| view! {
                            <h3>"Top 10 Species"</h3>
                            <table class="report-table">
                                <thead>
                                    <tr>
                                        <th>"#"</th>
                                        <th>"Species"</th>
                                        <th>"Detections"</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {top.into_iter().enumerate().map(|(i, (name, count))| view! {
                                        <tr>
                                            <td>{(i + 1).to_string()}</td>
                                            <td>{name}</td>
                                            <td>{format_number(count)}</td>
                                        </tr>
                                    }).collect::<Vec<_>>()}
                                </tbody>
                            </table>
                        })}

                        {r.latitude.map(|lat| {
                            let lon = r.longitude.unwrap_or(0.0);
                            view! {
                                <p class="report-location">
                                    "Location: "
                                    {format!("{lat:.4}° N, {lon:.4}° W")}
                                </p>
                            }
                        })}

                        // ── Import button ────────────────────────────────
                        <div class="import-actions">
                            <button
                                class="btn btn-success"
                                on:click=on_import
                                disabled=move || importing.get()
                            >
                                {move || if importing.get() {
                                    "Importing… (this may take a while)"
                                } else {
                                    "Import All Data"
                                }}
                            </button>
                        </div>
                    </div>
                }
            })}

            // ── Import results ───────────────────────────────────────────
            {move || import_result.get().map(|r| {
                let has_errors = !r.errors.is_empty();
                let errs = r.errors.clone();

                view! {
                    <div class="import-result">
                        <h2>"Import Complete"</h2>
                        <div class="report-grid">
                            <div class="report-card report-card-success">
                                <span class="report-label">"Detections"</span>
                                <span class="report-value">{format_number(r.detections_imported)}</span>
                            </div>
                            <div class="report-card report-card-success">
                                <span class="report-label">"Files Extracted"</span>
                                <span class="report-value">{format_number(r.files_extracted)}</span>
                            </div>
                            <div class="report-card">
                                <span class="report-label">"Skipped (existing)"</span>
                                <span class="report-value">{format_number(r.skipped_existing)}</span>
                            </div>
                        </div>

                        {has_errors.then(|| view! {
                            <details class="import-errors">
                                <summary>{format!("{} errors during import", errs.len())}</summary>
                                <ul>
                                    {errs.into_iter().map(|e| view! {
                                        <li>{e}</li>
                                    }).collect::<Vec<_>>()}
                                </ul>
                            </details>
                        })}
                    </div>
                }
            })}
        </div>
    }
}

/// Format a number with thousand separators.
fn format_number(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, ch) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out.chars().rev().collect()
}
