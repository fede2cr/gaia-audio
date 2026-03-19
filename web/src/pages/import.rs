//! Import page – discover BirdNET-Pi nodes and import observations.

use leptos::prelude::*;
use leptos::prelude::{
    signal, ElementChild, IntoView, Resource, ServerFnError,
    Suspense,
};

use crate::model::{BackupFile, BirdnetNode, ImportReport, ImportResult};

// ─── Server functions ────────────────────────────────────────────────────────

/// Discover BirdNET-Pi nodes on the local network via mDNS.
#[server(prefix = "/api")]
pub async fn discover_nodes() -> Result<Vec<BirdnetNode>, ServerFnError> {
    use crate::server::import;

    let nodes = tokio::task::spawn_blocking(import::discover_birdnet_nodes)
        .await
        .map_err(|e| ServerFnError::new(format!("Discovery error: {e}")))?;

    Ok(nodes)
}

/// Stream-import a BirdNET-Pi backup directly from a node on the network.
#[server(prefix = "/api")]
pub async fn import_from_node(
    address: String,
    port: u16,
) -> Result<ImportResult, ServerFnError> {
    use crate::server::import;

    let state = use_context::<crate::app::AppState>()
        .ok_or_else(|| ServerFnError::new("Missing AppState"))?;

    let db_path = state.db_path.clone();
    let extracted_dir = state.extracted_dir.clone();

    let result = tokio::task::spawn_blocking(move || {
        import::stream_import(&address, port, &db_path, &extracted_dir)
    })
    .await
    .map_err(|e| ServerFnError::new(format!("Task error: {e}")))?
    .map_err(|e| ServerFnError::new(e))?;

    Ok(ImportResult {
        detections_imported: result.detections_imported,
        files_extracted: result.files_extracted,
        skipped_existing: result.skipped_existing,
        errors: result.errors,
    })
}

/// Scan the `/backups` volume for `.tar` files (legacy file-based import).
#[server(prefix = "/api")]
pub async fn list_backups() -> Result<Vec<BackupFile>, ServerFnError> {
    use std::path::Path;

    let dir = Path::new("/backups");
    if !dir.is_dir() {
        return Ok(Vec::new());
    }

    let mut files: Vec<BackupFile> = std::fs::read_dir(dir)
        .map_err(|e| ServerFnError::new(format!("Cannot read /backups: {e}")))?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let name = entry.file_name().to_string_lossy().to_string();
            if name.ends_with(".tar") || name.ends_with(".tar.gz") || name.ends_with(".tgz") {
                let meta = entry.metadata().ok()?;
                Some(BackupFile {
                    path: entry.path().to_string_lossy().to_string(),
                    name,
                    size_bytes: meta.len(),
                })
            } else {
                None
            }
        })
        .collect();

    files.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(files)
}

/// Analyse a BirdNET-Pi backup tar without importing (legacy).
#[server(prefix = "/api")]
pub async fn analyse_backup(tar_path: String) -> Result<ImportReport, ServerFnError> {
    use crate::server::import;
    use std::path::Path;

    let path = Path::new(&tar_path);
    let report = import::analyse_backup(path).map_err(|e| ServerFnError::new(e))?;

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

/// Import from a tar file on disk (legacy).
#[server(prefix = "/api")]
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

/// BirdNET-Pi import page – primary workflow is network streaming import.
#[component]
pub fn ImportPage() -> impl IntoView {
    // ── Network import state ─────────────────────────────────────────
    let (nodes, set_nodes) = signal::<Vec<BirdnetNode>>(Vec::new());
    let (discovering, set_discovering) = signal(false);
    let (importing, set_importing) = signal(false);
    let (import_result, set_import_result) = signal::<Option<ImportResult>>(None);
    let (error_msg, set_error_msg) = signal::<Option<String>>(None);

    // Manual entry
    let (manual_addr, set_manual_addr) = signal("birdnet.local".to_string());
    let (manual_port, set_manual_port) = signal(80u16);

    // ── Legacy file-based import state ───────────────────────────────
    let (tar_path, set_tar_path) = signal(String::new());
    let (report, set_report) = signal::<Option<ImportReport>>(None);
    let (file_result, set_file_result) = signal::<Option<ImportResult>>(None);
    let (analysing, set_analysing) = signal(false);
    let (file_importing, set_file_importing) = signal(false);

    // ── Handlers ─────────────────────────────────────────────────────

    let on_discover = move |_| {
        set_discovering.set(true);
        set_error_msg.set(None);

        leptos::task::spawn_local(async move {
            match discover_nodes().await {
                Ok(found) => {
                    if found.is_empty() {
                        set_error_msg.set(Some(
                            "No BirdNET-Pi nodes found. Try entering the address manually."
                                .into(),
                        ));
                    }
                    set_nodes.set(found);
                }
                Err(e) => set_error_msg.set(Some(format!("Discovery failed: {e}"))),
            }
            set_discovering.set(false);
        });
    };

    let do_import = move |addr: String, port: u16| {
        set_importing.set(true);
        set_error_msg.set(None);
        set_import_result.set(None);

        leptos::task::spawn_local(async move {
            match import_from_node(addr, port).await {
                Ok(r) => set_import_result.set(Some(r)),
                Err(e) => set_error_msg.set(Some(format!("Import failed: {e}"))),
            }
            set_importing.set(false);
        });
    };

    let on_manual_import = move |_| {
        let addr = manual_addr.get();
        let port = manual_port.get();
        if addr.is_empty() {
            set_error_msg.set(Some("Please enter an address.".into()));
            return;
        }
        do_import(addr, port);
    };

    // Legacy handlers
    let on_select = move |ev: leptos::ev::Event| {
        let val = event_target_value(&ev);
        set_tar_path.set(val);
        set_report.set(None);
        set_file_result.set(None);
        set_error_msg.set(None);
    };

    let on_analyse = move |_| {
        let path = tar_path.get();
        if path.is_empty() {
            set_error_msg.set(Some("Please select a backup file.".into()));
            return;
        }
        set_error_msg.set(None);
        set_report.set(None);
        set_file_result.set(None);
        set_analysing.set(true);

        leptos::task::spawn_local(async move {
            match analyse_backup(path).await {
                Ok(r) => set_report.set(Some(r)),
                Err(e) => set_error_msg.set(Some(format!("Analysis failed: {e}"))),
            }
            set_analysing.set(false);
        });
    };

    let on_file_import = move |_| {
        let path = tar_path.get();
        set_file_importing.set(true);
        set_error_msg.set(None);

        leptos::task::spawn_local(async move {
            match run_import(path).await {
                Ok(r) => set_file_result.set(Some(r)),
                Err(e) => set_error_msg.set(Some(format!("Import failed: {e}"))),
            }
            set_file_importing.set(false);
        });
    };

    // Lazy-load backup file list for legacy section
    let backups = Resource::new(|| (), |_| async move { list_backups().await.ok() });

    view! {
        <div class="import-page">
            <h1>"Import BirdNET-Pi Observations"</h1>
            <p class="import-desc">
                "Discover a BirdNET-Pi node on your network and import its observations "
                "directly — no manual file downloads needed."
            </p>

            // ── Network import section ───────────────────────────────────
            {view! {
            <section class="import-section">
                <h2>"Import from Network"</h2>

                <div class="import-input-row">
                    <button
                        class="btn btn-primary"
                        on:click=on_discover
                        disabled=move || discovering.get() || importing.get()
                    >
                        {move || if discovering.get() { "Scanning…" } else { "Discover Nodes" }}
                    </button>
                </div>

                // Discovered nodes
                {move || {
                    let found = nodes.get();
                    (!found.is_empty()).then(|| {
                        view! {
                            <div class="node-list">
                                {found.into_iter().map(|node| {
                                    let addr = node.address.clone();
                                    let port = node.port;
                                    let label = format!("{}:{}", addr, port);
                                    let name = node.name.clone();
                                    let addr2 = addr.clone();
                                    view! {
                                        <div class="node-card">
                                            <div class="node-info">
                                                <span class="node-name">{name}</span>
                                                <span class="node-address">{label}</span>
                                            </div>
                                            <button
                                                class="btn btn-success"
                                                disabled=move || importing.get()
                                                on:click=move |_| do_import(addr2.clone(), port)
                                            >
                                                "Import"
                                            </button>
                                        </div>
                                    }
                                }).collect::<Vec<_>>()}
                            </div>
                        }
                    })
                }}

                // Manual entry
                <h3>"Manual Entry"</h3>
                <div class="import-input-row">
                    <input
                        type="text"
                        class="import-input"
                        placeholder="Address (e.g. 192.168.1.100 or birdnet.local)"
                        prop:value=move || manual_addr.get()
                        on:input=move |ev| set_manual_addr.set(event_target_value(&ev))
                    />
                    <input
                        type="number"
                        class="import-input import-port"
                        prop:value=move || manual_port.get().to_string()
                        on:input=move |ev| {
                            if let Ok(p) = event_target_value(&ev).parse::<u16>() {
                                set_manual_port.set(p);
                            }
                        }
                    />
                    <button
                        class="btn btn-success"
                        on:click=on_manual_import
                        disabled=move || importing.get() || manual_addr.get().is_empty()
                    >
                        {move || if importing.get() {
                            "Importing…"
                        } else {
                            "Import"
                        }}
                    </button>
                </div>
            </section>
            }.into_any()}

            // ── Error message ────────────────────────────────────────────
            {move || error_msg.get().map(|msg| view! {
                <div class="import-error">{msg}</div>
            })}

            // ── Import in progress ───────────────────────────────────────
            {move || importing.get().then(|| view! {
                <div class="import-progress">
                    <p>"Streaming backup and importing observations… This may take a while for large datasets."</p>
                </div>
            })}

            // ── Network import results ───────────────────────────────────
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
                                    {errs.into_iter().map(|e| view! { <li>{e}</li> }).collect::<Vec<_>>()}
                                </ul>
                            </details>
                        })}
                    </div>
                }
            })}

            // ── Legacy file-based import (collapsed) ─────────────────────
            {view! {
            <details class="import-legacy">
                <summary>"Import from backup file"</summary>
                <p class="import-desc">
                    "Place backup "
                    <code>".tar"</code>
                    " files in the "
                    <code>"/backups"</code>
                    " volume."
                </p>

                {view! {
                <div class="import-input-row">
                    <Suspense fallback=|| view! { <span>"Scanning…"</span> }>
                        {move || {
                            let files = backups.get().flatten().unwrap_or_default();
                            let is_empty = files.is_empty();
                            view! {
                                <select
                                    class="import-select"
                                    on:change=on_select
                                    prop:value=move || tar_path.get()
                                >
                                    <option value="" disabled=true selected=true>
                                        {if is_empty {
                                            "No backups found in /backups"
                                        } else {
                                            "Select a backup archive…"
                                        }}
                                    </option>
                                    {files.into_iter().map(|f| {
                                        let size_mb = f.size_bytes as f64 / (1024.0 * 1024.0);
                                        let label = format!("{} ({:.1} MB)", f.name, size_mb);
                                        let path = f.path.clone();
                                        view! { <option value=path>{label}</option> }
                                    }).collect::<Vec<_>>()}
                                </select>
                            }
                        }}
                    </Suspense>
                    <button
                        class="btn btn-primary"
                        on:click=on_analyse
                        disabled=move || analysing.get() || tar_path.get().is_empty()
                    >
                        {move || if analysing.get() { "Analysing…" } else { "Analyse" }}
                    </button>
                </div>
                }.into_any()}

                // Analysis report
                {move || report.get().map(|r| {
                    let top = r.top_species.clone();
                    view! {
                        <div class="import-report">
                            <h3>"Backup Report"</h3>
                            {view! {
                            <div class="report-grid">
                                <div class="report-card">
                                    <span class="report-label">"Detections"</span>
                                    <span class="report-value">{format_number(r.total_detections)}</span>
                                </div>
                                <div class="report-card">
                                    <span class="report-label">"Species"</span>
                                    <span class="report-value">{r.total_species.to_string()}</span>
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
                                    <span class="report-label">"Audio Files"</span>
                                    <span class="report-value">{format_number(r.audio_file_count)}</span>
                                </div>
                            </div>
                            }.into_any()}

                            {(!top.is_empty()).then(|| view! {
                                <h4>"Top Species"</h4>
                                <table class="report-table">
                                    <thead><tr><th>"#"</th><th>"Species"</th><th>"Count"</th></tr></thead>
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

                            <div class="import-actions">
                                <button
                                    class="btn btn-success"
                                    on:click=on_file_import
                                    disabled=move || file_importing.get()
                                >
                                    {move || if file_importing.get() {
                                        "Importing…"
                                    } else {
                                        "Import All Data"
                                    }}
                                </button>
                            </div>
                        </div>
                    }.into_any()
                })}

                // File import results
                {move || file_result.get().map(|r| {
                    let has_errors = !r.errors.is_empty();
                    let errs = r.errors.clone();
                    view! {
                        <div class="import-result">
                            <h3>"Import Complete"</h3>
                            {view! {
                            <div class="report-grid">
                                <div class="report-card report-card-success">
                                    <span class="report-label">"Detections"</span>
                                    <span class="report-value">{format_number(r.detections_imported)}</span>
                                </div>
                                <div class="report-card report-card-success">
                                    <span class="report-label">"Files"</span>
                                    <span class="report-value">{format_number(r.files_extracted)}</span>
                                </div>
                                <div class="report-card">
                                    <span class="report-label">"Skipped"</span>
                                    <span class="report-value">{format_number(r.skipped_existing)}</span>
                                </div>
                            </div>
                            }.into_any()}
                            {has_errors.then(|| view! {
                                <details class="import-errors">
                                    <summary>{format!("{} errors", errs.len())}</summary>
                                    <ul>
                                        {errs.into_iter().map(|e| view! { <li>{e}</li> }).collect::<Vec<_>>()}
                                    </ul>
                                </details>
                            })}
                        </div>
                    }.into_any()
                })}
            </details>
            }.into_any()}
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
