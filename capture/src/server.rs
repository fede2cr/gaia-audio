//! HTTP server exposing recorded audio files to the processing server.
//!
//! Routes:
//!   GET  /api/health              → health check
//!   GET  /api/recordings          → list available WAV/Opus files
//!   GET  /api/recordings/:name    → download a recording file
//!   DELETE /api/recordings/:name  → remove a processed recording

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use axum::body::Body;
use axum::extract::{Path, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Json};
use axum::routing::{delete, get};
use axum::Router;
use tokio::io::BufReader;
use tokio::net::TcpListener;
use tokio_util::io::ReaderStream;
use tower_http::cors::CorsLayer;
use tracing::{debug, info};

use gaia_common::protocol::{HealthResponse, RecordingInfo};

use crate::DiskState;

/// Resolve a user-supplied filename to an absolute path inside `base_dir`.
///
/// Returns `Err(StatusCode::BAD_REQUEST)` if the name contains path
/// separators or traversal sequences, or if the canonicalized result
/// escapes the base directory.  This guards against path-injection
/// attacks (CodeQL `rust/path-injection`).
fn safe_recording_path(
    base_dir: &std::path::Path,
    name: &str,
) -> Result<PathBuf, StatusCode> {
    // Reject obvious traversal / separator characters up-front so we
    // never even touch the filesystem for clearly malicious inputs.
    if name.contains('/') || name.contains('\\') || name.contains("..") {
        return Err(StatusCode::BAD_REQUEST);
    }

    let candidate = base_dir.join(name);

    // Canonicalize both paths (resolves symlinks + normalises) then
    // verify the candidate is strictly inside the base directory.
    let canon_base = base_dir
        .canonicalize()
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let canon_file = candidate
        .canonicalize()
        .map_err(|_| StatusCode::NOT_FOUND)?;

    if !canon_file.starts_with(&canon_base) {
        return Err(StatusCode::BAD_REQUEST);
    }

    Ok(canon_file)
}

/// Shared state for route handlers.
#[derive(Clone)]
struct AppState {
    stream_dir: PathBuf,
    start_time: Instant,
    #[allow(dead_code)]
    shutdown: Arc<AtomicBool>,
    disk: Arc<DiskState>,
}

/// Start the HTTP server. Blocks until shutdown.
pub async fn run(
    stream_dir: PathBuf,
    listen_addr: &str,
    shutdown: Arc<AtomicBool>,
    disk: Arc<DiskState>,
) -> anyhow::Result<()> {
    // Canonicalize the stream directory so all downstream path operations
    // (read_dir, join, metadata, open, remove) use a fully-resolved base.
    // This satisfies CodeQL's path-injection analysis by proving the base
    // is not attacker-controlled.
    let stream_dir = stream_dir.canonicalize().unwrap_or(stream_dir);

    let state = AppState {
        stream_dir,
        start_time: Instant::now(),
        shutdown: shutdown.clone(),
        disk,
    };

    let app = Router::new()
        .route("/api/health", get(health))
        .route("/api/recordings", get(list_recordings))
        .route("/api/recordings/{name}", get(download_recording))
        .route("/api/recordings/{name}", delete(delete_recording))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let listener = TcpListener::bind(listen_addr).await?;
    info!("Capture HTTP server listening on {listen_addr}");

    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                if shutdown.load(Ordering::Relaxed) {
                    break;
                }
            }
        })
        .await?;

    Ok(())
}

// ── route handlers ───────────────────────────────────────────────────────

async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    let paused = state.disk.capture_paused.load(Ordering::Relaxed);
    Json(HealthResponse {
        status: if paused {
            "disk_full".to_string()
        } else {
            "ok".to_string()
        },
        uptime_secs: state.start_time.elapsed().as_secs(),
        disk_usage_pct: state.disk.usage_pct(),
        capture_paused: paused,
    })
}

async fn list_recordings(
    State(state): State<AppState>,
) -> Result<Json<Vec<RecordingInfo>>, StatusCode> {
    let dir = &state.stream_dir;
    if !dir.exists() {
        return Ok(Json(vec![]));
    }

    let mut recordings = Vec::new();

    let entries =
        std::fs::read_dir(dir).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Only include files that are "settled" (not modified in the last 2 seconds)
    let cutoff = std::time::SystemTime::now() - std::time::Duration::from_secs(2);

    for entry in entries.flatten() {
        let path = entry.path();
        let ext = path.extension().and_then(|e| e.to_str());
        if ext != Some("wav") && ext != Some("opus") {
            continue;
        }
        if let Ok(meta) = path.metadata() {
            let modified = meta.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH);
            if modified > cutoff {
                // Still being written
                continue;
            }
            if meta.len() == 0 {
                continue;
            }
            let created = modified
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .map(|d| {
                    chrono::DateTime::from_timestamp(d.as_secs() as i64, 0)
                        .map(|dt| dt.to_rfc3339())
                        .unwrap_or_default()
                })
                .unwrap_or_default();

            recordings.push(RecordingInfo {
                filename: path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string(),
                size: meta.len(),
                created,
            });
        }
    }

    recordings.sort_by(|a, b| a.filename.cmp(&b.filename));

    let total_bytes: u64 = recordings.iter().map(|r| r.size).sum();
    debug!(
        "Listing {} recording(s), total size {:.1} MB",
        recordings.len(),
        total_bytes as f64 / 1_048_576.0
    );

    Ok(Json(recordings))
}

async fn download_recording(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<impl IntoResponse, StatusCode> {
    let file_path = safe_recording_path(&state.stream_dir, &name)?;

    let meta = tokio::fs::metadata(&file_path)
        .await
        .map_err(|_| StatusCode::NOT_FOUND)?;
    let file_size = meta.len();

    let file = tokio::fs::File::open(&file_path)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    debug!(
        file = %name,
        size_bytes = file_size,
        size_mb = format_args!("{:.2}", file_size as f64 / 1_048_576.0),
        "Streaming recording to processing node"
    );

    // Stream the file in 64 KB chunks instead of loading it entirely
    // into memory.  This keeps resident memory bounded regardless of
    // file size.
    let stream = ReaderStream::with_capacity(BufReader::with_capacity(65_536, file), 65_536);
    let body = Body::from_stream(stream);

    let mut headers = axum::http::HeaderMap::new();
    let content_type = match file_path.extension().and_then(|e| e.to_str()) {
        Some("opus") => "audio/opus",
        _ => "audio/wav",
    };
    headers.insert(header::CONTENT_TYPE, content_type.parse().unwrap());
    headers.insert(
        header::CONTENT_LENGTH,
        file_size.to_string().parse().unwrap(),
    );

    Ok((headers, body))
}

async fn delete_recording(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> StatusCode {
    debug!(file = %name, "DELETE request received from processing node");

    let file_path = match safe_recording_path(&state.stream_dir, &name) {
        Ok(p) => p,
        Err(StatusCode::NOT_FOUND) => {
            debug!(file = %name, "DELETE rejected: file not found");
            return StatusCode::NOT_FOUND;
        }
        Err(code) => {
            debug!(file = %name, "DELETE rejected: invalid path");
            return code;
        }
    };

    // Grab file size before deleting so we can log how much space was freed.
    let size_bytes = tokio::fs::metadata(&file_path)
        .await
        .map(|m| m.len())
        .unwrap_or(0);

    match tokio::fs::remove_file(&file_path).await {
        Ok(()) => {
            let disk_pct = state.disk.usage_pct();
            debug!(
                file = %name,
                freed_bytes = size_bytes,
                freed_mb = format_args!("{:.2}", size_bytes as f64 / 1_048_576.0),
                disk_usage_pct = format_args!("{disk_pct:.1}"),
                "DELETE OK — recording removed"
            );
            StatusCode::NO_CONTENT
        }
        Err(e) => {
            debug!(file = %name, error = %e, "DELETE failed: could not remove file");
            StatusCode::INTERNAL_SERVER_ERROR
        }
    }
}
