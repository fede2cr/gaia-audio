//! Server entry-point – Axum + Leptos SSR.

#[cfg(feature = "ssr")]
#[tokio::main]
async fn main() {
    use axum::{
        extract::State,
        response::{IntoResponse, Response},
        Router,
    };
    use leptos::*;
    use leptos_axum::{generate_route_list, LeptosRoutes};
    use std::path::PathBuf;
    use tower_http::services::ServeDir;

    use gaia_web::app::{App, AppState};
    use gaia_web::server::inaturalist;

    // ── Tracing ──────────────────────────────────────────────────────────
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "gaia_web=info,tower_http=info".into()),
        )
        .init();

    // ── Configuration ────────────────────────────────────────────────────
    let conf = get_configuration(None).await.unwrap();
    let leptos_options = conf.leptos_options.clone();
    let addr = leptos_options.site_addr;
    let site_root = leptos_options.site_root.clone();

    let db_path = PathBuf::from(
        std::env::var("GAIA_DB_PATH").unwrap_or_else(|_| "data/birds.db".into()),
    );

    // Ensure the database and schema exist so the dashboard works even
    // before the processing server has written any detections.
    if let Err(e) = gaia_web::server::import::ensure_gaia_schema(&db_path) {
        tracing::error!("Cannot initialise database: {e}");
        std::process::exit(1);
    }
    tracing::info!("Database ready at {}", db_path.display());

    let extracted_dir = PathBuf::from(
        std::env::var("GAIA_EXTRACTED_DIR").unwrap_or_else(|_| "data/extracted".into()),
    );

    let state = AppState {
        db_path,
        extracted_dir,
        photo_cache: inaturalist::new_cache(),
        leptos_options: leptos_options.clone(),
    };

    // ── Routes ───────────────────────────────────────────────────────────
    let routes = generate_route_list(App);

    let app = Router::new()
        .leptos_routes_with_context(
            &leptos_options,
            routes,
            {
                let state = state.clone();
                move || {
                    provide_context(state.clone());
                }
            },
            App,
        )
        // Serve static assets (WASM bundle, CSS, images, etc.)
        .nest_service(
            "/pkg",
            ServeDir::new(format!("{}/pkg", site_root.to_string())),
        )
        // Serve extracted audio clips + spectrograms
        .nest_service(
            "/extracted",
            ServeDir::new(extracted_dir.to_string_lossy().to_string()),
        )
        .fallback(fallback_handler)
        .with_state(leptos_options);

    tracing::info!("Gaia Web listening on http://{addr}");

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app.into_make_service())
        .await
        .unwrap();

    /// Fallback: try to serve a static file, otherwise return 404.
    async fn fallback_handler(
        State(options): State<LeptosOptions>,
        req: axum::http::Request<axum::body::Body>,
    ) -> Response {
        let root = options.site_root.clone();
        let (parts, _body) = req.into_parts();
        let path = format!("{}{}", root, parts.uri.path());

        // Try serving a static file
        if let Ok(meta) = tokio::fs::metadata(&path).await {
            if meta.is_file() {
                if let Ok(bytes) = tokio::fs::read(&path).await {
                    return (
                        axum::http::StatusCode::OK,
                        [(axum::http::header::CONTENT_TYPE, mime_for(&path))],
                        bytes,
                    )
                        .into_response();
                }
            }
        }

        // Otherwise 404
        (
            axum::http::StatusCode::NOT_FOUND,
            "Not Found",
        )
            .into_response()
    }

    fn mime_for(path: &str) -> &'static str {
        match path.rsplit('.').next().unwrap_or("") {
            "html" => "text/html; charset=utf-8",
            "css" => "text/css",
            "js" => "application/javascript",
            "wasm" => "application/wasm",
            "svg" => "image/svg+xml",
            "png" => "image/png",
            "jpg" | "jpeg" => "image/jpeg",
            "json" => "application/json",
            "wav" => "audio/wav",
            "mp3" => "audio/mpeg",
            _ => "application/octet-stream",
        }
    }
}

#[cfg(not(feature = "ssr"))]
fn main() {
    // This binary is only built with the `ssr` feature.
    // The WASM entry point is `lib::hydrate()`.
}
