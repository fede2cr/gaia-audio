//! Server entry-point – Axum + Leptos SSR.

#[cfg(feature = "ssr")]
#[tokio::main]
async fn main() {
    use axum::Router;
    use leptos::prelude::*;
    use leptos::prelude::ElementChild;
    use leptos_axum::{generate_route_list, LeptosRoutes};
    use std::path::PathBuf;
    use tower_http::services::ServeDir;

    use gaia_web::app::{shell, App, AppState};
    use gaia_web::server::inaturalist;

    // ── Tracing ──────────────────────────────────────────────────────────
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "gaia_web=info,tower_http=info".into()),
        )
        .init();

    if std::env::var("RUST_LOG").map_or(false, |v| v.contains("debug")) {
        tracing::info!("🔍 Debug logging ENABLED (RUST_LOG={})", std::env::var("RUST_LOG").unwrap_or_default());
    }

    // ── Configuration ────────────────────────────────────────────────────
    let conf = get_configuration(None).unwrap();
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

    // Refresh the species stats cache at startup so the species list loads
    // instantly from the cache table.
    {
        let db = db_path.clone();
        if let Err(e) = gaia_web::server::db::refresh_species_stats(&db) {
            tracing::warn!("Initial species-stats refresh failed: {e}");
        } else {
            tracing::info!("Species stats cache populated");
        }
    }

    // Spawn a background task that refreshes the species stats cache every
    // night at ~02:00 local time (or every 24 h if tz is unknown).
    {
        let db = db_path.clone();
        tokio::spawn(async move {
            loop {
                // Sleep until the next 02:00 or 24 h, whichever is simpler.
                tokio::time::sleep(std::time::Duration::from_secs(24 * 3600)).await;
                tracing::info!("Nightly species-stats refresh starting…");
                let db2 = db.clone();
                let res = tokio::task::spawn_blocking(move || {
                    gaia_web::server::db::refresh_species_stats(&db2)
                }).await;
                match res {
                    Ok(Ok(())) => tracing::info!("Nightly species-stats refresh complete"),
                    Ok(Err(e)) => tracing::warn!("Nightly species-stats refresh failed: {e}"),
                    Err(e) => tracing::warn!("Nightly species-stats task panicked: {e}"),
                }
            }
        });
    }

    let extracted_dir = PathBuf::from(
        std::env::var("GAIA_EXTRACTED_DIR").unwrap_or_else(|_| "data/extracted".into()),
    );
    let extracted_serve_path = extracted_dir.to_string_lossy().to_string();

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
            {
                let opts = leptos_options.clone();
                move || shell(opts.clone())
            },
        )
        // Serve static assets (WASM bundle, CSS, images, etc.)
        .nest_service(
            "/pkg",
            ServeDir::new(format!("{}/pkg", site_root.to_string())),
        )
        // Serve extracted audio clips + spectrograms
        .nest_service(
            "/extracted",
            ServeDir::new(&extracted_serve_path),
        )
        // Serve live analysis spectrogram from the shared data volume
        .nest_service(
            "/live",
            ServeDir::new(
                std::env::var("GAIA_DATA_DIR").unwrap_or_else(|_| "/data".into()),
            ),
        )
        .fallback(leptos_axum::file_and_error_handler(shell))
        .with_state(leptos_options);

    tracing::info!("Gaia Web listening on http://{addr}");

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app.into_make_service())
        .await
        .unwrap();
}

#[cfg(not(feature = "ssr"))]
fn main() {
    // This binary is only built with the `ssr` feature.
    // The WASM entry point is `lib::hydrate()`.
}
