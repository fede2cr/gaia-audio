//! Gaia Web – Leptos-based dashboard for real-time species monitoring.

pub mod app;
pub mod components;
pub mod model;
pub mod pages;

cfg_if::cfg_if! {
    if #[cfg(feature = "ssr")] {
        pub mod server;
    }
}

/// Entry-point called from the WASM bundle to hydrate the server-rendered HTML.
#[cfg(feature = "hydrate")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn hydrate() {
    console_error_panic_hook::set_once();
    leptos::mount_to_body(app::App);
}
