//! Root Leptos application component with routing.

use leptos::*;
use leptos_meta::*;
use leptos_router::*;

use crate::components::nav::Nav;
use crate::pages::{
    calendar::CalendarPage,
    day::DayView,
    home::Home,
    import::ImportPage,
    species::SpeciesPage,
    species_list::SpeciesListPage,
};

/// Server-side application state, provided as Leptos context for server functions.
#[derive(Clone, Debug)]
#[cfg(feature = "ssr")]
pub struct AppState {
    pub db_path: std::path::PathBuf,
    pub extracted_dir: std::path::PathBuf,
    pub photo_cache: crate::server::inaturalist::PhotoCache,
    pub leptos_options: leptos::LeptosOptions,
}

/// Dummy state for the client – never actually constructed on WASM, but the
/// type must exist so server functions can reference it in their signatures.
#[derive(Clone, Debug)]
#[cfg(not(feature = "ssr"))]
pub struct AppState;

/// The root `<App/>` component.
#[component]
pub fn App() -> impl IntoView {
    provide_meta_context();

    view! {
        <Stylesheet id="leptos" href="/pkg/gaia-web.css"/>
        <Title text="Gaia Audio – Species Monitor"/>
        <Meta name="viewport" content="width=device-width, initial-scale=1"/>
        <Meta name="description" content="Real-time audio species monitoring dashboard"/>

        <Router>
            <Nav/>
            <main class="main-content">
                <Routes>
                    <Route path="/" view=Home/>
                    <Route path="/calendar" view=CalendarPage/>
                    <Route path="/calendar/:date" view=DayView/>
                    <Route path="/species" view=SpeciesListPage/>
                    <Route path="/species/:name" view=SpeciesPage/>
                    <Route path="/import" view=ImportPage/>
                </Routes>
            </main>
        </Router>
    }
}
