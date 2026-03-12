//! Root Leptos application component with routing.

use leptos::prelude::*;
use leptos::prelude::{ElementChild, IntoView};
use leptos_meta::*;
use leptos_router::{
    components::{FlatRoutes, Route, Router},
    ParamSegment, StaticSegment,
};

use crate::components::nav::Nav;
use crate::pages::{
    calendar::CalendarPage,
    day::DayView,
    excluded::ExcludedPage,
    home::Home,
    import::ImportPage,
    settings::SettingsPage,
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
    pub leptos_options: leptos::config::LeptosOptions,
}

/// Dummy state for the client – never actually constructed on WASM, but the
/// type must exist so server functions can reference it in their signatures.
#[derive(Clone, Debug)]
#[cfg(not(feature = "ssr"))]
pub struct AppState;

/// Full-document shell rendered on the server.
///
/// Contains `<!DOCTYPE html>`, `<head>` (with hydration scripts, stylesheet,
/// meta-tag outlet), and `<body>` wrapping `<App/>`.
pub fn shell(options: LeptosOptions) -> impl IntoView {
    view! {
        <!DOCTYPE html>
        <html lang="en">
            <head>
                <meta charset="utf-8"/>
                <meta name="viewport" content="width=device-width, initial-scale=1"/>
                <meta name="description" content="Real-time audio species monitoring dashboard"/>
                <Stylesheet id="leptos" href="/pkg/gaia-web.css"/>
                <AutoReload options=options.clone() />
                <HydrationScripts options />
                <MetaTags />
            </head>
            <body>
                <App />
            </body>
        </html>
    }
}

/// The root `<App/>` component.
#[component]
pub fn App() -> impl IntoView {
    view! {
        <Title text="Gaia Audio – Species Monitor"/>
        <Router>
            <Nav/>
            <main class="main-content">
                <FlatRoutes fallback=|| "Page not found.">
                    <Route path=StaticSegment("") view=Home/>
                    <Route path=StaticSegment("calendar") view=CalendarPage/>
                    <Route path=(StaticSegment("calendar"), ParamSegment("date")) view=DayView/>
                    <Route path=StaticSegment("species") view=SpeciesListPage/>
                    <Route path=(StaticSegment("species"), ParamSegment("name")) view=SpeciesPage/>
                    <Route path=StaticSegment("excluded") view=ExcludedPage/>
                    <Route path=StaticSegment("import") view=ImportPage/>
                    <Route path=StaticSegment("settings") view=SettingsPage/>
                </FlatRoutes>
            </main>
        </Router>
    }
}
