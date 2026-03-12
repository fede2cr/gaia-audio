//! Top navigation bar component.

use leptos::prelude::*;
use leptos::prelude::{ElementChild, IntoView};

/// Site-wide navigation bar.
#[component]
pub fn Nav() -> impl IntoView {
    view! {
        <nav class="nav-bar">
            <div class="nav-brand">
                <a href="/" class="nav-logo">"🌍 Gaia Audio"</a>
            </div>
            <div class="nav-links">
                <a href="/" class="nav-link">"Live Feed"</a>
                <a href="/calendar" class="nav-link">"Calendar"</a>
                <a href="/species" class="nav-link">"Species"</a>
                <a href="/excluded" class="nav-link">"Excluded"</a>
                <a href="/import" class="nav-link">"Import"</a>
                <a href="/settings" class="nav-link">"Settings"</a>
            </div>
        </nav>
    }
}
