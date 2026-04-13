//! Top navigation bar component.

use leptos::*;
use leptos_router::*;

/// Site-wide navigation bar.
#[component]
pub fn Nav() -> impl IntoView {
    view! {
        <nav class="nav-bar">
            <div class="nav-brand">
                <A href="/" class="nav-logo">"🌍 Gaia Audio"</A>
            </div>
            <div class="nav-links">
                <A href="/" class="nav-link">"Live Feed"</A>
                <A href="/calendar" class="nav-link">"Calendar"</A>
                <A href="/species" class="nav-link">"Species"</A>
                <A href="/import" class="nav-link">"Import"</A>
            </div>
        </nav>
    }
}
