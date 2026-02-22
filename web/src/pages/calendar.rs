//! Calendar page – monthly overview with heatmap.

use leptos::*;

use crate::components::calendar_grid::CalendarGrid;
use crate::model::CalendarDay;

// ─── Server function ─────────────────────────────────────────────────────────

#[server(GetCalendarData, "/api")]
pub async fn get_calendar_data(
    year: i32,
    month: u32,
) -> Result<Vec<CalendarDay>, ServerFnError> {
    use crate::server::db;
    let state = use_context::<crate::app::AppState>()
        .ok_or_else(|| ServerFnError::new("Missing AppState"))?;
    db::calendar_data(&state.db_path, year, month)
        .map_err(|e| ServerFnError::new(format!("DB error: {e}")))
}

// ─── Page component ──────────────────────────────────────────────────────────

/// Monthly calendar view with navigation and species-count heatmap.
#[component]
pub fn CalendarPage() -> impl IntoView {
    // Current month selection
    let now = js_sys_now();
    let (year, set_year) = create_signal(now.0);
    let (month, set_month) = create_signal(now.1);

    // Fetch calendar data whenever year/month changes
    let calendar = create_resource(
        move || (year.get(), month.get()),
        |(y, m)| async move { get_calendar_data(y, m).await },
    );

    let go_prev = move |_| {
        let (y, m) = if month.get() == 1 {
            (year.get() - 1, 12u32)
        } else {
            (year.get(), month.get() - 1)
        };
        set_year.set(y);
        set_month.set(m);
    };

    let go_next = move |_| {
        let (y, m) = if month.get() == 12 {
            (year.get() + 1, 1u32)
        } else {
            (year.get(), month.get() + 1)
        };
        set_year.set(y);
        set_month.set(m);
    };

    view! {
        <div class="calendar-page">
            <div class="cal-nav">
                <button on:click=go_prev class="cal-nav-btn">"← Prev"</button>
                <button on:click=go_next class="cal-nav-btn">"Next →"</button>
            </div>

            <Suspense fallback=move || view! { <p class="loading">"Loading calendar…"</p> }>
                {move || calendar.get().map(|res| match res {
                    Ok(days) => view! {
                        <CalendarGrid year=year.get() month=month.get() days=days />
                    }.into_view(),
                    Err(e) => view! {
                        <p class="error">"Error: " {e.to_string()}</p>
                    }.into_view(),
                })}
            </Suspense>
        </div>
    }
}

/// Returns (year, month) for the current date.
/// On WASM we cannot use `chrono::Local`, so we parse from js_sys.
fn js_sys_now() -> (i32, u32) {
    // Fallback for SSR – use chrono if available.
    #[cfg(feature = "ssr")]
    {
        let now = chrono::Local::now();
        return (now.format("%Y").to_string().parse().unwrap_or(2025),
                now.format("%m").to_string().parse().unwrap_or(1));
    }

    // On WASM, just default to 2025-01 – the client will override via JS if needed.
    #[cfg(not(feature = "ssr"))]
    {
        (2025, 1)
    }
}
