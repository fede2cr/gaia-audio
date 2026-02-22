//! Monthly calendar grid component with heatmap colouring.

use leptos::*;

use crate::model::CalendarDay;

/// Props for the calendar grid.
#[component]
pub fn CalendarGrid(
    /// Year being displayed.
    year: i32,
    /// Month (1–12) being displayed.
    month: u32,
    /// Per-day aggregates (only days with detections are present).
    days: Vec<CalendarDay>,
    /// Optional: set of dates to highlight (e.g. species-specific active days).
    #[prop(optional)]
    highlight_dates: Option<Vec<String>>,
) -> impl IntoView {
    let month_name = month_label(month);
    let first_weekday = first_day_of_month(year, month);
    let days_in_month = num_days_in_month(year, month);
    let max_detections = days.iter().map(|d| d.total_detections).max().unwrap_or(1).max(1);

    // Build lookup: date string → CalendarDay
    let day_map: std::collections::HashMap<String, &CalendarDay> =
        days.iter().map(|d| (d.date.clone(), d)).collect();

    let cells: Vec<_> = (1..=days_in_month)
        .map(|day_num| {
            let date_str = format!("{year:04}-{month:02}-{day_num:02}");
            let cal_day = day_map.get(&date_str);
            let total = cal_day.map(|d| d.total_detections).unwrap_or(0);
            let species = cal_day.map(|d| d.unique_species).unwrap_or(0);
            let intensity = (total as f64 / max_detections as f64 * 100.0) as u32;

            let highlighted = highlight_dates
                .as_ref()
                .map(|h| h.contains(&date_str))
                .unwrap_or(false);

            let cell_class = if highlighted {
                "cal-cell highlighted"
            } else if total > 0 {
                "cal-cell has-data"
            } else {
                "cal-cell"
            };

            view! {
                <a href={format!("/calendar/{}", date_str)} class={cell_class}
                   style={format!("--intensity: {}%", intensity)}>
                    <span class="cal-day-num">{day_num}</span>
                    {if total > 0 {
                        view! {
                            <span class="cal-day-stats">
                                {total}" · "{species}" spp"
                            </span>
                        }.into_view()
                    } else {
                        view! { <span></span> }.into_view()
                    }}
                </a>
            }
        })
        .collect();

    // Empty cells for padding before the 1st
    let padding: Vec<_> = (0..first_weekday)
        .map(|_| view! { <div class="cal-cell empty"></div> })
        .collect();

    view! {
        <div class="calendar">
            <div class="cal-header">
                <h2>{month_name}" "{year}</h2>
            </div>
            <div class="cal-weekdays">
                <span>"Mon"</span><span>"Tue"</span><span>"Wed"</span>
                <span>"Thu"</span><span>"Fri"</span><span>"Sat"</span><span>"Sun"</span>
            </div>
            <div class="cal-grid">
                {padding}
                {cells}
            </div>
        </div>
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn month_label(m: u32) -> &'static str {
    match m {
        1 => "January", 2 => "February", 3 => "March", 4 => "April",
        5 => "May", 6 => "June", 7 => "July", 8 => "August",
        9 => "September", 10 => "October", 11 => "November", 12 => "December",
        _ => "?",
    }
}

/// Returns 0 = Monday … 6 = Sunday for the 1st of the given month.
fn first_day_of_month(year: i32, month: u32) -> u32 {
    // Tomohiko Sakamoto's algorithm (modified for Monday-start)
    let t = [0u32, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4];
    let y = if month < 3 { year - 1 } else { year } as u32;
    let dow = (y + y / 4 - y / 100 + y / 400 + t[(month - 1) as usize] + 1) % 7;
    // Convert: 0=Sun → 6, 1=Mon → 0, … 6=Sat → 5
    (dow + 6) % 7
}

fn num_days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) {
                29
            } else {
                28
            }
        }
        _ => 30,
    }
}
