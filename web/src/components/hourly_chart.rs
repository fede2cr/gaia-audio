//! Hourly detection chart – a CSS-only horizontal bar chart showing
//! detection counts per hour (0–23), similar to BirdNET-Pi's display.

use leptos::prelude::*;
use leptos::prelude::{ElementChild, IntoView};
use leptos::either::Either;

use crate::model::HourlyCount;

/// A horizontal bar chart showing detection counts for each hour of the day.
///
/// `title` is shown above the chart.  `hours` contains only hours with
/// detections; missing hours are rendered as empty rows.
#[component]
pub fn HourlyChart(
    /// Section title (e.g. "Activity by Hour" or species name).
    title: String,
    /// Hourly counts (only hours with data need be present).
    hours: Vec<HourlyCount>,
    /// Optional: total detection count shown beside the title.
    #[prop(optional)]
    total: Option<u32>,
) -> impl IntoView {
    let max_count = hours.iter().map(|h| h.count).max().unwrap_or(1).max(1);

    // Build a full 0–23 lookup.
    let mut by_hour = [0u32; 24];
    for h in &hours {
        if (h.hour as usize) < 24 {
            by_hour[h.hour as usize] = h.count;
        }
    }

    let bars: Vec<_> = (0..24)
        .map(|hr| {
            let count = by_hour[hr];
            let pct = (count as f64 / max_count as f64 * 100.0) as u32;
            let label = if count > 0 {
                count.to_string()
            } else {
                String::new()
            };
            let bar_class = if count > 0 { "hourly-bar filled" } else { "hourly-bar" };
            view! {
                <div class="hourly-row">
                    <span class="hourly-hour">{format!("{hr:02}")}</span>
                    <div class="hourly-bar-track">
                        <div class=bar_class style={format!("width: {pct}%")}></div>
                    </div>
                    <span class="hourly-count">{label}</span>
                </div>
            }
        })
        .collect();

    view! {
        <div class="hourly-chart">
            <div class="hourly-chart-header">
                <span class="hourly-chart-title">{title}</span>
                {total.map(|t| view! { <span class="hourly-chart-total">{t}" total"</span> })}
            </div>
            {bars}
        </div>
    }
}

/// All-species hourly grid: rows = species (sorted by total desc),
/// columns = hours 0–23.  Each cell shows the count (if > 0) with
/// background intensity proportional to the max count in the grid.
///
/// This mirrors the BirdNET-Pi "species × hour" heatmap.
#[component]
pub fn SpeciesHourlyGrid(
    /// Per-species hourly data, sorted by total descending.
    data: Vec<crate::model::SpeciesHourlyCounts>,
) -> impl IntoView {
    if data.is_empty() {
        return Either::Left(view! { <p class="text-muted">"No detections."</p> });
    }

    // Global max for intensity scaling.
    let global_max = data
        .iter()
        .flat_map(|s| s.hours.iter().map(|h| h.count))
        .max()
        .unwrap_or(1)
        .max(1);

    let header_cells: Vec<_> = (0..24)
        .map(|h| view! { <th class="shg-hour-header">{h}</th> })
        .collect();

    let rows: Vec<_> = data
        .iter()
        .map(|sp| {
            let mut by_hour = [0u32; 24];
            for h in &sp.hours {
                if (h.hour as usize) < 24 {
                    by_hour[h.hour as usize] = h.count;
                }
            }

            let species_href = format!(
                "/species/{}",
                sp.scientific_name.replace(' ', "%20")
            );

            let cells: Vec<_> = (0..24)
                .map(|hr| {
                    let count = by_hour[hr];
                    if count > 0 {
                        let intensity =
                            (count as f64 / global_max as f64 * 100.0).round() as u32;
                        view! {
                            <td class="shg-cell has-data"
                                style={format!("--intensity: {}%", intensity)}>
                                {count}
                            </td>
                        }.into_any()
                    } else {
                        view! { <td class="shg-cell"></td> }.into_any()
                    }
                })
                .collect();

            view! {
                <tr>
                    <td class="shg-species">
                        <a href={species_href.clone()}>
                            {sp.common_name.clone()}
                        </a>
                    </td>
                    <td class="shg-total">{sp.total}</td>
                    {cells}
                </tr>
            }
        })
        .collect();

    Either::Right(view! {
        <div class="species-hourly-grid-wrap">
            <table class="species-hourly-grid">
                <thead>
                    <tr>
                        <th class="shg-species-header">"Species"</th>
                        <th class="shg-total-header">"#"</th>
                        {header_cells}
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
    })
}
