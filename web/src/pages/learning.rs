//! Learning page — bird song quiz.
//!
//! The user picks "Today" or "All Species", then listens to 4 random
//! high-confidence clips (each from a different species whose recording
//! contains only that one species). For each clip a dropdown shows the 4
//! species to choose from; pressing "Evaluate" reveals results.

use leptos::prelude::*;
use leptos::prelude::{
    signal, Effect, ElementChild, IntoView, ReadSignal, Resource,
    ServerFnError, Suspense, WriteSignal,
};

use crate::model::QuizItem;

// ─── Server function ─────────────────────────────────────────────────────────

#[server(prefix = "/api")]
pub async fn get_quiz(today_only: bool) -> Result<Vec<QuizItem>, ServerFnError> {
    let state = use_context::<crate::app::AppState>()
        .ok_or_else(|| ServerFnError::new("Missing AppState"))?;

    let mut items = crate::server::detections_duckdb::quiz_candidates(&state.db_path, today_only)
        .await
        .map_err(|e| ServerFnError::new(format!("DB error: {e}")))?;

    // Enrich with iNaturalist photos.
    for item in &mut items {
        if let Some(photo) = crate::server::inaturalist::lookup(&state.photo_cache, &item.scientific_name).await {
            item.image_url = Some(photo.medium_url);
        }
    }

    Ok(items)
}

// ─── Page component ──────────────────────────────────────────────────────────

#[component]
pub fn LearningPage() -> impl IntoView {
    // Mode: None = not started, Some(true) = today, Some(false) = all species
    let (mode, set_mode) = signal::<Option<bool>>(None);

    // Quiz data — refetches when mode changes
    let quiz = Resource::new(
        move || mode.get(),
        move |m| async move {
            match m {
                Some(today_only) => get_quiz(today_only).await,
                None => Ok(vec![]),
            }
        },
    );

    // User answers: index → selected common_name
    let (answers, set_answers) = signal::<Vec<String>>(vec![
        String::new(),
        String::new(),
        String::new(),
        String::new(),
    ]);

    // Whether the quiz has been evaluated
    let (evaluated, set_evaluated) = signal(false);

    // Reset answers when quiz data changes
    Effect::new(move || {
        let _ = quiz.get();
        set_answers.set(vec![
            String::new(),
            String::new(),
            String::new(),
            String::new(),
        ]);
        set_evaluated.set(false);
    });

    view! {
        <div class="learning-page">
            <h1>"🎵 Learning"</h1>
            <p class="page-description">
                "Test your bird song recognition skills! Listen to audio clips and identify the species."
            </p>

            // Mode selector
            <div class="quiz-mode-selector">
                <button
                    class="btn-mode"
                    class:active=move || mode.get() == Some(true)
                    on:click=move |_| set_mode.set(Some(true))
                >
                    "Today"
                </button>
                <button
                    class="btn-mode"
                    class:active=move || mode.get() == Some(false)
                    on:click=move |_| set_mode.set(Some(false))
                >
                    "All Species"
                </button>
            </div>

            // Quiz content
            <Suspense fallback=move || view! { <p class="loading">"Loading quiz..."</p> }>
                {move || quiz.get().map(|result| match result {
                    Ok(items) if items.is_empty() && mode.get().is_some() => {
                        view! {
                            <p class="empty-state">
                                "Not enough high-confidence single-species recordings available. Try \"All Species\" or wait for more detections."
                            </p>
                        }.into_any()
                    }
                    Ok(items) if items.is_empty() => {
                        view! {
                            <p class="info">"Choose a mode above to start the quiz."</p>
                        }.into_any()
                    }
                    Ok(items) => {
                        // Build the list of species names for the dropdown
                        let species_names: Vec<String> = items.iter()
                            .map(|i| i.common_name.clone())
                            .collect();

                        let num_questions = items.len();

                        view! {
                            <div class="quiz-questions">
                                {items.into_iter().enumerate().map(|(idx, item)| {
                                    let names = species_names.clone();
                                    let correct = item.common_name.clone();
                                    let correct_sci = item.scientific_name.clone();
                                    let clip = item.clip_url.clone();
                                    let spec = item.spectrogram_url.clone();
                                    let photo = item.image_url.clone();

                                    view! {
                                        <QuizCard
                                            index=idx
                                            clip_url=clip
                                            spectrogram_url=spec
                                            image_url=photo
                                            correct_common=correct
                                            correct_scientific=correct_sci
                                            species_options=names
                                            answers=answers
                                            set_answers=set_answers
                                            evaluated=evaluated
                                        />
                                    }
                                }).collect::<Vec<_>>()}
                            </div>

                            <div class="quiz-actions">
                                <button
                                    class="btn-evaluate"
                                    on:click=move |_| set_evaluated.set(true)
                                    disabled=move || evaluated.get()
                                >
                                    "Evaluate"
                                </button>

                                {move || {
                                    if evaluated.get() {
                                        let ans = answers.get();
                                        let q = num_questions;
                                        let correct_count = ans.iter().enumerate()
                                            .filter(|(i, a)| {
                                                *i < species_names.len() && **a == species_names[*i]
                                            })
                                            .count();
                                        view! {
                                            <div class="quiz-result">
                                                <span class="score">
                                                    {format!("{correct_count} / {q} correct")}
                                                </span>
                                                {if correct_count == q {
                                                    " 🎉 Perfect!"
                                                } else if correct_count >= q / 2 {
                                                    " 👍 Good job!"
                                                } else {
                                                    " Keep practicing!"
                                                }}
                                            </div>
                                        }.into_any()
                                    } else {
                                        ().into_any()
                                    }
                                }}

                                <button
                                    class="btn-retry"
                                    on:click=move |_| {
                                        // Re-trigger the resource to get new random clips
                                        let current = mode.get();
                                        set_mode.set(None);
                                        // Use a micro-delay to force resource refetch
                                        set_mode.set(current);
                                    }
                                >
                                    "New Quiz"
                                </button>
                            </div>
                        }.into_any()
                    }
                    Err(e) => {
                        let msg = e.to_string();
                        view! {
                            <p class="error-state">"Error: " {msg}</p>
                        }.into_any()
                    }
                })}
            </Suspense>
        </div>
    }
}

// ─── Quiz card sub-component ─────────────────────────────────────────────────

/// A single quiz question card with audio player, spectrogram, and answer dropdown.
#[component]
fn QuizCard(
    index: usize,
    clip_url: String,
    spectrogram_url: String,
    image_url: Option<String>,
    correct_common: String,
    correct_scientific: String,
    species_options: Vec<String>,
    answers: ReadSignal<Vec<String>>,
    set_answers: WriteSignal<Vec<String>>,
    evaluated: ReadSignal<bool>,
) -> impl IntoView {
    let correct_for_check = correct_common.clone();
    let correct_for_reveal = correct_common.clone();
    let sci_for_reveal = correct_scientific.clone();

    let is_correct = move || {
        let ans = answers.get();
        ans.get(index).map(|a| *a == correct_for_check).unwrap_or(false)
    };

    let answer_class = move || {
        if !evaluated.get() {
            "quiz-card".to_string()
        } else if is_correct() {
            "quiz-card quiz-correct".to_string()
        } else {
            "quiz-card quiz-wrong".to_string()
        }
    };

    view! {
        <div class=answer_class>
            <div class="quiz-card-header">
                <span class="quiz-number">{format!("#{}", index + 1)}</span>
            </div>

            // Spectrogram
            <div class="quiz-spectrogram">
                <img
                    src=spectrogram_url
                    alt="Spectrogram"
                    class="spectrogram-img"
                    loading="lazy"
                />
            </div>

            // Audio player
            <div class="quiz-audio">
                <audio controls preload="none">
                    <source src=clip_url type="audio/ogg; codecs=opus"/>
                    "Your browser does not support the audio element."
                </audio>
            </div>

            // Answer dropdown
            <div class="quiz-answer">
                <label>"Which species is this?"</label>
                <select
                    on:change=move |ev| {
                        let val = event_target_value(&ev);
                        set_answers.update(|ans| {
                            if index < ans.len() {
                                ans[index] = val;
                            }
                        });
                    }
                    disabled=move || evaluated.get()
                >
                    <option value="" selected=true>"— Select species —"</option>
                    {species_options.iter().map(|name| {
                        let n = name.clone();
                        let n2 = name.clone();
                        view! {
                            <option value=n>{n2}</option>
                        }
                    }).collect::<Vec<_>>()}
                </select>
            </div>

            // Reveal correct answer after evaluation
            {move || {
                if evaluated.get() {
                    let photo = image_url.clone();
                    view! {
                        <div class="quiz-reveal">
                            {photo.map(|url| view! {
                                <img src=url class="quiz-species-photo" alt="Species photo"/>
                            })}
                            <div class="quiz-correct-answer">
                                <strong>{correct_for_reveal.clone()}</strong>
                                <em>" ("{sci_for_reveal.clone()}")"</em>
                            </div>
                        </div>
                    }.into_any()
                } else {
                    ().into_any()
                }
            }}
        </div>
    }
}
