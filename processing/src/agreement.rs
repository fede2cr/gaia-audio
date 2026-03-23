//! Cross-model agreement scoring.
//!
//! When multiple models analyse the same audio file, detections that
//! agree across models are more likely to be correct.  This module
//! computes a weighted agreement score for each detection based on
//! whether other models also detected the same species in overlapping
//! time windows.
//!
//! ## Scoring
//!
//! For each detection `d` produced by model `M`:
//!
//! 1. Find all detections from *other* models that match the same
//!    species (scientific name) **and** overlap in time (the chunk
//!    windows have ≥ 50% temporal overlap).
//!
//! 2. The agreement score is a weighted sum of matching models'
//!    `trust_weight` values, normalised by the total weight of all
//!    models that processed the file:
//!
//!    ```text
//!    agreement = (own_weight + Σ matching_weights) / total_weight
//!    ```
//!
//!    - A detection confirmed by all models gets score ≈ 1.0.
//!    - A detection from a single model gets score = own_weight / total.
//!    - BirdNET V2.4 (trust_weight=1.0) contributes more to the score
//!      than beta models (trust_weight=0.5).
//!
//! 3. The list of agreeing model slugs is stored alongside the score
//!    so the UI can show which models agree.

use std::collections::HashMap;

use gaia_common::detection::Detection;

/// Per-model trust weight, extracted from manifests before scoring.
#[derive(Debug, Clone)]
pub struct ModelWeight {
    pub slug: String,
    pub trust_weight: f64,
}

/// Annotate detections with agreement scores.
///
/// Mutates each detection in-place, writing to the `agreement_score` and
/// `agreement_models` fields (which must exist on `Detection`).
///
/// `model_weights` maps model slug → trust_weight (from manifests).
/// Only models that actually produced detections in this batch are
/// considered for the total weight denominator.
pub fn score_agreement(
    detections: &mut [Detection],
    model_weights: &[ModelWeight],
) {
    if detections.is_empty() {
        return;
    }

    // Build weight lookup.
    let weights: HashMap<&str, f64> = model_weights
        .iter()
        .map(|mw| (mw.slug.as_str(), mw.trust_weight))
        .collect();

    // Collect the set of model slugs that actually produced detections.
    let mut active_slugs: Vec<String> = Vec::new();
    for d in detections.iter() {
        if !d.model_slug.is_empty() && !active_slugs.contains(&d.model_slug) {
            active_slugs.push(d.model_slug.clone());
        }
    }

    // Total weight of all active models.
    let total_weight: f64 = active_slugs
        .iter()
        .map(|s| weights.get(s.as_str()).copied().unwrap_or(1.0))
        .sum();

    if total_weight <= 0.0 || active_slugs.len() <= 1 {
        // Single model or no models → agreement is trivially 1.0.
        for d in detections.iter_mut() {
            d.agreement_score = 1.0;
            d.agreement_models = d.model_slug.clone();
        }
        return;
    }

    // Group detections by (species, model) for efficient cross-lookup.
    // key: normalized sci_name → Vec<(index, model_slug, start, stop)>
    let mut species_index: HashMap<String, Vec<(usize, &str, f64, f64)>> = HashMap::new();
    for (i, d) in detections.iter().enumerate() {
        species_index
            .entry(d.scientific_name.clone())
            .or_default()
            .push((i, &d.model_slug, d.start, d.stop));
    }

    // For each detection, find agreeing detections from other models.
    // We accumulate results in a Vec to avoid borrow-checker issues
    // with mutating detections while iterating.
    let mut results: Vec<(f64, String)> = Vec::with_capacity(detections.len());

    for (i, d) in detections.iter().enumerate() {
        let own_weight = weights.get(d.model_slug.as_str()).copied().unwrap_or(1.0);
        let mut matched_weight = own_weight;
        let mut matched_slugs: Vec<&str> = vec![&d.model_slug];

        if let Some(same_species) = species_index.get(&d.scientific_name) {
            for &(j, other_slug, other_start, other_stop) in same_species {
                if j == i || other_slug == d.model_slug {
                    continue;
                }
                // Check temporal overlap: ≥ 50% of the shorter chunk.
                let overlap_start = d.start.max(other_start);
                let overlap_end = d.stop.min(other_stop);
                let overlap = (overlap_end - overlap_start).max(0.0);
                let min_duration = (d.stop - d.start).min(other_stop - other_start);
                if min_duration > 0.0 && overlap / min_duration >= 0.5 {
                    let w = weights.get(other_slug).copied().unwrap_or(1.0);
                    if !matched_slugs.contains(&other_slug) {
                        matched_weight += w;
                        matched_slugs.push(other_slug);
                    }
                }
            }
        }

        let score = (matched_weight / total_weight).min(1.0);
        let models_str = matched_slugs.join(",");
        results.push((score, models_str));
    }

    // Write back.
    for (i, (score, models)) in results.into_iter().enumerate() {
        detections[i].agreement_score = score;
        detections[i].agreement_models = models;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gaia_common::detection::Detection;
    use chrono::NaiveDateTime;

    fn make_det(species: &str, model: &str, start: f64, stop: f64) -> Detection {
        let dt = NaiveDateTime::parse_from_str("2024-06-15 10:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let mut d = Detection::new("birds", dt, start, stop, species, species, 0.9);
        d.model_slug = model.to_string();
        d.model_name = model.to_string();
        d
    }

    #[test]
    fn single_model_gets_full_agreement() {
        let mut dets = vec![
            make_det("Turdus merula", "birdnet", 0.0, 3.0),
            make_det("Parus major", "birdnet", 3.0, 6.0),
        ];
        let weights = vec![ModelWeight { slug: "birdnet".into(), trust_weight: 1.0 }];
        score_agreement(&mut dets, &weights);
        assert!((dets[0].agreement_score - 1.0).abs() < 0.001);
        assert!((dets[1].agreement_score - 1.0).abs() < 0.001);
    }

    #[test]
    fn two_models_agree() {
        let mut dets = vec![
            make_det("Turdus merula", "birdnet", 0.0, 3.0),
            make_det("Turdus merula", "perch", 0.0, 5.0),
        ];
        let weights = vec![
            ModelWeight { slug: "birdnet".into(), trust_weight: 1.0 },
            ModelWeight { slug: "perch".into(), trust_weight: 0.5 },
        ];
        score_agreement(&mut dets, &weights);
        // Both should get full agreement (1.0+0.5 / 1.5 = 1.0)
        assert!((dets[0].agreement_score - 1.0).abs() < 0.001);
        assert!((dets[1].agreement_score - 1.0).abs() < 0.001);
    }

    #[test]
    fn disagreement_weighted() {
        let mut dets = vec![
            make_det("Turdus merula", "birdnet", 0.0, 3.0),
            make_det("Parus major", "perch", 0.0, 5.0), // different species
        ];
        let weights = vec![
            ModelWeight { slug: "birdnet".into(), trust_weight: 1.0 },
            ModelWeight { slug: "perch".into(), trust_weight: 0.5 },
        ];
        score_agreement(&mut dets, &weights);
        // birdnet only: 1.0 / 1.5 ≈ 0.667
        assert!((dets[0].agreement_score - 1.0 / 1.5).abs() < 0.001);
        // perch only: 0.5 / 1.5 ≈ 0.333
        assert!((dets[1].agreement_score - 0.5 / 1.5).abs() < 0.001);
    }

    #[test]
    fn no_temporal_overlap_no_agreement() {
        let mut dets = vec![
            make_det("Turdus merula", "birdnet", 0.0, 3.0),
            make_det("Turdus merula", "perch", 10.0, 15.0), // far apart
        ];
        let weights = vec![
            ModelWeight { slug: "birdnet".into(), trust_weight: 1.0 },
            ModelWeight { slug: "perch".into(), trust_weight: 0.5 },
        ];
        score_agreement(&mut dets, &weights);
        // No overlap → each model only has its own weight
        assert!((dets[0].agreement_score - 1.0 / 1.5).abs() < 0.001);
        assert!((dets[1].agreement_score - 0.5 / 1.5).abs() < 0.001);
    }
}
