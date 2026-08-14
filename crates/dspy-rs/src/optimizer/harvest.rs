//! Demo harvesting from rollout traces — the trace name-join.
//!
//! A rollout trace records one span per `Predict` invocation under the same
//! component name the mutation seam addresses (the dotted path assigned by
//! `predictor_names`). Harvesting is therefore a pure name join: successful
//! spans from well-scored rollouts become few-shot demo rows for the predictor
//! that produced them — no pointer identity, works identically for fx and
//! struct harnesses. Shared by [`BootstrapFewShot`](crate::BootstrapFewShot)
//! and [`MIPROv2`](crate::MIPROv2).

use std::collections::{HashMap, HashSet};

use serde_json::Value;

use crate::data::RawExample;
use crate::trace::{JsonMap, Trace};

/// Builds a demo row from a span's recorded input/output field maps.
pub(crate) fn demo_from_json(input: &JsonMap, output: &JsonMap) -> RawExample {
    let mut data: HashMap<String, Value> = HashMap::with_capacity(input.len() + output.len());
    data.extend(input.iter().map(|(k, v)| (k.clone(), v.clone())));
    data.extend(output.iter().map(|(k, v)| (k.clone(), v.clone())));
    RawExample::new(
        data,
        input.keys().cloned().collect(),
        output.keys().cloned().collect(),
    )
}

/// Collects scored demo candidates per predictor name from rollout traces.
///
/// Every successful span (parsed output present) inside a rollout whose score
/// reaches `min_score` contributes one `(score, demo)` pair to its component's
/// bucket, where the score is the *whole-rollout* metric score.
pub(crate) fn collect_demo_candidates<'a>(
    rollouts: impl IntoIterator<Item = (f64, &'a Trace)>,
    min_score: f64,
) -> HashMap<String, Vec<(f64, RawExample)>> {
    let mut candidates: HashMap<String, Vec<(f64, RawExample)>> = HashMap::new();
    for (score, trace) in rollouts {
        if score < min_score {
            continue;
        }
        for span in trace.successes() {
            let name = trace.component_name(span.component);
            if let (Some(input), Some(output)) = (&span.input, &span.output) {
                candidates
                    .entry(name.to_string())
                    .or_default()
                    .push((score, demo_from_json(input, output)));
            }
        }
    }
    candidates
}

/// Keeps the top `max_per_predictor` demos per predictor by rollout score,
/// deduplicated on input fields so repeated inputs don't crowd the demo set.
pub(crate) fn select_demos(
    candidates: HashMap<String, Vec<(f64, RawExample)>>,
    max_per_predictor: usize,
) -> HashMap<String, Vec<RawExample>> {
    let mut selected = HashMap::with_capacity(candidates.len());
    for (path, mut scored) in candidates {
        scored.sort_by(|(left, _), (right, _)| {
            right.partial_cmp(left).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut seen_inputs = HashSet::new();
        let mut demos = Vec::new();
        for (_, demo) in scored {
            let mut input_pairs: Vec<(&String, &Value)> = demo
                .input_keys
                .iter()
                .filter_map(|key| demo.data.get(key).map(|value| (key, value)))
                .collect();
            input_pairs.sort_by_key(|(name, _)| *name);
            let fingerprint = serde_json::to_string(&input_pairs).unwrap_or_default();
            if seen_inputs.insert(fingerprint) {
                demos.push(demo);
                if demos.len() >= max_per_predictor {
                    break;
                }
            }
        }
        if !demos.is_empty() {
            selected.insert(path, demos);
        }
    }
    selected
}
