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

use crate::trace::{JsonMap, Trace};

/// A scored demo candidate: the flat demo row plus the input-only fingerprint
/// used for deduplication.
pub(crate) struct DemoCandidate {
    /// Whole-rollout metric score of the trace this row came from.
    pub score: f64,
    /// Canonical fingerprint of the input fields, for input-level dedup.
    pub input_fingerprint: String,
    /// The demo row: input and output fields merged into one flat object.
    pub row: JsonMap,
}

/// Builds a flat demo row from a span's recorded input/output field maps.
pub(crate) fn demo_from_json(input: &JsonMap, output: &JsonMap) -> JsonMap {
    let mut row = input.clone();
    row.extend(output.iter().map(|(k, v)| (k.clone(), v.clone())));
    row
}

fn input_fingerprint(input: &JsonMap) -> String {
    let mut pairs: Vec<(&String, &Value)> = input.iter().collect();
    pairs.sort_by_key(|(name, _)| *name);
    serde_json::to_string(&pairs).unwrap_or_default()
}

/// Collects scored demo candidates per predictor name from rollout traces.
///
/// Every successful span (parsed output present) inside a rollout whose score
/// reaches `min_score` contributes one candidate to its component's bucket,
/// where the score is the *whole-rollout* metric score.
pub(crate) fn collect_demo_candidates<'a>(
    rollouts: impl IntoIterator<Item = (f64, &'a Trace)>,
    min_score: f64,
) -> HashMap<String, Vec<DemoCandidate>> {
    let mut candidates: HashMap<String, Vec<DemoCandidate>> = HashMap::new();
    for (score, trace) in rollouts {
        if score < min_score {
            continue;
        }
        for span in trace.successes() {
            let name = trace.component_name(span.component);
            if let (Some(input), Some(output)) = (&span.input, &span.output) {
                candidates.entry(name.to_string()).or_default().push(
                    DemoCandidate {
                        score,
                        input_fingerprint: input_fingerprint(input),
                        row: demo_from_json(input, output),
                    },
                );
            }
        }
    }
    candidates
}

/// Keeps the top `max_per_predictor` demos per predictor by rollout score,
/// deduplicated on input fields so repeated inputs don't crowd the demo set.
pub(crate) fn select_demos(
    candidates: HashMap<String, Vec<DemoCandidate>>,
    max_per_predictor: usize,
) -> HashMap<String, Vec<JsonMap>> {
    let mut selected = HashMap::with_capacity(candidates.len());
    for (path, mut scored) in candidates {
        scored.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut seen_inputs = HashSet::new();
        let mut demos = Vec::new();
        for candidate in scored {
            if seen_inputs.insert(candidate.input_fingerprint) {
                demos.push(candidate.row);
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
