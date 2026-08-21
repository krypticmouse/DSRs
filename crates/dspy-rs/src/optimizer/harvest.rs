//! Demo harvesting from rollout traces — the trace name-join.
//!
//! A rollout trace records one span per `Predict` invocation under the same
//! component name candidates address (the leaf name declared through the
//! [`Predictors`](crate::Predictors) contract and stamped by the target's
//! naming pass). Harvesting is therefore a pure name join: successful
//! spans from well-scored rollouts become few-shot demo rows for the predictor
//! that produced them — no pointer identity, works identically for fx and
//! struct harnesses. Shared by [`BootstrapFewShot`](crate::BootstrapFewShot)
//! and [`MIPROv2`](crate::MIPROv2).
//!
//! Credit is per-span when available (RFC 0004 §4): a span carrying its own
//! [`Eval`](crate::Eval) — attached by
//! [`TypedMetric::evaluate_spans`](crate::evaluate::TypedMetric::evaluate_spans)
//! — is gated and ranked on that score instead of the whole-rollout score, so
//! a good final answer no longer vouches for every intermediate step.

use std::collections::{HashMap, HashSet};

use serde_json::Value;

use crate::trace::{JsonMap, Trace};

/// A scored demo candidate: the flat demo row plus the input-only fingerprint
/// used for deduplication.
pub(crate) struct DemoCandidate {
    /// Effective credit score: the span's own eval when the metric attached
    /// one, the whole-rollout metric score otherwise.
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
/// Every successful span (parsed output present) whose *effective* score
/// reaches `min_score` contributes one candidate to its component's bucket.
/// The effective score is the span's own [`Eval`](crate::Eval) when the
/// metric attached one via
/// [`TypedMetric::evaluate_spans`](crate::evaluate::TypedMetric::evaluate_spans),
/// the whole-rollout metric score otherwise — so a span the metric scored
/// badly is excluded even from a winning rollout, and a span it scored well
/// qualifies even from a losing one.
pub(crate) fn collect_demo_candidates<'a>(
    rollouts: impl IntoIterator<Item = (f64, &'a Trace)>,
    min_score: f64,
) -> HashMap<String, Vec<DemoCandidate>> {
    let mut candidates: HashMap<String, Vec<DemoCandidate>> = HashMap::new();
    for (rollout_score, trace) in rollouts {
        for span in trace.successes() {
            let score = span.eval.as_ref().map_or(rollout_score, |eval| eval.score);
            if score < min_score {
                continue;
            }
            let name = trace.component_name(span.component);
            if let (Some(input), Some(output)) = (&span.input, &span.output) {
                candidates
                    .entry(name.to_string())
                    .or_default()
                    .push(DemoCandidate {
                        score,
                        input_fingerprint: input_fingerprint(input),
                        row: demo_from_json(input, output),
                    });
            }
        }
    }
    candidates
}

/// Keeps the top `max_per_predictor` demos per predictor by effective score
/// (span eval when present, rollout score otherwise), deduplicated on input
/// fields so repeated inputs don't crowd the demo set.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LmUsage;
    use crate::trace::{CompId, Eval, ModelId, Span, SpanId, Trace};

    fn json_map(pairs: &[(&str, &str)]) -> JsonMap {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), Value::String(v.to_string())))
            .collect()
    }

    fn span(id: u32, component: u32, input: &str, output: &str, eval: Option<Eval>) -> Span {
        Span {
            id: SpanId(id),
            component: CompId(component),
            seq: 0,
            parent: None,
            prefix: None,
            suffix: Vec::new(),
            input: Some(json_map(&[("prompt", input)])),
            model: ModelId(0),
            request_hash: 0,
            events: Vec::new(),
            raw_output: None,
            output: Some(json_map(&[("answer", output)])),
            usage: LmUsage::default(),
            error: None,
            eval,
            started_at_us: 0,
            duration_us: 0,
            complete: true,
        }
    }

    fn trace(components: &[&str], spans: Vec<Span>) -> Trace {
        Trace {
            components: components.iter().map(|name| name.to_string()).collect(),
            spans,
            ..Trace::default()
        }
    }

    fn harvested_prompts(demos: &HashMap<String, Vec<JsonMap>>, name: &str) -> Vec<String> {
        demos
            .get(name)
            .map(|rows| {
                rows.iter()
                    .map(|row| row["prompt"].as_str().unwrap_or_default().to_string())
                    .collect()
            })
            .unwrap_or_default()
    }

    #[test]
    fn without_span_evals_rollout_score_gates_every_span() {
        // Baseline behavior: no span evals anywhere, so the whole-rollout
        // score decides for every span — 0.0 rollouts contribute nothing.
        let good = trace(
            &["draft", "refine"],
            vec![span(0, 0, "q1", "a1", None), span(1, 1, "q1'", "a1'", None)],
        );
        let bad = trace(
            &["draft", "refine"],
            vec![span(0, 0, "q2", "a2", None), span(1, 1, "q2'", "a2'", None)],
        );

        let demos = select_demos(collect_demo_candidates([(1.0, &good), (0.0, &bad)], 1.0), 4);
        assert_eq!(harvested_prompts(&demos, "draft"), vec!["q1"]);
        assert_eq!(harvested_prompts(&demos, "refine"), vec!["q1'"]);
    }

    #[test]
    fn span_eval_overrides_rollout_score_in_both_directions() {
        // Winning rollout, but the metric scored the draft span 0.0 (a
        // misstep the refine step recovered from) — the draft is excluded.
        let recovered = trace(
            &["draft", "refine"],
            vec![
                span(0, 0, "q1", "wrong", Some(Eval::score(0.0))),
                span(1, 1, "q1'", "right", None),
            ],
        );
        // Losing rollout, but the metric scored the draft span 1.0 — the
        // draft qualifies anyway.
        let salvaged = trace(
            &["draft", "refine"],
            vec![
                span(0, 0, "q2", "right", Some(Eval::score(1.0))),
                span(1, 1, "q2'", "wrong", None),
            ],
        );

        let demos = select_demos(
            collect_demo_candidates([(1.0, &recovered), (0.0, &salvaged)], 1.0),
            4,
        );
        assert_eq!(harvested_prompts(&demos, "draft"), vec!["q2"]);
        assert_eq!(harvested_prompts(&demos, "refine"), vec!["q1'"]);
    }

    #[test]
    fn span_eval_score_ranks_candidates() {
        // Both spans qualify; the span-scored one outranks the
        // rollout-scored one when only one demo slot is available.
        let modest = trace(&["draft"], vec![span(0, 0, "q1", "a1", None)]);
        let strong = trace(
            &["draft"],
            vec![span(0, 0, "q2", "a2", Some(Eval::score(0.9)))],
        );

        let demos = select_demos(
            collect_demo_candidates([(0.5, &modest), (0.5, &strong)], 0.5),
            1,
        );
        assert_eq!(harvested_prompts(&demos, "draft"), vec!["q2"]);
    }
}
