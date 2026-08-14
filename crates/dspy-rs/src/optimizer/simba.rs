//! SIMBA: Stochastic Introspective Mini-Batch Ascent (vision §4.3) — the
//! cheap agentic default. Each step samples a minibatch, contrasts the best
//! and worst rollout of the current program on it, proposes exactly one move
//! (append-demo or append-rule), and accepts it through the engine's
//! minibatch gate.

use std::collections::HashSet;

use anyhow::{Result, anyhow};
use bon::Builder;
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};

use crate::evaluate::{Eval, TypedMetric};
use crate::optimizer::engine::{
    Budget, Candidate, EngineConfig, EvalEngine, EvalOutcome, GateOutcome, Spend, apply_candidate,
    canonical_hash,
};
use crate::optimizer::gepa::format_schema_for_reflection;
use crate::optimizer::harvest::{collect_demo_candidates, select_demos};
use crate::optimizer::{Optimizer, predictor_names, with_named_predictor};
use crate::predictors::Example;
use crate::trace::Trace;
use crate::utils::truncate;
use crate::{Facet, Module, Predict, Signature};

/// Distill one improvement rule from contrasting rollouts.
///
/// You are optimizing one module inside an LLM pipeline. Study the module's
/// input/output contract, its current instruction, and two execution rollouts
/// of the same program: one that scored well and one that scored poorly.
/// Distill ONE concise, generalizable rule that would have improved the poor
/// rollout without harming the good one. Return only the rule text, with no
/// preamble or commentary.
#[derive(Signature, Clone, Debug)]
struct IntrospectRollouts {
    /// The module's input and output fields with their descriptions.
    #[input]
    task_description: String,

    /// The instruction currently used by the module.
    #[input]
    current_instruction: String,

    /// The best-scoring rollout on this minibatch (score, feedback, LM calls).
    #[input]
    better_rollout: String,

    /// The worst-scoring rollout on this minibatch (score, feedback, LM calls).
    #[input]
    worse_rollout: String,

    /// One concise rule to append to the module's instruction.
    #[output]
    rule: String,
}

/// Which of SIMBA's two moves a step proposed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SimbaMove {
    /// Append a demo harvested from the minibatch's best rollout.
    AppendDemo,
    /// Append an instruction rule distilled from best/worst introspection.
    AppendRule,
}

/// What one SIMBA step did.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimbaStep {
    /// Step index (0-based).
    pub step: usize,
    /// The move this step proposed.
    pub move_kind: SimbaMove,
    /// Current program's mean score on the step's minibatch (the gate threshold).
    pub parent_minibatch_score: f64,
    /// Proposed child's mean score on the same minibatch.
    pub child_minibatch_score: f64,
    /// Whether the gate promoted the child to the new current program.
    pub accepted: bool,
    /// Full-trainset mean of the child; `Some` only when promoted.
    pub full_score: Option<f64>,
}

/// What a [`SIMBA`] run did.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimbaReport {
    /// Mean metric score of the unmodified module over the trainset.
    pub baseline_score: f64,
    /// Full-trainset mean of the final program (baseline if nothing was accepted).
    pub final_score: f64,
    /// Per-step outcomes, in order.
    pub steps: Vec<SimbaStep>,
    /// Steps whose move was promoted by the gate.
    pub accepted: usize,
    /// Steps whose move was rejected by the gate.
    pub rejected: usize,
    /// Engine spend for the whole run (reflection calls included via charge).
    pub spend: Spend,
}

/// Minibatch introspective ascent — the cheap agentic default (vision §4.3).
///
/// SIMBA is a thin strategy over the shared [`EvalEngine`]. It keeps one
/// *current* program as an overlay [`Candidate`] and hill-climbs:
///
/// 1. **Sample** a trainset minibatch (seeded RNG, indices sorted for
///    deterministic evaluation order).
/// 2. **Introspect** — from the current program's rollouts (all served from
///    engine bookkeeping; the current program always has full-trainset
///    coverage, so this costs nothing) pick the minibatch's best and worst
///    rollout by score.
/// 3. **Move** — exactly one of:
///    - **append-demo**: when the best rollout reached `min_demo_score`, its
///      successful `Predict` spans become one new few-shot demo per predictor
///      via the shared harvest name-join, appended to the current demo set
///      (capped at `max_demos`, oldest dropped, duplicates skipped);
///    - **append-rule**: otherwise, a reflection LM (`prompt_model`) reads the
///      contrasting rollouts and distills one rule, appended to the target
///      predictor's instruction (the predictor with the most spans in the
///      worst rollout). Without a `prompt_model` the worst rollout's metric
///      feedback is appended verbatim as the rule.
/// 4. **Gate** — the child is accepted through the engine's minibatch gate:
///    only if its minibatch mean strictly beats the current program's does it
///    promote to a full-trainset evaluation and become the new current
///    program.
///
/// The winner is installed permanently through the one candidate seam
/// ([`apply_candidate`]) when `compile` returns.
///
/// # Hyperparameters
///
/// - **`max_steps`** (default: 8) — ascent steps to attempt.
/// - **`minibatch_size`** (default: 8) — examples per step.
/// - **`max_demos`** (default: 4) — demo-set cap per predictor.
/// - **`min_demo_score`** (default: 1.0) — rollout score needed to qualify as
///   a demo source; below it the step falls back to append-rule.
/// - **`prompt_model`** — reflection LM for append-rule moves. Strongly
///   recommended; without it rules degrade to metric-feedback concatenation.
/// - **`max_metric_calls`** / **`max_lm_calls`** — engine budget caps; the run
///   stops cleanly when a step no longer fits.
/// - **`eval_concurrency`** (default: 16) — rollouts in flight during evaluation.
/// - **`seed`** — fixes minibatch sampling for reproducible runs.
///
/// # Cost
///
/// `trainset_size` for the baseline pass, then per step: `minibatch_size`
/// rollouts for the gate (+ the remaining `trainset_size - minibatch_size`
/// only when promoted, + 1 reflection call for rule moves). Rejected moves
/// never pay for a full evaluation.
///
/// ```ignore
/// let simba = SIMBA::builder().max_steps(8).minibatch_size(8).build();
/// let report = simba.compile(&mut module, trainset, &metric).await?;
/// println!("{:.3} -> {:.3}", report.baseline_score, report.final_score);
/// ```
#[derive(Builder)]
pub struct SIMBA {
    /// Ascent steps to attempt.
    #[builder(default = 8)]
    pub max_steps: usize,

    /// Examples sampled per step.
    #[builder(default = 8)]
    pub minibatch_size: usize,

    /// Maximum demos kept per predictor (append-demo drops the oldest).
    #[builder(default = 4)]
    pub max_demos: usize,

    /// Minimum rollout score for its spans to qualify as demos; below it the
    /// step proposes a rule instead.
    #[builder(default = 1.0)]
    pub min_demo_score: f64,

    /// Reflection LM that distills rules from contrasting rollouts. Without
    /// it, append-rule degrades to metric-feedback concatenation.
    pub prompt_model: Option<crate::LM>,

    /// Hard cap on metric calls (rollouts). `None` = unlimited.
    pub max_metric_calls: Option<usize>,
    /// Hard cap on LM call units (rollouts + reflection). `None` = unlimited.
    pub max_lm_calls: Option<usize>,

    /// Concurrent rollouts in flight during evaluation.
    #[builder(default = crate::evaluate::DEFAULT_EVAL_CONCURRENCY)]
    pub eval_concurrency: usize,

    /// Seed for minibatch sampling. `None` uses a nondeterministic seed.
    pub seed: Option<u64>,
}

/// Per-example bookkeeping for the *current* program: metric result plus the
/// trace when the engine ran it fresh (cache-served cells carry no trace).
type RolloutStore = Vec<Option<(Eval, Option<Trace>)>>;

impl SIMBA {
    /// Formats one rollout (score, feedback, per-span I/O) for the reflection
    /// prompt.
    fn summarize_rollout(eval: &Eval, trace: Option<&Trace>) -> String {
        use std::fmt::Write as _;

        let mut text = String::new();
        let _ = writeln!(
            text,
            "score={:.3}; {}",
            eval.score,
            eval.feedback.as_deref().unwrap_or("-")
        );
        let Some(trace) = trace else {
            return text.trim_end().to_string();
        };
        for span in &trace.spans {
            let name = trace.component_name(span.component);
            let input = span
                .input
                .as_ref()
                .map(|map| serde_json::Value::Object(map.clone()).to_string())
                .unwrap_or_else(|| "-".to_string());
            let output = match (&span.output, &span.error) {
                (Some(map), _) => serde_json::Value::Object(map.clone()).to_string(),
                (None, Some(error)) => format!(
                    "<{}: {}>",
                    error.kind.as_str(),
                    truncate(span.raw_output.as_deref().unwrap_or(&error.message), 500)
                ),
                (None, None) => "-".to_string(),
            };
            let _ = writeln!(text, "  {name} call {}: input={input}; output={output}", span.seq);
        }
        text.trim_end().to_string()
    }

    /// The predictor an append-rule move targets: the one with the most spans
    /// in the worst rollout's trace (first name wins ties or when no trace).
    fn rule_target<'a>(names: &'a [String], worst_trace: Option<&Trace>) -> &'a str {
        let mut target = names[0].as_str();
        if let Some(trace) = worst_trace {
            let mut most = 0usize;
            for name in names {
                let count = trace.for_component(name).count();
                if count > most {
                    most = count;
                    target = name;
                }
            }
        }
        target
    }

    /// Builds the append-demo child: one new demo per predictor harvested from
    /// the best rollout via the shared trace name-join, appended to the
    /// current effective demo set. Returns `None` when nothing changes (no
    /// qualifying spans, or every harvested demo is already present).
    fn append_demo_child<M>(
        &self,
        module: &mut M,
        current: &Candidate,
        score: f64,
        trace: &Trace,
    ) -> Result<Option<Candidate>>
    where
        M: for<'a> Facet<'a>,
    {
        let harvested = select_demos(
            collect_demo_candidates(std::iter::once((score, trace)), self.min_demo_score),
            1,
        );

        let mut child = current.clone();
        let mut changed = false;
        for (name, new_demos) in harvested {
            // Effective demo set: the current overlay if it carries one,
            // otherwise whatever is installed on the module.
            let mut demos = match current.overlays.get(&name).and_then(|o| o.demos.clone()) {
                Some(demos) => demos,
                None => with_named_predictor(module, &name, |predictor| {
                    Ok(predictor.demos_as_json())
                })?,
            };
            let seen: HashSet<u64> = demos.iter().map(canonical_hash).collect();

            let mut appended = false;
            for demo in new_demos {
                if seen.contains(&canonical_hash(&demo)) {
                    continue;
                }
                demos.push(demo);
                appended = true;
            }
            if appended {
                while demos.len() > self.max_demos.max(1) {
                    demos.remove(0);
                }
                child.set_demos(name, demos);
                changed = true;
            }
        }
        Ok(changed.then_some(child))
    }

    /// Proposes the rule text for an append-rule move, preferring LM
    /// reflection. Returns the rule and the number of reflection LM calls
    /// consumed (0 or 1); reflection failures degrade to metric-feedback
    /// concatenation with a warning rather than aborting the run.
    async fn propose_rule<M>(
        &self,
        module: &mut M,
        target: &str,
        current_instruction: &str,
        better_rollout: String,
        worse_rollout: String,
        worst_eval: &Eval,
        reflector: Option<&Predict<IntrospectRollouts>>,
    ) -> (String, usize)
    where
        M: for<'a> Facet<'a>,
    {
        let fallback = || {
            worst_eval.feedback.clone().unwrap_or_else(|| {
                format!(
                    "Avoid the failure mode of the lowest-scoring rollout (score {:.3}).",
                    worst_eval.score
                )
            })
        };

        let Some(reflector) = reflector else {
            return (fallback(), 0);
        };

        let task_description = with_named_predictor(module, target, |predictor| {
            Ok(format_schema_for_reflection(predictor.schema()))
        })
        .unwrap_or_default();

        let input = IntrospectRolloutsInput {
            task_description,
            current_instruction: current_instruction.to_string(),
            better_rollout,
            worse_rollout,
        };

        match reflector.call(input).await {
            Ok(predicted) => {
                let rule = predicted.rule.trim().to_string();
                if rule.is_empty() {
                    tracing::warn!(
                        target,
                        "reflection LM returned an empty rule; using metric feedback"
                    );
                } else {
                    return (rule, 1);
                }
            }
            Err(err) => {
                tracing::warn!(
                    target,
                    error = %err,
                    "reflection LM call failed; using metric feedback"
                );
            }
        }

        (fallback(), 1)
    }
}

impl Optimizer for SIMBA {
    type Report = SimbaReport;

    async fn compile<S, M, MT>(
        &self,
        module: &mut M,
        trainset: Vec<Example<S>>,
        metric: &MT,
    ) -> Result<Self::Report>
    where
        S: Signature,
        S::Input: Clone,
        M: Module<Input = S::Input> + for<'a> Facet<'a>,
        MT: TypedMetric<S, M>,
    {
        let names = predictor_names(module)?;
        if names.is_empty() {
            return Err(anyhow!("no optimizable predictors found"));
        }

        let mut engine = EvalEngine::new(
            trainset,
            metric,
            EngineConfig {
                concurrency: self.eval_concurrency,
                budget: Budget {
                    max_metric_calls: self.max_metric_calls,
                    max_lm_calls: self.max_lm_calls,
                    max_tokens: None,
                },
                cache_salt: 0,
            },
        );
        let num_examples = engine.num_examples();
        let all_indices: Vec<usize> = (0..num_examples).collect();

        let reflector = self
            .prompt_model
            .as_ref()
            .map(|lm| Predict::<IntrospectRollouts>::builder().lm(lm.clone()).build());
        let mut rng = match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        // Baseline: the empty candidate over the full trainset, traced. This
        // seeds the rollout store; because every later *current* program is
        // promoted through a full evaluation, per-step introspection never
        // needs extra rollouts.
        let mut current = Candidate::new();
        let current_idx = engine.register(current.clone());
        let baseline_eval = match engine.evaluate(module, current_idx, None).await? {
            EvalOutcome::Complete(eval) => eval,
            EvalOutcome::BudgetExhausted { needed } => {
                return Err(anyhow!(
                    "budget too small for the baseline pass ({needed} rollouts needed)"
                ));
            }
        };
        let baseline_score = baseline_eval.mean();
        let mut final_score = baseline_score;

        let mut store: RolloutStore = vec![None; num_examples];
        for rollout in &baseline_eval.rollouts {
            store[rollout.example] = Some((rollout.eval.clone(), rollout.trace.clone()));
        }

        let mut steps = Vec::new();
        let mut accepted = 0usize;
        let mut rejected = 0usize;

        for step in 0..self.max_steps {
            if !engine.budget_allows(1) {
                break;
            }

            // 1. Sample a minibatch; sort so evaluation order is deterministic.
            let minibatch_size = num_examples.min(self.minibatch_size.max(1));
            let mut minibatch: Vec<usize> = all_indices
                .choose_multiple(&mut rng, minibatch_size)
                .copied()
                .collect();
            minibatch.sort_unstable();

            // 2. Introspect the current program's rollouts on the minibatch.
            let threshold = minibatch
                .iter()
                .filter_map(|&idx| store[idx].as_ref().map(|(eval, _)| eval.score))
                .sum::<f64>()
                / minibatch.len() as f64;
            let mut best = minibatch[0];
            let mut worst = minibatch[0];
            for &idx in &minibatch {
                let score = store[idx].as_ref().map_or(0.0, |(eval, _)| eval.score);
                if score > store[best].as_ref().map_or(0.0, |(e, _)| e.score) {
                    best = idx;
                }
                if score < store[worst].as_ref().map_or(0.0, |(e, _)| e.score) {
                    worst = idx;
                }
            }
            let (best_eval, best_trace) = store[best]
                .as_ref()
                .map(|(eval, trace)| (eval.clone(), trace.clone()))
                .expect("store is seeded by the baseline pass");
            let (worst_eval, worst_trace) = store[worst]
                .as_ref()
                .map(|(eval, trace)| (eval.clone(), trace.clone()))
                .expect("store is seeded by the baseline pass");

            // 3. Propose exactly one move.
            let demo_child = if best_eval.score >= self.min_demo_score {
                match &best_trace {
                    Some(trace) => {
                        self.append_demo_child(module, &current, best_eval.score, trace)?
                    }
                    None => None,
                }
            } else {
                None
            };

            let (child, move_kind) = match demo_child {
                Some(child) => (child, SimbaMove::AppendDemo),
                None => {
                    let target = Self::rule_target(&names, worst_trace.as_ref());
                    let base_instruction = match current
                        .overlays
                        .get(target)
                        .and_then(|overlay| overlay.instruction.clone())
                    {
                        Some(instruction) => instruction,
                        None => with_named_predictor(module, target, |predictor| {
                            Ok(predictor.instruction())
                        })?,
                    };
                    let (rule, reflection_calls) = self
                        .propose_rule(
                            module,
                            target,
                            &base_instruction,
                            Self::summarize_rollout(&best_eval, best_trace.as_ref()),
                            Self::summarize_rollout(&worst_eval, worst_trace.as_ref()),
                            &worst_eval,
                            reflector.as_ref(),
                        )
                        .await;
                    engine.charge(0, reflection_calls);

                    let mut child = current.clone();
                    child.set_instruction(target, format!("{base_instruction}\n\n[SIMBA rule] {rule}"));
                    (child, SimbaMove::AppendRule)
                }
            };

            // 4. Accept through the engine's minibatch gate.
            let child_idx = engine.register(child.clone());
            match engine
                .evaluate_gated(module, child_idx, &minibatch, threshold)
                .await?
            {
                GateOutcome::BudgetExhausted { .. } => break,
                GateOutcome::Rejected { minibatch: mb_eval } => {
                    rejected += 1;
                    steps.push(SimbaStep {
                        step,
                        move_kind,
                        parent_minibatch_score: threshold,
                        child_minibatch_score: mb_eval.mean(),
                        accepted: false,
                        full_score: None,
                    });
                }
                GateOutcome::Promoted {
                    minibatch: mb_eval,
                    full,
                } => {
                    // The child is the new current program. Refresh the store
                    // from the full pass, preferring the gate minibatch's
                    // fresh traces over cache-served cells.
                    for rollout in &full.rollouts {
                        store[rollout.example] =
                            Some((rollout.eval.clone(), rollout.trace.clone()));
                    }
                    for rollout in &mb_eval.rollouts {
                        if rollout.trace.is_some() {
                            store[rollout.example] =
                                Some((rollout.eval.clone(), rollout.trace.clone()));
                        }
                    }
                    final_score = full.mean();
                    accepted += 1;
                    steps.push(SimbaStep {
                        step,
                        move_kind,
                        parent_minibatch_score: threshold,
                        child_minibatch_score: mb_eval.mean(),
                        accepted: true,
                        full_score: Some(final_score),
                    });
                    current = child;
                }
            }
        }

        // Install the winner permanently through the one candidate seam.
        if !current.is_empty() {
            let _undo = apply_candidate(module, &current)?;
        }

        Ok(SimbaReport {
            baseline_score,
            final_score,
            steps,
            accepted,
            rejected,
            spend: *engine.spend(),
        })
    }
}
