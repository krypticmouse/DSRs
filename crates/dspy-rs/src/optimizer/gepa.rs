use anyhow::{Context, Result, anyhow};
use bon::Builder;
use rand::{Rng, rngs::StdRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};

use crate::core::ToInput;
use crate::evaluate::TypedMetric;
use crate::optimizer::engine::{
    Candidate, CandidateEval, Engine, EngineConfig, EvalOutcome, ParetoStatistics, ParetoView,
};
use crate::optimizer::target::LeafInfo;
use crate::optimizer::{OptimizeTarget, Optimizer, OptimizerCommon, Report};
use crate::utils::truncate;
use crate::{Module, Predict, Predictors, Signature};

/// Improve an LLM-pipeline module's instruction using execution feedback.
///
/// You are optimizing the system instruction of one module inside an LLM pipeline.
/// Study the module's input/output contract, its current instruction, and the
/// per-example feedback from the last evaluation. Then write an improved
/// instruction: fix the failure modes the feedback names, keep what already works,
/// and be specific without becoming verbose. Return only the improved instruction
/// text, with no preamble or commentary.
#[derive(Signature, Clone, Debug)]
struct ReflectOnInstruction {
    /// The module's input and output fields with their descriptions.
    #[input]
    task_description: String,

    /// The instruction currently used by the module.
    #[input]
    current_instruction: String,

    /// Per-example scores and textual feedback from the last evaluation.
    #[input]
    execution_feedback: String,

    /// The rewritten instruction.
    #[output]
    improved_instruction: String,
}

/// Character budget for the feedback appended by the *no-prompt-model*
/// fallback mutation. Without a cap the child instruction would grow by a
/// full trace dump every generation (unbounded quadratic growth across a
/// lineage); the reflection-LM path is unaffected.
const CONCAT_FEEDBACK_BUDGET: usize = 2000;

/// A single instruction candidate tracked through GEPA's evolutionary search.
///
/// Carries the instruction text, per-example scores, lineage (parent_id), and
/// generation number. The Pareto frontier selects candidates that aren't dominated
/// on any individual example — not just by average score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GEPACandidate {
    pub id: usize,
    pub instruction: String,
    pub module_name: String,
    pub example_scores: Vec<f32>,
    pub parent_id: Option<usize>,
    pub generation: usize,
}

impl GEPACandidate {
    pub fn average_score(&self) -> f32 {
        if self.example_scores.is_empty() {
            return 0.0;
        }
        self.example_scores.iter().sum::<f32>() / self.example_scores.len() as f32
    }

    pub fn mutate(&self, new_instruction: String, generation: usize) -> Self {
        Self {
            id: 0,
            instruction: new_instruction,
            module_name: self.module_name.clone(),
            example_scores: Vec::new(),
            parent_id: Some(self.id),
            generation,
        }
    }
}

/// Full report from a [`GEPA`] optimization run.
///
/// Contains the winning candidate, the complete candidate history (if `track_stats`
/// was enabled), budget usage, and optionally the best outputs on the validation set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GEPAResult {
    /// The candidate with the best average score on the Pareto frontier.
    pub best_candidate: GEPACandidate,
    /// All candidates evaluated (empty unless `track_stats` is enabled).
    pub all_candidates: Vec<GEPACandidate>,
    /// Total evaluation rollouts consumed.
    pub total_rollouts: usize,
    /// Total LM calls consumed (rollouts + candidate generation).
    pub total_lm_calls: usize,
    /// (generation, best_average_score) pairs for plotting convergence.
    pub evolution_history: Vec<(usize, f32)>,
    /// Highest score achieved per validation example across all candidates.
    pub highest_score_achieved_per_val_task: Vec<f32>,
    /// Best outputs on the validation set (only if `track_best_outputs` is enabled).
    pub best_outputs_valset: Option<Vec<serde_json::Value>>,
    /// Pareto frontier statistics per generation.
    pub frontier_history: Vec<ParetoStatistics>,
}

/// Genetic-Pareto instruction optimizer with feedback-driven evolution.
///
/// GEPA is a thin strategy over the shared [`Engine`]: candidates are
/// instruction [`Candidate`]s injected ambiently, evaluation is the engine's
/// cached bounded-concurrency fan-out, budgets are engine budgets, and the
/// per-instance Pareto frontier is a view over the engine's
/// (candidates × examples) score matrix.
///
/// GEPA uses an evolutionary search guided by per-example feedback from your metric.
/// Unlike [`COPRO`](crate::COPRO) which only uses numerical scores, GEPA requires your
/// [`TypedMetric`] to return [`Eval::with_feedback`](crate::Eval::with_feedback) —
/// textual feedback explaining *why* each example scored the way it did. When a
/// `prompt_model` is configured, a reflection LM reads the current instruction, that
/// feedback, and the mutated component's per-invocation execution trace
/// ([`Trace::for_component`](crate::Trace::for_component)), then writes an improved
/// instruction each generation; without one, the feedback is appended to the
/// instruction as a deterministic mutation (capped at a fixed character budget
/// so lineages can't grow unboundedly). Either way the quality of your
/// feedback directly determines the quality of GEPA's search.
///
/// The Pareto frontier tracks candidates that aren't dominated on any individual
/// training example, not just by average score. This means GEPA finds instructions
/// that are robust across diverse inputs rather than overfitting to easy examples.
///
/// Only searches instruction space — no demo mutation, no crossover between candidates.
/// Each child has exactly one parent.
///
/// # Validation sets
///
/// Build the target with
/// [`OptimizeTarget::module_with_valset`] (or use
/// [`compile_module_with_valset`](GEPA::compile_module_with_valset)): initial
/// evaluation and child scoring use the target's validation columns, parent
/// re-evaluation samples from its trainset columns. Without a valset the
/// trainset serves both roles.
///
/// # Hyperparameters
///
/// - **`num_iterations`** (default: 20) — evolutionary generations. More = deeper search.
/// - **`minibatch_size`** (default: 25) — examples per parent evaluation within each
///   generation. Controls exploration vs cost.
/// - **`num_trials`** (default: 10) — **currently unused.** Reserved for multi-child
///   evolution (one child per generation right now). Setting this has no effect.
/// - **`temperature`** (default: 1.0) — **currently unused.** Reserved for mutation
///   diversity control. Setting this has no effect.
/// - **`max_rollouts`** / **`max_lm_calls`** — hard budget caps. Optimization stops
///   when either limit would be exceeded by the next batch.
/// - **`track_stats`** (default: true) — record all candidates and frontier history.
/// - **`track_best_outputs`** (default: false) — re-run the best instruction on the
///   eval set and record outputs.
/// - **`prompt_model`** — reflection LM that rewrites instructions from feedback.
///   Strongly recommended; without it mutation degrades to (budget-capped)
///   feedback concatenation.
/// - **`eval_concurrency`** (default: 16) — LM calls in flight during evaluation.
/// - **`seed`** — fixes minibatch sampling and parent selection for reproducible runs.
///
/// # Requires feedback
///
/// GEPA will error if any [`Eval`](crate::Eval) from your metric has `feedback: None`.
/// Use [`Eval::with_feedback`](crate::Eval::with_feedback).
///
/// # Cost
///
/// Roughly `num_iterations × (minibatch_size + eval_set_size) + initial_eval` LM calls,
/// *minus* whatever the engine's rollout cache serves for free (a parent re-evaluated
/// on already-seen examples costs nothing). Budget caps (`max_rollouts`,
/// `max_lm_calls`) prevent runaway costs.
///
/// ```ignore
/// let gepa = GEPA::builder()
///     .num_iterations(20)
///     .max_lm_calls(Some(500))
///     .build();
/// let report = gepa.compile_module(&mut module, &trainset, &feedback_metric).await?;
/// println!("Best score: {:.3}", report.best_candidate.average_score());
/// ```
#[derive(Builder)]
pub struct GEPA {
    /// Evolutionary generations to run.
    #[builder(default = 20)]
    pub num_iterations: usize,

    /// Examples per parent evaluation within each generation.
    #[builder(default = 25)]
    pub minibatch_size: usize,

    /// **Currently unused.** Reserved for multi-child evolution (one child per
    /// generation right now). Setting this has no effect.
    #[builder(default = 10)]
    pub num_trials: usize,

    /// **Currently unused.** Reserved for mutation diversity control.
    /// Setting this has no effect.
    #[builder(default = 1.0)]
    pub temperature: f32,

    /// Record all candidates and frontier history in the report.
    #[builder(default = true)]
    pub track_stats: bool,

    /// Re-run the best instruction on the eval set and record outputs.
    #[builder(default = false)]
    pub track_best_outputs: bool,

    /// Hard cap on total evaluation rollouts.
    pub max_rollouts: Option<usize>,
    /// Hard cap on total LM calls (rollouts + generation).
    pub max_lm_calls: Option<usize>,
    /// Optional separate LM used to reflect on feedback and propose improved
    /// instructions. When unset, GEPA falls back to deterministic feedback
    /// concatenation (no reflection LM call).
    pub prompt_model: Option<crate::LM>,
    /// Concurrent LM calls in flight during candidate evaluation.
    #[builder(default = crate::evaluate::DEFAULT_EVAL_CONCURRENCY)]
    pub eval_concurrency: usize,
    /// Seed for minibatch sampling. `None` uses a nondeterministic seed.
    pub seed: Option<u64>,
}

/// GEPA's evolving state: report-facing candidates joined to engine rows.
struct Lineage {
    /// `(report candidate, engine candidate index)` in creation order.
    entries: Vec<(GEPACandidate, usize)>,
}

impl Lineage {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    fn add(
        &mut self,
        mut candidate: GEPACandidate,
        engine_row: usize,
        eval: &CandidateEval,
    ) -> &GEPACandidate {
        candidate.id = self.entries.len();
        candidate.example_scores = eval.scores().iter().map(|&score| score as f32).collect();
        self.entries.push((candidate, engine_row));
        &self.entries.last().expect("just pushed").0
    }

    /// Samples a parent proportional to Pareto coverage (wins on the
    /// validation columns). Falls back to the first candidate when nobody has
    /// wins recorded yet; `None` when there are no candidates at all.
    fn sample_parent(&self, pareto: &ParetoView, rng: &mut StdRng) -> Option<&GEPACandidate> {
        if self.entries.is_empty() {
            return None;
        }
        let coverages: Vec<usize> = self
            .entries
            .iter()
            .map(|(_, row)| pareto.wins(*row))
            .collect();
        let total: usize = coverages.iter().sum();
        if total == 0 {
            return self.entries.first().map(|(candidate, _)| candidate);
        }
        let mut target = rng.gen_range(0..total);
        for ((candidate, _), &coverage) in self.entries.iter().zip(&coverages) {
            if target < coverage {
                return Some(candidate);
            }
            target -= coverage;
        }
        self.entries.last().map(|(candidate, _)| candidate)
    }

    /// Frontier members (wins > 0), i.e. the historical `ParetoFrontier`
    /// contents after pruning.
    fn frontier<'a>(&'a self, pareto: &'a ParetoView) -> impl Iterator<Item = &'a GEPACandidate> {
        self.entries
            .iter()
            .filter(|(_, row)| pareto.wins(*row) > 0)
            .map(|(candidate, _)| candidate)
    }

    fn best_by_average<'a>(&'a self, pareto: &'a ParetoView) -> Option<&'a GEPACandidate> {
        self.frontier(pareto).max_by(|a, b| {
            a.average_score()
                .partial_cmp(&b.average_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

impl GEPA {
    fn common(&self) -> OptimizerCommon {
        OptimizerCommon {
            eval_concurrency: self.eval_concurrency,
            max_metric_calls: self.max_rollouts,
            max_lm_calls: self.max_lm_calls,
            seed: self.seed,
            ..OptimizerCommon::default()
        }
    }

    fn require_feedback(eval: &CandidateEval, module_name: &str, generation: usize) -> Result<()> {
        if eval
            .rollouts
            .iter()
            .any(|rollout| rollout.eval.feedback.is_none())
        {
            return Err(anyhow!(
                "GEPA requires feedback for every evaluated example (module=`{module_name}`, generation={generation})"
            ));
        }
        Ok(())
    }

    /// Formats per-example scores/feedback plus, for the component under
    /// mutation, every invocation's input/output (or error + raw output) and
    /// tool behavior from the execution trace — the `pred_trace` contract.
    /// Cache-served rollouts carry no trace; their score and feedback still
    /// appear.
    fn summarize_feedback(module_name: &str, eval: &CandidateEval) -> String {
        use std::fmt::Write as _;

        let mut text = String::new();
        for (idx, rollout) in eval.rollouts.iter().enumerate() {
            let _ = writeln!(
                text,
                "{}: score={:.3}; {}",
                idx + 1,
                rollout.eval.score,
                rollout.eval.feedback.as_deref().unwrap_or("-")
            );
            let Some(trace) = &rollout.trace else {
                continue;
            };
            for span in trace.for_component(module_name) {
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
                let _ = writeln!(
                    text,
                    "  call {}: input={input}; output={output}",
                    span.seq
                );
                for event in &span.events {
                    if let crate::trace::SpanEvent::ToolRun {
                        name,
                        result,
                        error,
                        ..
                    } = event
                    {
                        let _ = writeln!(
                            text,
                            "    tool {name}: {}{}",
                            truncate(result, 200),
                            error
                                .as_ref()
                                .map(|e| format!(" (error: {e})"))
                                .unwrap_or_default()
                        );
                    }
                }
            }
        }
        text.trim_end().to_string()
    }

    /// Deterministic fallback mutation: append the feedback to the parent
    /// instruction. Used when no `prompt_model` is configured or the reflection
    /// call fails. The appended feedback is capped at
    /// [`CONCAT_FEEDBACK_BUDGET`] characters so instructions can't grow by a
    /// full trace dump every generation.
    fn concat_child_instruction(
        parent_instruction: &str,
        feedback_summary: &str,
        parent_score: f64,
        generation: usize,
    ) -> String {
        format!(
            "{}\n\n[GEPA gen {}] Improve based on feedback:\n{}\n(Parent score {:.3})",
            parent_instruction,
            generation + 1,
            truncate(feedback_summary, CONCAT_FEEDBACK_BUDGET),
            parent_score,
        )
    }

    /// Proposes a child instruction, preferring LM reflection when a
    /// `prompt_model` is configured.
    ///
    /// Returns the proposed instruction and the number of reflection LM calls
    /// consumed (0 or 1). Reflection failures degrade to the deterministic
    /// concatenation mutation with a warning rather than aborting the run.
    async fn propose_child_instruction(
        &self,
        leaves: &[LeafInfo],
        module_name: &str,
        parent_instruction: &str,
        feedback_summary: &str,
        parent_score: f64,
        generation: usize,
        reflector: Option<&Predict<ReflectOnInstruction>>,
    ) -> (String, usize) {
        let Some(reflector) = reflector else {
            return (
                Self::concat_child_instruction(
                    parent_instruction,
                    feedback_summary,
                    parent_score,
                    generation,
                ),
                0,
            );
        };

        let task_description = leaves
            .iter()
            .find(|leaf| leaf.name == module_name)
            .map(LeafInfo::schema_for_reflection)
            .unwrap_or_default();

        let input = ReflectOnInstructionInput {
            task_description,
            current_instruction: parent_instruction.to_string(),
            execution_feedback: format!(
                "Parent average score: {parent_score:.3}\n{feedback_summary}"
            ),
        };

        match reflector.call(input).await {
            Ok(predicted) => {
                let improved = predicted.improved_instruction.trim().to_string();
                if improved.is_empty() {
                    tracing::warn!(
                        module_name,
                        generation,
                        "reflection LM returned an empty instruction; using feedback concatenation"
                    );
                } else {
                    return (improved, 1);
                }
            }
            Err(err) => {
                tracing::warn!(
                    module_name,
                    generation,
                    error = %err,
                    "reflection LM call failed; using feedback concatenation"
                );
            }
        }

        (
            Self::concat_child_instruction(
                parent_instruction,
                feedback_summary,
                parent_score,
                generation,
            ),
            1,
        )
    }

    /// Convenience: optimizes a typed module over a trainset with this
    /// optimizer's default engine.
    pub async fn compile_module<E, M, MT>(
        &self,
        module: &mut M,
        trainset: &[E],
        metric: &MT,
    ) -> Result<GEPAResult>
    where
        E: ToInput<M::Input> + serde::Serialize + Send + Sync,
        M: Module + Predictors,
        MT: TypedMetric<E, M>,
    {
        self.compile_module_with_valset(module, trainset, None, metric)
            .await
    }

    /// [`compile_module`](Self::compile_module) with an explicit validation
    /// set: initial evaluation and child scoring use the valset, parent
    /// re-evaluation samples trainset minibatches. Sugar over
    /// [`OptimizeTarget::module_with_valset`] + the [`Optimizer`] trait.
    pub async fn compile_module_with_valset<E, M, MT>(
        &self,
        module: &mut M,
        trainset: &[E],
        valset: Option<&[E]>,
        metric: &MT,
    ) -> Result<GEPAResult>
    where
        E: ToInput<M::Input> + serde::Serialize + Send + Sync,
        M: Module + Predictors,
        MT: TypedMetric<E, M>,
    {
        let mut target = OptimizeTarget::module_with_valset(module, trainset, valset, metric);
        let mut engine = Engine::new(Optimizer::engine_config(self));
        let report = Optimizer::compile(self, &mut target, &mut engine).await?;
        report
            .into_gepa()
            .ok_or_else(|| anyhow!("GEPA must return a GEPA report"))
    }
}

#[async_trait::async_trait(?Send)]
impl Optimizer for GEPA {
    fn engine_config(&self) -> EngineConfig {
        self.common().engine_config()
    }

    async fn compile(
        &self,
        target: &mut OptimizeTarget<'_>,
        engine: &mut Engine,
    ) -> Result<Report> {
        let leaves = target.leaves().to_vec();
        if leaves.is_empty() {
            return Err(anyhow!("no optimizable predictors found"));
        }

        // The validation columns are the Pareto/score columns; the trainset
        // columns are the minibatch pool (identical without a valset).
        let val_cols = target.val_columns();
        let train_pool = target.train_columns();

        let reflector = self
            .prompt_model
            .as_ref()
            .map(|lm| Predict::<ReflectOnInstruction>::builder().lm(lm.clone()).build());
        let mut rng = self.common().rng();

        let mut lineage = Lineage::new();

        // Seed the frontier: each predictor's current instruction, scored on
        // the validation columns.
        for leaf in &leaves {
            let row = engine.register(Candidate::with_instruction(&leaf.name, &leaf.instruction));
            let eval = match engine.evaluate(target, row, Some(&val_cols)).await? {
                EvalOutcome::Complete(eval) => eval,
                EvalOutcome::BudgetExhausted { .. } => break,
            };
            Self::require_feedback(&eval, &leaf.name, 0)?;
            lineage.add(
                GEPACandidate {
                    id: 0,
                    instruction: leaf.instruction.clone(),
                    module_name: leaf.name.clone(),
                    example_scores: Vec::new(),
                    parent_id: None,
                    generation: 0,
                },
                row,
                &eval,
            );
        }

        let mut all_candidates = Vec::new();
        let mut evolution_history = Vec::new();
        let mut frontier_history = Vec::new();

        for generation in 0..self.num_iterations {
            if !engine.budget_allows(1) {
                break;
            }

            let parent = lineage
                .sample_parent(&engine.pareto_over(&val_cols), &mut rng)
                .context("failed to sample from frontier")?
                .clone();
            let parent_row = engine.register(Candidate::with_instruction(
                &parent.module_name,
                &parent.instruction,
            ));

            // Parent re-evaluation on a trainset minibatch. Already-seen
            // (candidate, example) pairs are served from the rollout cache.
            let minibatch_size = train_pool.len().min(self.minibatch_size.max(1));
            let minibatch: Vec<usize> = train_pool
                .choose_multiple(&mut rng, minibatch_size)
                .copied()
                .collect();

            let parent_eval = match engine
                .evaluate(target, parent_row, Some(&minibatch))
                .await?
            {
                EvalOutcome::Complete(eval) => eval,
                EvalOutcome::BudgetExhausted { .. } => break,
            };
            Self::require_feedback(&parent_eval, &parent.module_name, generation)?;

            let feedback_summary = Self::summarize_feedback(&parent.module_name, &parent_eval);
            let parent_score = parent_eval.mean();

            // Don't spend a reflection call on a child we can't afford to score.
            if !engine.budget_allows(val_cols.len()) {
                break;
            }

            let (child_instruction, reflection_calls) = self
                .propose_child_instruction(
                    &leaves,
                    &parent.module_name,
                    &parent.instruction,
                    &feedback_summary,
                    parent_score,
                    generation,
                    reflector.as_ref(),
                )
                .await;
            engine.charge(0, reflection_calls);

            let child = parent.mutate(child_instruction, generation + 1);
            let child_row = engine.register(Candidate::with_instruction(
                &child.module_name,
                &child.instruction,
            ));
            let child_eval = match engine.evaluate(target, child_row, Some(&val_cols)).await? {
                EvalOutcome::Complete(eval) => eval,
                EvalOutcome::BudgetExhausted { .. } => break,
            };
            Self::require_feedback(&child_eval, &child.module_name, generation + 1)?;

            let child = lineage.add(child, child_row, &child_eval).clone();

            if self.track_stats {
                all_candidates.push(child);
                let pareto = engine.pareto_over(&val_cols);
                let best_avg = lineage
                    .best_by_average(&pareto)
                    .map(|candidate| candidate.average_score())
                    .unwrap_or(0.0);
                evolution_history.push((generation, best_avg));
                frontier_history.push(pareto.statistics());
            }
        }

        let pareto = engine.pareto_over(&val_cols);
        let best_candidate = lineage
            .best_by_average(&pareto)
            .cloned()
            .context("no candidates available on Pareto frontier")?;

        let winner =
            Candidate::with_instruction(&best_candidate.module_name, &best_candidate.instruction);

        let highest_score_achieved_per_val_task: Vec<f32> = if lineage.entries.is_empty() {
            Vec::new()
        } else {
            pareto
                .best_scores()
                .iter()
                .map(|best| best.unwrap_or(f64::from(f32::MIN)) as f32)
                .collect()
        };

        let best_outputs_valset = if self.track_best_outputs {
            if !engine.budget_allows(val_cols.len()) {
                tracing::debug!(
                    eval_examples = val_cols.len(),
                    spend = ?engine.spend(),
                    max_lm_calls = ?self.max_lm_calls,
                    max_rollouts = ?self.max_rollouts,
                    "skipping best output collection because budget would be exceeded"
                );
                None
            } else {
                let outputs = target.candidate_outputs(&val_cols, &winner).await?;
                engine.charge(val_cols.len(), val_cols.len());
                Some(outputs)
            }
        } else {
            None
        };

        // The one mutation of the run: install the winner.
        target.install(&winner)?;

        Ok(Report::Gepa(GEPAResult {
            best_candidate,
            all_candidates,
            total_rollouts: engine.spend().metric_calls,
            total_lm_calls: engine.spend().lm_calls,
            evolution_history,
            highest_score_achieved_per_val_task,
            best_outputs_valset,
            frontier_history,
        }))
    }
}

#[cfg(test)]
mod tests {
    use anyhow::{Result, anyhow};

    use super::*;
    use crate::evaluate::{Eval, TypedMetric};
    use crate::trace::Trace;
    use crate::{CallMetadata, Predict, PredictError, Predicted, PredictorInfo, Signature};

    #[derive(Signature, Clone, Debug)]
    struct GepaStateSig {
        #[input]
        prompt: String,

        #[output]
        answer: String,
    }

    struct GepaStateModule {
        predictor: Predict<GepaStateSig>,
    }

    crate::predictors!(GepaStateModule { predictor });

    impl Module for GepaStateModule {
        type Input = GepaStateSigInput;
        type Output = GepaStateSigOutput;

        async fn forward(
            &self,
            input: GepaStateSigInput,
        ) -> Result<Predicted<GepaStateSigOutput>, PredictError> {
            Ok(Predicted::new(
                GepaStateSigOutput {
                    answer: input.prompt,
                },
                CallMetadata::default(),
            ))
        }
    }

    struct AlwaysFailMetric;

    type GepaRow = (GepaStateSigInput, GepaStateSigOutput);

    impl TypedMetric<GepaRow, GepaStateModule> for AlwaysFailMetric {
        async fn evaluate(
            &self,
            _example: &GepaRow,
            _prediction: &Predicted<GepaStateSigOutput>,
            _trace: Option<&Trace>,
        ) -> Result<Eval> {
            Err(anyhow!("metric failure"))
        }
    }

    fn eval_set() -> Vec<GepaRow> {
        vec![(
            GepaStateSigInput {
                prompt: "one".to_string(),
            },
            GepaStateSigOutput {
                answer: "one".to_string(),
            },
        )]
    }

    #[tokio::test]
    async fn compile_leaves_state_untouched_when_metric_errors() {
        let optimizer = GEPA::builder().num_iterations(1).minibatch_size(1).build();
        let mut module = GepaStateModule {
            predictor: Predict::<GepaStateSig>::builder()
                .instruction("seed-instruction")
                .build(),
        };

        let err = optimizer
            .compile_module(&mut module, &eval_set(), &AlwaysFailMetric)
            .await
            .expect_err("candidate evaluation should propagate metric failure");
        assert!(err.to_string().contains("metric failure"));

        assert_eq!(
            PredictorInfo::instruction(&module.predictor),
            "seed-instruction"
        );
    }

    #[test]
    fn concat_mutation_caps_appended_feedback() {
        let huge = "x".repeat(50_000);
        let child = GEPA::concat_child_instruction("seed", &huge, 0.5, 3);
        assert!(child.len() < 3_000, "feedback must be capped: {}", child.len());
        assert!(child.starts_with("seed\n\n[GEPA gen 4]"));
    }
}
