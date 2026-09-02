use anyhow::{Context, Result, anyhow};
use bon::Builder;
use rand::{Rng, rngs::StdRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};

use crate::core::ToInput;
use crate::evaluate::TypedMetric;
use std::collections::HashMap;

use crate::optimizer::engine::{
    Candidate, CandidateEval, Engine, EngineConfig, EvalOutcome, ParetoStatistics, ParetoView,
    SCORE_EPS, ScoreMatrix,
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
    /// The candidate's FULL instruction assignment, one entry per optimizable
    /// leaf. `module_name`/`instruction` name the slot this candidate's
    /// mutation touched; multi-leaf programs are searched jointly — children
    /// inherit the parent's whole assignment and rewrite one slot. Empty on
    /// data serialized before 2026-08 (single-leaf era).
    #[serde(default)]
    pub slots: std::collections::BTreeMap<String, String>,
}

impl GEPACandidate {
    pub fn average_score(&self) -> f32 {
        if self.example_scores.is_empty() {
            return 0.0;
        }
        self.example_scores.iter().sum::<f32>() / self.example_scores.len() as f32
    }

    pub fn mutate(&self, new_instruction: String, generation: usize) -> Self {
        self.mutate_slot(self.module_name.clone(), new_instruction, generation)
    }

    /// Child with `slot`'s instruction rewritten; every other slot inherits
    /// the parent's assignment (joint multi-leaf search).
    pub fn mutate_slot(
        &self,
        slot: String,
        new_instruction: String,
        generation: usize,
    ) -> Self {
        let mut slots = self.slots.clone();
        slots.insert(slot.clone(), new_instruction.clone());
        Self {
            id: 0,
            instruction: new_instruction,
            module_name: slot,
            example_scores: Vec::new(),
            parent_id: Some(self.id),
            generation,
            slots,
        }
    }

    /// The instruction currently assigned to `slot` (falls back to this
    /// candidate's own mutation for pre-slots data).
    pub fn slot_text(&self, slot: &str) -> &str {
        self.slots
            .get(slot)
            .map(String::as_str)
            .unwrap_or(&self.instruction)
    }

    /// The engine-level candidate carrying the full assignment.
    pub(crate) fn engine_candidate(&self) -> Candidate {
        if self.slots.is_empty() {
            return Candidate::with_instruction(&self.module_name, &self.instruction);
        }
        let mut candidate = Candidate::new();
        for (name, text) in &self.slots {
            candidate.set_instruction(name, text);
        }
        candidate
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
    /// Children rejected by the held-out gate (no validation sweep spent).
    pub gate_rejected: usize,
    /// Generations skipped because the parent aced its whole reflection batch.
    pub skipped_perfect: usize,
    /// Reflection retries triggered by the leak guard.
    pub leak_retries: usize,
    /// Generations abandoned because the proposal still leaked after a retry.
    pub leak_skipped: usize,
    /// The frontier candidate with the highest raw validation mean, when
    /// seed protection kept the seed instead (`None` when they coincide).
    pub best_by_mean: Option<GEPACandidate>,
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
/// # Overfitting controls (2026-08, see `docs/gepa-diagnosis-2026-08-22.md`)
///
/// Three defenses are on by default; all three address measured failure modes
/// of the reference algorithm, not hypothetical ones:
///
/// - **Reflect on A, gate on B.** Each generation draws two *disjoint*
///   trainset minibatches. Reflection reads batch A's feedback; the child is
///   then scored on held-out batch B against the parent's B scores and only a
///   child that wins on B earns the full validation sweep. (The reference
///   gates on A itself — the batch the child was written to fix — which
///   passed 93% of proposals in our trajectories while only 33% went on to
///   beat their parent on validation.)
/// - **Noise-aware winner (seed-protected argmax).** Children compete by raw
///   validation mean, but the generation-0 seed keeps incumbency unless the
///   argmax beats it by more than `selection_z` standard errors of the paired
///   per-example delta — the seed survives unless something *really* beats
///   it, while real gains between children are never forfeited. Set
///   `selection_z(0.0)` to recover raw argmax behavior.
/// - **Leak guard.** Proposed instructions are checked for verbatim copies of
///   short string values from batch A's rows (gold answers, entities). A
///   leaking proposal is re-asked once with the offending spans named; if it
///   still leaks, the generation is skipped. Disable with
///   `allow_memorization(true)` for tasks where baking dataset facts into
///   the instruction is intended. (The concat fallback used when no
///   `prompt_model` is set embeds feedback verbatim and is exempt.)
///
/// Minibatch A is drawn with epoch-style least-used-first coverage and
/// prefers examples the parent is not already known-perfect on
/// (`perfect_score`); a parent that aces its whole reflection batch skips the
/// generation instead of paying for a no-op reflection.
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
/// - **`gate_size`** (default: `minibatch_size`) — held-out examples for the
///   accept gate (batch B). `Some(0)` disables the gate.
/// - **`selection_z`** (default: 1.0) — winner rule: the argmax-by-mean child
///   wins unless the generation-0 seed is within `z · SE` of it (paired
///   per-example), in which case the seed keeps incumbency. `0.0` = plain
///   argmax.
/// - **`perfect_score`** (default: 1.0) — score at which an example is
///   considered solved; used to bias reflection batches toward unsolved
///   examples and to skip reflection on all-perfect batches.
/// - **`allow_memorization`** (default: false) — disable the leak guard.
/// - **`parents_per_step`** / **`proposals_per_parent`** (defaults: 1) — the
///   P and N of P×N parallel proposals (gepa-ai blog, 2026-07-30): sample P
///   parents per step, reflect N mutations per parent on disjoint
///   mini-batches, run the reflection calls concurrently, and commit the
///   whole batch before the search re-steers with validation results. `1×1`
///   (the default) is the validated sequential behavior.
/// - **`num_trials`** (default: 10) — **currently unused.** Multi-child
///   evolution shipped as `parents_per_step`/`proposals_per_parent` instead;
///   setting this has no effect.
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
///     .max_lm_calls(500)
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

    /// **Currently unused.** Multi-child evolution shipped as
    /// `parents_per_step`/`proposals_per_parent` instead; setting this has
    /// no effect.
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
    /// Held-out examples per accept gate (batch B). `None` = `minibatch_size`;
    /// `Some(0)` disables the gate (every child pays the full validation sweep).
    pub gate_size: Option<usize>,
    /// Winner-selection noise tolerance in standard errors of the paired
    /// per-example delta: the seed keeps incumbency when the argmax candidate
    /// is within tolerance of it. `0.0` = raw argmax of validation means.
    #[builder(default = 1.0)]
    pub selection_z: f64,
    /// Score at which an example counts as solved (used by reflection-batch
    /// biasing and the all-perfect skip). Assumes the metric convention of
    /// scores in `[0, 1]`; for unnormalized metrics set this to your actual
    /// ceiling, or to `f64::INFINITY` to disable both behaviors.
    #[builder(default = 1.0)]
    pub perfect_score: f64,
    /// Disable the leak guard: allow proposals to copy example strings
    /// (gold answers, entities) into the instruction verbatim.
    #[builder(default = false)]
    pub allow_memorization: bool,
    /// Parents sampled per step — the "P" of P×N parallel proposals (see
    /// [`Ratchet::parents_per_step`](crate::optimizer::ratchet::Ratchet::parents_per_step)).
    /// Only the mainline (RATCHET-engine) path reads it; `1` (the default)
    /// preserves the validated sequential behavior.
    #[builder(default = 1)]
    pub parents_per_step: usize,
    /// Reflective mutations per parent per step — the "N" of P×N (see
    /// [`Ratchet::proposals_per_parent`](crate::optimizer::ratchet::Ratchet::proposals_per_parent)).
    /// Only the mainline (RATCHET-engine) path reads it; `1` (the default)
    /// preserves the validated sequential behavior.
    #[builder(default = 1)]
    pub proposals_per_parent: usize,
}

/// GEPA's evolving state: report-facing candidates joined to engine rows.
/// Shared with [`Ratchet`](crate::optimizer::ratchet::Ratchet), which keeps
/// the same pool semantics.
pub(crate) struct Lineage {
    /// `(report candidate, engine candidate index)` in creation order.
    pub(crate) entries: Vec<(GEPACandidate, usize)>,
}

impl Lineage {
    pub(crate) fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub(crate) fn add(
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

    /// Adds a candidate whose scores were assembled from the score matrix
    /// (racing evaluation fills the row block-by-block rather than through a
    /// single [`CandidateEval`]).
    pub(crate) fn add_scored(
        &mut self,
        mut candidate: GEPACandidate,
        engine_row: usize,
        example_scores: Vec<f32>,
    ) -> &GEPACandidate {
        candidate.id = self.entries.len();
        candidate.example_scores = example_scores;
        self.entries.push((candidate, engine_row));
        &self.entries.last().expect("just pushed").0
    }

    /// Samples a parent proportional to Pareto coverage (wins on the
    /// validation columns). Falls back to the first candidate when nobody has
    /// wins recorded yet; `None` when there are no candidates at all.
    pub(crate) fn sample_parent(&self, pareto: &ParetoView, rng: &mut StdRng) -> Option<&GEPACandidate> {
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

    pub(crate) fn best_by_average<'a>(&'a self, pareto: &'a ParetoView) -> Option<&'a GEPACandidate> {
        self.frontier(pareto).max_by(|a, b| {
            a.average_score()
                .partial_cmp(&b.average_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Frontier members with their engine rows (for paired score lookups).
    pub(crate) fn frontier_entries<'a>(
        &'a self,
        pareto: &'a ParetoView,
    ) -> impl Iterator<Item = (&'a GEPACandidate, usize)> {
        self.entries
            .iter()
            .filter(|(_, row)| pareto.wins(*row) > 0)
            .map(|(candidate, row)| (candidate, *row))
    }
}

/// Whether `challenger` is statistically indistinguishable from `best` under
/// a paired per-example comparison: `mean(best − challenger) ≤ z · SE`.
/// Only columns scored for both are compared; `true` on an empty overlap is
/// never returned (no evidence ≠ tie) — the caller keeps `best`.
pub(crate) fn within_noise(best: &[Option<f64>], challenger: &[Option<f64>], z: f64) -> bool {
    let deltas: Vec<f64> = best
        .iter()
        .zip(challenger)
        .filter_map(|(b, c)| Some(b.as_ref()? - c.as_ref()?))
        .collect();
    if deltas.is_empty() {
        return false;
    }
    let n = deltas.len() as f64;
    let mean = deltas.iter().sum::<f64>() / n;
    if mean <= 0.0 {
        return true;
    }
    let var = deltas.iter().map(|d| (d - mean) * (d - mean)).sum::<f64>() / n;
    let se = (var / n).sqrt();
    mean <= z * se
}

/// Collects the short string leaf values (6–80 chars, lowercased) from a JSON
/// value — the "example-specific strings" the leak guard forbids an
/// instruction from quoting. Long strings (contexts, documents) are skipped:
/// gold answers and entity names are short; quoting a paragraph verbatim is a
/// different failure with a different fix.
pub(crate) fn collect_short_strings(value: &serde_json::Value, out: &mut Vec<String>) {
    match value {
        serde_json::Value::String(s) => {
            let trimmed = s.trim();
            if (6..=80).contains(&trimmed.len()) {
                out.push(trimmed.to_lowercase());
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                collect_short_strings(item, out);
            }
        }
        serde_json::Value::Object(map) => {
            for item in map.values() {
                collect_short_strings(item, out);
            }
        }
        _ => {}
    }
}

/// The forbidden strings that appear verbatim (case-insensitive) in
/// `instruction`.
pub(crate) fn find_leaks<'s>(instruction: &str, forbidden: &'s [String]) -> Vec<&'s str> {
    let haystack = instruction.to_lowercase();
    forbidden
        .iter()
        .filter(|needle| haystack.contains(needle.as_str()))
        .map(String::as_str)
        .collect()
}

/// A candidate row's scores over the given columns (`None` = not evaluated).
pub(crate) fn row_scores(matrix: &ScoreMatrix, row: usize, cols: &[usize]) -> Vec<Option<f64>> {
    cols.iter().map(|&col| matrix.score(row, col)).collect()
}

/// What [`GEPA::propose_child_instruction`] produced.
struct ProposalOutcome {
    /// The proposed instruction; `None` when the generation should be skipped
    /// (the proposal kept leaking example strings, or the retry call failed).
    instruction: Option<String>,
    /// Reflection LM calls consumed (0–2).
    reflection_calls: usize,
    /// Whether the leak guard triggered a retry.
    leak_retried: bool,
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

    pub(crate) fn require_feedback(eval: &CandidateEval, module_name: &str, generation: usize) -> Result<()> {
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
    pub(crate) fn summarize_feedback(module_name: &str, eval: &CandidateEval) -> String {
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

    /// Short string values from the given example columns — the strings the
    /// leak guard forbids a proposal from quoting verbatim.
    pub(crate) fn leak_strings(target: &OptimizeTarget<'_>, cols: &[usize]) -> Vec<String> {
        let mut out = Vec::new();
        for &idx in cols {
            if let Some(value) = target.example_json(idx) {
                collect_short_strings(&value, &mut out);
            }
        }
        out.sort();
        out.dedup();
        out
    }

    /// Draws this generation's two disjoint trainset minibatches: A
    /// (reflection) and B (accept gate). Columns are consumed least-used
    /// first — epoch-style coverage, shuffled within equal usage — and both
    /// batches prefer columns the parent is not already known-perfect on
    /// (A so reflection sees failures; B so the gate stays winnable).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sample_batches(
        train_pool: &[usize],
        usage: &HashMap<usize, usize>,
        matrix: &ScoreMatrix,
        parent_row: usize,
        minibatch_size: usize,
        gate_size: usize,
        perfect_score: f64,
        rng: &mut StdRng,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut cols: Vec<usize> = train_pool.to_vec();
        if cols.is_empty() {
            return (Vec::new(), Vec::new());
        }
        cols.shuffle(rng);
        // Stable sort: least-used first, shuffled within equal usage.
        cols.sort_by_key(|col| usage.get(col).copied().unwrap_or(0));

        let a_len = minibatch_size.max(1).min(cols.len());
        let unsolved = |col: &usize| {
            matrix
                .score(parent_row, *col)
                .is_none_or(|score| score < perfect_score - SCORE_EPS)
        };
        let mut batch_a: Vec<usize> = cols.iter().copied().filter(unsolved).take(a_len).collect();
        if batch_a.len() < a_len {
            let fill: Vec<usize> = cols
                .iter()
                .copied()
                .filter(|col| !batch_a.contains(col))
                .take(a_len - batch_a.len())
                .collect();
            batch_a.extend(fill);
        }
        // B gets the same unsolved preference: a gate batch the parent has
        // fully solved is unwinnable (a strictly better child can only tie),
        // which would waste the whole generation.
        let remaining: Vec<usize> = cols
            .iter()
            .copied()
            .filter(|col| !batch_a.contains(col))
            .collect();
        let mut batch_b: Vec<usize> = remaining
            .iter()
            .copied()
            .filter(unsolved)
            .take(gate_size)
            .collect();
        if batch_b.len() < gate_size {
            let fill: Vec<usize> = remaining
                .iter()
                .copied()
                .filter(|col| !batch_b.contains(col))
                .take(gate_size - batch_b.len())
                .collect();
            batch_b.extend(fill);
        }
        (batch_a, batch_b)
    }

    /// Proposes a child instruction, preferring LM reflection when a
    /// `prompt_model` is configured.
    ///
    /// Reflection failures on the first attempt degrade to the deterministic
    /// concatenation mutation with a warning rather than aborting the run.
    /// When `forbidden` is non-empty, a proposal that quotes one of the
    /// strings verbatim is re-asked once with the leaked spans named; a
    /// second leak abandons the proposal (`instruction: None`).
    #[allow(clippy::too_many_arguments)]
    async fn propose_child_instruction(
        &self,
        leaves: &[LeafInfo],
        module_name: &str,
        parent_instruction: &str,
        feedback_summary: &str,
        parent_score: f64,
        generation: usize,
        reflector: Option<&Predict<ReflectOnInstruction>>,
        forbidden: &[String],
    ) -> ProposalOutcome {
        let concat = |reflection_calls: usize| ProposalOutcome {
            instruction: Some(Self::concat_child_instruction(
                parent_instruction,
                feedback_summary,
                parent_score,
                generation,
            )),
            reflection_calls,
            leak_retried: false,
        };
        let Some(reflector) = reflector else {
            return concat(0);
        };

        let task_description = leaves
            .iter()
            .find(|leaf| leaf.name == module_name)
            .map(LeafInfo::schema_for_reflection)
            .unwrap_or_default();
        let base_feedback = format!("Parent average score: {parent_score:.3}\n{feedback_summary}");

        let mut execution_feedback = base_feedback.clone();
        let mut leak_retried = false;
        for attempt in 0..2usize {
            let input = ReflectOnInstructionInput {
                task_description: task_description.clone(),
                current_instruction: parent_instruction.to_string(),
                execution_feedback: execution_feedback.clone(),
            };
            let reflection_calls = attempt + 1;
            match reflector.call(input).await {
                Ok(predicted) => {
                    let improved = predicted.improved_instruction.trim().to_string();
                    if improved.is_empty() {
                        tracing::warn!(
                            module_name,
                            generation,
                            "reflection LM returned an empty instruction; using feedback concatenation"
                        );
                        return concat(reflection_calls);
                    }
                    let leaks = find_leaks(&improved, forbidden);
                    if leaks.is_empty() {
                        return ProposalOutcome {
                            instruction: Some(improved),
                            reflection_calls,
                            leak_retried,
                        };
                    }
                    if attempt == 0 {
                        tracing::debug!(
                            module_name,
                            generation,
                            leaks = ?leaks,
                            "proposal copied example strings verbatim; retrying with leak feedback"
                        );
                        leak_retried = true;
                        execution_feedback = format!(
                            "{base_feedback}\n\nIMPORTANT: your previous proposal copied these \
                             example-specific strings verbatim: {leaks:?}. Never copy answers, \
                             entities, or facts that appear only in specific examples — the \
                             assistant will face different inputs. State the general rule instead."
                        );
                        continue;
                    }
                    tracing::warn!(
                        module_name,
                        generation,
                        leaks = ?leaks,
                        "proposal still copied example strings after retry; skipping this generation"
                    );
                    return ProposalOutcome {
                        instruction: None,
                        reflection_calls,
                        leak_retried,
                    };
                }
                Err(err) => {
                    if attempt == 0 {
                        tracing::warn!(
                            module_name,
                            generation,
                            error = %err,
                            "reflection LM call failed; using feedback concatenation"
                        );
                        return concat(reflection_calls);
                    }
                    tracing::warn!(
                        module_name,
                        generation,
                        error = %err,
                        "reflection LM retry failed; skipping this generation"
                    );
                    return ProposalOutcome {
                        instruction: None,
                        reflection_calls,
                        leak_retried,
                    };
                }
            }
        }
        unreachable!("the proposal loop returns on every path")
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
        // Merged 2026-08-26 (showdown campaign, docs/showdown-preregistration.md):
        // GEPA now runs on the RATCHET engine — held-out gate with noise-aware
        // kill, paired racing, operator portfolio + trial ledger, inline leak
        // guard, fresh-column champion, seed-protected argmax — behind GEPA's
        // existing knob surface. The legacy single-operator loop remains only
        // for the degenerate paths (zero iterations, or no prompt_model),
        // which keep their original seed-eval semantics.
        if self.num_iterations > 0 && self.prompt_model.is_some() {
            return self.compile_ratchet(target, engine).await;
        }
        self.compile_legacy(target, engine).await
    }
}

impl GEPA {
    /// Mainline compile path: delegate to a RATCHET configured from this
    /// optimizer's knobs, then repackage the result behind GEPA's report
    /// shape (val-task bests and optional best-output capture included).
    async fn compile_ratchet(
        &self,
        target: &mut OptimizeTarget<'_>,
        engine: &mut Engine,
    ) -> Result<Report> {
        let ratchet = {
            let mut ratchet = crate::optimizer::ratchet::Ratchet::builder()
                .prompt_model(
                    self.prompt_model
                        .clone()
                        .expect("compile_ratchet is only entered with a prompt_model"),
                )
                .build();
            ratchet.num_iterations = self.num_iterations;
            ratchet.minibatch_size = self.minibatch_size;
            ratchet.gate_size = self.gate_size;
            ratchet.selection_z = self.selection_z;
            ratchet.perfect_score = self.perfect_score;
            ratchet.allow_memorization = self.allow_memorization;
            ratchet.max_rollouts = self.max_rollouts;
            ratchet.max_lm_calls = self.max_lm_calls;
            ratchet.eval_concurrency = self.eval_concurrency;
            ratchet.seed = self.seed;
            ratchet.track_stats = self.track_stats;
            ratchet.parents_per_step = self.parents_per_step;
            ratchet.proposals_per_parent = self.proposals_per_parent;
            ratchet
        };

        let report = Optimizer::compile(&ratchet, target, engine).await?;
        let Report::Ratchet(result) = report else {
            return Err(anyhow!("RATCHET must return a Ratchet report"));
        };

        let val_cols = target.val_columns();
        let highest_score_achieved_per_val_task: Vec<f32> = engine
            .pareto_over(&val_cols)
            .best_scores()
            .iter()
            .map(|best| best.unwrap_or(f64::from(f32::MIN)) as f32)
            .collect();

        // RATCHET installed the winner already; optionally capture its val
        // outputs under the same budget rule as the legacy path.
        let winner = result.best_candidate.engine_candidate();
        let best_outputs_valset = if self.track_best_outputs {
            if !engine.budget_allows(val_cols.len()) {
                tracing::debug!(
                    eval_examples = val_cols.len(),
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

        Ok(Report::Gepa(GEPAResult {
            best_candidate: result.best_candidate,
            all_candidates: result.all_candidates,
            total_rollouts: engine.spend().metric_calls,
            total_lm_calls: engine.spend().lm_calls,
            evolution_history: result.evolution_history,
            highest_score_achieved_per_val_task,
            best_outputs_valset,
            frontier_history: Vec::new(),
            gate_rejected: result.gate_rejected,
            skipped_perfect: result.skipped,
            leak_retries: result.leak_retries,
            leak_skipped: result.leak_skipped,
            best_by_mean: result.best_by_mean,
        }))
    }

    /// Legacy single-operator loop, kept for the degenerate paths (zero
    /// iterations, or no prompt_model).
    async fn compile_legacy(
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
                    slots: std::collections::BTreeMap::from([(
                        leaf.name.clone(),
                        leaf.instruction.clone(),
                    )]),
                },
                row,
                &eval,
            );
        }

        let mut all_candidates = Vec::new();
        let mut evolution_history = Vec::new();
        let mut frontier_history = Vec::new();

        let gate_size = self.gate_size.unwrap_or(self.minibatch_size);
        let mut usage: HashMap<usize, usize> = HashMap::new();
        let mut gate_rejected = 0usize;
        let mut skipped_perfect = 0usize;
        let mut leak_retries = 0usize;
        let mut leak_skipped = 0usize;

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

            // Disjoint minibatches: A for reflection, B for the accept gate.
            let (batch_a, batch_b) = Self::sample_batches(
                &train_pool,
                &usage,
                engine.matrix(),
                parent_row,
                self.minibatch_size,
                gate_size,
                self.perfect_score,
                &mut rng,
            );
            for &col in batch_a.iter().chain(&batch_b) {
                *usage.entry(col).or_insert(0) += 1;
            }

            // Parent re-evaluation on batch A. Already-seen
            // (candidate, example) pairs are served from the rollout cache.
            let parent_eval = match engine.evaluate(target, parent_row, Some(&batch_a)).await? {
                EvalOutcome::Complete(eval) => eval,
                EvalOutcome::BudgetExhausted { .. } => break,
            };
            Self::require_feedback(&parent_eval, &parent.module_name, generation)?;

            // Reflection on an all-perfect batch is a no-op rewrite at best;
            // skip the generation. (Batch biasing makes this rare once the
            // score matrix has trainset coverage.)
            if parent_eval
                .rollouts
                .iter()
                .all(|rollout| rollout.eval.score >= self.perfect_score - SCORE_EPS)
            {
                skipped_perfect += 1;
                tracing::debug!(generation, "parent aced its reflection batch; skipping");
                continue;
            }

            let feedback_summary = Self::summarize_feedback(&parent.module_name, &parent_eval);
            let parent_score = parent_eval.mean();

            // Don't spend a reflection call on a child we can't afford to gate.
            if !engine.budget_allows(batch_b.len().max(1)) {
                break;
            }

            let forbidden = if self.allow_memorization {
                Vec::new()
            } else {
                Self::leak_strings(target, &batch_a)
            };
            let proposal = self
                .propose_child_instruction(
                    &leaves,
                    &parent.module_name,
                    &parent.instruction,
                    &feedback_summary,
                    parent_score,
                    generation,
                    reflector.as_ref(),
                    &forbidden,
                )
                .await;
            engine.charge(0, proposal.reflection_calls);
            leak_retries += proposal.leak_retried as usize;
            let Some(child_instruction) = proposal.instruction else {
                leak_skipped += 1;
                continue;
            };
            if child_instruction == parent.instruction {
                // The teacher proposed a no-op; nothing worth evaluating.
                continue;
            }

            let child = parent.mutate(child_instruction, generation + 1);
            let child_row = engine.register(Candidate::with_instruction(
                &child.module_name,
                &child.instruction,
            ));

            // The accept gate: the child must beat the parent on held-out
            // batch B — which reflection never saw — before earning the full
            // validation sweep. The parent's B scores are cache-served after
            // the first generation that visits them.
            if !batch_b.is_empty() {
                let parent_gate = match engine.evaluate(target, parent_row, Some(&batch_b)).await? {
                    EvalOutcome::Complete(eval) => eval,
                    EvalOutcome::BudgetExhausted { .. } => break,
                };
                let child_gate = match engine.evaluate(target, child_row, Some(&batch_b)).await? {
                    EvalOutcome::Complete(eval) => eval,
                    EvalOutcome::BudgetExhausted { .. } => break,
                };
                Self::require_feedback(&child_gate, &child.module_name, generation + 1)?;
                if child_gate.mean() <= parent_gate.mean() {
                    gate_rejected += 1;
                    tracing::debug!(
                        generation,
                        parent_gate = parent_gate.mean(),
                        child_gate = child_gate.mean(),
                        "held-out gate rejected the child; no validation sweep spent"
                    );
                    continue;
                }
            }

            if !engine.budget_allows(val_cols.len()) {
                break;
            }
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

        if skipped_perfect > 0 && lineage.entries.len() == leaves.len() {
            tracing::warn!(
                skipped_perfect,
                perfect_score = self.perfect_score,
                "every generation skipped as already-perfect; if your metric is not \
                 0-to-1, raise `perfect_score` (or set it to infinity to disable the skip)"
            );
        }

        let pareto = engine.pareto_over(&val_cols);
        let best_by_mean_candidate = lineage
            .best_by_average(&pareto)
            .cloned()
            .context("no candidates available on Pareto frontier")?;
        // Noise-aware winner: raw argmax of validation means is a
        // winner's-curse machine at small val sizes, but over-protection
        // forfeits real gains — hence seed-protected argmax (below).
        // Seed-protected argmax: children compete by raw validation mean; only
        // the generation-0 seed keeps incumbency when the argmax fails to beat
        // it by more than `selection_z` standard errors (paired per column).
        // Protecting every earlier child was measured to forfeit real gains
        // when headroom exists (benchmarks/ratchet-hotpot ablation:
        // chosen − argmax = −0.045), while seed protection alone is what
        // prevents shipping noise-artifact regressions at zero headroom
        // (benchmarks/gepa-hotpot: it dodged a −0.05 regression).
        let best_candidate = if self.selection_z > 0.0 {
            let best_row = lineage
                .frontier_entries(&pareto)
                .find(|(candidate, _)| candidate.id == best_by_mean_candidate.id)
                .map(|(_, row)| row)
                .expect("the argmax candidate is on the frontier");
            let best_scores = row_scores(engine.matrix(), best_row, &val_cols);
            let seed_entry = lineage.entries.iter().find(|(candidate, _)| {
                candidate.generation == 0
                    && candidate.module_name == best_by_mean_candidate.module_name
            });
            match seed_entry {
                Some((seed, seed_row))
                    if seed.id != best_by_mean_candidate.id
                        && within_noise(
                            &best_scores,
                            &row_scores(engine.matrix(), *seed_row, &val_cols),
                            self.selection_z,
                        ) =>
                {
                    seed.clone()
                }
                _ => best_by_mean_candidate.clone(),
            }
        } else {
            best_by_mean_candidate.clone()
        };
        let best_by_mean =
            (best_candidate.id != best_by_mean_candidate.id).then_some(best_by_mean_candidate);

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
            gate_rejected,
            skipped_perfect,
            leak_retries,
            leak_skipped,
            best_by_mean,
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
    fn parallel_proposal_knobs_default_to_sequential() {
        let gepa = GEPA::builder().build();
        assert_eq!(gepa.parents_per_step, 1);
        assert_eq!(gepa.proposals_per_parent, 1);
    }

    #[test]
    fn concat_mutation_caps_appended_feedback() {
        let huge = "x".repeat(50_000);
        let child = GEPA::concat_child_instruction("seed", &huge, 0.5, 3);
        assert!(child.len() < 3_000, "feedback must be capped: {}", child.len());
        assert!(child.starts_with("seed\n\n[GEPA gen 4]"));
    }

    #[test]
    fn within_noise_calls_real_gaps_and_tolerates_noisy_ones() {
        // Identical scores: always within noise.
        let a = vec![Some(0.8), Some(0.6), Some(1.0)];
        assert!(within_noise(&a, &a, 1.0));
        // Uniform, consistent gap (SE = 0 < mean): a real difference.
        let worse = vec![Some(0.6), Some(0.4), Some(0.8)];
        assert!(!within_noise(&a, &worse, 1.0));
        // Small mean gap with large per-example variance: within noise.
        let best = vec![Some(1.0), Some(0.0), Some(1.0), Some(0.0), Some(1.0)];
        let noisy = vec![Some(0.0), Some(1.0), Some(1.0), Some(0.0), Some(1.0)];
        assert!(within_noise(&best, &noisy, 1.0));
        // Challenger actually better on the paired columns: trivially within.
        let better = vec![Some(1.0), Some(0.8), Some(1.0)];
        assert!(within_noise(&a, &better, 1.0));
        // No paired evidence: NOT a tie.
        let unscored = vec![None, None, None];
        assert!(!within_noise(&a, &unscored, 1.0));
        // z = 0 degenerates to strict mean comparison: any positive mean
        // deficit disqualifies, however noisy…
        let slightly_worse = vec![Some(1.0), Some(0.0), Some(1.0), Some(0.0), Some(0.8)];
        assert!(!within_noise(&best, &slightly_worse, 0.0));
        // …while exactly equal means remain a tie (Occam tie-break applies).
        assert!(within_noise(&best, &noisy, 0.0));
    }

    #[test]
    fn leak_guard_catches_gold_answers_but_not_long_contexts() {
        let row = serde_json::json!({
            "context": "A very long paragraph about many things that easily exceeds the eighty character limit for leak candidates.",
            "question": "Who won the 1992 award?",
            "answer": "Leo Varadkar",
            "key_points": ["short fact", "predictors: BTreeMap<String, PredictState>"],
            "n": 3,
        });
        let mut strings = Vec::new();
        collect_short_strings(&row, &mut strings);
        strings.sort();
        strings.dedup();
        // Long context excluded, short-but-trivial "short fact" (10 chars) included,
        // number and nothing under 6 chars.
        assert!(strings.contains(&"leo varadkar".to_string()));
        assert!(strings.contains(&"who won the 1992 award?".to_string()));
        assert!(!strings.iter().any(|s| s.contains("eighty character limit")));

        let leaky = "Answer precisely. Note that the answer to the award question is Leo Varadkar.";
        let leaks = find_leaks(leaky, &strings);
        assert_eq!(leaks, vec!["leo varadkar"]);
        let clean = "Answer precisely and return only the exact span.";
        assert!(find_leaks(clean, &strings).is_empty());
    }

    #[test]
    fn sample_batches_are_disjoint_and_prefer_unsolved() {
        use crate::optimizer::engine::ScoreMatrix;
        use rand::SeedableRng;

        let train_pool: Vec<usize> = (10..20).collect();
        let mut matrix = ScoreMatrix::new(20);
        matrix.ensure_rows(1);
        // Parent is perfect on columns 10..15, unsolved on 15..20.
        for col in 10..15 {
            matrix.record(0, col, 1.0);
        }
        for col in 15..18 {
            matrix.record(0, col, 0.2);
        }
        let usage = HashMap::new();
        let mut rng = StdRng::seed_from_u64(0);
        let (a, b) = GEPA::sample_batches(&train_pool, &usage, &matrix, 0, 3, 3, 1.0, &mut rng);
        assert_eq!(a.len(), 3);
        assert_eq!(b.len(), 3);
        assert!(a.iter().all(|col| !b.contains(col)), "A and B must be disjoint");
        assert!(
            a.iter().all(|col| (15..20).contains(col)),
            "A must prefer unsolved/unseen columns, got {a:?}"
        );
        // B prefers the remaining unsolved columns (2 left) so the gate is
        // winnable, then fills from solved ones.
        assert_eq!(
            b.iter().filter(|col| (15..20).contains(*col)).count(),
            2,
            "B must take the remaining unsolved columns first, got {b:?}"
        );

        // Usage steers coverage: columns already used a lot come last.
        let mut heavy = HashMap::new();
        for col in 15..20 {
            heavy.insert(col, 5);
        }
        let (a2, _) = GEPA::sample_batches(&train_pool, &heavy, &matrix, 0, 3, 3, 1.0, &mut rng);
        assert!(
            a2.iter().all(|col| (15..20).contains(col)),
            "unsolved preference outranks usage for batch A, got {a2:?}"
        );

        // Degenerate pool: no room for B once A is filled.
        let tiny: Vec<usize> = vec![0, 1];
        let (a3, b3) = GEPA::sample_batches(&tiny, &usage, &matrix, 0, 2, 3, 1.0, &mut rng);
        assert_eq!(a3.len(), 2);
        assert!(b3.is_empty());
    }
}
