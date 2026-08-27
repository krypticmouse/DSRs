//! RATCHET: reflective prompt evolution with a trial ledger, paired racing
//! evaluation, and an adaptive operator portfolio (design:
//! `docs/ratchet-design.md`).
//!
//! A ratchet only turns one way: validated gains lock in (the ledger and the
//! noise-aware winner), slips get caught early (held-out gate + paired
//! racing), and the search never re-spends budget on falsified ideas.
//! Every mechanism here traces to a *measured* GEPA failure mode
//! (`docs/gepa-diagnosis-2026-08-22.md`), not intuition:
//!
//! - GEPA's reflection is memoryless → both engines plateau after 1–2
//!   accepted mutations. RATCHET feeds every proposal the **trial ledger**:
//!   what was tried, the hypothesis it claimed, and its measured fate.
//! - GEPA spends a full validation sweep per child → 5 proposals per
//!   120-rollout budget. RATCHET **races** children block-by-block against
//!   the parent's cached per-column scores and kills them the moment the
//!   paired deficit is significant.
//! - GEPA has one mutation operator. RATCHET runs a **portfolio** — Rewrite,
//!   Patch, Simplify, Merge — selected per generation by a UCB1 bandit on
//!   measured promote-rate. Merge is semantic crossover between Pareto
//!   members with complementary win-sets, which works even for
//!   single-instruction programs (where structural merge is a no-op).
//!
//! # TEMPER
//!
//! The same engine also hosts **TEMPER** (design: `docs/temper-design.md`):
//! four additional mechanisms behind builder knobs that default *off*, so
//! RATCHET's measured v2 behavior is untouched and every TEMPER ablation is
//! a builder config rather than a code fork. [`temper`] builds the preset.
//!
//! - [`ParentSelection::Lexicase`]: ε-lexicase parent selection with novelty
//!   pressure, replacing Pareto-coverage sampling — GEPA-style
//!   dominance/coverage selection measurably concentrates mass on
//!   noise-minted specialist candidates as the pool grows (gepa-ai/gepa
//!   PR #432; our measured anti-scaling at budget 2400).
//! - [`BatchWeighting::Learnability`]: minibatches weighted by per-column
//!   solve variance `p(1-p)` — reflection concentrates on the frontier band
//!   the pool solves *sometimes*, not on never-solvable columns.
//! - `variant_pairs`: harness-supplied answer-preserving paraphrase twins of
//!   seen examples; never optimized against, they vote only in the endgame
//!   champion race, where phrasing-memorized prompts lose.
//! - `length_cap`: a hard proposal-length budget with one shortening retry —
//!   bloat correlates with overfit in both our data and production reports.

use std::collections::HashMap;

use anyhow::{Context, Result, anyhow};
use bon::Builder;
use rand::{Rng, rngs::StdRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};

use crate::core::ToInput;
use crate::evaluate::TypedMetric;
use crate::optimizer::engine::{
    BatchEvalOutcome, Candidate, CandidateEval, Engine, EngineConfig, EvalOutcome,
    RolloutOutcome, SCORE_EPS, ScoreMatrix,
};
use crate::optimizer::gepa::{
    GEPA, GEPACandidate, Lineage, find_leaks, row_scores, within_noise,
};
use crate::optimizer::target::LeafInfo;
use crate::optimizer::{OptimizeTarget, Optimizer, OptimizerCommon, Report};
use crate::utils::truncate;
use crate::{Module, Predict, Predictors, Signature};

// ---------------------------------------------------------------------------
// Operator signatures — all four produce (improved_instruction, edit_summary)
// ---------------------------------------------------------------------------

/// Rewrite a module's instruction from execution feedback and the trial ledger.
///
/// You are one step of an evolutionary search over the system instruction of
/// one module inside an LLM pipeline. Study the module's contract, the
/// current instruction, the per-example execution feedback, and the trial
/// ledger of what this search has already tried. Write an improved
/// instruction: fix the failure modes the feedback names, keep what already
/// works, build on the validated lessons, and never re-propose a falsified
/// idea. Write rules that generalize; never copy answers, entities, or other
/// example-specific facts — the module will face different inputs. Also state,
/// in one line of at most 25 words, the single hypothesis your edit is
/// betting on (what change, expected effect).
#[derive(Signature, Clone, Debug)]
struct ProposeRewrite {
    /// The module's input and output fields with their descriptions.
    #[input]
    task_description: String,

    /// The instruction currently used by the module.
    #[input]
    current_instruction: String,

    /// Per-example scores and textual feedback from the last evaluation.
    #[input]
    execution_feedback: String,

    /// What this search already tried: validated lessons and falsified hypotheses.
    #[input]
    trial_ledger: String,

    /// The full replacement instruction.
    #[output]
    improved_instruction: String,

    /// One line (max 25 words): the hypothesis this edit bets on.
    #[output]
    edit_summary: String,
}

/// Make one minimal, targeted edit to a module's instruction.
///
/// You are one step of an evolutionary search over the system instruction of
/// one module inside an LLM pipeline. Below are two rollouts of the same
/// instruction: one that scored well and one that scored poorly. Distill the
/// difference into ONE rule and integrate it into the instruction with the
/// smallest possible edit — do not restructure or rewrite the rest. The rule
/// must generalize; never copy answers, entities, or other example-specific
/// facts. Never re-propose a falsified idea from the trial ledger. Also
/// state, in one line of at most 25 words, the hypothesis your edit is
/// betting on.
#[derive(Signature, Clone, Debug)]
struct ProposePatch {
    /// The module's input and output fields with their descriptions.
    #[input]
    task_description: String,

    /// The instruction currently used by the module.
    #[input]
    current_instruction: String,

    /// The best-scoring rollout on this minibatch.
    #[input]
    better_rollout: String,

    /// The worst-scoring rollout on this minibatch.
    #[input]
    worse_rollout: String,

    /// What this search already tried: validated lessons and falsified hypotheses.
    #[input]
    trial_ledger: String,

    /// The instruction with one minimal rule integrated.
    #[output]
    improved_instruction: String,

    /// One line (max 25 words): the hypothesis this edit bets on.
    #[output]
    edit_summary: String,
}

/// Compress a module's instruction without losing what is proven to work.
///
/// You are one step of an evolutionary search over the system instruction of
/// one module inside an LLM pipeline. Instructions accumulate speculative
/// clauses over generations; bloat dilutes the constraints that matter.
/// Rewrite the instruction to be substantially shorter: keep every constraint
/// the validated lessons support, merge redundant phrasing, and drop clauses
/// that no lesson supports. Do not add anything new. Also state, in one line
/// of at most 25 words, what you removed and why it was safe.
#[derive(Signature, Clone, Debug)]
struct ProposeSimplify {
    /// The module's input and output fields with their descriptions.
    #[input]
    task_description: String,

    /// The instruction currently used by the module.
    #[input]
    current_instruction: String,

    /// Edits this search has measured to help, with their deltas.
    #[input]
    validated_lessons: String,

    /// The compressed instruction.
    #[output]
    improved_instruction: String,

    /// One line (max 25 words): what was removed and why it was safe.
    #[output]
    edit_summary: String,
}

/// Merge two instructions that win on different examples into one.
///
/// You are one step of an evolutionary search over the system instruction of
/// one module inside an LLM pipeline. The two instructions below each beat
/// the other on a different subset of examples (their strengths are listed).
/// Write ONE instruction that keeps what each does right: identify the
/// specific behavior responsible for each side's wins and combine those
/// behaviors without contradiction or duplication. Prefer the tighter overall
/// structure of the two. Never copy answers, entities, or other
/// example-specific facts. Also state, in one line of at most 25 words, which
/// behavior you took from each side.
#[derive(Signature, Clone, Debug)]
struct ProposeMerge {
    /// The module's input and output fields with their descriptions.
    #[input]
    task_description: String,

    /// The first instruction.
    #[input]
    instruction_a: String,

    /// Examples instruction A wins on, with the metric's feedback.
    #[input]
    a_strengths: String,

    /// The second instruction.
    #[input]
    instruction_b: String,

    /// Examples instruction B wins on, with the metric's feedback.
    #[input]
    b_strengths: String,

    /// What this search already tried: validated lessons and falsified hypotheses.
    #[input]
    trial_ledger: String,

    /// The merged instruction.
    #[output]
    improved_instruction: String,

    /// One line (max 25 words): what was taken from each side.
    #[output]
    edit_summary: String,
}

// ---------------------------------------------------------------------------
// Public result types
// ---------------------------------------------------------------------------

/// One of RATCHET's proposal operators.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RatchetOp {
    /// Full rewrite from batch-A failures (GEPA-style, ledger-aware).
    Rewrite,
    /// One minimal rule distilled from the best-vs-worst rollout contrast.
    Patch,
    /// Compress: drop clauses not backed by validated lessons.
    Simplify,
    /// Semantic crossover of two Pareto members with complementary win-sets.
    Merge,
}

impl RatchetOp {
    pub fn name(self) -> &'static str {
        match self {
            Self::Rewrite => "rewrite",
            Self::Patch => "patch",
            Self::Simplify => "simplify",
            Self::Merge => "merge",
        }
    }
}

/// How the parent of each generation is drawn from the candidate pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParentSelection {
    /// RATCHET/GEPA default: sample proportional to Pareto coverage (wins
    /// on validation columns).
    Coverage,
    /// TEMPER T1: ε-lexicase — shuffle the discriminative validation columns,
    /// iteratively keep only candidates within `lexicase_epsilon` of the
    /// column best, and break the final tie by sampling ∝ `1/(1+children)`
    /// (novelty pressure). No dominance pruning, so a noise-minted sole
    /// solver cannot monopolize selection mass and broadly-strong candidates
    /// always retain some.
    Lexicase,
}

/// How reflection/gate minibatches are drawn from the trainset.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchWeighting {
    /// RATCHET/GEPA default: least-used-first epoch coverage, biased toward
    /// columns the parent has not solved.
    UnsolvedBias,
    /// TEMPER T2: least-used-first epoch coverage, weighted by per-column
    /// solve variance `p(1-p)` over the pool (+0.05 floor; never-evaluated
    /// columns get the maximum-uncertainty prior 0.25). Assumes scores are
    /// roughly 0-to-1; keep [`BatchWeighting::UnsolvedBias`] for unnormalized
    /// metrics.
    Learnability,
}

/// How a trial ended.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum TrialOutcome {
    /// The proposal still quoted example strings after a leak retry.
    LeakSkipped,
    /// The proposal still exceeded `length_cap` after a shortening retry.
    LengthSkipped,
    /// The proposal was identical to its parent.
    NoOp,
    /// Killed on the held-out gate batch (child ≤ parent).
    GateKilled { parent_gate: f64, child_gate: f64 },
    /// Killed mid-race: paired deficit vs the parent exceeded `race_z` SEs.
    RaceKilled { columns_evaluated: usize, paired_delta: f64 },
    /// Survived the full race and joined the pool.
    Promoted { paired_delta: f64, candidate_id: usize },
}

/// One entry of the trial ledger: what was proposed, by which operator, the
/// hypothesis it claimed, and what the evaluation actually said.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrialRecord {
    pub generation: usize,
    pub operator: RatchetOp,
    pub parent_id: usize,
    /// The teacher's own one-line summary of the edit's bet.
    pub hypothesis: String,
    pub outcome: TrialOutcome,
}

/// Full report from a [`Ratchet`] run.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RatchetResult {
    /// The winner under the noise-aware Occam rule; installed on the module.
    pub best_candidate: GEPACandidate,
    /// The raw argmax-by-validation-mean candidate when seed protection
    /// kept the seed instead (`None` when they coincide).
    pub best_by_mean: Option<GEPACandidate>,
    /// Every promoted candidate (empty unless `track_stats`).
    pub all_candidates: Vec<GEPACandidate>,
    /// The complete trial ledger, in order.
    pub trials: Vec<TrialRecord>,
    /// Per-operator `(name, attempts, wins)` — a win is a promotion with a
    /// positive paired delta.
    pub operator_stats: Vec<(String, usize, usize)>,
    /// Total evaluation rollouts consumed.
    pub total_rollouts: usize,
    /// Total LM calls consumed (rollouts + reflection).
    pub total_lm_calls: usize,
    /// `(generation, best_average_score)` pairs for plotting convergence.
    pub evolution_history: Vec<(usize, f32)>,
    /// Children killed on the held-out gate.
    pub gate_rejected: usize,
    /// Children killed mid-race.
    pub race_killed: usize,
    /// Generations with no eligible operator (parent aced its batch and
    /// nothing else applied).
    pub skipped: usize,
    /// Reflection retries triggered by the leak guard.
    pub leak_retries: usize,
    /// Proposals abandoned because they still leaked after a retry.
    pub leak_skipped: usize,
    /// Reflection retries triggered by the length cap.
    #[serde(default)]
    pub length_retries: usize,
    /// Proposals abandoned because they still exceeded the cap after a retry.
    #[serde(default)]
    pub length_skipped: usize,
    /// Never-touched trainset columns used for fresh-data champion selection
    /// (`0` when the phase did not run).
    pub fresh_selection_cols: usize,
    /// Variant-tempering columns that voted in the champion race (`0` when
    /// no variants were supplied or the phase did not run).
    #[serde(default)]
    pub variant_selection_cols: usize,
    /// Clauses removed from the champion by the ablation finale.
    #[serde(default)]
    pub ablation_removed: usize,
    /// Rollouts spent by the ablation finale.
    #[serde(default)]
    pub ablation_rollouts: usize,
}

// ---------------------------------------------------------------------------
// Optimizer
// ---------------------------------------------------------------------------

/// Reflective prompt evolution with a trial ledger, paired racing evaluation,
/// and an adaptive operator portfolio. See the module docs and
/// `docs/ratchet-design.md`.
///
/// Requires a `prompt_model`: all four operators are LM-driven (there is no
/// concatenation fallback). Inherits GEPA's full anti-overfit stack:
/// reflect-on-A/gate-on-B disjoint minibatches with unsolved bias, the leak
/// guard, and the noise-aware Occam winner.
///
/// ```ignore
/// let ratchet = Ratchet::builder()
///     .num_iterations(30)
///     .minibatch_size(5)
///     .max_rollouts(600)
///     .prompt_model(teacher_lm)
///     .build();
/// let report = ratchet
///     .compile_module_with_valset(&mut module, &train, Some(&val), &metric)
///     .await?;
/// ```
#[derive(Builder)]
pub struct Ratchet {
    /// Evolutionary generations to attempt.
    #[builder(default = 30)]
    pub num_iterations: usize,

    /// Examples per reflection batch (batch A).
    #[builder(default = 5)]
    pub minibatch_size: usize,

    /// Held-out gate batch B size. `None` = `minibatch_size`; `Some(0)`
    /// disables the gate (children go straight to the race).
    pub gate_size: Option<usize>,

    /// Validation columns evaluated per racing block.
    #[builder(default = 8)]
    pub race_block: usize,

    /// Racing kill threshold: a child whose paired deficit to its parent
    /// exceeds `race_z` standard errors dies without finishing the sweep.
    #[builder(default = 1.0)]
    pub race_z: f64,

    /// Gate tolerance: the child passes batch B unless the parent beats it by
    /// more than `gate_z` standard errors (paired). Lateral moves — children
    /// that merely tie the parent on B — pass and let the race decide; v1's
    /// strict `>` gate was measured to prune the stepping stones deep
    /// lineages are made of. `0.0` = strict mean comparison.
    #[builder(default = 1.0)]
    pub gate_z: f64,

    /// Rewrite proposals sampled per generation (temperature diversity);
    /// the one with the better gate-batch mean races. Other operators
    /// propose once.
    #[builder(default = 2)]
    pub rewrite_fanout: usize,

    /// Fresh-data champion selection: reserve budget to race the top two
    /// candidates (by validation mean) on up to this many *never-touched*
    /// trainset columns at the end — champion selection on unseen data has
    /// no winner's curse. `0` disables.
    #[builder(default = 40)]
    pub champion_fresh: usize,

    /// Winner-selection noise tolerance (the Occam rule; `0.0` = raw argmax).
    #[builder(default = 1.0)]
    pub selection_z: f64,

    /// Score at which an example counts as solved. Set to your ceiling for
    /// unnormalized metrics, or `f64::INFINITY` to disable batch biasing and
    /// the all-perfect skip.
    #[builder(default = 1.0)]
    pub perfect_score: f64,

    /// Disable the leak guard.
    #[builder(default = false)]
    pub allow_memorization: bool,

    /// Falsified trials shown in the ledger digest (validated lessons are
    /// always shown in full).
    #[builder(default = 12)]
    pub ledger_window: usize,

    /// How generation parents are drawn from the pool. TEMPER uses
    /// [`ParentSelection::Lexicase`].
    #[builder(default = ParentSelection::Coverage)]
    pub parent_selection: ParentSelection,

    /// ε for lexicase filtering and column discrimination (only read when
    /// `parent_selection` is [`ParentSelection::Lexicase`]).
    #[builder(default = 0.05)]
    pub lexicase_epsilon: f64,

    /// How reflection/gate minibatches are drawn. TEMPER uses
    /// [`BatchWeighting::Learnability`].
    #[builder(default = BatchWeighting::UnsolvedBias)]
    pub batch_weighting: BatchWeighting,

    /// TEMPER T3 variant tempering: `(source_idx, variant_idx)` pairs of
    /// trainset indices, where the variant row is an answer-preserving
    /// paraphrase of the source row. Variant columns are excluded from every
    /// batch, gate, and race draw — the optimizer never optimizes against
    /// them — and are added to the endgame champion race, where a prompt
    /// that memorized seen phrasings loses. Empty disables.
    #[builder(default)]
    pub variant_pairs: Vec<(usize, usize)>,

    /// Hard cap (chars) on proposed instructions: an over-cap proposal gets
    /// one shortening retry, then the trial is abandoned. Operators are told
    /// the cap inline. `None` disables.
    pub length_cap: Option<usize>,

    /// Apply `length_cap` only once the search has a promoted child. The
    /// bootstrap phase explores freely — the cap twice killed winning-lineage
    /// rewrites when enforced from generation 0 (temper-design §6, ablation
    /// pre-reg: docs/temper-v2-ablation-prereg.md).
    #[builder(default = false)]
    pub cap_after_promotion: bool,

    /// Use lexicase parent selection only once the search has a promoted
    /// child (Pareto coverage before). Lexicase guards a *late-run* pathology
    /// (sole-solver collapse); always-on it dilutes early exploitation.
    #[builder(default = false)]
    pub lexicase_after_promotion: bool,

    /// Clause-ablation finale: race single-clause removals of the shipped
    /// champion on the fresh∪variant selection columns and keep removals
    /// that don't lose (within `selection_z`·SE, paired). Compactness earned
    /// causally instead of via a hard cap.
    #[builder(default = false)]
    pub ablate_champion: bool,

    /// Rollout budget reserved for the clause-ablation finale (only read
    /// when `ablate_champion` is set; reserved only once a child has been
    /// promoted — an unconditional reserve starves the search).
    #[builder(default = 90)]
    pub ablate_budget: usize,

    /// Reflection LM driving all four operators. **Required.**
    pub prompt_model: Option<crate::LM>,

    /// Hard cap on evaluation rollouts.
    pub max_rollouts: Option<usize>,
    /// Hard cap on LM calls (rollouts + reflection).
    pub max_lm_calls: Option<usize>,

    /// Concurrent rollouts in flight during evaluation.
    #[builder(default = crate::evaluate::DEFAULT_EVAL_CONCURRENCY)]
    pub eval_concurrency: usize,

    /// Seed for batch sampling, racing order, and operator tie-breaks.
    pub seed: Option<u64>,

    /// Record promoted candidates and evolution history in the report.
    #[builder(default = true)]
    pub track_stats: bool,
}

/// Builds a TEMPER-configured [`Ratchet`] (design: `docs/temper-design.md`):
/// ε-lexicase parent selection with novelty pressure, learnability-band
/// minibatches, and a 1,500-char length cap, on top of RATCHET's full
/// backbone. Supply variant-tempering pairs (or an empty vec to skip T3).
/// Set `prompt_model` on the returned value — like RATCHET, TEMPER refuses
/// to run without one — and override any other knob through the public
/// fields:
///
/// ```ignore
/// let mut optimizer = temper(variant_pairs);
/// optimizer.prompt_model = Some(teacher_lm);
/// optimizer.max_rollouts = Some(600);
/// ```
pub fn temper(variant_pairs: Vec<(usize, usize)>) -> Ratchet {
    Ratchet::builder()
        .parent_selection(ParentSelection::Lexicase)
        .batch_weighting(BatchWeighting::Learnability)
        .length_cap(1500)
        .variant_pairs(variant_pairs)
        .build()
}

/// Split an instruction into clause-level segments for the ablation finale:
/// newline-separated blocks, with long blocks further split on sentence
/// boundaries. Segments rejoin with `\n` — content-preserving for prompts,
/// not byte-identical.
fn split_clauses(text: &str) -> Vec<String> {
    let mut clauses = Vec::new();
    for block in text.split('\n') {
        let block = block.trim_end();
        if block.trim().is_empty() {
            continue;
        }
        if block.len() <= 200 {
            clauses.push(block.to_string());
            continue;
        }
        let bytes = block.as_bytes();
        let mut start = 0usize;
        let mut i = 0usize;
        while i + 1 < bytes.len() {
            if matches!(bytes[i], b'.' | b'!' | b'?') && bytes[i + 1] == b' ' {
                let segment = block[start..=i].trim();
                if !segment.is_empty() {
                    clauses.push(segment.to_string());
                }
                start = i + 2;
                i += 2;
                continue;
            }
            i += 1;
        }
        let tail = block[start..].trim();
        if !tail.is_empty() {
            clauses.push(tail.to_string());
        }
    }
    clauses
}

struct Reflectors {
    rewrite: Predict<ProposeRewrite>,
    patch: Predict<ProposePatch>,
    simplify: Predict<ProposeSimplify>,
    merge: Predict<ProposeMerge>,
}

impl Reflectors {
    fn new(lm: &crate::LM) -> Self {
        Self {
            rewrite: Predict::<ProposeRewrite>::builder().lm(lm.clone()).build(),
            patch: Predict::<ProposePatch>::builder().lm(lm.clone()).build(),
            simplify: Predict::<ProposeSimplify>::builder().lm(lm.clone()).build(),
            merge: Predict::<ProposeMerge>::builder().lm(lm.clone()).build(),
        }
    }
}

/// What an operator proposal produced.
struct OpProposal {
    /// `None` = abandon this trial (leak/length after retry, or reflection
    /// error).
    instruction: Option<String>,
    hypothesis: String,
    reflection_calls: usize,
    leak_retried: bool,
    leaked: bool,
    length_retried: bool,
    overlong: bool,
}

impl OpProposal {
    fn abandoned(
        reflection_calls: usize,
        leak_retried: bool,
        leaked: bool,
        length_retried: bool,
        overlong: bool,
    ) -> Self {
        Self {
            instruction: None,
            hypothesis: String::new(),
            reflection_calls,
            leak_retried,
            leaked,
            length_retried,
            overlong,
        }
    }
}

fn leak_addendum(leaks: &[&str]) -> String {
    format!(
        "\n\nIMPORTANT: your previous proposal copied these example-specific strings \
         verbatim: {leaks:?}. Never copy answers, entities, or facts that appear only \
         in specific examples — the module will face different inputs. State the \
         general rule instead."
    )
}

fn length_addendum(len: usize, cap: usize) -> String {
    format!(
        "\n\nIMPORTANT: your previous proposal was {len} characters; the hard limit \
         is {cap}. Rewrite it to fit: keep the rules with measured value, cut \
         speculative clauses and redundant phrasing."
    )
}

/// Runs one operator call with the leak and length guards: propose, check for
/// verbatim example strings and the length cap, retry once naming the
/// violations, abandon on a second violation.
macro_rules! guarded_propose {
    ($reflector:expr, $forbidden:expr, $cap:expr, $make_input:expr) => {{
        let mut leak_retried = false;
        let mut length_retried = false;
        let mut extra = String::new();
        let mut outcome = None;
        for attempt in 0..2usize {
            let calls = attempt + 1;
            match $reflector.call($make_input(extra.clone())).await {
                Ok(predicted) => {
                    let instruction = predicted.improved_instruction.trim().to_string();
                    let hypothesis =
                        truncate(predicted.edit_summary.trim(), 200).to_string();
                    if instruction.is_empty() {
                        outcome = Some(OpProposal::abandoned(
                            calls, leak_retried, false, length_retried, false,
                        ));
                        break;
                    }
                    let leaks = find_leaks(&instruction, $forbidden);
                    let cap: Option<usize> = $cap;
                    let overlong = cap.is_some_and(|cap| instruction.len() > cap);
                    if leaks.is_empty() && !overlong {
                        outcome = Some(OpProposal {
                            instruction: Some(instruction),
                            hypothesis,
                            reflection_calls: calls,
                            leak_retried,
                            leaked: false,
                            length_retried,
                            overlong: false,
                        });
                        break;
                    }
                    if attempt == 0 {
                        extra = String::new();
                        if !leaks.is_empty() {
                            leak_retried = true;
                            extra.push_str(&leak_addendum(&leaks));
                        }
                        if overlong {
                            length_retried = true;
                            extra.push_str(&length_addendum(
                                instruction.len(),
                                cap.expect("overlong implies a cap"),
                            ));
                        }
                        continue;
                    }
                    tracing::warn!(
                        leaks = ?leaks,
                        overlong,
                        "proposal still violated guards after retry; abandoning trial"
                    );
                    outcome = Some(OpProposal::abandoned(
                        calls,
                        leak_retried,
                        !leaks.is_empty(),
                        length_retried,
                        overlong,
                    ));
                    break;
                }
                Err(err) => {
                    tracing::warn!(error = %err, "reflection call failed; abandoning trial");
                    outcome = Some(OpProposal::abandoned(
                        calls, leak_retried, false, length_retried, false,
                    ));
                    break;
                }
            }
        }
        outcome.expect("the proposal loop always sets an outcome")
    }};
}

impl Ratchet {
    fn common(&self) -> OptimizerCommon {
        OptimizerCommon {
            eval_concurrency: self.eval_concurrency,
            max_metric_calls: self.max_rollouts,
            max_lm_calls: self.max_lm_calls,
            seed: self.seed,
            ..OptimizerCommon::default()
        }
    }

    /// Renders the ledger digest fed to every operator: all validated lessons
    /// plus the last `ledger_window` falsified trials.
    fn render_ledger(trials: &[TrialRecord], window: usize) -> String {
        use std::fmt::Write as _;
        let mut text = String::from(
            "Validated lessons (edits that measurably helped — keep and build on them):\n",
        );
        let mut any = false;
        for trial in trials {
            if let TrialOutcome::Promoted { paired_delta, .. } = trial.outcome
                && paired_delta > 0.0
            {
                let _ = writeln!(
                    text,
                    "- [gen {} | {} | val {:+.3}] {}",
                    trial.generation,
                    trial.operator.name(),
                    paired_delta,
                    trial.hypothesis
                );
                any = true;
            }
        }
        if !any {
            text.push_str("(none yet)\n");
        }
        text.push_str(
            "\nFalsified hypotheses (edits that failed evaluation — do NOT re-propose these ideas):\n",
        );
        let mut falsified: Vec<&TrialRecord> = trials
            .iter()
            .filter(|t| {
                matches!(
                    t.outcome,
                    TrialOutcome::GateKilled { .. } | TrialOutcome::RaceKilled { .. }
                ) && !t.hypothesis.is_empty()
            })
            .collect();
        let skip = falsified.len().saturating_sub(window);
        falsified.drain(..skip);
        if falsified.is_empty() {
            text.push_str("(none yet)\n");
        }
        for trial in falsified {
            let fate = match trial.outcome {
                TrialOutcome::GateKilled { .. } => "killed at held-out gate".to_string(),
                TrialOutcome::RaceKilled { columns_evaluated, paired_delta } => {
                    format!("race-killed after {columns_evaluated} examples, val {paired_delta:+.3}")
                }
                _ => unreachable!("filtered above"),
            };
            let _ = writeln!(
                text,
                "- [gen {} | {} | {}] {}",
                trial.generation,
                trial.operator.name(),
                fate,
                trial.hypothesis
            );
        }
        text.trim_end().to_string()
    }

    /// Renders only the validated lessons (for Simplify).
    fn render_validated(trials: &[TrialRecord]) -> String {
        use std::fmt::Write as _;
        let mut text = String::new();
        for trial in trials {
            if let TrialOutcome::Promoted { paired_delta, .. } = trial.outcome
                && paired_delta > 0.0
            {
                let _ = writeln!(text, "- [val {:+.3}] {}", paired_delta, trial.hypothesis);
            }
        }
        if text.is_empty() {
            text.push_str("(none yet — keep only what the current instruction clearly needs)");
        }
        text.trim_end().to_string()
    }

    /// TEMPER T1: ε-lexicase parent selection with novelty pressure.
    ///
    /// Shuffles the discriminative validation columns (pool disagreement
    /// beyond ε), then walks them keeping only candidates within ε of the
    /// column best; the final tie is sampled ∝ `1/(1+children)` so
    /// heavily-mutated lineages yield ground to unexplored ones. Unlike
    /// coverage sampling, a candidate that is second-best everywhere keeps
    /// selection mass.
    fn sample_parent_lexicase<'p>(
        pool: &'p Lineage,
        matrix: &ScoreMatrix,
        val_cols: &[usize],
        epsilon: f64,
        rng: &mut StdRng,
    ) -> Option<&'p GEPACandidate> {
        if pool.entries.is_empty() {
            return None;
        }
        let mut children: HashMap<usize, usize> = HashMap::new();
        for (candidate, _) in &pool.entries {
            if let Some(parent) = candidate.parent_id {
                *children.entry(parent).or_insert(0) += 1;
            }
        }
        let mut cols: Vec<usize> = val_cols
            .iter()
            .copied()
            .filter(|&col| {
                let scores: Vec<f64> = pool
                    .entries
                    .iter()
                    .filter_map(|(_, row)| matrix.score(*row, col))
                    .collect();
                scores.len() >= 2 && {
                    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let min = scores.iter().cloned().fold(f64::INFINITY, f64::min);
                    max - min > epsilon
                }
            })
            .collect();
        cols.shuffle(rng);
        let mut survivors: Vec<usize> = (0..pool.entries.len()).collect();
        for col in cols {
            if survivors.len() <= 1 {
                break;
            }
            let best = survivors
                .iter()
                .filter_map(|&i| matrix.score(pool.entries[i].1, col))
                .fold(f64::NEG_INFINITY, f64::max);
            if !best.is_finite() {
                continue;
            }
            let filtered: Vec<usize> = survivors
                .iter()
                .copied()
                .filter(|&i| {
                    matrix
                        .score(pool.entries[i].1, col)
                        .is_some_and(|score| score >= best - epsilon)
                })
                .collect();
            if !filtered.is_empty() {
                survivors = filtered;
            }
        }
        let weights: Vec<f64> = survivors
            .iter()
            .map(|&i| {
                let kids = children
                    .get(&pool.entries[i].0.id)
                    .copied()
                    .unwrap_or(0);
                1.0 / (1.0 + kids as f64)
            })
            .collect();
        let total: f64 = weights.iter().sum();
        let mut draw = rng.gen_range(0.0..total.max(f64::MIN_POSITIVE));
        for (&i, weight) in survivors.iter().zip(&weights) {
            if draw < *weight {
                return Some(&pool.entries[i].0);
            }
            draw -= weight;
        }
        survivors.last().map(|&i| &pool.entries[i].0)
    }

    /// TEMPER T2: learnability-band minibatches.
    ///
    /// Weighted sampling without replacement (Efraimidis–Spirakis keys),
    /// stratified least-used-first so epoch coverage survives: within a
    /// usage tier, a column's draw odds follow `p(1-p) + 0.05` where `p` is
    /// the pool's mean score on it — the frontier band the pool solves
    /// *sometimes*. Never-evaluated columns get the maximum-uncertainty
    /// prior 0.25. Batch A takes the first `minibatch_size` columns, batch B
    /// the next `gate_size` (disjoint by construction).
    fn sample_batches_learnability(
        pool: &Lineage,
        train_pool: &[usize],
        usage: &HashMap<usize, usize>,
        matrix: &ScoreMatrix,
        minibatch_size: usize,
        gate_size: usize,
        rng: &mut StdRng,
    ) -> (Vec<usize>, Vec<usize>) {
        if train_pool.is_empty() {
            return (Vec::new(), Vec::new());
        }
        let mut keyed: Vec<(usize, usize, f64)> = train_pool
            .iter()
            .map(|&col| {
                let scores: Vec<f64> = pool
                    .entries
                    .iter()
                    .filter_map(|(_, row)| matrix.score(*row, col))
                    .collect();
                let weight = if scores.is_empty() {
                    0.25
                } else {
                    let p = (scores.iter().sum::<f64>() / scores.len() as f64).clamp(0.0, 1.0);
                    p * (1.0 - p) + 0.05
                };
                let key = rng.gen_range(0.0f64..1.0).powf(1.0 / weight);
                (col, usage.get(&col).copied().unwrap_or(0), key)
            })
            .collect();
        keyed.sort_by(|a, b| a.1.cmp(&b.1).then(b.2.total_cmp(&a.2)));
        let a_len = minibatch_size.max(1).min(keyed.len());
        let batch_a: Vec<usize> = keyed.iter().take(a_len).map(|entry| entry.0).collect();
        let batch_b: Vec<usize> = keyed
            .iter()
            .skip(a_len)
            .take(gate_size)
            .map(|entry| entry.0)
            .collect();
        (batch_a, batch_b)
    }

    /// UCB1 over the eligible operators on promote-with-positive-delta rate.
    /// Untried eligible operators are explored first.
    fn choose_op(
        eligible: &[RatchetOp],
        stats: &HashMap<RatchetOp, (usize, usize)>,
        rng: &mut StdRng,
    ) -> RatchetOp {
        let untried: Vec<RatchetOp> = eligible
            .iter()
            .copied()
            .filter(|op| stats.get(op).is_none_or(|s| s.0 == 0))
            .collect();
        if let Some(&op) = untried.choose(rng) {
            return op;
        }
        let total: usize = eligible
            .iter()
            .map(|op| stats.get(op).map_or(0, |s| s.0))
            .sum();
        let mut best = eligible[0];
        let mut best_score = f64::NEG_INFINITY;
        for &op in eligible {
            let (attempts, wins) = stats[&op];
            let score = wins as f64 / attempts as f64
                + (2.0 * (total.max(1) as f64).ln() / attempts as f64).sqrt();
            if score > best_score {
                best_score = score;
                best = op;
            }
        }
        best
    }

    /// Formats one rollout (score, feedback, component sub-trace) for Patch.
    fn fmt_rollout(module_name: &str, rollout: &RolloutOutcome, row: usize) -> String {
        let single = CandidateEval {
            candidate: row,
            rollouts: vec![rollout.clone()],
        };
        GEPA::summarize_feedback(module_name, &single)
    }

    /// Finds two same-module frontier members with complementary win-sets:
    /// each is Pareto-best on validation columns the other is not. Returns
    /// the pair (better average first) and their exclusive win columns.
    fn find_merge_pair(
        pool: &Lineage,
        engine: &Engine,
        val_cols: &[usize],
        module_name: &str,
    ) -> Option<(GEPACandidate, usize, GEPACandidate, usize, Vec<usize>, Vec<usize>)> {
        let pareto = engine.pareto_over(val_cols);
        let best = pareto.best_scores();
        let entries: Vec<(&GEPACandidate, usize)> = pool
            .frontier_entries(&pareto)
            .filter(|(c, _)| c.module_name == module_name)
            .collect();
        let wins_of = |row: usize| -> Vec<usize> {
            val_cols
                .iter()
                .enumerate()
                .filter(|(pos, col)| {
                    matches!(
                        (engine.matrix().score(row, **col), best[*pos]),
                        (Some(score), Some(top)) if score + SCORE_EPS >= top
                    )
                })
                .map(|(_, col)| *col)
                .collect()
        };
        let mut best_pair = None;
        let mut best_strength = 0usize;
        for i in 0..entries.len() {
            for j in (i + 1)..entries.len() {
                let (a, row_a) = entries[i];
                let (b, row_b) = entries[j];
                let wins_a = wins_of(row_a);
                let wins_b = wins_of(row_b);
                let excl_a: Vec<usize> =
                    wins_a.iter().copied().filter(|c| !wins_b.contains(c)).collect();
                let excl_b: Vec<usize> =
                    wins_b.iter().copied().filter(|c| !wins_a.contains(c)).collect();
                let strength = excl_a.len().min(excl_b.len());
                if strength > best_strength {
                    best_strength = strength;
                    let (first, first_row, second, second_row, first_excl, second_excl) =
                        if a.average_score() >= b.average_score() {
                            (a.clone(), row_a, b.clone(), row_b, excl_a, excl_b)
                        } else {
                            (b.clone(), row_b, a.clone(), row_a, excl_b, excl_a)
                        };
                    best_pair =
                        Some((first, first_row, second, second_row, first_excl, second_excl));
                }
            }
        }
        best_pair
    }

    /// Formats a candidate's exclusive-win columns for the Merge prompt,
    /// pulling the metric's cached feedback for a few of them.
    fn fmt_strengths(
        engine: &Engine,
        target: &OptimizeTarget<'_>,
        row: usize,
        exclusive_wins: &[usize],
    ) -> String {
        use std::fmt::Write as _;
        let mut text = format!(
            "Beats the other instruction on {} validation example(s).",
            exclusive_wins.len()
        );
        let salt = engine.config().cache_salt;
        for &col in exclusive_wins.iter().take(3) {
            if let Some(eval) = engine.cache().get(
                target.baseline(),
                engine.candidate_hash(row),
                target.example_uid(col),
                salt,
            ) {
                let _ = write!(
                    text,
                    "\n- score {:.2}: {}",
                    eval.score,
                    truncate(eval.feedback.as_deref().unwrap_or("-"), 300)
                );
            }
        }
        text
    }

    /// Convenience: optimizes a typed module over a trainset.
    pub async fn compile_module<E, M, MT>(
        &self,
        module: &mut M,
        trainset: &[E],
        metric: &MT,
    ) -> Result<RatchetResult>
    where
        E: ToInput<M::Input> + serde::Serialize + Send + Sync,
        M: Module + Predictors,
        MT: TypedMetric<E, M>,
    {
        self.compile_module_with_valset(module, trainset, None, metric)
            .await
    }

    /// [`compile_module`](Self::compile_module) with an explicit validation
    /// set: scoring and racing use the valset columns, reflection and gate
    /// batches sample the trainset.
    pub async fn compile_module_with_valset<E, M, MT>(
        &self,
        module: &mut M,
        trainset: &[E],
        valset: Option<&[E]>,
        metric: &MT,
    ) -> Result<RatchetResult>
    where
        E: ToInput<M::Input> + serde::Serialize + Send + Sync,
        M: Module + Predictors,
        MT: TypedMetric<E, M>,
    {
        let mut target = OptimizeTarget::module_with_valset(module, trainset, valset, metric);
        let mut engine = Engine::new(Optimizer::engine_config(self));
        let report = Optimizer::compile(self, &mut target, &mut engine).await?;
        report
            .into_ratchet()
            .ok_or_else(|| anyhow!("Ratchet must return a Ratchet report"))
    }
}

#[async_trait::async_trait(?Send)]
impl Optimizer for Ratchet {
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
        let reflectors = match &self.prompt_model {
            Some(lm) => Reflectors::new(lm),
            None => {
                return Err(anyhow!(
                    "Ratchet requires a prompt_model — all four operators are LM-driven"
                ));
            }
        };

        let val_cols = target.val_columns();
        let full_train_pool = target.train_columns();
        let mut rng = self.common().rng();
        // TEMPER T3: map variant trainset indices to engine columns and pull
        // them out of the batch pool — variants are never optimized against;
        // they only vote in the endgame champion race.
        let mut variant_cols: Vec<usize> = Vec::new();
        for &(source, variant) in &self.variant_pairs {
            if source >= full_train_pool.len() || variant >= full_train_pool.len() {
                tracing::warn!(source, variant, "variant_pairs index out of range; ignoring pair");
                continue;
            }
            variant_cols.push(full_train_pool[variant]);
        }
        variant_cols.sort_unstable();
        variant_cols.dedup();
        let batch_pool: Vec<usize> = full_train_pool
            .iter()
            .copied()
            .filter(|col| !variant_cols.contains(col))
            .collect();
        // Reserve the fresh-data champion slice out of the batch pool up
        // front: the epoch sampler deliberately touches every pool column,
        // so post-hoc "never used" columns barely exist. Reserved columns
        // are unseen by construction.
        let (train_pool, champion_cols) =
            if self.champion_fresh > 0 && batch_pool.len() > self.champion_fresh + 20 {
                let mut cols = batch_pool.clone();
                cols.shuffle(&mut rng);
                let reserved = cols[..self.champion_fresh].to_vec();
                let rest = cols[self.champion_fresh..].to_vec();
                (rest, reserved)
            } else {
                (batch_pool, Vec::new())
            };
        // The endgame champion race runs on fresh ∪ variant columns.
        let selection_cols: Vec<usize> = champion_cols
            .iter()
            .chain(&variant_cols)
            .copied()
            .collect();
        let mut pool = Lineage::new();

        // Seed the pool: each leaf's current instruction, scored on val.
        for leaf in &leaves {
            let row = engine.register(Candidate::with_instruction(&leaf.name, &leaf.instruction));
            let eval = match engine.evaluate(target, row, Some(&val_cols)).await? {
                EvalOutcome::Complete(eval) => eval,
                EvalOutcome::BudgetExhausted { .. } => break,
            };
            GEPA::require_feedback(&eval, &leaf.name, 0)?;
            pool.add(
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

        let gate_size = self.gate_size.unwrap_or(self.minibatch_size);
        let mut usage: HashMap<usize, usize> = HashMap::new();
        let mut trials: Vec<TrialRecord> = Vec::new();
        let mut op_stats: HashMap<RatchetOp, (usize, usize)> = HashMap::new();
        let mut all_candidates = Vec::new();
        let mut evolution_history = Vec::new();
        let (mut gate_rejected, mut race_killed, mut skipped) = (0usize, 0usize, 0usize);
        let (mut leak_retries, mut leak_skipped) = (0usize, 0usize);
        let (mut length_retries, mut length_skipped) = (0usize, 0usize);

        'generations: for generation in 0..self.num_iterations {
            // Keep enough budget in reserve for the fresh-data champion
            // phase — including this generation's own worst-case spend
            // (batch A + three gate evaluations + a full race).
            // Promotion state gates the scheduled knobs (cap/lexicase after
            // promotion), the ablation reserve, and Simplify/Merge
            // eligibility further down.
            let has_promotion = pool.entries.len() > leaves.len();

            let worst_generation =
                self.minibatch_size + 3 * gate_size + val_cols.len();
            // Reserving finale budget from generation 0 measurably starves
            // the search (v2.0 run shipped the seed at 402/600 rollouts);
            // there is nothing to ablate until a child is promoted, so the
            // reserve waits for one. It also carries a fan-out margin:
            // `worst_generation` counts one race, but rewrite fan-out can run
            // `rewrite_fanout` full races in a generation, and that overshoot
            // consumed the entire finale budget in the v2.1 run.
            // Margin = the paired-race worst case: each fanned-out child can
            // bill child AND parent rows over the full val slice, which
            // `worst_generation`'s single `val_cols` term undercounts (v2.1
            // and v2.2 both lost the finale to this overshoot).
            let ablate_reserve = if self.ablate_champion && has_promotion {
                self.ablate_budget
                    + (2 * self.rewrite_fanout).saturating_sub(1) * val_cols.len()
            } else {
                0
            };
            if !engine.budget_allows(2 * selection_cols.len() + worst_generation + ablate_reserve)
            {
                break;
            }
            let effective_cap = if self.cap_after_promotion && !has_promotion {
                None
            } else {
                self.length_cap
            };
            let use_lexicase = matches!(self.parent_selection, ParentSelection::Lexicase)
                && (!self.lexicase_after_promotion || has_promotion);

            let sampled_parent = if !use_lexicase {
                pool.sample_parent(&engine.pareto_over(&val_cols), &mut rng)
            } else {
                Self::sample_parent_lexicase(
                    &pool,
                    engine.matrix(),
                    &val_cols,
                    self.lexicase_epsilon,
                    &mut rng,
                )
            }
            .context("failed to sample from frontier")?
            .clone();
            let sampled_row = engine.register(Candidate::with_instruction(
                &sampled_parent.module_name,
                &sampled_parent.instruction,
            ));

            let (batch_a, batch_b) = match self.batch_weighting {
                BatchWeighting::UnsolvedBias => GEPA::sample_batches(
                    &train_pool,
                    &usage,
                    engine.matrix(),
                    sampled_row,
                    self.minibatch_size,
                    gate_size,
                    self.perfect_score,
                    &mut rng,
                ),
                BatchWeighting::Learnability => Self::sample_batches_learnability(
                    &pool,
                    &train_pool,
                    &usage,
                    engine.matrix(),
                    self.minibatch_size,
                    gate_size,
                    &mut rng,
                ),
            };
            for &col in batch_a.iter().chain(&batch_b) {
                *usage.entry(col).or_insert(0) += 1;
            }

            // Parent on batch A (mostly cache-served for revisited parents).
            let parent_eval = match engine.evaluate(target, sampled_row, Some(&batch_a)).await? {
                EvalOutcome::Complete(eval) => eval,
                EvalOutcome::BudgetExhausted { .. } => break,
            };
            GEPA::require_feedback(&parent_eval, &sampled_parent.module_name, generation)?;
            let a_perfect = parent_eval
                .rollouts
                .iter()
                .all(|r| r.eval.score >= self.perfect_score - SCORE_EPS);
            let a_spread = {
                let scores = parent_eval.scores();
                let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let min = scores.iter().cloned().fold(f64::INFINITY, f64::min);
                scores.len() >= 2 && max > min + SCORE_EPS
            };

            // Operator eligibility, then bandit choice.
            let merge_pair = Self::find_merge_pair(
                &pool,
                engine,
                &val_cols,
                &sampled_parent.module_name,
            );
            let mut eligible = Vec::new();
            if !a_perfect {
                eligible.push(RatchetOp::Rewrite);
                if a_spread {
                    eligible.push(RatchetOp::Patch);
                }
            }
            // Simplify and Merge only earn a slot once the search has at
            // least one promoted child — before that, forced exploration of
            // them is wasted budget (measured 1 win in 25 early attempts).
            if has_promotion && sampled_parent.instruction.len() > 500 {
                eligible.push(RatchetOp::Simplify);
            }
            if has_promotion && merge_pair.is_some() {
                eligible.push(RatchetOp::Merge);
            }
            if eligible.is_empty() {
                skipped += 1;
                tracing::debug!(generation, "no eligible operator; skipping generation");
                continue;
            }
            let op = Self::choose_op(&eligible, &op_stats, &mut rng);

            if !engine.budget_allows(batch_b.len().max(1)) {
                break;
            }

            let forbidden = if self.allow_memorization {
                Vec::new()
            } else {
                GEPA::leak_strings(target, &batch_a)
            };
            let ledger = Self::render_ledger(&trials, self.ledger_window);
            let mut task_description = leaves
                .iter()
                .find(|leaf| leaf.name == sampled_parent.module_name)
                .map(LeafInfo::schema_for_reflection)
                .unwrap_or_default();
            if let Some(cap) = effective_cap {
                task_description.push_str(&format!(
                    "\n\nHARD LIMIT: the improved instruction must be at most {cap} \
                     characters. Proposals over the limit are rejected."
                ));
            }

            // The lineage parent the child is gated/raced against: the
            // sampled parent, except Merge races against the pair's stronger
            // member.
            let (mut lineage_parent, mut lineage_row) = (sampled_parent.clone(), sampled_row);

            let proposal = match op {
                RatchetOp::Rewrite => {
                    let feedback = format!(
                        "Parent average score: {:.3}\n{}",
                        parent_eval.mean(),
                        GEPA::summarize_feedback(&sampled_parent.module_name, &parent_eval)
                    );
                    guarded_propose!(reflectors.rewrite, &forbidden, effective_cap, |extra: String| {
                        ProposeRewriteInput {
                            task_description: task_description.clone(),
                            current_instruction: sampled_parent.instruction.clone(),
                            execution_feedback: format!("{feedback}{extra}"),
                            trial_ledger: ledger.clone(),
                        }
                    })
                }
                RatchetOp::Patch => {
                    let best = parent_eval
                        .rollouts
                        .iter()
                        .max_by(|a, b| a.eval.score.total_cmp(&b.eval.score))
                        .expect("batch A is non-empty");
                    let worst = parent_eval
                        .rollouts
                        .iter()
                        .min_by(|a, b| a.eval.score.total_cmp(&b.eval.score))
                        .expect("batch A is non-empty");
                    let better =
                        Self::fmt_rollout(&sampled_parent.module_name, best, sampled_row);
                    let worse =
                        Self::fmt_rollout(&sampled_parent.module_name, worst, sampled_row);
                    guarded_propose!(reflectors.patch, &forbidden, effective_cap, |extra: String| {
                        ProposePatchInput {
                            task_description: task_description.clone(),
                            current_instruction: sampled_parent.instruction.clone(),
                            better_rollout: better.clone(),
                            worse_rollout: format!("{worse}{extra}"),
                            trial_ledger: ledger.clone(),
                        }
                    })
                }
                RatchetOp::Simplify => {
                    let validated = Self::render_validated(&trials);
                    guarded_propose!(reflectors.simplify, &forbidden, effective_cap, |extra: String| {
                        ProposeSimplifyInput {
                            task_description: task_description.clone(),
                            current_instruction: sampled_parent.instruction.clone(),
                            validated_lessons: format!("{validated}{extra}"),
                        }
                    })
                }
                RatchetOp::Merge => {
                    let (first, first_row, second, second_row, first_excl, second_excl) =
                        merge_pair.expect("eligibility checked");
                    let a_strengths =
                        Self::fmt_strengths(engine, target, first_row, &first_excl);
                    let b_strengths =
                        Self::fmt_strengths(engine, target, second_row, &second_excl);
                    lineage_parent = first.clone();
                    lineage_row = first_row;
                    guarded_propose!(reflectors.merge, &forbidden, effective_cap, |extra: String| {
                        ProposeMergeInput {
                            task_description: task_description.clone(),
                            instruction_a: first.instruction.clone(),
                            a_strengths: a_strengths.clone(),
                            instruction_b: second.instruction.clone(),
                            b_strengths: format!("{b_strengths}{extra}"),
                            trial_ledger: ledger.clone(),
                        }
                    })
                }
            };

            engine.charge(0, proposal.reflection_calls);
            leak_retries += proposal.leak_retried as usize;
            length_retries += proposal.length_retried as usize;
            let entry = op_stats.entry(op).or_insert((0, 0));
            entry.0 += 1;

            let Some(child_instruction) = proposal.instruction else {
                if proposal.leaked {
                    leak_skipped += 1;
                    trials.push(TrialRecord {
                        generation,
                        operator: op,
                        parent_id: lineage_parent.id,
                        hypothesis: proposal.hypothesis,
                        outcome: TrialOutcome::LeakSkipped,
                    });
                } else if proposal.overlong {
                    length_skipped += 1;
                    trials.push(TrialRecord {
                        generation,
                        operator: op,
                        parent_id: lineage_parent.id,
                        hypothesis: proposal.hypothesis,
                        outcome: TrialOutcome::LengthSkipped,
                    });
                }
                continue;
            };

            let mut hypothesis = proposal.hypothesis;
            let mut child = lineage_parent.mutate(child_instruction, generation + 1);
            child.module_name = lineage_parent.module_name.clone();
            let mut child_row = engine.register(Candidate::with_instruction(
                &child.module_name,
                &child.instruction,
            ));
            if child_row == lineage_row {
                trials.push(TrialRecord {
                    generation,
                    operator: op,
                    parent_id: lineage_parent.id,
                    hypothesis,
                    outcome: TrialOutcome::NoOp,
                });
                continue;
            }

            // Rewrite fan-out: sample a second proposal from the same
            // reflection context (the teacher runs at temperature 1.0, so
            // the draws differ); the better of the two on batch B races.
            let mut sibling: Option<(GEPACandidate, usize, String)> = None;
            if op == RatchetOp::Rewrite && self.rewrite_fanout > 1 && !batch_b.is_empty() {
                let feedback = format!(
                    "Parent average score: {:.3}\n{}",
                    parent_eval.mean(),
                    GEPA::summarize_feedback(&sampled_parent.module_name, &parent_eval)
                );
                let second = guarded_propose!(reflectors.rewrite, &forbidden, effective_cap, |extra: String| {
                    ProposeRewriteInput {
                        task_description: task_description.clone(),
                        current_instruction: sampled_parent.instruction.clone(),
                        execution_feedback: format!("{feedback}{extra}"),
                        trial_ledger: ledger.clone(),
                    }
                });
                engine.charge(0, second.reflection_calls);
                leak_retries += second.leak_retried as usize;
                length_retries += second.length_retried as usize;
                if let Some(instruction) = second.instruction {
                    let mut sib = lineage_parent.mutate(instruction, generation + 1);
                    sib.module_name = lineage_parent.module_name.clone();
                    let sib_row = engine.register(Candidate::with_instruction(
                        &sib.module_name,
                        &sib.instruction,
                    ));
                    if sib_row != lineage_row && sib_row != child_row {
                        sibling = Some((sib, sib_row, second.hypothesis));
                    }
                }
            }

            // Held-out gate on batch B — noise-aware: a child dies here
            // only when the parent beats it by more than `gate_z` SEs.
            // Lateral moves pass and the race decides; v1's strict `>` gate
            // was measured to prune the stepping stones deep lineages need.
            if !batch_b.is_empty() {
                let parent_gate =
                    match engine.evaluate(target, lineage_row, Some(&batch_b)).await? {
                        EvalOutcome::Complete(eval) => eval,
                        EvalOutcome::BudgetExhausted { .. } => break,
                    };
                let mut child_gate = match engine.evaluate(target, child_row, Some(&batch_b)).await?
                {
                    EvalOutcome::Complete(eval) => eval,
                    EvalOutcome::BudgetExhausted { .. } => break,
                };
                GEPA::require_feedback(&child_gate, &child.module_name, generation + 1)?;
                if let Some((sib, sib_row, sib_hypothesis)) = sibling {
                    let sib_gate = match engine.evaluate(target, sib_row, Some(&batch_b)).await? {
                        EvalOutcome::Complete(eval) => eval,
                        EvalOutcome::BudgetExhausted { .. } => break,
                    };
                    if sib_gate.mean() > child_gate.mean() {
                        tracing::debug!(generation, "fan-out sibling won batch B; racing it");
                        child = sib;
                        child_row = sib_row;
                        hypothesis = sib_hypothesis;
                        child_gate = sib_gate;
                    }
                }
                let parent_b = row_scores(engine.matrix(), lineage_row, &batch_b);
                let child_b = row_scores(engine.matrix(), child_row, &batch_b);
                if !within_noise(&parent_b, &child_b, self.gate_z) {
                    gate_rejected += 1;
                    trials.push(TrialRecord {
                        generation,
                        operator: op,
                        parent_id: lineage_parent.id,
                        hypothesis: hypothesis.clone(),
                        outcome: TrialOutcome::GateKilled {
                            parent_gate: parent_gate.mean(),
                            child_gate: child_gate.mean(),
                        },
                    });
                    continue;
                }
            }

            // Paired race over the validation columns in shuffled blocks.
            let mut order = val_cols.clone();
            order.shuffle(&mut rng);
            let mut evaluated: Vec<usize> = Vec::new();
            let mut race_dead = false;
            for block in order.chunks(self.race_block.max(1)) {
                let block_eval = match engine.evaluate(target, child_row, Some(block)).await? {
                    EvalOutcome::Complete(eval) => eval,
                    EvalOutcome::BudgetExhausted { .. } => break 'generations,
                };
                GEPA::require_feedback(&block_eval, &child.module_name, generation + 1)?;
                evaluated.extend_from_slice(block);
                if evaluated.len() == val_cols.len() {
                    break;
                }
                let parent_scores = row_scores(engine.matrix(), lineage_row, &evaluated);
                let child_scores = row_scores(engine.matrix(), child_row, &evaluated);
                if !within_noise(&parent_scores, &child_scores, self.race_z) {
                    let deltas: Vec<f64> = evaluated
                        .iter()
                        .filter_map(|&col| {
                            Some(
                                engine.matrix().score(child_row, col)?
                                    - engine.matrix().score(lineage_row, col)?,
                            )
                        })
                        .collect();
                    let delta = deltas.iter().sum::<f64>() / deltas.len().max(1) as f64;
                    race_killed += 1;
                    race_dead = true;
                    trials.push(TrialRecord {
                        generation,
                        operator: op,
                        parent_id: lineage_parent.id,
                        hypothesis: hypothesis.clone(),
                        outcome: TrialOutcome::RaceKilled {
                            columns_evaluated: evaluated.len(),
                            paired_delta: delta,
                        },
                    });
                    break;
                }
            }
            if race_dead {
                continue;
            }
            if evaluated.len() < val_cols.len() {
                // Budget died mid-race (handled by the labeled break) or the
                // val set was empty; either way there is nothing to promote.
                continue;
            }

            // Survivor: assemble its full row from the matrix and promote.
            let child_scores: Vec<f32> = val_cols
                .iter()
                .map(|&col| engine.matrix().score(child_row, col).unwrap_or(0.0) as f32)
                .collect();
            let paired_delta = {
                let parent_scores = row_scores(engine.matrix(), lineage_row, &val_cols);
                let deltas: Vec<f64> = val_cols
                    .iter()
                    .zip(&parent_scores)
                    .filter_map(|(&col, parent)| {
                        Some(engine.matrix().score(child_row, col)? - (*parent)?)
                    })
                    .collect();
                deltas.iter().sum::<f64>() / deltas.len().max(1) as f64
            };
            let promoted = pool.add_scored(child, child_row, child_scores).clone();
            if paired_delta > 0.0 {
                entry.1 += 1;
            }
            trials.push(TrialRecord {
                generation,
                operator: op,
                parent_id: lineage_parent.id,
                hypothesis,
                outcome: TrialOutcome::Promoted {
                    paired_delta,
                    candidate_id: promoted.id,
                },
            });
            if self.track_stats {
                all_candidates.push(promoted);
                let pareto = engine.pareto_over(&val_cols);
                let best_avg = pool
                    .best_by_average(&pareto)
                    .map(|candidate| candidate.average_score())
                    .unwrap_or(0.0);
                evolution_history.push((generation, best_avg));
            }
        }

        if skipped > 0 && pool.entries.len() == leaves.len() {
            tracing::warn!(
                skipped,
                perfect_score = self.perfect_score,
                "every generation skipped; if your metric is not 0-to-1, raise \
                 `perfect_score` (or set it to infinity)"
            );
        }

        // Noise-aware Occam winner — same rule as GEPA.
        let pareto = engine.pareto_over(&val_cols);
        let mut best_by_mean_candidate = pool
            .best_by_average(&pareto)
            .cloned()
            .context("no candidates available on Pareto frontier")?;
        let val_argmax_candidate = best_by_mean_candidate.clone();

        // Fresh-data champion selection: race the top two candidates (by
        // validation mean) on never-touched trainset columns — plus, under
        // TEMPER, the variant-tempering columns (paraphrase twins of seen
        // examples, equally unseen by the search). Selecting on unseen data
        // carries no winner's curse — a structural advantage of the engine's
        // usage-tracked score matrix that the reference implementation
        // cannot replicate — and a prompt that memorized seen phrasings
        // loses on the variant columns.
        let mut fresh_selection_cols = 0usize;
        let mut variant_selection_cols = 0usize;
        if !selection_cols.is_empty() {
            let mut ranked: Vec<(GEPACandidate, usize)> = pool
                .entries
                .iter()
                .filter(|(candidate, _)| {
                    candidate.module_name == best_by_mean_candidate.module_name
                })
                .map(|(candidate, row)| (candidate.clone(), *row))
                .collect();
            ranked.sort_by(|a, b| {
                b.0.average_score()
                    .partial_cmp(&a.0.average_score())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            if ranked.len() >= 2 && engine.budget_allows(2 * selection_cols.len()) {
                let rows = [ranked[0].1, ranked[1].1];
                if let BatchEvalOutcome::Complete(evals) =
                    engine.evaluate_many(target, &rows, Some(&selection_cols)).await?
                {
                    fresh_selection_cols = champion_cols.len();
                    variant_selection_cols = variant_cols.len();
                    let (first_mean, second_mean) = (evals[0].mean(), evals[1].mean());
                    tracing::info!(
                        first_mean,
                        second_mean,
                        fresh = champion_cols.len(),
                        variants = variant_cols.len(),
                        "fresh-data champion selection"
                    );
                    best_by_mean_candidate = if second_mean > first_mean {
                        ranked[1].0.clone()
                    } else {
                        ranked[0].0.clone()
                    };
                }
            }
        }

        // Seed-protected argmax: children compete by raw validation mean; only
        // the generation-0 seed keeps incumbency when the argmax fails to beat
        // it by more than `selection_z` standard errors (paired per column).
        // See the identical rule in GEPA for the measured rationale.
        let mut best_candidate = if self.selection_z > 0.0 {
            let best_row = pool
                .entries
                .iter()
                .find(|(candidate, _)| candidate.id == best_by_mean_candidate.id)
                .map(|(_, row)| *row)
                .expect("the chosen candidate is in the pool");
            let best_scores = row_scores(engine.matrix(), best_row, &val_cols);
            let seed_entry = pool.entries.iter().find(|(candidate, _)| {
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
        // Clause-ablation finale (docs/temper-v2-ablation-prereg.md): prune
        // the shipped champion clause-by-clause on the selection columns,
        // keeping removals the paired data can't distinguish from the full
        // text. Seeds are already minimal; only promoted children ablate.
        let mut ablation_removed = 0usize;
        let mut ablation_rollouts = 0usize;
        if self.ablate_champion && !selection_cols.is_empty() && best_candidate.generation > 0 {
            let spend_before = engine.spend().metric_calls;
            let ablate_z = if self.selection_z > 0.0 { self.selection_z } else { 1.0 };
            let mut current = best_candidate.clone();
            let mut current_row = pool
                .entries
                .iter()
                .find(|(candidate, _)| candidate.id == current.id)
                .map(|(_, row)| *row)
                .expect("the shipped candidate is in the pool");
            // Champion-phase cells are cache-served when already evaluated.
            if let EvalOutcome::Complete(_) = engine
                .evaluate(target, current_row, Some(&selection_cols))
                .await?
            {
                'ablation: loop {
                    let clauses = split_clauses(&current.instruction);
                    if clauses.len() < 2 {
                        break;
                    }
                    let baseline_scores =
                        row_scores(engine.matrix(), current_row, &selection_cols);
                    let baseline_evaluated: Vec<f64> =
                        baseline_scores.iter().filter_map(|score| *score).collect();
                    let baseline_mean = baseline_evaluated.iter().sum::<f64>()
                        / baseline_evaluated.len().max(1) as f64;
                    let mut accepted: Option<(GEPACandidate, usize, f64)> = None;
                    for skip in 0..clauses.len() {
                        let spent = engine.spend().metric_calls - spend_before;
                        if spent >= self.ablate_budget
                            || !engine.budget_allows(selection_cols.len())
                        {
                            break 'ablation;
                        }
                        let text = clauses
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| *i != skip)
                            .map(|(_, clause)| clause.as_str())
                            .collect::<Vec<_>>()
                            .join("\n");
                        let row = engine
                            .register(Candidate::with_instruction(&current.module_name, &text));
                        let eval = match engine
                            .evaluate(target, row, Some(&selection_cols))
                            .await?
                        {
                            EvalOutcome::Complete(eval) => eval,
                            EvalOutcome::BudgetExhausted { .. } => break 'ablation,
                        };
                        let scores = row_scores(engine.matrix(), row, &selection_cols);
                        let mean = eval.mean();
                        let survives = mean >= baseline_mean
                            || within_noise(&baseline_scores, &scores, ablate_z);
                        let improves = accepted
                            .as_ref()
                            .map(|(_, _, best)| mean > *best)
                            .unwrap_or(true);
                        if survives && improves {
                            let mut candidate = current.clone();
                            candidate.instruction = text;
                            accepted = Some((candidate, row, mean));
                        }
                    }
                    match accepted {
                        Some((candidate, row, _)) => {
                            ablation_removed += 1;
                            current = candidate;
                            current_row = row;
                        }
                        None => break,
                    }
                }
            }
            ablation_rollouts = engine.spend().metric_calls - spend_before;
            // Log even zero-removal runs — a silent finale is
            // indistinguishable from one that never engaged (cost a full
            // debugging cycle across v2.1–v2.3).
            tracing::info!(
                removed = ablation_removed,
                rollouts = ablation_rollouts,
                final_len = current.instruction.len(),
                "clause-ablation finale"
            );
            if ablation_removed > 0 {
                best_candidate = current;
            }
        }

        // Report the raw val-argmax whenever the shipped winner differs from
        // it (champion override or seed protection) so ablation arms exist.
        let best_by_mean =
            (best_candidate.id != val_argmax_candidate.id).then_some(val_argmax_candidate);

        let winner =
            Candidate::with_instruction(&best_candidate.module_name, &best_candidate.instruction);
        target.install(&winner)?;

        let operator_stats = op_stats
            .into_iter()
            .map(|(op, (attempts, wins))| (op.name().to_string(), attempts, wins))
            .collect();

        Ok(Report::Ratchet(RatchetResult {
            best_candidate,
            best_by_mean,
            all_candidates,
            trials,
            operator_stats,
            total_rollouts: engine.spend().metric_calls,
            total_lm_calls: engine.spend().lm_calls,
            evolution_history,
            gate_rejected,
            race_killed,
            skipped,
            leak_retries,
            leak_skipped,
            length_retries,
            length_skipped,
            fresh_selection_cols,
            variant_selection_cols,
            ablation_removed,
            ablation_rollouts,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(
        generation: usize,
        operator: RatchetOp,
        hypothesis: &str,
        outcome: TrialOutcome,
    ) -> TrialRecord {
        TrialRecord {
            generation,
            operator,
            parent_id: 0,
            hypothesis: hypothesis.to_string(),
            outcome,
        }
    }

    #[test]
    fn ledger_separates_validated_from_falsified_and_windows() {
        let mut trials = vec![record(
            0,
            RatchetOp::Rewrite,
            "demand short spans",
            TrialOutcome::Promoted {
                paired_delta: 0.04,
                candidate_id: 1,
            },
        )];
        for generation in 1..20 {
            trials.push(record(
                generation,
                RatchetOp::Patch,
                &format!("falsified idea {generation}"),
                TrialOutcome::GateKilled {
                    parent_gate: 0.6,
                    child_gate: 0.5,
                },
            ));
        }
        let text = Ratchet::render_ledger(&trials, 5);
        assert!(text.contains("demand short spans"));
        assert!(text.contains("+0.040"));
        // Window keeps only the last 5 falsified trials.
        assert!(!text.contains("falsified idea 14"));
        assert!(text.contains("falsified idea 15"));
        assert!(text.contains("falsified idea 19"));
        // A promoted-with-negative-delta trial is not a validated lesson.
        let neutral = vec![record(
            0,
            RatchetOp::Merge,
            "made it worse",
            TrialOutcome::Promoted {
                paired_delta: -0.02,
                candidate_id: 1,
            },
        )];
        assert!(Ratchet::render_ledger(&neutral, 5).contains("(none yet)"));
    }

    #[test]
    fn bandit_explores_untried_then_exploits_win_rate() {
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(0);
        let eligible = [RatchetOp::Rewrite, RatchetOp::Patch];
        let mut stats: HashMap<RatchetOp, (usize, usize)> = HashMap::new();
        stats.insert(RatchetOp::Rewrite, (5, 0));
        // Patch untried: must be explored first.
        assert_eq!(
            Ratchet::choose_op(&eligible, &stats, &mut rng),
            RatchetOp::Patch
        );
        // Both tried, Patch winning: exploitation prefers it.
        stats.insert(RatchetOp::Patch, (5, 4));
        assert_eq!(
            Ratchet::choose_op(&eligible, &stats, &mut rng),
            RatchetOp::Patch
        );
        // An operator that is not eligible is never chosen even if it wins.
        stats.insert(RatchetOp::Merge, (5, 5));
        assert_eq!(
            Ratchet::choose_op(&eligible, &stats, &mut rng),
            RatchetOp::Patch
        );
    }

    fn pool_with(rows: &[(&str, Option<usize>)]) -> Lineage {
        let mut pool = Lineage::new();
        for (idx, (name, parent)) in rows.iter().enumerate() {
            pool.entries.push((
                GEPACandidate {
                    id: idx,
                    instruction: format!("instruction {name}"),
                    module_name: "m".to_string(),
                    example_scores: Vec::new(),
                    parent_id: *parent,
                    generation: idx,
                },
                idx,
            ));
        }
        pool
    }

    #[test]
    fn lexicase_prefers_broad_strength_over_noise_specialist() {
        use rand::SeedableRng;
        // Candidate 0 is best on four columns; candidate 1 is a "sole
        // solver" specialist on one column; candidate 2 is dominated.
        // Coverage-proportional sampling would hand the specialist a large
        // share; lexicase gives the broad candidate the majority.
        let pool = pool_with(&[("broad", None), ("specialist", None), ("weak", None)]);
        let mut matrix = ScoreMatrix::new(5);
        for (col, (broad, spec, weak)) in [
            (0, (0.9, 0.2, 0.5)),
            (1, (0.8, 0.1, 0.4)),
            (2, (0.9, 0.3, 0.2)),
            (3, (0.7, 0.2, 0.3)),
            (4, (0.3, 0.9, 0.1)),
        ] {
            matrix.record(0, col, broad);
            matrix.record(1, col, spec);
            matrix.record(2, col, weak);
        }
        let val_cols: Vec<usize> = (0..5).collect();
        let mut rng = StdRng::seed_from_u64(7);
        let mut counts = [0usize; 3];
        for _ in 0..200 {
            let picked =
                Ratchet::sample_parent_lexicase(&pool, &matrix, &val_cols, 0.05, &mut rng)
                    .expect("non-empty pool");
            counts[picked.id] += 1;
        }
        assert!(counts[0] > counts[1], "broad should beat specialist: {counts:?}");
        assert!(counts[1] > 0, "specialist keeps some mass: {counts:?}");
        assert_eq!(counts[2], 0, "dominated candidate is never picked: {counts:?}");
    }

    #[test]
    fn lexicase_tie_breaks_toward_less_mutated_lineages() {
        use rand::SeedableRng;
        // Two candidates with identical scores; candidate 0 already has two
        // children, candidate 1 has none. Novelty pressure prefers 1.
        let pool = pool_with(&[("old", None), ("fresh", None), ("kid1", Some(0)), ("kid2", Some(0))]);
        let mut matrix = ScoreMatrix::new(2);
        for row in 0..4 {
            // Children score lower so only 0 and 1 survive filtering.
            let score = if row < 2 { 0.9 } else { 0.1 };
            matrix.record(row, 0, score);
            matrix.record(row, 1, score);
        }
        let mut rng = StdRng::seed_from_u64(3);
        let (mut old, mut fresh) = (0usize, 0usize);
        for _ in 0..300 {
            match Ratchet::sample_parent_lexicase(&pool, &matrix, &[0, 1], 0.05, &mut rng)
                .expect("non-empty pool")
                .id
            {
                0 => old += 1,
                1 => fresh += 1,
                other => panic!("unexpected pick {other}"),
            }
        }
        // Weights are 1/3 vs 1: the fresh candidate should win ~3x as often.
        assert!(fresh > 2 * old, "novelty pressure too weak: old={old} fresh={fresh}");
    }

    #[test]
    fn learnability_batches_prefer_the_frontier_band_and_epoch_coverage() {
        use rand::SeedableRng;
        let pool = pool_with(&[("a", None), ("b", None)]);
        let mut matrix = ScoreMatrix::new(4);
        // col 0: solved by everyone (p=1, weight floor). col 1: frontier
        // band (p=0.5, max weight). col 2: never evaluated (prior 0.25).
        // col 3: frontier band but already used once this epoch.
        matrix.record(0, 0, 1.0);
        matrix.record(1, 0, 1.0);
        matrix.record(0, 1, 1.0);
        matrix.record(1, 1, 0.0);
        matrix.record(0, 3, 1.0);
        matrix.record(1, 3, 0.0);
        let usage = HashMap::from([(3usize, 1usize)]);
        let train_pool = vec![0, 1, 2, 3];
        let mut rng = StdRng::seed_from_u64(11);
        let mut counts = [0usize; 4];
        for _ in 0..300 {
            let (batch_a, batch_b) = Ratchet::sample_batches_learnability(
                &pool, &train_pool, &usage, &matrix, 1, 0, &mut rng,
            );
            assert!(batch_b.is_empty());
            counts[batch_a[0]] += 1;
        }
        // Used column never beats the unused tier with minibatch 1.
        assert_eq!(counts[3], 0, "epoch coverage violated: {counts:?}");
        assert!(counts[1] > counts[0], "frontier band under-weighted: {counts:?}");
        assert!(counts[2] > counts[0], "unseen prior under-weighted: {counts:?}");
    }

    #[test]
    fn temper_preset_flips_the_knobs() {
        let optimizer = temper(vec![(0, 5)]);
        assert_eq!(optimizer.parent_selection, ParentSelection::Lexicase);
        assert_eq!(optimizer.batch_weighting, BatchWeighting::Learnability);
        assert_eq!(optimizer.length_cap, Some(1500));
        assert_eq!(optimizer.variant_pairs, vec![(0, 5)]);
        // Backbone defaults untouched.
        assert_eq!(optimizer.gate_z, 1.0);
        assert_eq!(optimizer.champion_fresh, 40);
        // v2 scheduled knobs stay off unless asked for.
        assert!(!optimizer.cap_after_promotion);
        assert!(!optimizer.lexicase_after_promotion);
        assert!(!optimizer.ablate_champion);
        assert_eq!(optimizer.ablate_budget, 90);
    }

    #[test]
    fn split_clauses_blocks_and_sentences() {
        // Short newline blocks stay whole; blank lines drop.
        let text = "Do the task.\n\nBe brief.";
        assert_eq!(split_clauses(text), vec!["Do the task.", "Be brief."]);

        // Long blocks split on sentence boundaries.
        let long = format!(
            "{} first sentence ends here. Second sentence follows! Third one? Tail without terminator",
            "padding ".repeat(30)
        );
        let clauses = split_clauses(&long);
        assert!(clauses.len() >= 4, "expected sentence split, got {clauses:?}");
        assert_eq!(clauses.last().unwrap(), "Tail without terminator");

        // A single clause never splits below two — the finale's stop case.
        assert_eq!(split_clauses("One instruction only.").len(), 1);

        // Multibyte content survives (no mid-char slicing panics).
        let unicode = format!("{} café naïve — done. Next sentence here.", "x".repeat(200));
        let clauses = split_clauses(&unicode);
        assert_eq!(clauses.last().unwrap(), "Next sentence here.");
    }

    #[tokio::test]
    async fn compile_requires_prompt_model() {
        use crate::{CallMetadata, PredictError, Predicted};
        use anyhow::Result;

        #[derive(Signature, Clone, Debug)]
        struct EchoSig {
            #[input]
            prompt: String,
            #[output]
            answer: String,
        }

        struct EchoModule {
            predictor: Predict<EchoSig>,
        }
        crate::predictors!(EchoModule { predictor });

        impl Module for EchoModule {
            type Input = EchoSigInput;
            type Output = EchoSigOutput;
            async fn forward(
                &self,
                input: EchoSigInput,
            ) -> Result<Predicted<EchoSigOutput>, PredictError> {
                Ok(Predicted::new(
                    EchoSigOutput {
                        answer: input.prompt,
                    },
                    CallMetadata::default(),
                ))
            }
        }

        struct AlwaysFeedback;
        impl TypedMetric<(EchoSigInput, EchoSigOutput), EchoModule> for AlwaysFeedback {
            async fn evaluate(
                &self,
                _example: &(EchoSigInput, EchoSigOutput),
                _prediction: &Predicted<EchoSigOutput>,
                _trace: Option<&crate::Trace>,
            ) -> Result<crate::Eval> {
                Ok(crate::Eval::with_feedback(0.5, "fine"))
            }
        }

        let optimizer = Ratchet::builder().num_iterations(1).build();
        let mut module = EchoModule {
            predictor: Predict::<EchoSig>::builder().instruction("seed").build(),
        };
        let rows = vec![(
            EchoSigInput {
                prompt: "x".to_string(),
            },
            EchoSigOutput {
                answer: "x".to_string(),
            },
        )];
        let err = optimizer
            .compile_module(&mut module, &rows, &AlwaysFeedback)
            .await
            .expect_err("Ratchet without a prompt_model must refuse to run");
        assert!(err.to_string().contains("prompt_model"));
    }
}
