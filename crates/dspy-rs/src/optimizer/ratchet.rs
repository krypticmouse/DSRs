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
//!
//! # P×N parallel proposals
//!
//! `parents_per_step` (P) and `proposals_per_parent` (N) batch the proposal
//! loop (gepa-ai blog, 2026-07-30): each step samples P parents, plans N
//! reflective mutations per parent on disjoint mini-batches, runs every
//! reflection call concurrently, screens each parent's children on its
//! held-out gate batch in one batched fan-out, and races all survivors
//! together block-by-block (each still paired against its own lineage for
//! the kill check). Beyond wall-clock, committing proposals in batches
//! reduces how often the search re-steers against the validation set — the
//! blog measures a smaller generalization gap for batched commits. The
//! defaults (1×1) preserve the measured sequential v2 behavior exactly,
//! including the per-seed rng stream.
//!
//! # COMPOUND mode (v4)
//!
//! Four knob-gated mechanisms (defaults off; design:
//! `docs/ratchet-v4-design.md`) answering the measured **win-size law**
//! (`docs/handoff-verification-economics.md`): v2/v3 RATCHET pays its
//! verification fee per accepted idea, so it wins tasks whose improvements
//! are few and large but structurally refuses piles of tiny true wins —
//! each is invisible in a fixed-size race, and the endgame's 1-SE seed
//! protection refused a real +.04 bundle on DocsQA.
//!
//! - [`RaceRule::Eprocess`]: the mid-race kill becomes an anytime-valid
//!   harm test ([`PairedEvidence`]) instead of repeated 1-SE peeks — the
//!   z-peek rule falsely kills ~25–30% of true tiny-winners across 4–5
//!   looks *and* records them as falsified in the ledger, steering the
//!   teacher away from true ideas. Disasters still die in a few columns.
//! - `certify_ship`: the endgame becomes a **sequential paired
//!   certification** of the champion vs the seed on reserved, never-touched
//!   trainset columns, stopped the moment the e-process crosses `1/α` in
//!   either direction. The fee self-scales ~1/Δ²: a big behavioral fix
//!   certifies in a handful of columns, a compounded pile of tiny wins
//!   certifies as a bundle in ~150 — and unproven still honestly ships the
//!   seed, now with the evidence state reported. The certification unit is
//!   the *bundle*, never the idea; that is the whole economics fix.
//! - [`LeakPolicy::AllowWithProvenance`]: verified fact-pinning. On
//!   corpus-fact tasks (leak semantics are task-dependent), a repair probe
//!   that counterfactually proves a fact fixes real failures *licenses*
//!   that fact's strings through the leak guard; every licensed string the
//!   winner carries ships as a dated [`PinRecord`] with provenance — the
//!   harvesting power GEPA gets from blind fact-stuffing, with receipts
//!   and a re-verification path instead of silent rot.
//! - `ledger_falsify_margin`: race/gate kills enter the falsified ledger
//!   only when decisive, so noise-kills stop poisoning future proposals.
//!
//! [`compound`] builds the preset (long cheap search + e-process racing +
//! sequential certification), mirroring [`temper`].

use std::collections::HashMap;

use anyhow::{Context, Result, anyhow};
use bon::Builder;
use rand::{Rng, rngs::StdRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};

use crate::core::ToInput;
use crate::evaluate::TypedMetric;
use crate::optimizer::engine::{
    BatchEvalOutcome, CandidateEval, Engine, EngineConfig, EvalOutcome,
    RolloutOutcome, SCORE_EPS, ScoreMatrix,
};
use crate::optimizer::evidence::PairedEvidence;
use crate::optimizer::gepa::{
    GEPA, GEPACandidate, Lineage, find_leaks, row_scores, within_noise,
};
use crate::optimizer::target::LeafInfo;
use crate::optimizer::{OptimizeTarget, Optimizer, OptimizerCommon, Report};
use crate::trace::Trace;
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

/// Diagnose which module of an LLM pipeline caused a failed rollout, and
/// write the output that module should have produced.
///
/// You are debugging ONE failed rollout of a multi-module LLM pipeline.
/// Study the pipeline schema, the per-module execution trace (each module's
/// actual inputs and outputs, in order), and the failure feedback. Name the
/// single module whose output, had it been better, would most likely have
/// fixed this rollout — when several could have, blame the EARLIEST module
/// where the needed information was available but got lost — and write the
/// corrected output that module should have produced. Prefer a repair the
/// module could have produced from its own inputs as shown in the trace;
/// when the inputs genuinely lack what was needed, use the feedback to
/// construct the output that would have fixed the rollout anyway, and say
/// so in your hypothesis (the repair is verified by re-execution, never
/// shipped). Anchor the repair in the failure feedback: it states exactly
/// what the rollout got wrong, and a repair that does not directly target
/// that stated gap fixes nothing. Also state, in one line of at most 25
/// words, the general weakness this failure reveals (a rule about HOW the
/// module fails, not about this example).
#[derive(Signature, Clone, Debug)]
struct DiagnoseRepair {
    /// Every module's name and input/output contract, in pipeline order.
    #[input]
    pipeline_schema: String,

    /// The failed rollout: each module's actual inputs and outputs.
    #[input]
    execution_trace: String,

    /// The metric's feedback on why the rollout failed.
    #[input]
    failure_feedback: String,

    /// The name of the single module most responsible (exactly as it appears
    /// in the pipeline schema).
    #[output]
    blamed_module: String,

    /// The corrected output that module should have produced from its own
    /// inputs (the output value only, no commentary).
    #[output]
    repaired_output: String,

    /// One line (max 25 words): the general weakness this failure reveals.
    #[output]
    hypothesis: String,
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

/// How the mutated slot (leaf) is chosen each step in a multi-module program.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SlotCredit {
    /// RATCHET v2 default: `leaves[generation % len]` — the reference
    /// package's component schedule. No credit assignment.
    RoundRobin,
    /// Counterfactual stage-repair (v3): before proposing, the teacher
    /// debugs failed rollouts — it blames one module and writes the output
    /// that module SHOULD have produced; the engine replays the pipeline
    /// with the repaired output substituted (trace replay `UntilDivergence`:
    /// upstream serves free, downstream runs live) and measures whether the
    /// rollout actually recovers. The slot with the largest verified
    /// recovery is mutated, and the verified (actual → repaired) pairs are
    /// fed to the Rewrite operator as evidence. Falls back to round-robin
    /// when no probe recovers. Motivated by the HoVer autopsy: round-robin
    /// piled every promotion on one slot while the compounding failures
    /// lived in the summarizers (docs/handoff-hover-multimodule.md).
    Repair,
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

/// How the mid-race kill decision is made.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RaceRule {
    /// RATCHET v2 default: after each block, kill when the paired deficit
    /// to the lineage parent exceeds `race_z` standard errors. Cheap and
    /// fast, but the repeated looks are anti-conservative: across 4–5
    /// blocks at z = 1 a true +.005 child is falsely killed ~25–30% of the
    /// time — and logged as falsified.
    ZBlock,
    /// COMPOUND (v4): kill only when the anytime-valid harm e-process
    /// ([`PairedEvidence`]) certifies the child worse at `race_alpha`.
    /// Valid under any number of looks; disasters still die in a few
    /// columns, while tiny-winners survive to compound in the pool (the
    /// ship decision is guarded by certification instead).
    Eprocess,
}

/// How a gate survivor earns its place in the pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AcceptMode {
    /// RATCHET default: every gate survivor races the full validation
    /// slice, paired against its lineage parent. The right economics when
    /// improvements are few and large.
    Race,
    /// COMPOUND accretion mode (port spec v2 §1, "cheap accept"): gate
    /// survivors skip the race — they get one small shared "scout" slice
    /// of the validation set for pool ranking and join the pool directly.
    /// ALL verification moves to the sequential ship certification. The
    /// right economics for accretive-content regimes, where dropping a
    /// true +.01 child on a noisy race draw loses its content forever
    /// (the live mem3 arm faded 14 of 24 true fact-adders this way while
    /// pip-gepa kept every accepted child and compounded to +.07).
    /// `promote_only_gains` is ignored in this mode — accretion keeps
    /// noise-negative steps because their content persists; the
    /// certification is the honesty. Use with `certify_ship`.
    Gate,
}

/// What the leak guard does with example strings a proposal quotes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LeakPolicy {
    /// Default: any verbatim example string triggers a retry, then abandons
    /// the trial. Right for benchmarks, where train answers in the prompt
    /// are contamination.
    Forbid,
    /// COMPOUND (v4), for corpus-fact tasks where the "leak" *is* the win
    /// (DocsQA-class): strings that a counterfactually verified repair
    /// introduced are licensed through the guard; everything unlicensed is
    /// still forbidden. Licensed strings the winner carries are reported as
    /// dated [`PinRecord`]s with provenance. Requires
    /// [`SlotCredit::Repair`] — licenses only come from verified repairs.
    AllowWithProvenance,
}

/// A corpus string the shipped winner quotes verbatim, licensed by a
/// counterfactually verified repair — the provenance GEPA's blind
/// fact-stuffing cannot produce. Re-verify on a schedule by ablating the
/// pin's clause on fresh data (or continuing its evidence stream); a pin
/// whose removal no longer hurts has rotted and should be dropped.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PinRecord {
    /// The licensed string (lowercased, as the leak guard matches it).
    pub text: String,
    /// Trainset column of the failed rollout the licensing repair fixed.
    pub source_example: usize,
    /// Generation of the licensing repair probe.
    pub generation: usize,
    /// The repair's verified recovery delta on that rollout.
    pub verified_delta: f64,
    /// Unix seconds when the run recorded the pin (staleness clock).
    pub pinned_at_unix: u64,
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
    /// Finished the full race with a negative paired delta and was refused
    /// pool entry (`promote_only_gains`): fully measured, not compounded.
    Faded { paired_delta: f64 },
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
    /// Race survivors refused pool entry for a negative paired delta
    /// (`promote_only_gains` only).
    #[serde(default)]
    pub faded: usize,
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
    /// Counterfactual stage-repair probes run (`SlotCredit::Repair` only).
    #[serde(default)]
    pub repair_probes_run: usize,
    /// Probes whose repaired output measurably recovered the rollout.
    #[serde(default)]
    pub repair_recovered: usize,
    /// Per-slot repair evidence: `(slot, probes_blamed, total_recovery_delta)`.
    #[serde(default)]
    pub repair_slot_stats: Vec<(String, usize, f64)>,
    /// Verified repair lessons `(generation, slot, hypothesis, delta)` —
    /// each counterfactually proven, persisted into every later step's
    /// ledger digest (v3.1).
    #[serde(default)]
    pub repair_lessons: Vec<(usize, String, String, f64)>,
    /// Sequential ship certification verdict (`certify_ship` only):
    /// `Some(true)` = champion certified better than the seed and shipped;
    /// `Some(false)` = certified worse (seed shipped); `None` with
    /// `certify_cols_used > 0` = evidence ran out unproven (seed shipped,
    /// per Occam — the challenger is in `best_by_mean`).
    #[serde(default)]
    pub certified: Option<bool>,
    /// Paired columns the certification consumed before stopping.
    #[serde(default)]
    pub certify_cols_used: usize,
    /// Mean paired champion−seed delta over the certification columns.
    #[serde(default)]
    pub certify_mean_delta: f64,
    /// Final gain e-value (≥ `1/certify_alpha` means certified).
    #[serde(default)]
    pub certify_e_value: f64,
    /// Verified fact pins the shipped winner carries
    /// ([`LeakPolicy::AllowWithProvenance`] only).
    #[serde(default)]
    pub pins: Vec<PinRecord>,
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
    /// Only read under [`RaceRule::ZBlock`].
    #[builder(default = 1.0)]
    pub race_z: f64,

    /// How the mid-race kill decision is made. [`RaceRule::Eprocess`]
    /// (COMPOUND) replaces the repeated z-peeks with an anytime-valid harm
    /// test; [`RaceRule::ZBlock`] (default) preserves measured v2 behavior.
    #[builder(default = RaceRule::ZBlock)]
    pub race_rule: RaceRule,

    /// How gate survivors join the pool. [`AcceptMode::Gate`] is the
    /// cheap-accept accretion mode; [`AcceptMode::Race`] (default)
    /// preserves measured v2/v4.0 behavior.
    #[builder(default = AcceptMode::Race)]
    pub accept_mode: AcceptMode,

    /// Accretive operator portfolio: only Rewrite (and Merge, the fact-set
    /// union) are eligible — Patch edits one rule at a time and Simplify
    /// actively strips content, and in an accretive-content regime the
    /// mixed portfolio measurably dilutes the harvest (live mem-series:
    /// Rewrite won 1–7 promotions per run while ~2/3 of proposals went to
    /// non-accreting operators; pip-gepa is 100% accreting rewrites).
    /// `false` (default) keeps the measured four-operator bandit.
    #[builder(default = false)]
    pub accretive_ops: bool,

    /// Validation columns in the shared scout slice under
    /// [`AcceptMode::Gate`] — the only val evaluation accepted children
    /// get, used for pool ranking and champion screening (every candidate
    /// shares the same slice, so ranks stay comparable).
    #[builder(default = 12)]
    pub scout_size: usize,

    /// Kill level for [`RaceRule::Eprocess`]: a child dies when its harm
    /// e-value reaches `1/race_alpha`. A racing kill is a budget heuristic,
    /// not a published claim, so a loose level is appropriate.
    #[builder(default = 0.1)]
    pub race_alpha: f64,

    /// Gate tolerance: the child passes batch B unless the parent beats it by
    /// more than `gate_z` standard errors (paired). Lateral moves — children
    /// that merely tie the parent on B — pass and let the race decide; v1's
    /// strict `>` gate was measured to prune the stepping stones deep
    /// lineages are made of. `0.0` = strict mean comparison.
    #[builder(default = 1.0)]
    pub gate_z: f64,

    /// Rewrite proposals sampled per generation (temperature diversity);
    /// the one with the better gate-batch mean races. Other operators
    /// propose once. Superseded when `proposals_per_parent` > 1.
    #[builder(default = 2)]
    pub rewrite_fanout: usize,

    /// Parents sampled per step — the "P" of P×N parallel proposals
    /// (gepa-ai blog, 2026-07-30). Each parent draws its own reflection and
    /// gate batches; all of a step's proposals are planned against the same
    /// pool/ledger/bandit snapshot and committed together, which reduces how
    /// often the search re-steers against the validation set. `1` (the
    /// default) preserves RATCHET's measured sequential v2 behavior exactly.
    #[builder(default = 1)]
    pub parents_per_step: usize,

    /// Reflective mutations per parent per step — the "N" of P×N. Each
    /// mutation reflects on its own disjoint trainset mini-batch (feedback
    /// diversity, not just a fresh temperature draw); every gate survivor
    /// races, the blog's AllImprovements selection. Values above 1 supersede
    /// `rewrite_fanout`. `1` (the default) preserves RATCHET's measured
    /// sequential v2 behavior exactly.
    #[builder(default = 1)]
    pub proposals_per_parent: usize,

    /// How the mutated slot is chosen each step. [`SlotCredit::Repair`] runs
    /// counterfactual stage-repair probes to assign blame before mutating;
    /// [`SlotCredit::RoundRobin`] (the default) preserves v2 behavior.
    #[builder(default = SlotCredit::RoundRobin)]
    pub slot_credit: SlotCredit,

    /// Failed batch-A examples probed per parent per step under
    /// [`SlotCredit::Repair`]. Each probe costs up to `1 + repair_attempts`
    /// rollouts (one fresh traced base rollout + one counterfactual replay
    /// per attempt) plus one teacher call per attempt.
    #[builder(default = 2)]
    pub repair_probes: usize,

    /// Repair hypotheses tried per probe: a repair falsified by re-execution
    /// is shown back to the teacher (with everything the re-executed run
    /// still got wrong) for a revised attempt, until one recovers or the
    /// attempts run out. Counterfactuals whose blamed stage has no LM
    /// downstream (e.g. the last query writer of a retrieval pipeline) cost
    /// no student calls, so extra attempts are cheap where iteration matters
    /// most.
    #[builder(default = 2)]
    pub repair_attempts: usize,

    /// Fresh-data champion selection: reserve budget to race the top two
    /// candidates (by validation mean) on up to this many *never-touched*
    /// trainset columns at the end — champion selection on unseen data has
    /// no winner's curse. `0` disables. Under `certify_ship` this stays the
    /// challenger-picking screen; the ship decision moves to certification.
    #[builder(default = 40)]
    pub champion_fresh: usize,

    /// Candidates in the fresh-data champion race. The default 2
    /// reproduces measured behavior; larger pools (COMPOUND Gate mode)
    /// need a wider screen — the mem4 live arm's 57-candidate pool put its
    /// true best at scout-rank 1 and the 2-wide screen still lost it to a
    /// noise draw.
    #[builder(default = 2)]
    pub champion_k: usize,

    /// Winner-selection noise tolerance (the Occam rule; `0.0` = raw
    /// argmax). Under `certify_ship` the seed-protection comparison is
    /// replaced by sequential certification, which serves the same honesty
    /// with adaptive power.
    #[builder(default = 1.0)]
    pub selection_z: f64,

    /// COMPOUND (v4) sequential ship certification: reserve never-touched
    /// trainset columns up front and, at the end, race the champion
    /// *paired against the seed* on them block-by-block under an
    /// anytime-valid e-process — ship the champion the moment its gain is
    /// certified at `certify_alpha`, ship the seed the moment harm is, and
    /// ship the seed (reporting the open evidence) when columns or budget
    /// run out unproven. Replaces the endgame's 1-SE seed protection, whose
    /// fixed 20–40-column power refused a real +.04 bundle on DocsQA.
    #[builder(default = false)]
    pub certify_ship: bool,

    /// Certification columns reserved out of the trainset (never seen by
    /// any batch, race, or champion screen). The e-process stops early on
    /// big effects, so reserve for the smallest bundle worth shipping —
    /// ~150 columns certifies a +.04-class bundle under typical judge
    /// noise. Only read when `certify_ship` is set.
    #[builder(default = 150)]
    pub certify_reserve: usize,

    /// Explicit certification columns: trainset indices the HARNESS
    /// supplies as holdout-class ship-gate data (port spec v2 §2 — append
    /// the rows to the trainset and name their indices here). They are
    /// excluded from every batch, race, and champion draw, and the
    /// certification runs on them in addition to the `certify_reserve`
    /// carve. Small trainsets cannot carve enough power for +.03-.04-class
    /// bundles (the live compound-mem refusal); dedicated columns can.
    /// Empty (default) keeps the carve-only behavior.
    #[builder(default)]
    pub certify_cols: Vec<usize>,

    /// Rollout budget reserved for certification once a child has been
    /// promoted (an unconditional reserve would starve the search — same
    /// scheduling as `ablate_budget`). Certification consumes 2 rollouts
    /// per column (champion + seed; seed cells cache-hit when reserve
    /// columns repeat across runs). Only read when `certify_ship` is set.
    #[builder(default = 240)]
    pub certify_budget: usize,

    /// Columns evaluated per certification block (the e-process is checked
    /// between blocks). Only read when `certify_ship` is set.
    #[builder(default = 8)]
    pub certify_block: usize,

    /// Certification level: the champion ships only when its gain e-value
    /// reaches `1/certify_alpha`. Only read when `certify_ship` is set.
    #[builder(default = 0.05)]
    pub certify_alpha: f64,

    /// What the leak guard does with quoted example strings.
    /// [`LeakPolicy::AllowWithProvenance`] enables verified fact-pinning on
    /// corpus-fact tasks (needs [`SlotCredit::Repair`] to mint licenses);
    /// [`LeakPolicy::Forbid`] (default) preserves measured v2 behavior.
    /// `allow_memorization` still bypasses the guard entirely.
    #[builder(default = LeakPolicy::Forbid)]
    pub leak_policy: LeakPolicy,

    /// A gate/race kill enters the falsified-ledger digest only when the
    /// measured deficit is at least this margin — noise-kills of
    /// near-neutral children otherwise poison the teacher against ideas
    /// that were never really falsified. `0.0` (default) keeps every kill,
    /// preserving measured v2 ledgers.
    #[builder(default = 0.0)]
    pub ledger_falsify_margin: f64,

    /// COMPOUND (v4.1) pool hygiene: a race survivor joins the pool only
    /// when its full-race paired delta vs its lineage parent is
    /// non-negative; negative survivors are recorded as
    /// [`TrialOutcome::Faded`] instead of compounding. The e-process race
    /// deliberately kills only *certified* harm, and the first live cell
    /// measured the cost of pooling the rest: 23 of 32 promotions carried
    /// negative deltas, the chain compounded noise, and certification had
    /// to refuse the pile. `false` (default) preserves measured v2/v3
    /// pooling exactly.
    #[builder(default = false)]
    pub promote_only_gains: bool,

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

    /// COMPOUND (v4): probability that a step's parent is the current
    /// val-mean argmax (the head of the compounding chain) instead of a
    /// coverage/lexicase draw. In an additive tiny-win regime value
    /// compounds along one lineage, and pure coverage sampling forks the
    /// pile across the frontier — the sim measured challengers carrying
    /// ~1/4 of the pool's harvested facts under pure coverage. `0.0`
    /// (default) preserves measured v2 selection exactly.
    #[builder(default = 0.0)]
    pub parent_greed: f64,

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

/// Builds a COMPOUND-configured [`Ratchet`] (v4, design:
/// `docs/ratchet-v4-design.md`) — the many-tiny-wins preset for tasks on
/// the far side of the win-size law: long cheap search cycles (the
/// measured long-cycle result: iters 150 / minibatch 2–3 / P×N 2×2 shipped
/// the first non-seed winner on the DocsQA cell), anytime-valid e-process
/// racing so tiny true wins survive to compound, decisive-only falsified
/// ledgers, and the sequential paired ship certification against the seed.
/// Set `prompt_model` and `max_rollouts` on the returned value.
///
/// For corpus-fact tasks, additionally enable verified fact-pinning:
///
/// ```ignore
/// let mut optimizer = compound();
/// optimizer.prompt_model = Some(teacher_lm);
/// optimizer.max_rollouts = Some(2000);
/// optimizer.slot_credit = SlotCredit::Repair;               // mints licenses
/// optimizer.leak_policy = LeakPolicy::AllowWithProvenance;  // spends them
/// ```
pub fn compound() -> Ratchet {
    Ratchet::builder()
        .num_iterations(150)
        .minibatch_size(3)
        .parents_per_step(2)
        .proposals_per_parent(2)
        .race_rule(RaceRule::Eprocess)
        .parent_greed(0.5)
        .promote_only_gains(true)
        .certify_ship(true)
        // A +.04-class bundle under judge noise needs the full reserve:
        // ~150 paired columns ≈ 300–400 rollouts. Big effects stop early
        // and refund the difference.
        .certify_budget(400)
        // e ≥ 10, anytime-valid — still a far stronger ship guarantee than
        // an unreplicated argmax; the strict 0.05 default suits big-win
        // tasks where evidence is cheap.
        .certify_alpha(0.1)
        .ledger_falsify_margin(0.02)
        .build()
}

/// A JSON value as prompt text: bare strings unquoted, everything else
/// serialized.
fn value_text(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Renders a rollout's execution trace per module for the repair teacher:
/// each span's component name with its actual inputs and outputs, in
/// execution order. Field values are truncated (inputs can carry retrieved
/// passages) but generously — the teacher must see enough of a module's
/// inputs to write a repair producible from them.
fn render_pipeline_trace(trace: &Trace) -> String {
    use std::fmt::Write as _;
    let mut text = String::new();
    for span in &trace.spans {
        let name = trace.component_name(span.component);
        let inputs = span
            .input
            .as_ref()
            .map(|map| {
                map.iter()
                    .map(|(key, val)| format!("  {key}: {}", truncate(&value_text(val), 2000)))
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .unwrap_or_else(|| "  -".to_string());
        let output = match (&span.output, &span.error) {
            (Some(map), _) => map
                .iter()
                .map(|(key, val)| format!("  {key}: {}", truncate(&value_text(val), 2000)))
                .collect::<Vec<_>>()
                .join("\n"),
            (None, Some(error)) => format!("  <{}: {}>", error.kind.as_str(), truncate(&error.message, 300)),
            (None, None) => "  -".to_string(),
        };
        let _ = writeln!(text, "## module `{name}` (call {})\ninputs:\n{inputs}\noutput:\n{output}\n", span.seq);
    }
    text.trim_end().to_string()
}

/// Every module's name and contract, in declaration order, for the repair
/// teacher's pipeline_schema input.
fn pipeline_schema(leaves: &[LeafInfo]) -> String {
    leaves
        .iter()
        .map(|leaf| format!("## module `{}`\n{}", leaf.name, leaf.schema_for_reflection()))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Clones `trace` with the FIRST span of `component` given `text` as its
/// `field` output — the counterfactual edit. Returns the edited trace plus
/// the span's previous field value. `None` when the component has no
/// complete span (a stage that fell back or errored can't serve a repair).
fn repair_span_output(
    trace: &Trace,
    component: &str,
    field: &str,
    text: &str,
) -> Option<(Trace, String)> {
    let id = trace.component_id(component)?;
    let mut edited = trace.clone();
    let span = edited
        .spans
        .iter_mut()
        .filter(|span| span.component == id && span.complete && span.output.is_some())
        .min_by_key(|span| span.seq)?;
    let map = span.output.as_mut()?;
    let previous = map.get(field).map(value_text).unwrap_or_default();
    map.insert(
        field.to_string(),
        serde_json::Value::String(text.to_string()),
    );
    Some((edited, previous))
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
    diagnose: Predict<DiagnoseRepair>,
}

impl Reflectors {
    fn new(lm: &crate::LM) -> Self {
        Self {
            rewrite: Predict::<ProposeRewrite>::builder().lm(lm.clone()).build(),
            patch: Predict::<ProposePatch>::builder().lm(lm.clone()).build(),
            simplify: Predict::<ProposeSimplify>::builder().lm(lm.clone()).build(),
            merge: Predict::<ProposeMerge>::builder().lm(lm.clone()).build(),
            diagnose: Predict::<DiagnoseRepair>::builder().lm(lm.clone()).build(),
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

/// Prepared reflection inputs for one planned proposal slot of a step —
/// the operator-specific strings beyond the step-level task description and
/// ledger (P×N parallel proposals; see `parents_per_step`).
enum PlannedInput {
    Rewrite {
        feedback: String,
    },
    Patch {
        better: String,
        worse: String,
    },
    Simplify {
        validated: String,
    },
    Merge {
        instruction_a: String,
        a_strengths: String,
        instruction_b: String,
        b_strengths: String,
    },
}

/// One planned proposal of a step: its parent group, chosen operator,
/// prepared inputs, leak-guard strings, and the lineage it is gated and
/// raced against (Merge overrides the sampled parent with the pair's
/// stronger member).
struct StepSlot {
    group: usize,
    op: RatchetOp,
    input: PlannedInput,
    forbidden: Vec<String>,
    lineage_parent: GEPACandidate,
    lineage_row: usize,
}

/// One sampled parent of a step: the slot it mutates, that slot's current
/// text and reflection schema, and the group's held-out gate batch.
struct StepGroup {
    leaf_idx: usize,
    parent_text: String,
    task_description: String,
    batch_b: Vec<usize>,
}

/// A proposal that survived registration and awaits its group's gate.
struct PendingChild {
    slot: usize,
    child: GEPACandidate,
    child_row: usize,
    hypothesis: String,
}

/// A gate survivor racing the validation columns.
struct RaceEntry {
    child: GEPACandidate,
    child_row: usize,
    hypothesis: String,
    op: RatchetOp,
    lineage_parent_id: usize,
    lineage_row: usize,
    /// Anytime-valid paired evidence vs the lineage parent, fed one block
    /// at a time (only consulted under [`RaceRule::Eprocess`]).
    evidence: PairedEvidence,
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

    /// Children a step can race per parent: the N parallel proposals, or the
    /// rewrite fan-out pair when N = 1.
    fn step_fanout(&self) -> usize {
        if self.proposals_per_parent > 1 {
            self.proposals_per_parent
        } else {
            self.rewrite_fanout.max(1)
        }
    }

    /// Worst-case rollout spend of one step, used by the budget precheck:
    /// each parent pays its reflection pool (N mini-batches) plus gate
    /// evaluations for itself and every child, and one full validation race
    /// tops it off. Reduces to the measured v2 formula
    /// (`minibatch + 3·gate + val`) at P = N = 1.
    fn step_worst_case(&self, gate_size: usize, val_len: usize) -> usize {
        let p = self.parents_per_step.max(1);
        let n = self.proposals_per_parent.max(1);
        let repair = if matches!(self.slot_credit, SlotCredit::Repair) {
            // Worst case per probe: one fresh base rollout plus one
            // counterfactual per attempt (the falsified-repair retries).
            p * self.repair_probes * (1 + self.repair_attempts.max(1))
        } else {
            0
        };
        p * (n * self.minibatch_size + (self.step_fanout() + 1) * gate_size) + val_len + repair
    }

    /// Fan-out margin for the ablation reserve: a step can run up to
    /// P × fan full races beyond the single race `step_worst_case` counts,
    /// and each racing child can bill child AND parent rows over the full
    /// val slice (v2.1 and v2.2 both lost the finale to this overshoot).
    fn ablate_margin(&self, val_len: usize) -> usize {
        (2 * self.parents_per_step.max(1) * self.step_fanout()).saturating_sub(1) * val_len
    }

    /// Renders the verified-repair section appended to the ledger digest
    /// (v3.1): every counterfactually proven repair, as a rule to encode.
    /// Entries are `(generation, slot, hypothesis, delta)`.
    fn render_repair_lessons(lessons: &[(usize, String, String, f64)]) -> String {
        use std::fmt::Write as _;
        if lessons.is_empty() {
            return String::new();
        }
        let mut text = String::from(
            "\n\nVerified repairs (counterfactually PROVEN: a module output with \
             this property fixed a real failed rollout — encode the property as \
             an instruction rule for that module):\n",
        );
        for (generation, slot, hypothesis, delta) in lessons {
            let _ = writeln!(
                text,
                "- [gen {generation} | {slot} | recovered {delta:+.2}] {hypothesis}"
            );
        }
        text.trim_end().to_string()
    }

    /// Renders the ledger digest fed to every operator: all validated lessons
    /// plus the last `ledger_window` falsified trials. A kill enters the
    /// falsified section only when its measured deficit reaches
    /// `falsify_margin` — a near-neutral kill teaches nothing and, worse,
    /// steers the teacher away from ideas that were never really falsified
    /// (`0.0` keeps every kill, the measured v2 digest).
    fn render_ledger(trials: &[TrialRecord], window: usize, falsify_margin: f64) -> String {
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
                let decisive = match t.outcome {
                    TrialOutcome::GateKilled {
                        parent_gate,
                        child_gate,
                    } => parent_gate - child_gate >= falsify_margin,
                    TrialOutcome::RaceKilled { paired_delta, .. }
                    | TrialOutcome::Faded { paired_delta } => {
                        -paired_delta >= falsify_margin
                    }
                    _ => false,
                };
                decisive && !t.hypothesis.is_empty()
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
                TrialOutcome::Faded { paired_delta } => {
                    format!("finished the full race below its parent, val {paired_delta:+.3}")
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
        // Harness-supplied certification columns (port spec v2 §2): mapped
        // like variants, and pulled out of the batch pool the same way —
        // the search never touches them.
        let mut explicit_certify: Vec<usize> = Vec::new();
        if self.certify_ship {
            for &idx in &self.certify_cols {
                if idx >= full_train_pool.len() {
                    tracing::warn!(idx, "certify_cols index out of range; ignoring");
                    continue;
                }
                explicit_certify.push(full_train_pool[idx]);
            }
            explicit_certify.sort_unstable();
            explicit_certify.dedup();
        }
        let batch_pool: Vec<usize> = full_train_pool
            .iter()
            .copied()
            .filter(|col| !variant_cols.contains(col) && !explicit_certify.contains(col))
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
        // COMPOUND: carve the certification slice out of what remains, the
        // same never-touched-by-construction way. The extra shuffle only
        // happens when `certify_ship` is on, so default rng streams are
        // untouched.
        let (train_pool, mut certify_cols) = if self.certify_ship && self.certify_reserve > 0 {
            let take = train_pool
                .len()
                .saturating_sub(20)
                .min(self.certify_reserve);
            if take >= self.certify_block.max(1) {
                let mut cols = train_pool.clone();
                cols.shuffle(&mut rng);
                (cols[take..].to_vec(), cols[..take].to_vec())
            } else {
                if explicit_certify.is_empty() {
                    tracing::warn!(
                        trainset = train_pool.len(),
                        "trainset too small to reserve certification columns; \
                         the endgame falls back to seed protection"
                    );
                }
                (train_pool, Vec::new())
            }
        } else {
            (train_pool, Vec::new())
        };
        // Harness-supplied columns join the carve; one shuffle keeps the
        // certification stream's column order independent of its source.
        if !explicit_certify.is_empty() {
            certify_cols.extend(&explicit_certify);
            certify_cols.sort_unstable();
            certify_cols.dedup();
            certify_cols.shuffle(&mut rng);
            tracing::info!(
                explicit = explicit_certify.len(),
                total = certify_cols.len(),
                "certification columns include a harness-supplied slice"
            );
        }
        // Deltas are scaled by the score ceiling for the e-process wallets.
        let delta_scale = if self.perfect_score.is_finite() && self.perfect_score > 0.0 {
            self.perfect_score
        } else {
            1.0
        };
        // AcceptMode::Gate: candidates are ranked on one shared "scout"
        // slice of the validation set instead of racing all of it — the
        // only val evaluation accepted children get. Under Race the slice
        // IS the validation set, preserving measured behavior exactly.
        let scout_cols: Vec<usize> = if matches!(self.accept_mode, AcceptMode::Gate) {
            let mut cols = val_cols.clone();
            cols.shuffle(&mut rng);
            cols.truncate(self.scout_size.max(1));
            tracing::info!(scout = cols.len(), "gate-accept mode: scout slice reserved");
            cols
        } else {
            val_cols.clone()
        };
        // The endgame champion race runs on fresh ∪ variant columns.
        let selection_cols: Vec<usize> = champion_cols
            .iter()
            .chain(&variant_cols)
            .copied()
            .collect();
        let mut pool = Lineage::new();

        // Seed the pool with ONE joint candidate carrying every leaf's
        // incumbent instruction. Multi-leaf programs are searched jointly —
        // children inherit the full assignment and rewrite one slot per
        // generation (round-robin), matching the reference package's
        // multi-component candidates. Single-leaf behavior is unchanged.
        {
            let seed_slots: std::collections::BTreeMap<String, String> = leaves
                .iter()
                .map(|leaf| (leaf.name.clone(), leaf.instruction.clone()))
                .collect();
            let seed = GEPACandidate {
                id: 0,
                instruction: leaves[0].instruction.clone(),
                module_name: leaves[0].name.clone(),
                example_scores: Vec::new(),
                parent_id: None,
                generation: 0,
                slots: seed_slots,
            };
            let row = engine.register(seed.engine_candidate());
            let eval = match engine.evaluate(target, row, Some(&scout_cols)).await? {
                EvalOutcome::Complete(eval) => eval,
                EvalOutcome::BudgetExhausted { .. } => return Err(anyhow!(
                    "budget exhausted during the seed evaluation"
                )),
            };
            GEPA::require_feedback(&eval, &seed.module_name, 0)?;
            pool.add(seed, row, &eval);
        }

        let gate_size = self.gate_size.unwrap_or(self.minibatch_size);
        let mut usage: HashMap<usize, usize> = HashMap::new();
        let mut trials: Vec<TrialRecord> = Vec::new();
        let mut op_stats: HashMap<RatchetOp, (usize, usize)> = HashMap::new();
        let mut all_candidates = Vec::new();
        let mut evolution_history = Vec::new();
        let (mut gate_rejected, mut race_killed, mut skipped) = (0usize, 0usize, 0usize);
        let mut faded = 0usize;
        let (mut leak_retries, mut leak_skipped) = (0usize, 0usize);
        let (mut length_retries, mut length_skipped) = (0usize, 0usize);
        let (mut repair_probes_run, mut repair_recovered) = (0usize, 0usize);
        let mut repair_slot_totals: HashMap<String, (usize, f64)> = HashMap::new();
        // v3.1: verified repair lessons persist for the rest of the run —
        // rendered into every subsequent step's ledger digest so proven
        // properties compound instead of conditioning one Rewrite and
        // vanishing (the HoVer conversion gap).
        let mut repair_lessons: Vec<(usize, String, String, f64)> = Vec::new();
        // COMPOUND fact-pinning: short strings introduced by verified
        // repairs, licensed through the leak guard under
        // `LeakPolicy::AllowWithProvenance`. Key is the lowercased string
        // (the guard's matching form); value is provenance.
        let mut pin_licenses: HashMap<String, (usize, usize, f64)> = HashMap::new();

        'generations: for generation in 0..self.num_iterations {
            // Keep enough budget in reserve for the fresh-data champion
            // phase — including this step's own worst-case spend (P parents
            // × (N reflection mini-batches + gate evaluations) + a full
            // race; see `step_worst_case`).
            // Promotion state gates the scheduled knobs (cap/lexicase after
            // promotion), the ablation reserve, and Simplify/Merge
            // eligibility further down.
            let has_promotion = pool.entries.len() > leaves.len();
            let p_count = self.parents_per_step.max(1);
            let n_count = self.proposals_per_parent.max(1);

            // Under Gate accept, a step's post-gate cost is the scout fan-out
            // instead of one full-val race.
            let step_val_need = match self.accept_mode {
                AcceptMode::Race => val_cols.len(),
                AcceptMode::Gate => {
                    (self.step_fanout() * p_count + 1) * scout_cols.len()
                }
            };
            let worst_step = self.step_worst_case(gate_size, step_val_need);
            // Reserving finale budget from generation 0 measurably starves
            // the search (v2.0 run shipped the seed at 402/600 rollouts);
            // there is nothing to ablate until a child is promoted, so the
            // reserve waits for one. It also carries a fan-out margin:
            // `step_worst_case` counts one race, but a step can run up to
            // P × fan full races, and each racing child can bill child AND
            // parent rows over the full val slice (v2.1 and v2.2 both lost
            // the finale to this overshoot; see `ablate_margin`).
            let ablate_reserve = if self.ablate_champion && has_promotion {
                self.ablate_budget + self.ablate_margin(val_cols.len())
            } else {
                0
            };
            // COMPOUND: hold the certification budget once there is a
            // champion to certify (same promotion-gated scheduling as the
            // ablation reserve — an unconditional hold starves the search).
            // The hold carries the fan-out margin unless the ablation
            // reserve already does: `step_worst_case` prices ONE race, but
            // a step can run P × fan full races, and the live compound-mem
            // arm measured the overshoot eating half the certification
            // columns (40 of 80; prereg amendment 2). The margin is capped
            // at half the certification budget: uncapped, it scales with
            // the val slice and at small budgets starves the SEARCH instead
            // (live pupa-toast: 330 of 600 rollouts held back, 10 proposals
            // vs v2's ~25; prereg amendment 7).
            let certify_hold = if self.certify_ship && has_promotion && !certify_cols.is_empty()
            {
                self.certify_budget
                    + if ablate_reserve == 0 {
                        self.ablate_margin(val_cols.len()).min(self.certify_budget / 2)
                    } else {
                        0
                    }
            } else {
                0
            };
            if !engine.budget_allows(
                2 * selection_cols.len() + worst_step + ablate_reserve + certify_hold,
            ) {
                break;
            }
            let effective_cap = if self.cap_after_promotion && !has_promotion {
                None
            } else {
                self.length_cap
            };
            let use_lexicase = matches!(self.parent_selection, ParentSelection::Lexicase)
                && (!self.lexicase_after_promotion || has_promotion);

            // Joint multi-leaf search: each step mutates one slot per
            // parent group. RoundRobin uses the reference package's
            // `generation % len` schedule; Repair overrides it per group
            // with counterfactually verified blame. Single-leaf programs
            // are unaffected either way.
            let round_robin_leaf = generation % leaves.len();

            let ledger = format!(
                "{}{}",
                Self::render_ledger(&trials, self.ledger_window, self.ledger_falsify_margin),
                Self::render_repair_lessons(&repair_lessons)
            );

            // ---- Plan phase (P×N parallel proposals, gepa blog 2026-07-30):
            // sample P parents, draw each a reflection pool of N disjoint
            // mini-batches plus one held-out gate batch, evaluate each parent
            // on its pool in one fan-out, and plan up to N proposal slots per
            // parent. Every slot in the step sees the same ledger, pool, and
            // operator-stats snapshot — the step commits its proposals in a
            // batch, which reduces how often the search re-steers against
            // the validation set (the blog's generalization argument).
            // P = N = 1 is exactly the sequential v2 loop.
            let mut groups: Vec<StepGroup> = Vec::new();
            let mut planned: Vec<StepSlot> = Vec::new();
            let mut merge_taken = false;
            for _ in 0..p_count {
                let greedy = self.parent_greed > 0.0
                    && rng.gen_range(0.0f64..1.0) < self.parent_greed;
                let pareto_snapshot = engine.pareto_over(&val_cols);
                let sampled_parent = if greedy
                    && let Some(parent) = pool.best_by_average(&pareto_snapshot)
                {
                    Some(parent)
                } else if !use_lexicase {
                    pool.sample_parent(&pareto_snapshot, &mut rng)
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
                let sampled_row = engine.register(sampled_parent.engine_candidate());

                let (a_pool, batch_b) = match self.batch_weighting {
                    BatchWeighting::UnsolvedBias => GEPA::sample_batches(
                        &train_pool,
                        &usage,
                        engine.matrix(),
                        sampled_row,
                        self.minibatch_size * n_count,
                        gate_size,
                        self.perfect_score,
                        &mut rng,
                    ),
                    BatchWeighting::Learnability => Self::sample_batches_learnability(
                        &pool,
                        &train_pool,
                        &usage,
                        engine.matrix(),
                        self.minibatch_size * n_count,
                        gate_size,
                        &mut rng,
                    ),
                };
                for &col in a_pool.iter().chain(&batch_b) {
                    *usage.entry(col).or_insert(0) += 1;
                }

                // Parent on its whole reflection pool in one fan-out
                // (mostly cache-served for revisited parents).
                let parent_eval = match engine.evaluate(target, sampled_row, Some(&a_pool)).await?
                {
                    EvalOutcome::Complete(eval) => eval,
                    EvalOutcome::BudgetExhausted { .. } => break 'generations,
                };
                GEPA::require_feedback(&parent_eval, &sampled_parent.module_name, generation)?;

                // ---- Counterfactual stage-repair (SlotCredit::Repair):
                // debug failed batch-A rollouts BEFORE choosing what to
                // mutate. For each probe: a fresh traced base rollout (cache
                // hits carry no trace), one teacher diagnosis (blame a
                // module + write the output it should have produced), and
                // one counterfactual replay with the repaired output
                // substituted — upstream serves free from the recording,
                // downstream runs live, and the metric says whether the
                // rollout actually recovers. Verified recovery drives the
                // slot choice and conditions the Rewrite.
                let mut leaf_idx = round_robin_leaf;
                let mut repair_context: Option<String> = None;
                if matches!(self.slot_credit, SlotCredit::Repair) && self.repair_probes > 0 {
                    let mut failed: Vec<&RolloutOutcome> = parent_eval
                        .rollouts
                        .iter()
                        .filter(|r| r.eval.score < self.perfect_score - SCORE_EPS)
                        .collect();
                    failed.sort_by(|a, b| a.eval.score.total_cmp(&b.eval.score));
                    let schema = pipeline_schema(&leaves);
                    let mut notes: HashMap<String, Vec<String>> = HashMap::new();
                    let mut step_totals: HashMap<String, f64> = HashMap::new();
                    let mut budget_dead = false;
                    for rollout in failed.into_iter().take(self.repair_probes) {
                        if budget_dead {
                            break;
                        }
                        let (base_eval, base_trace) = match engine
                            .probe_rollout(target, sampled_row, rollout.example)
                            .await
                        {
                            Ok(Some(pair)) => pair,
                            Ok(None) => break,
                            Err(err) => {
                                tracing::warn!(error = %err, "repair base rollout failed; skipping probe");
                                continue;
                            }
                        };
                        repair_probes_run += 1;
                        if base_eval.score >= self.perfect_score - SCORE_EPS {
                            continue;
                        }
                        let trace_text = render_pipeline_trace(&base_trace);
                        if trace_text.is_empty() {
                            continue;
                        }
                        let base_feedback =
                            base_eval.feedback.clone().unwrap_or_default();
                        // Up to `repair_attempts` hypotheses per probe: each
                        // repair falsified by re-execution is shown back to
                        // the teacher (with what the re-executed run still
                        // got wrong) for a revised attempt — hypothesis,
                        // falsification, revision.
                        let mut falsified: Vec<String> = Vec::new();
                        for _attempt in 0..self.repair_attempts.max(1) {
                            let failure_feedback = if falsified.is_empty() {
                                base_feedback.clone()
                            } else {
                                format!("{base_feedback}\n\n{}", falsified.join("\n\n"))
                            };
                            let diagnosis = reflectors
                                .diagnose
                                .call(DiagnoseRepairInput {
                                    pipeline_schema: schema.clone(),
                                    execution_trace: trace_text.clone(),
                                    failure_feedback,
                                })
                                .await;
                            engine.charge(0, 1);
                            let diagnosis = match diagnosis {
                                Ok(diagnosis) => diagnosis,
                                Err(err) => {
                                    tracing::warn!(error = %err, "repair diagnosis call failed; skipping probe");
                                    break;
                                }
                            };
                            let blamed = diagnosis.blamed_module.trim().to_string();
                            let Some(leaf) =
                                leaves.iter().find(|leaf| leaf.name == blamed)
                            else {
                                tracing::debug!(blamed, "diagnosis blamed an unknown module; skipping probe");
                                break;
                            };
                            let Some((field, _)) = leaf.output_fields.first() else {
                                break;
                            };
                            let repaired = diagnosis.repaired_output.trim().to_string();
                            if repaired.is_empty() {
                                break;
                            }
                            let Some((edited, actual)) =
                                repair_span_output(&base_trace, &blamed, field, &repaired)
                            else {
                                tracing::debug!(blamed, "no replayable span for the blamed module; skipping probe");
                                break;
                            };
                            let cf_eval = match engine
                                .counterfactual_rollout(
                                    target,
                                    sampled_row,
                                    rollout.example,
                                    edited,
                                )
                                .await
                            {
                                Ok(Some(eval)) => eval,
                                Ok(None) => {
                                    budget_dead = true;
                                    break;
                                }
                                Err(err) => {
                                    tracing::warn!(error = %err, "counterfactual replay failed; skipping probe");
                                    break;
                                }
                            };
                            let delta = cf_eval.score - base_eval.score;
                            let slot_entry =
                                repair_slot_totals.entry(blamed.clone()).or_insert((0, 0.0));
                            slot_entry.0 += 1;
                            slot_entry.1 += delta;
                            *step_totals.entry(blamed.clone()).or_insert(0.0) += delta;
                            tracing::info!(
                                example = rollout.example,
                                blamed,
                                base = base_eval.score,
                                counterfactual = cf_eval.score,
                                repaired = truncate(&repaired, 160),
                                "stage-repair probe"
                            );
                            if delta > SCORE_EPS {
                                repair_recovered += 1;
                                repair_lessons.push((
                                    generation,
                                    blamed.clone(),
                                    truncate(diagnosis.hypothesis.trim(), 200).to_string(),
                                    delta,
                                ));
                                // COMPOUND fact-pinning: the verified repair
                                // licenses the fixed example's own
                                // leak-class strings that its repaired text
                                // carries — counterfactual proof that this
                                // content fixes a real failure is exactly
                                // the provenance a pinned fact needs.
                                if matches!(self.leak_policy, LeakPolicy::AllowWithProvenance) {
                                    let repaired_lower = repaired.to_lowercase();
                                    for text in
                                        GEPA::leak_strings(target, &[rollout.example])
                                    {
                                        if repaired_lower.contains(&text) {
                                            pin_licenses.entry(text).or_insert((
                                                rollout.example,
                                                generation,
                                                delta,
                                            ));
                                        }
                                    }
                                }
                                notes.entry(blamed).or_default().push(format!(
                                    "- A failed rollout was FIXED (score {:.2} -> {:.2}) by \
                                     replacing this module's output.\n  Actual output: {}\n  \
                                     Repaired output that fixed it: {}\n  Weakness: {}",
                                    base_eval.score,
                                    cf_eval.score,
                                    truncate(&actual, 600),
                                    truncate(&repaired, 900),
                                    truncate(diagnosis.hypothesis.trim(), 200),
                                ));
                                break;
                            }
                            falsified.push(format!(
                                "REPAIR ATTEMPT {} (falsified by re-execution): you \
                                 blamed `{blamed}` and proposed this output:\n{}\nRe-running \
                                 the pipeline with it did NOT improve the outcome (score \
                                 {:.2} -> {:.2}). What the re-executed run still got wrong: \
                                 {}\nRevise your diagnosis: blame a different module, or \
                                 write a materially different repaired output that accounts \
                                 for why the previous one changed nothing.",
                                falsified.len() + 1,
                                truncate(&repaired, 600),
                                base_eval.score,
                                cf_eval.score,
                                truncate(cf_eval.feedback.as_deref().unwrap_or("-"), 800),
                            ));
                        }
                    }
                    if let Some((slot, total)) = step_totals
                        .iter()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        && *total > SCORE_EPS
                    {
                        leaf_idx = leaves
                            .iter()
                            .position(|leaf| &leaf.name == slot)
                            .expect("blamed slots are validated against leaves");
                    }
                    repair_context = notes.remove(&leaves[leaf_idx].name).map(|list| {
                        format!(
                            "VERIFIED WEAK-POINT DIAGNOSIS for module `{}` — counterfactual \
                             probes proved each repair below fixes a real failed rollout. \
                             Write instruction rules that make the module produce outputs \
                             like the repaired ones (never copy example-specific facts):\n{}",
                            leaves[leaf_idx].name,
                            list.join("\n")
                        )
                    });
                }
                let mutate_leaf = &leaves[leaf_idx];
                let parent_text = sampled_parent.slot_text(&mutate_leaf.name).to_string();
                let mut task_description = mutate_leaf.schema_for_reflection();
                if let Some(cap) = effective_cap {
                    task_description.push_str(&format!(
                        "\n\nHARD LIMIT: the improved instruction must be at most {cap} \
                         characters. Proposals over the limit are rejected."
                    ));
                }
                // Corpus-fact task-config: the operator prompts bake in
                // "never copy example-specific facts" (right for
                // benchmarks). With the guard off, that language must be
                // countermanded or the teacher refuses the winning move —
                // the mem2 live arm harvested style-only bundles for
                // exactly this reason.
                if self.allow_memorization {
                    task_description.push_str(
                        "\n\nTASK-CONFIG OVERRIDE: this is a corpus-fact task, and \
                         the rule against copying example-specific content does NOT \
                         apply — the opposite does. Identify ALL niche and \
                         domain-specific factual information in the execution \
                         feedback (missed key points, corrected claims, exact names, \
                         values, definitions, API details) and include it verbatim \
                         in the instruction: the assistant will not have access to \
                         this information later unless the instruction carries it. \
                         Maintain a growing, organized FACTS section — never drop a \
                         previously pinned fact, and add every new fact this batch's \
                         feedback reveals. A longer instruction dense with correct \
                         facts beats a short general one on this task.",
                    );
                }

                let group = groups.len();
                groups.push(StepGroup {
                    leaf_idx,
                    parent_text: parent_text.clone(),
                    task_description,
                    batch_b,
                });

                // Merge pairing is step-stable for a given leaf; racing the
                // same pair N times would only differ by temperature, so at
                // most one Merge slot is planned per step.
                let merge_pair =
                    Self::find_merge_pair(&pool, engine, &val_cols, &mutate_leaf.name);

                // Split the pool into N disjoint reflection mini-batches —
                // each proposal slot reflects on distinct feedback (the
                // blog's diversity device), not just a fresh temperature
                // draw. An empty pool still plans one slot so Simplify and
                // Merge stay reachable.
                let chunk_size = self.minibatch_size.max(1);
                let chunks: Vec<(usize, usize)> = if a_pool.is_empty() {
                    vec![(0, 0)]
                } else {
                    a_pool
                        .chunks(chunk_size)
                        .take(n_count)
                        .scan(0usize, |start, chunk| {
                            let range = (*start, *start + chunk.len());
                            *start += chunk.len();
                            Some(range)
                        })
                        .collect()
                };
                for (start, end) in chunks {
                    let chunk_cols = &a_pool[start..end];
                    let chunk_rollouts = &parent_eval.rollouts[start..end];
                    let a_perfect = chunk_rollouts
                        .iter()
                        .all(|r| r.eval.score >= self.perfect_score - SCORE_EPS);
                    let a_spread = {
                        let scores: Vec<f64> =
                            chunk_rollouts.iter().map(|r| r.eval.score).collect();
                        let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                        let min = scores.iter().cloned().fold(f64::INFINITY, f64::min);
                        scores.len() >= 2 && max > min + SCORE_EPS
                    };

                    // Operator eligibility, then bandit choice.
                    let mut eligible = Vec::new();
                    if !a_perfect {
                        eligible.push(RatchetOp::Rewrite);
                        if a_spread && !self.accretive_ops {
                            eligible.push(RatchetOp::Patch);
                        }
                    }
                    // Simplify and Merge only earn a slot once the search
                    // has at least one promoted child — before that, forced
                    // exploration of them is wasted budget (measured 1 win
                    // in 25 early attempts).
                    if has_promotion && parent_text.len() > 500 && !self.accretive_ops {
                        eligible.push(RatchetOp::Simplify);
                    }
                    if has_promotion && merge_pair.is_some() && !merge_taken {
                        eligible.push(RatchetOp::Merge);
                    }
                    if eligible.is_empty() {
                        skipped += 1;
                        tracing::debug!(generation, "no eligible operator; skipping slot");
                        continue;
                    }
                    let op = Self::choose_op(&eligible, &op_stats, &mut rng);

                    let mut forbidden = if self.allow_memorization {
                        Vec::new()
                    } else {
                        GEPA::leak_strings(target, chunk_cols)
                    };
                    // COMPOUND fact-pinning: licensed strings pass the guard.
                    if matches!(self.leak_policy, LeakPolicy::AllowWithProvenance)
                        && !pin_licenses.is_empty()
                    {
                        forbidden.retain(|text| !pin_licenses.contains_key(text));
                    }
                    let chunk_eval = CandidateEval {
                        candidate: sampled_row,
                        rollouts: chunk_rollouts.to_vec(),
                    };
                    // The lineage the child is gated/raced against: the
                    // sampled parent, except Merge races against the pair's
                    // stronger member.
                    let (mut lineage_parent, mut lineage_row) =
                        (sampled_parent.clone(), sampled_row);
                    let input = match op {
                        RatchetOp::Rewrite => PlannedInput::Rewrite {
                            feedback: {
                                let mut feedback = format!(
                                    "Parent average score: {:.3}\n{}",
                                    chunk_eval.mean(),
                                    GEPA::summarize_feedback(
                                        &sampled_parent.module_name,
                                        &chunk_eval
                                    )
                                );
                                if let Some(context) = &repair_context {
                                    feedback.push_str("\n\n");
                                    feedback.push_str(context);
                                }
                                feedback
                            },
                        },
                        RatchetOp::Patch => {
                            let best = chunk_eval
                                .rollouts
                                .iter()
                                .max_by(|a, b| a.eval.score.total_cmp(&b.eval.score))
                                .expect("the reflection chunk is non-empty");
                            let worst = chunk_eval
                                .rollouts
                                .iter()
                                .min_by(|a, b| a.eval.score.total_cmp(&b.eval.score))
                                .expect("the reflection chunk is non-empty");
                            PlannedInput::Patch {
                                better: Self::fmt_rollout(
                                    &sampled_parent.module_name,
                                    best,
                                    sampled_row,
                                ),
                                worse: Self::fmt_rollout(
                                    &sampled_parent.module_name,
                                    worst,
                                    sampled_row,
                                ),
                            }
                        }
                        RatchetOp::Simplify => PlannedInput::Simplify {
                            validated: Self::render_validated(&trials),
                        },
                        RatchetOp::Merge => {
                            merge_taken = true;
                            let (first, first_row, second, second_row, first_excl, second_excl) =
                                merge_pair.clone().expect("eligibility checked");
                            let a_strengths =
                                Self::fmt_strengths(engine, target, first_row, &first_excl);
                            let b_strengths =
                                Self::fmt_strengths(engine, target, second_row, &second_excl);
                            lineage_parent = first.clone();
                            lineage_row = first_row;
                            PlannedInput::Merge {
                                instruction_a: first.instruction.clone(),
                                a_strengths,
                                instruction_b: second.instruction.clone(),
                                b_strengths,
                            }
                        }
                    };
                    planned.push(StepSlot {
                        group,
                        op,
                        input,
                        forbidden,
                        lineage_parent,
                        lineage_row,
                    });
                }
            }
            if planned.is_empty() {
                continue;
            }

            // Don't spend reflection calls on children we can't afford to
            // gate.
            let gate_cells: usize = planned
                .iter()
                .map(|slot| groups[slot.group].batch_b.len())
                .sum();
            if !engine.budget_allows(gate_cells.max(1)) {
                break;
            }

            // ---- Propose phase: every planned slot's reflection call runs
            // concurrently (the blog's parallel dispatch). The leak and
            // length guards run inside each call's own retry loop.
            let proposals: Vec<OpProposal> = {
                let reflectors = &reflectors;
                let ledger = &ledger;
                let groups = &groups;
                let futures: Vec<_> = planned
                    .iter()
                    .map(|slot| async move {
                        let parent_text = &groups[slot.group].parent_text;
                        let task_description = &groups[slot.group].task_description;
                        match &slot.input {
                            PlannedInput::Rewrite { feedback } => {
                                guarded_propose!(
                                    reflectors.rewrite,
                                    &slot.forbidden,
                                    effective_cap,
                                    |extra: String| ProposeRewriteInput {
                                        task_description: task_description.clone(),
                                        current_instruction: parent_text.clone(),
                                        execution_feedback: format!("{feedback}{extra}"),
                                        trial_ledger: ledger.clone(),
                                    }
                                )
                            }
                            PlannedInput::Patch { better, worse } => {
                                guarded_propose!(
                                    reflectors.patch,
                                    &slot.forbidden,
                                    effective_cap,
                                    |extra: String| ProposePatchInput {
                                        task_description: task_description.clone(),
                                        current_instruction: parent_text.clone(),
                                        better_rollout: better.clone(),
                                        worse_rollout: format!("{worse}{extra}"),
                                        trial_ledger: ledger.clone(),
                                    }
                                )
                            }
                            PlannedInput::Simplify { validated } => {
                                guarded_propose!(
                                    reflectors.simplify,
                                    &slot.forbidden,
                                    effective_cap,
                                    |extra: String| ProposeSimplifyInput {
                                        task_description: task_description.clone(),
                                        current_instruction: parent_text.clone(),
                                        validated_lessons: format!("{validated}{extra}"),
                                    }
                                )
                            }
                            PlannedInput::Merge {
                                instruction_a,
                                a_strengths,
                                instruction_b,
                                b_strengths,
                            } => {
                                guarded_propose!(
                                    reflectors.merge,
                                    &slot.forbidden,
                                    effective_cap,
                                    |extra: String| ProposeMergeInput {
                                        task_description: task_description.clone(),
                                        instruction_a: instruction_a.clone(),
                                        a_strengths: a_strengths.clone(),
                                        instruction_b: instruction_b.clone(),
                                        b_strengths: format!("{b_strengths}{extra}"),
                                        trial_ledger: ledger.clone(),
                                    }
                                )
                            }
                        }
                    })
                    .collect();
                futures::future::join_all(futures).await
            };

            // Account for every slot's proposal, then keep the viable ones.
            let mut children: Vec<PendingChild> = Vec::new();
            for (slot_idx, (slot, proposal)) in planned.iter().zip(proposals).enumerate() {
                engine.charge(0, proposal.reflection_calls);
                leak_retries += proposal.leak_retried as usize;
                length_retries += proposal.length_retried as usize;
                op_stats.entry(slot.op).or_insert((0, 0)).0 += 1;

                let Some(child_instruction) = proposal.instruction else {
                    if proposal.leaked {
                        leak_skipped += 1;
                        trials.push(TrialRecord {
                            generation,
                            operator: slot.op,
                            parent_id: slot.lineage_parent.id,
                            hypothesis: proposal.hypothesis,
                            outcome: TrialOutcome::LeakSkipped,
                        });
                    } else if proposal.overlong {
                        length_skipped += 1;
                        trials.push(TrialRecord {
                            generation,
                            operator: slot.op,
                            parent_id: slot.lineage_parent.id,
                            hypothesis: proposal.hypothesis,
                            outcome: TrialOutcome::LengthSkipped,
                        });
                    }
                    continue;
                };

                let child = slot.lineage_parent.mutate_slot(
                    leaves[groups[slot.group].leaf_idx].name.clone(),
                    child_instruction,
                    generation + 1,
                );
                let child_row = engine.register(child.engine_candidate());
                if child_row == slot.lineage_row {
                    trials.push(TrialRecord {
                        generation,
                        operator: slot.op,
                        parent_id: slot.lineage_parent.id,
                        hypothesis: proposal.hypothesis,
                        outcome: TrialOutcome::NoOp,
                    });
                    continue;
                }
                if children.iter().any(|pending| pending.child_row == child_row) {
                    tracing::debug!(generation, "duplicate proposal within step; dropping");
                    continue;
                }
                children.push(PendingChild {
                    slot: slot_idx,
                    child,
                    child_row,
                    hypothesis: proposal.hypothesis,
                });
            }

            // ---- Gate phase, per parent group: children are screened on
            // their group's held-out batch B in one batched fan-out —
            // noise-aware: a child dies here only when the parent beats it
            // by more than `gate_z` SEs. Lateral moves pass and the race
            // decides; v1's strict `>` gate was measured to prune the
            // stepping stones deep lineages need.
            let mut race_entries: Vec<RaceEntry> = Vec::new();
            for (group_idx, group) in groups.iter().enumerate() {
                let (group_children, rest): (Vec<PendingChild>, Vec<PendingChild>) =
                    std::mem::take(&mut children)
                        .into_iter()
                        .partition(|pending| planned[pending.slot].group == group_idx);
                children = rest;
                if group_children.is_empty() {
                    continue;
                }
                if group.batch_b.is_empty() {
                    for pending in group_children {
                        let slot = &planned[pending.slot];
                        race_entries.push(RaceEntry {
                            child: pending.child,
                            child_row: pending.child_row,
                            hypothesis: pending.hypothesis,
                            op: slot.op,
                            lineage_parent_id: slot.lineage_parent.id,
                            lineage_row: slot.lineage_row,
                            evidence: PairedEvidence::new(delta_scale),
                        });
                    }
                    continue;
                }

                // Rewrite fan-out (N = 1 only): sample a second proposal
                // from the same reflection context (the teacher runs at
                // temperature 1.0, so the draws differ); the better of the
                // two on batch B races. With N > 1 the parallel slots
                // supersede this.
                let mut sibling: Option<(GEPACandidate, usize, String, usize)> = None;
                if n_count == 1
                    && self.rewrite_fanout > 1
                    && let Some(pending) = group_children
                        .iter()
                        .find(|pending| planned[pending.slot].op == RatchetOp::Rewrite)
                {
                    let slot = &planned[pending.slot];
                    let PlannedInput::Rewrite { feedback } = &slot.input else {
                        unreachable!("a Rewrite slot carries a Rewrite input");
                    };
                    let second = guarded_propose!(
                        reflectors.rewrite,
                        &slot.forbidden,
                        effective_cap,
                        |extra: String| ProposeRewriteInput {
                            task_description: group.task_description.clone(),
                            current_instruction: group.parent_text.clone(),
                            execution_feedback: format!("{feedback}{extra}"),
                            trial_ledger: ledger.clone(),
                        }
                    );
                    engine.charge(0, second.reflection_calls);
                    leak_retries += second.leak_retried as usize;
                    length_retries += second.length_retried as usize;
                    if let Some(instruction) = second.instruction {
                        let sib = slot.lineage_parent.mutate_slot(
                            leaves[group.leaf_idx].name.clone(),
                            instruction,
                            generation + 1,
                        );
                        let sib_row = engine.register(sib.engine_candidate());
                        if sib_row != slot.lineage_row && sib_row != pending.child_row {
                            sibling = Some((sib, sib_row, second.hypothesis, pending.slot));
                        }
                    }
                }

                // One batched screen: every lineage parent and child of the
                // group on batch B. The parents' cells are cache-served
                // after the first generation that visits them.
                let mut rows: Vec<usize> = Vec::new();
                for pending in &group_children {
                    let lineage_row = planned[pending.slot].lineage_row;
                    if !rows.contains(&lineage_row) {
                        rows.push(lineage_row);
                    }
                }
                for pending in &group_children {
                    if !rows.contains(&pending.child_row) {
                        rows.push(pending.child_row);
                    }
                }
                if let Some((_, sib_row, _, _)) = &sibling
                    && !rows.contains(sib_row)
                {
                    rows.push(*sib_row);
                }
                let evals = match engine
                    .evaluate_many(target, &rows, Some(&group.batch_b))
                    .await?
                {
                    BatchEvalOutcome::Complete(evals) => evals,
                    BatchEvalOutcome::BudgetExhausted { .. } => break 'generations,
                };
                let mean_of = |row: usize| -> f64 {
                    evals[rows.iter().position(|&r| r == row).expect("row was requested")]
                        .mean()
                };

                for mut pending in group_children {
                    let slot_idx = pending.slot;
                    let slot = &planned[slot_idx];
                    let child_eval = &evals[rows
                        .iter()
                        .position(|&r| r == pending.child_row)
                        .expect("row was requested")];
                    GEPA::require_feedback(
                        child_eval,
                        &pending.child.module_name,
                        generation + 1,
                    )?;
                    let mut child_gate_mean = child_eval.mean();
                    if let Some((sib, sib_row, sib_hypothesis, sib_slot)) = &sibling
                        && *sib_slot == slot_idx
                        && mean_of(*sib_row) > child_gate_mean
                    {
                        tracing::debug!(generation, "fan-out sibling won batch B; racing it");
                        pending.child = sib.clone();
                        pending.child_row = *sib_row;
                        pending.hypothesis = sib_hypothesis.clone();
                        child_gate_mean = mean_of(*sib_row);
                    }
                    let parent_b =
                        row_scores(engine.matrix(), slot.lineage_row, &group.batch_b);
                    let child_b =
                        row_scores(engine.matrix(), pending.child_row, &group.batch_b);
                    if !within_noise(&parent_b, &child_b, self.gate_z) {
                        gate_rejected += 1;
                        trials.push(TrialRecord {
                            generation,
                            operator: slot.op,
                            parent_id: slot.lineage_parent.id,
                            hypothesis: pending.hypothesis.clone(),
                            outcome: TrialOutcome::GateKilled {
                                parent_gate: mean_of(slot.lineage_row),
                                child_gate: child_gate_mean,
                            },
                        });
                        continue;
                    }
                    race_entries.push(RaceEntry {
                        child: pending.child,
                        child_row: pending.child_row,
                        hypothesis: pending.hypothesis,
                        op: slot.op,
                        lineage_parent_id: slot.lineage_parent.id,
                        lineage_row: slot.lineage_row,
                        evidence: PairedEvidence::new(delta_scale),
                    });
                }
            }
            if race_entries.is_empty() {
                continue;
            }

            // ---- Race phase: all gate survivors race the validation
            // columns together in shuffled blocks (one batched fan-out per
            // block); each child is still paired against its own lineage
            // parent for the kill check. Under AcceptMode::Gate the race is
            // skipped entirely: survivors get one batched scout evaluation
            // and join the pool — accretion economics, with the sequential
            // certification carrying all the verification.
            let mut alive = race_entries;
            if matches!(self.accept_mode, AcceptMode::Gate) {
                let mut rows: Vec<usize> = alive.iter().map(|entry| entry.child_row).collect();
                for entry in &alive {
                    if !rows.contains(&entry.lineage_row) {
                        rows.push(entry.lineage_row);
                    }
                }
                match engine.evaluate_many(target, &rows, Some(&scout_cols)).await? {
                    BatchEvalOutcome::Complete(evals) => {
                        for entry in &alive {
                            let eval = &evals[rows
                                .iter()
                                .position(|&r| r == entry.child_row)
                                .expect("row was requested")];
                            GEPA::require_feedback(
                                eval,
                                &entry.child.module_name,
                                generation + 1,
                            )?;
                        }
                    }
                    BatchEvalOutcome::BudgetExhausted { .. } => break 'generations,
                }
                // Promotion happens below on the shared scout slice.
            } else {
            let mut order = val_cols.clone();
            order.shuffle(&mut rng);
            let mut evaluated: Vec<usize> = Vec::new();
            for block in order.chunks(self.race_block.max(1)) {
                if alive.is_empty() {
                    break;
                }
                let rows: Vec<usize> = alive.iter().map(|entry| entry.child_row).collect();
                let evals = match engine.evaluate_many(target, &rows, Some(block)).await? {
                    BatchEvalOutcome::Complete(evals) => evals,
                    BatchEvalOutcome::BudgetExhausted { .. } => break 'generations,
                };
                for (entry, eval) in alive.iter().zip(&evals) {
                    GEPA::require_feedback(eval, &entry.child.module_name, generation + 1)?;
                }
                evaluated.extend_from_slice(block);
                // Feed this block's paired deltas into each survivor's
                // evidence wallets (race order is the shuffled column
                // order, fixed before any child's scores existed).
                {
                    let matrix = engine.matrix();
                    for entry in alive.iter_mut() {
                        for &col in block {
                            if let (Some(child), Some(parent)) = (
                                matrix.score(entry.child_row, col),
                                matrix.score(entry.lineage_row, col),
                            ) {
                                entry.evidence.observe(child - parent);
                            }
                        }
                    }
                }
                if evaluated.len() == val_cols.len() {
                    break;
                }
                let matrix = engine.matrix();
                alive.retain(|entry| {
                    let survives = match self.race_rule {
                        RaceRule::ZBlock => {
                            let parent_scores =
                                row_scores(matrix, entry.lineage_row, &evaluated);
                            let child_scores =
                                row_scores(matrix, entry.child_row, &evaluated);
                            within_noise(&parent_scores, &child_scores, self.race_z)
                        }
                        RaceRule::Eprocess => {
                            !entry.evidence.certified_harm(self.race_alpha)
                        }
                    };
                    if survives {
                        return true;
                    }
                    let deltas: Vec<f64> = evaluated
                        .iter()
                        .filter_map(|&col| {
                            Some(
                                matrix.score(entry.child_row, col)?
                                    - matrix.score(entry.lineage_row, col)?,
                            )
                        })
                        .collect();
                    let delta = deltas.iter().sum::<f64>() / deltas.len().max(1) as f64;
                    race_killed += 1;
                    trials.push(TrialRecord {
                        generation,
                        operator: entry.op,
                        parent_id: entry.lineage_parent_id,
                        hypothesis: entry.hypothesis.clone(),
                        outcome: TrialOutcome::RaceKilled {
                            columns_evaluated: evaluated.len(),
                            paired_delta: delta,
                        },
                    });
                    false
                });
            }
            if evaluated.len() < val_cols.len() {
                // The val set was empty or every child died mid-race; either
                // way there is nothing to promote.
                continue;
            }
            }

            // Survivors: assemble their ranking rows from the matrix and
            // promote them all — the step commits its whole batch before
            // the pool steers another adaptation. Every candidate is ranked
            // on the same slice (full val under Race, the scout slice under
            // Gate), so pool comparisons stay apples-to-apples.
            let ranking_cols: &[usize] = match self.accept_mode {
                AcceptMode::Race => &val_cols,
                AcceptMode::Gate => &scout_cols,
            };
            for entry in alive {
                let child_scores: Vec<f32> = ranking_cols
                    .iter()
                    .map(|&col| {
                        engine.matrix().score(entry.child_row, col).unwrap_or(0.0) as f32
                    })
                    .collect();
                let paired_delta = {
                    let parent_scores =
                        row_scores(engine.matrix(), entry.lineage_row, ranking_cols);
                    let deltas: Vec<f64> = ranking_cols
                        .iter()
                        .zip(&parent_scores)
                        .filter_map(|(&col, parent)| {
                            Some(engine.matrix().score(entry.child_row, col)? - (*parent)?)
                        })
                        .collect();
                    deltas.iter().sum::<f64>() / deltas.len().max(1) as f64
                };
                if matches!(self.accept_mode, AcceptMode::Race)
                    && self.promote_only_gains
                    && paired_delta < 0.0
                {
                    faded += 1;
                    trials.push(TrialRecord {
                        generation,
                        operator: entry.op,
                        parent_id: entry.lineage_parent_id,
                        hypothesis: entry.hypothesis,
                        outcome: TrialOutcome::Faded { paired_delta },
                    });
                    continue;
                }
                let promoted = pool
                    .add_scored(entry.child, entry.child_row, child_scores)
                    .clone();
                if paired_delta > 0.0 {
                    op_stats.entry(entry.op).or_insert((0, 0)).1 += 1;
                }
                trials.push(TrialRecord {
                    generation,
                    operator: entry.op,
                    parent_id: entry.lineage_parent_id,
                    hypothesis: entry.hypothesis,
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
                .map(|(candidate, row)| (candidate.clone(), *row))
                .collect();
            ranked.sort_by(|a, b| {
                b.0.average_score()
                    .partial_cmp(&a.0.average_score())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let k = self.champion_k.max(2).min(ranked.len());
            if ranked.len() >= 2 && engine.budget_allows(k * selection_cols.len()) {
                let rows: Vec<usize> = ranked[..k].iter().map(|(_, row)| *row).collect();
                if let BatchEvalOutcome::Complete(evals) =
                    engine.evaluate_many(target, &rows, Some(&selection_cols)).await?
                {
                    fresh_selection_cols = champion_cols.len();
                    variant_selection_cols = variant_cols.len();
                    let means: Vec<f64> = evals.iter().map(|eval| eval.mean()).collect();
                    tracing::info!(
                        ?means,
                        k,
                        fresh = champion_cols.len(),
                        variants = variant_cols.len(),
                        "fresh-data champion selection"
                    );
                    // Argmax with first-on-tie: at k = 2 this reproduces the
                    // measured "second only if strictly better" rule.
                    let best_idx = means
                        .iter()
                        .enumerate()
                        .max_by(|a, b| {
                            a.1.partial_cmp(b.1)
                                .unwrap_or(std::cmp::Ordering::Equal)
                                .then(b.0.cmp(&a.0))
                        })
                        .map(|(idx, _)| idx)
                        .unwrap_or(0);
                    best_by_mean_candidate = ranked[best_idx].0.clone();
                }
            }
        }

        // Seed-protected argmax: children compete by raw validation mean; only
        // the generation-0 seed keeps incumbency when the argmax fails to beat
        // it by more than `selection_z` standard errors (paired per column).
        // See the identical rule in GEPA for the measured rationale.
        // Under `certify_ship` the sequential certification below replaces
        // this comparison — same honesty, adaptive power, and on fresh
        // columns instead of the optimized-against validation slice.
        let certifying = self.certify_ship && !certify_cols.is_empty();
        let mut best_candidate = if self.selection_z > 0.0 && !certifying {
            let best_row = pool
                .entries
                .iter()
                .find(|(candidate, _)| candidate.id == best_by_mean_candidate.id)
                .map(|(_, row)| *row)
                .expect("the chosen candidate is in the pool");
            let best_scores = row_scores(engine.matrix(), best_row, &val_cols);
            let seed_entry = pool.entries.iter().find(|(candidate, _)| {
                candidate.generation == 0
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
                        let mut probe = current.clone();
                        probe.instruction = text.clone();
                        probe.slots.insert(current.module_name.clone(), text.clone());
                        let row = engine.register(probe.engine_candidate());
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
                            candidate.slots.insert(candidate.module_name.clone(), text.clone());
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

        // COMPOUND sequential ship certification: the champion races the
        // seed PAIRED on the reserved never-touched columns, block by
        // block, under an anytime-valid e-process — stop the moment the
        // gain (ship it) or the harm (ship the seed) is certified at
        // `certify_alpha`, or when columns/budget run out (unproven: ship
        // the seed per Occam, and report the open evidence). The fee
        // self-scales with the effect: a big fix certifies in a block or
        // two, a compounded pile of tiny wins as a single bundle.
        let mut certified: Option<bool> = None;
        let mut certify_cols_used = 0usize;
        let mut certify_mean_delta = 0.0f64;
        let mut certify_e_value = 0.0f64;
        if certifying && best_candidate.generation > 0 {
            let (seed_candidate, seed_row) = pool
                .entries
                .iter()
                .find(|(candidate, _)| candidate.generation == 0)
                .map(|(candidate, row)| (candidate.clone(), *row))
                .expect("the generation-0 seed is always in the pool");
            // Up to two challengers under a Bonferroni split: the shipped
            // champion, then (when different) the raw val argmax — the
            // mem4 live arm stranded a SIG-positive candidate in the
            // argmax slot after the champion screen drew noise. Testing
            // n challengers each at `certify_alpha / n` keeps the family
            // ship guarantee at `certify_alpha`.
            let mut challengers: Vec<GEPACandidate> = vec![best_candidate.clone()];
            if val_argmax_candidate.id != best_candidate.id {
                challengers.push(val_argmax_candidate.clone());
            }
            let alpha_each = self.certify_alpha / challengers.len() as f64;
            let spend_before = engine.spend().metric_calls;
            let mut shipped: Option<GEPACandidate> = None;
            'challengers: for challenger in &challengers {
                let champion_row = engine.register(challenger.engine_candidate());
                if champion_row == seed_row {
                    continue;
                }
                // Warm-start the bets from the search-time val delta
                // (challenger's pool row vs seed, paired). The val estimate
                // is selection-biased, but a bet prior only shapes λ — a
                // too-optimistic prior costs growth, never validity — and
                // its 25-pseudo-observation weight washes out quickly.
                let mut evidence = {
                    let matrix = engine.matrix();
                    let prior_row = pool
                        .entries
                        .iter()
                        .find(|(candidate, _)| candidate.id == challenger.id)
                        .map(|(_, row)| *row);
                    let deltas: Vec<f64> = prior_row
                        .map(|row| {
                            val_cols
                                .iter()
                                .filter_map(|&col| {
                                    Some(matrix.score(row, col)? - matrix.score(seed_row, col)?)
                                })
                                .collect()
                        })
                        .unwrap_or_default();
                    if deltas.len() >= 5 {
                        let n = deltas.len() as f64;
                        let mean = deltas.iter().sum::<f64>() / n;
                        let var =
                            deltas.iter().map(|d| (d - mean) * (d - mean)).sum::<f64>() / n;
                        PairedEvidence::with_bet_prior(
                            delta_scale,
                            mean,
                            var.sqrt().max(0.05 * delta_scale),
                            25.0,
                        )
                    } else {
                        PairedEvidence::new(delta_scale)
                    }
                };
                let mut verdict: Option<bool> = None;
                for block in certify_cols.chunks(self.certify_block.max(1)) {
                    if engine.spend().metric_calls - spend_before >= self.certify_budget {
                        break;
                    }
                    let evals = match engine
                        .evaluate_many(target, &[champion_row, seed_row], Some(block))
                        .await?
                    {
                        BatchEvalOutcome::Complete(evals) => evals,
                        BatchEvalOutcome::BudgetExhausted { .. } => break,
                    };
                    for (champ, seed) in evals[0].rollouts.iter().zip(&evals[1].rollouts) {
                        evidence.observe(champ.eval.score - seed.eval.score);
                    }
                    if evidence.certified_gain(alpha_each) {
                        verdict = Some(true);
                        break;
                    }
                    if evidence.certified_harm(alpha_each) {
                        verdict = Some(false);
                        break;
                    }
                }
                certify_cols_used += evidence.count();
                certify_mean_delta = evidence.mean_delta();
                certify_e_value = evidence.e_gain();
                certified = verdict;
                tracing::info!(
                    challenger = challenger.id,
                    cols = evidence.count(),
                    mean_delta = certify_mean_delta,
                    e_gain = certify_e_value,
                    verdict = ?verdict,
                    alpha_each,
                    "sequential ship certification"
                );
                if verdict == Some(true) {
                    shipped = Some(challenger.clone());
                    break 'challengers;
                }
            }
            best_candidate = match shipped {
                Some(champion) => champion,
                None => seed_candidate,
            };
        }

        // COMPOUND fact pins: licensed strings the shipped winner actually
        // carries, with the licensing repair's provenance and a staleness
        // clock for scheduled re-verification.
        let pins: Vec<PinRecord> = if pin_licenses.is_empty() {
            Vec::new()
        } else {
            let winner_text = best_candidate
                .slots
                .values()
                .map(|text| text.to_lowercase())
                .collect::<Vec<_>>()
                .join("\n");
            let winner_text = if winner_text.is_empty() {
                best_candidate.instruction.to_lowercase()
            } else {
                winner_text
            };
            let now_unix = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|elapsed| elapsed.as_secs())
                .unwrap_or(0);
            let mut pins: Vec<PinRecord> = pin_licenses
                .iter()
                .filter(|(text, _)| winner_text.contains(text.as_str()))
                .map(|(text, &(source_example, generation, verified_delta))| PinRecord {
                    text: text.clone(),
                    source_example,
                    generation,
                    verified_delta,
                    pinned_at_unix: now_unix,
                })
                .collect();
            pins.sort_by(|a, b| a.text.cmp(&b.text));
            pins
        };

        // Report the raw val-argmax whenever the shipped winner differs from
        // it (champion override, seed protection, or refused certification)
        // so ablation arms exist.
        let best_by_mean =
            (best_candidate.id != val_argmax_candidate.id).then_some(val_argmax_candidate);

        let winner = best_candidate.engine_candidate();
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
            faded,
            skipped,
            leak_retries,
            leak_skipped,
            length_retries,
            length_skipped,
            fresh_selection_cols,
            variant_selection_cols,
            ablation_removed,
            ablation_rollouts,
            repair_probes_run,
            repair_recovered,
            repair_lessons,
            certified,
            certify_cols_used,
            certify_mean_delta,
            certify_e_value,
            pins,
            repair_slot_stats: {
                let mut stats: Vec<(String, usize, f64)> = repair_slot_totals
                    .into_iter()
                    .map(|(slot, (probes, delta))| (slot, probes, delta))
                    .collect();
                stats.sort_by(|a, b| b.2.total_cmp(&a.2));
                stats
            },
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
        let text = Ratchet::render_ledger(&trials, 5, 0.0);
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
        assert!(Ratchet::render_ledger(&neutral, 5, 0.0).contains("(none yet)"));
    }

    #[test]
    fn falsify_margin_drops_noise_kills_from_the_digest() {
        let trials = vec![
            record(
                1,
                RatchetOp::Rewrite,
                "a decisive failure",
                TrialOutcome::RaceKilled {
                    columns_evaluated: 16,
                    paired_delta: -0.09,
                },
            ),
            record(
                2,
                RatchetOp::Patch,
                "a noise kill of a near-neutral child",
                TrialOutcome::RaceKilled {
                    columns_evaluated: 8,
                    paired_delta: -0.004,
                },
            ),
            record(
                3,
                RatchetOp::Rewrite,
                "a marginal gate kill",
                TrialOutcome::GateKilled {
                    parent_gate: 0.61,
                    child_gate: 0.60,
                },
            ),
        ];
        // Margin 0.0 (the v2 default) keeps every kill.
        let all = Ratchet::render_ledger(&trials, 12, 0.0);
        assert!(all.contains("a decisive failure"));
        assert!(all.contains("noise kill"));
        assert!(all.contains("marginal gate kill"));
        // A 0.02 margin keeps only the decisive kill.
        let filtered = Ratchet::render_ledger(&trials, 12, 0.02);
        assert!(filtered.contains("a decisive failure"));
        assert!(!filtered.contains("noise kill"));
        assert!(!filtered.contains("marginal gate kill"));

        // Faded survivors follow the same margin rule: a decisively-worse
        // full-race result is real negative evidence, a hairline one isn't.
        let faded = vec![
            record(
                4,
                RatchetOp::Rewrite,
                "clearly hurt over the full race",
                TrialOutcome::Faded { paired_delta: -0.06 },
            ),
            record(
                5,
                RatchetOp::Patch,
                "hairline fade",
                TrialOutcome::Faded { paired_delta: -0.001 },
            ),
        ];
        let digest = Ratchet::render_ledger(&faded, 12, 0.02);
        assert!(digest.contains("clearly hurt over the full race"));
        assert!(digest.contains("finished the full race below its parent"));
        assert!(!digest.contains("hairline fade"));
    }

    #[test]
    fn compound_preset_flips_the_v4_knobs_and_keeps_the_backbone() {
        let optimizer = compound();
        assert_eq!(optimizer.race_rule, RaceRule::Eprocess);
        assert!(optimizer.certify_ship);
        assert_eq!(optimizer.certify_reserve, 150);
        assert_eq!(optimizer.certify_budget, 400);
        assert_eq!(optimizer.certify_alpha, 0.1);
        assert_eq!(optimizer.parent_greed, 0.5);
        assert!(optimizer.promote_only_gains);
        assert_eq!(optimizer.ledger_falsify_margin, 0.02);
        // Long cheap cycles (the measured long-cycle configuration).
        assert_eq!(optimizer.num_iterations, 150);
        assert_eq!(optimizer.minibatch_size, 3);
        assert_eq!(optimizer.parents_per_step, 2);
        assert_eq!(optimizer.proposals_per_parent, 2);
        // Fact-pinning stays opt-in: leak semantics are task-dependent.
        assert_eq!(optimizer.leak_policy, LeakPolicy::Forbid);
        assert_eq!(optimizer.slot_credit, SlotCredit::RoundRobin);
        // Backbone untouched.
        assert_eq!(optimizer.gate_z, 1.0);
        assert_eq!(optimizer.champion_fresh, 40);
        assert_eq!(optimizer.selection_z, 1.0);

        // And the plain builder preserves measured v2/v3 behavior exactly.
        let v2 = Ratchet::builder().build();
        assert_eq!(v2.race_rule, RaceRule::ZBlock);
        assert!(!v2.certify_ship);
        assert_eq!(v2.leak_policy, LeakPolicy::Forbid);
        assert_eq!(v2.parent_greed, 0.0);
        assert!(!v2.promote_only_gains);
        assert_eq!(v2.ledger_falsify_margin, 0.0);
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
                    slots: std::collections::BTreeMap::from([(
                        "m".to_string(),
                        format!("instruction {name}"),
                    )]),
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

    fn probe_trace() -> Trace {
        use crate::trace::{CompId, ModelId, Span, SpanId};
        let mk = |id: u32, comp: u32, field: &str, value: &str| Span {
            id: SpanId(id),
            component: CompId(comp),
            seq: 0,
            parent: None,
            prefix: None,
            suffix: Vec::new(),
            input: serde_json::json!({"claim": "c"}).as_object().cloned(),
            model: ModelId(0),
            request_hash: 40 + u64::from(id),
            events: Vec::new(),
            raw_output: None,
            output: serde_json::json!({field: value}).as_object().cloned(),
            usage: Default::default(),
            error: None,
            eval: None,
            started_at_us: 0,
            duration_us: 0,
            complete: true,
        };
        Trace {
            components: vec!["summarize1".into(), "hop1_query".into()],
            spans: vec![
                mk(0, 0, "summary", "old summary"),
                mk(1, 1, "query", "old query"),
            ],
            ..Default::default()
        }
    }

    #[test]
    fn repair_span_output_edits_only_the_blamed_stage() {
        let trace = probe_trace();
        let (edited, previous) =
            repair_span_output(&trace, "summarize1", "summary", "better summary")
                .expect("the stage has a complete span");
        assert_eq!(previous, "old summary");
        assert_eq!(
            edited.spans[0].output.as_ref().unwrap()["summary"],
            "better summary"
        );
        // The request hash is untouched, so the edited span still SERVES on
        // replay — that is the whole counterfactual mechanism.
        assert_eq!(edited.spans[0].request_hash, trace.spans[0].request_hash);
        assert_eq!(edited.spans[1].output, trace.spans[1].output);

        assert!(repair_span_output(&trace, "no_such_stage", "summary", "x").is_none());
        let mut broken = trace.clone();
        broken.spans[0].complete = false;
        assert!(
            repair_span_output(&broken, "summarize1", "summary", "x").is_none(),
            "an incomplete span cannot serve a repair"
        );
    }

    #[test]
    fn pipeline_trace_render_names_every_stage_with_io() {
        let text = render_pipeline_trace(&probe_trace());
        assert!(text.contains("## module `summarize1`"));
        assert!(text.contains("summary: old summary"));
        assert!(text.contains("## module `hop1_query`"));
        assert!(text.contains("query: old query"));
        assert!(text.contains("claim: c"));
    }

    #[test]
    fn repair_lessons_render_into_a_ledger_section() {
        assert_eq!(Ratchet::render_repair_lessons(&[]), "");
        let lessons = vec![
            (
                2usize,
                "summarize2".to_string(),
                "summaries must name candidate bridge entities".to_string(),
                0.33,
            ),
            (
                4usize,
                "hop1_query".to_string(),
                "quote paired entities in queries".to_string(),
                0.67,
            ),
        ];
        let text = Ratchet::render_repair_lessons(&lessons);
        assert!(text.contains("counterfactually PROVEN"));
        assert!(text.contains("[gen 2 | summarize2 | recovered +0.33]"));
        assert!(text.contains("quote paired entities in queries"));
    }

    #[test]
    fn repair_knobs_default_off_and_price_into_the_step_budget() {
        let v2 = Ratchet::builder().build();
        assert_eq!(v2.slot_credit, SlotCredit::RoundRobin);
        assert_eq!(v2.repair_probes, 2);
        let base = v2.step_worst_case(5, 40);

        let mut repair = Ratchet::builder().slot_credit(SlotCredit::Repair).build();
        assert_eq!(repair.repair_attempts, 2);
        repair.repair_probes = 3;
        // Each probe worst-cases 1 base + `repair_attempts` counterfactuals.
        assert_eq!(repair.step_worst_case(5, 40), base + 3 * 3);
        repair.repair_attempts = 3;
        assert_eq!(repair.step_worst_case(5, 40), base + 3 * 4);
    }

    #[test]
    fn parallel_proposal_defaults_preserve_v2_budget_math() {
        let v2 = Ratchet::builder().build();
        assert_eq!(v2.parents_per_step, 1);
        assert_eq!(v2.proposals_per_parent, 1);
        // At 1×1 the step precheck is the measured v2 formula:
        // minibatch + 3·gate + val (fan-out = the rewrite pair).
        assert_eq!(v2.step_fanout(), 2);
        assert_eq!(v2.step_worst_case(5, 40), 5 + 3 * 5 + 40);
        assert_eq!(v2.ablate_margin(40), 3 * 40);

        let mut pxn = Ratchet::builder()
            .parents_per_step(2)
            .proposals_per_parent(3)
            .build();
        // P parents × (N mini-batches + gate rows for fan + 1) + one race.
        assert_eq!(pxn.step_worst_case(5, 40), 2 * (3 * 5 + 4 * 5) + 40);
        // N > 1 supersedes rewrite_fanout everywhere fan-out is counted.
        pxn.rewrite_fanout = 9;
        assert_eq!(pxn.step_fanout(), 3);
        assert_eq!(pxn.ablate_margin(40), (2 * 2 * 3 - 1) * 40);
    }

    #[test]
    fn temper_preset_flips_the_knobs() {
        let optimizer = temper(vec![(0, 5)]);
        assert_eq!(optimizer.parent_selection, ParentSelection::Lexicase);
        assert_eq!(optimizer.batch_weighting, BatchWeighting::Learnability);
        assert_eq!(optimizer.length_cap, Some(1500));
        assert_eq!(optimizer.variant_pairs, vec![(0, 5)]);
        // The P×N knobs stay sequential unless asked for.
        assert_eq!(optimizer.parents_per_step, 1);
        assert_eq!(optimizer.proposals_per_parent, 1);
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
    fn joint_candidates_mutate_one_slot_and_install_all() {
        use std::collections::BTreeMap;
        let seed = GEPACandidate {
            id: 0,
            instruction: "q1".into(),
            module_name: "hop1".into(),
            example_scores: Vec::new(),
            parent_id: None,
            generation: 0,
            slots: BTreeMap::from([
                ("hop1".to_string(), "q1".to_string()),
                ("summarize1".to_string(), "s1".to_string()),
            ]),
        };
        let child = seed.mutate_slot("summarize1".into(), "s1-improved".into(), 1);
        // The mutated slot changes; the sibling slot is inherited.
        assert_eq!(child.module_name, "summarize1");
        assert_eq!(child.slot_text("summarize1"), "s1-improved");
        assert_eq!(child.slot_text("hop1"), "q1");
        // The engine candidate carries the FULL assignment (both slots).
        let engine_candidate = child.engine_candidate();
        assert_eq!(engine_candidate.slots.len(), 2);
        assert_eq!(
            engine_candidate.slots["hop1"].instruction.as_deref(),
            Some("q1")
        );
        assert_eq!(
            engine_candidate.slots["summarize1"].instruction.as_deref(),
            Some("s1-improved")
        );
        // Pre-slots data (deserialized old results) falls back cleanly.
        let legacy = GEPACandidate {
            slots: BTreeMap::new(),
            ..seed.clone()
        };
        assert_eq!(legacy.slot_text("hop1"), "q1");
        assert_eq!(legacy.engine_candidate().slots.len(), 1);
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
