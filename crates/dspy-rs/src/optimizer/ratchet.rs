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
//! - `length_prior` (v6.5): a SOFT, teacher-facing length budget — a LENGTH
//!   BUDGET block in every Rewrite/Policy reflection input, a loose hard
//!   guard at `prior × length_prior_slack` through the `length_cap` path,
//!   and within-noise ranking ties broken toward the shorter candidate.
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
    ParetoView, RolloutOutcome, SCORE_EPS, ScoreMatrix,
};
use crate::optimizer::evidence::{BandBets, PairedEvidence};
use crate::optimizer::forage::{
    CheckExtractor, ForageDigest, ForageLedger, TeacherContext, forage_tools,
};
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

/// Propose the ANSWER POLICY layer of a module's instruction.
///
/// You are one step of an evolutionary search over the system instruction of
/// one module inside an LLM pipeline. This step edits the answer POLICY, not
/// topic facts. Study the per-example scores, the grader's feedback and its
/// advice, and find the failure patterns that REPEAT across examples:
/// something every answer must include, name, or cite (and in what exact
/// form and position), the required shape and order of an answer, what must
/// be left out, and how to answer when the evidence does not establish the
/// detail. Then write the improved instruction: the current instruction with
/// a short, explicit ANSWER POLICY section. State each recurring requirement
/// in THREE reinforcing forms, because a requirement stated once is obeyed
/// only part of the time: (1) a numbered rule; (2) an exact RESPONSE
/// TEMPLATE showing the required line(s) in their position in the answer,
/// with a placeholder for the input-specific value; (3) a short FINAL
/// CHECKLIST the module must satisfy before answering, repeating the
/// requirement. The rules MUST generalize to inputs on topics that
/// do not appear in these examples: tell the module HOW to determine the
/// required element from its own evidence for any input (for instance,
/// "cite the identifier of the page the answer was drawn from"), and list
/// identifiers seen in the feedback only as examples of the required form
/// ("for example …"), never as a fixed mapping from topic to value. Keep
/// everything in the current instruction that works; do not add facts
/// about specific topics; never re-propose a falsified idea from the trial
/// ledger. Also state, in one line of at most 25 words, the single
/// hypothesis your edit is betting on.
#[derive(Signature, Clone, Debug)]
struct ProposePolicy {
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

/// Condense a module's appended teachings without losing any of them.
///
/// You are one step of an evolutionary search over the system instruction of
/// one module inside an LLM pipeline. The current instruction is a frozen
/// base followed by an appended teachings section this search has
/// accumulated. Rewrite ONLY the appended section into the densest
/// equivalent form: deduplicate, merge overlapping entries, convert prose to
/// telegraphic single-line entries, and where several example-specific facts
/// instantiate one general rule, state the rule (keeping any specifics the
/// rule does not imply). Every teaching currently present must remain stated
/// or derivable — dropping information is failure; shrinking its expression
/// is the goal. Reproduce the base portion VERBATIM, every character
/// unchanged. Also state, in one line of at most 25 words, what you merged
/// or generalized.
#[derive(Signature, Clone, Debug)]
struct ProposeCondense {
    /// The module's input and output fields with their descriptions.
    #[input]
    task_description: String,

    /// The instruction currently used by the module (frozen base + appended
    /// teachings).
    #[input]
    current_instruction: String,

    /// Edits this search has measured to help, with their deltas.
    #[input]
    validated_lessons: String,

    /// The instruction with the appended section condensed and the base
    /// intact.
    #[output]
    improved_instruction: String,

    /// One line (max 25 words): what was merged or generalized.
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
    /// FORAGE v5.2: compress the appended teachings' REPRESENTATION without
    /// losing any teaching (dedupe, merge, telegraph, generalize) — the
    /// anchored-accretion replacement for Simplify, which optimizes brevity
    /// by dropping content and is measured to strip fact bundles.
    Condense,
    /// v6 HARVEST: a bounded edit of the answer-policy layer only (what every
    /// answer must include/cite and in what exact form, shape, omissions,
    /// unknowns handling), from the repeated patterns in the feedback.
    Policy,
}

impl RatchetOp {
    pub fn name(self) -> &'static str {
        match self {
            Self::Rewrite => "rewrite",
            Self::Patch => "patch",
            Self::Simplify => "simplify",
            Self::Merge => "merge",
            Self::Condense => "condense",
            Self::Policy => "policy",
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

/// Which examples a freshly proposed child is screened on before it may
/// enter the pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScreenRule {
    /// RATCHET default: a disjoint held-out batch B, noise-aware (`gate_z`).
    /// B is drawn unsolved-biased, i.e. from columns where the parent
    /// already fails — where the child's new content is usually
    /// irrelevant. Measured on the DocsQA cell: 62% of promotions
    /// net-negative against their own parent, ~25% of neutral children
    /// killed on noise at n = 3 (forensics F4 D1, 2026-09-01).
    HeldOutB,
    /// v6 HARVEST: the child is scored on the very reflection chunk it was
    /// written from (the parent's cells there are already paid for) and
    /// passes iff its summed score strictly beats the parent's — pip-gepa's
    /// acceptance rule, measured 0/33 negatives at +.11 per column on the
    /// DocsQA b2000 run. It verifies that the edit did what it was meant
    /// to do on its target examples; generalization is then measured by
    /// the shared ranking slice (never a kill), and honesty by the ship
    /// certification. `gate_size` should be 0 under this rule (batch B is
    /// unused); `rewrite_fanout` siblings apply only under `HeldOutB`.
    OnTargetA,
}

/// What the leak guard does with example strings a proposal quotes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LeakPolicy {
    /// Default: quoting a guarded example string triggers a retry, then
    /// abandons the trial. This enforces an instruction-only data-use
    /// policy. Retaining permitted training examples is a modeling choice,
    /// not itself contamination of a disjoint evaluation set.
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
    /// The proposal still broke the anchored-accretion base after a retry.
    AnchorSkipped,
    /// Killed on the held-out gate batch (child ≤ parent).
    GateKilled { parent_gate: f64, child_gate: f64 },
    /// Killed mid-race: paired deficit vs the parent exceeded `race_z` SEs.
    RaceKilled { columns_evaluated: usize, paired_delta: f64 },
    /// Finished the full race with a negative paired delta and was refused
    /// pool entry (`promote_only_gains`): fully measured, not compounded.
    Faded { paired_delta: f64 },
    /// Survived the full race and joined the pool.
    Promoted { paired_delta: f64, candidate_id: usize },
    /// ASSAY re-audit eviction: the member's cumulative paired-vs-seed harm
    /// e-process certified at `race_alpha` — its content is genuinely worse
    /// than the seed, pruned mid-run instead of at certification.
    Audited { candidate_id: usize, columns: usize, paired_delta: f64 },
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

/// Per-clause evidence from lineage diffs (v6.4; SEPO-style edit-effect
/// lineage, ZERO extra rollouts). Every promoted child is scored on the same
/// shared ranking slice as its parent, so the parent→child per-column deltas
/// are attributed to the clauses the child added (and, sign-flipped, to the
/// ones it removed) and accumulated per clause across the whole run. Under
/// HARVEST the slice is the full validation set, so the deltas are directly
/// comparable run-wide.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClauseCredit {
    /// Stable clause id: FNV-1a of the normalized text (list markers, case,
    /// whitespace, and trailing punctuation folded), so a renumbered or
    /// re-bulleted rule keeps its id across parent and child.
    pub id: String,
    /// The clause text as first seen.
    pub text: String,
    /// Generation of the first promoted candidate carrying the clause
    /// (`0` = present in the seed).
    pub added_at_gen: usize,
    /// Promoted candidates (seed included) whose text carries the clause.
    pub times_present: usize,
    /// Attribution events: promoted children that added or removed it.
    pub times_credited: usize,
    /// Sum of attributed per-column deltas (shared evenly among the
    /// clauses one child added together).
    pub delta_sum: f64,
    /// Ranking columns the crediting children gained on.
    pub cols_won: usize,
    /// Ranking columns the crediting children lost on.
    pub cols_lost: usize,
    /// EMA (β = 0.5) of the attributed per-column mean delta — the signed
    /// evidence the Occam pruning orders by.
    pub ema: f64,
    /// Whether the shipped winner carries the clause.
    pub in_champion: bool,
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
    /// Pool members evicted by the ASSAY re-audit (cumulative paired-vs-seed
    /// harm certified at `race_alpha`; `reaudit_budget_frac` > 0 only).
    #[serde(default)]
    pub audit_evicted: usize,
    /// Rollouts spent by the re-audit passes.
    #[serde(default)]
    pub audit_rollouts: usize,
    /// Pins the endgame assay probed (`pin_assay` only).
    #[serde(default)]
    pub pins_tested: usize,
    /// Pins that passed stopped e-BH and were composed into the champion.
    #[serde(default)]
    pub pins_kept: usize,
    /// Rollouts spent by the pin-ablation probes.
    #[serde(default)]
    pub pin_probe_rollouts: usize,
    /// Per-pin assay evidence: `(pin, provenance_cols_probed, mean_delta,
    /// e_gain)`, in the champion's appended order.
    #[serde(default)]
    pub pin_assay_report: Vec<(String, usize, f64, f64)>,
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
    /// The soft length prior the run steered the teacher with
    /// (`length_prior`), if any (v6.5).
    #[serde(default)]
    pub length_prior: Option<usize>,
    /// The hard cap proposals were actually held to: `length_cap`, or under
    /// a prior the smaller of that and `prior × length_prior_slack`.
    #[serde(default)]
    pub length_cap_in_force: Option<usize>,
    /// Reflection retries triggered by the anchored-accretion guard.
    #[serde(default)]
    pub anchor_retries: usize,
    /// Proposals abandoned because they still broke the anchor after a retry.
    #[serde(default)]
    pub anchor_skipped: usize,
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
    /// Lineage-diff clause credit (v6.4): the shipped winner's clauses in
    /// document order (`in_champion`), then every other clause a promoted
    /// child carried, strongest evidence first. Empty when no child was
    /// promoted. See [`RatchetResult::clause_credit_report`].
    #[serde(default)]
    pub clause_credits: Vec<ClauseCredit>,
    /// Post-certification Occam pruning (`prune_after_certify` only):
    /// champion chars before pruning (`0` when the phase did not run).
    #[serde(default)]
    pub prune_chars_before: usize,
    /// Champion chars after pruning (equals `prune_chars_before` when
    /// nothing was removed or the pruned prompt failed re-certification).
    #[serde(default)]
    pub prune_chars_after: usize,
    /// Clauses the shipped winner lost to the pruning.
    #[serde(default)]
    pub prune_removed: usize,
    /// Rollouts the pruning spent: bundle probes plus the re-certification.
    #[serde(default)]
    pub prune_rollouts: usize,
    /// Re-certification verdict of the pruned prompt vs the seed
    /// (`None` = no bundle survived the race, or the remaining certification
    /// budget ran out unproven — the original certified champion shipped).
    #[serde(default)]
    pub prune_certified: Option<bool>,
}

impl RatchetResult {
    /// Human-readable lineage-credit table: the shipped winner's clauses in
    /// document order with their accumulated evidence (tag `+`/`0`/`-` by
    /// EMA sign, `seed` for never-credited seed clauses), then a one-line
    /// tally of everything else the run saw, and the Occam-pruning summary
    /// when that phase ran. Empty when no credit was recorded.
    pub fn clause_credit_report(&self) -> String {
        use std::fmt::Write as _;
        if self.clause_credits.is_empty() {
            return String::new();
        }
        let carried: Vec<&ClauseCredit> =
            self.clause_credits.iter().filter(|credit| credit.in_champion).collect();
        let mut text = format!(
            "lineage credit (SEPO-style, zero extra rollouts): {} clauses seen, {} in the shipped winner\n",
            self.clause_credits.len(),
            carried.len()
        );
        for credit in &carried {
            let tag = if credit.times_credited == 0 {
                if credit.added_at_gen == 0 { "seed" } else { "n/a" }
            } else if credit.ema > SCORE_EPS {
                "+"
            } else if credit.ema < -SCORE_EPS {
                "-"
            } else {
                "0"
            };
            let excerpt: String = credit.text.chars().take(96).collect();
            let _ = writeln!(
                text,
                "  [{tag:>4} ema {:+.4} sum {:+.3} won {} lost {} credited {} present {} gen {}] {excerpt}",
                credit.ema,
                credit.delta_sum,
                credit.cols_won,
                credit.cols_lost,
                credit.times_credited,
                credit.times_present,
                credit.added_at_gen
            );
        }
        let rest: Vec<&ClauseCredit> =
            self.clause_credits.iter().filter(|credit| !credit.in_champion).collect();
        if !rest.is_empty() {
            let positive = rest.iter().filter(|c| c.ema > SCORE_EPS).count();
            let negative = rest.iter().filter(|c| c.ema < -SCORE_EPS).count();
            let _ = writeln!(
                text,
                "  not carried: {} clauses ({positive} positive, {} zero, {negative} negative ema)",
                rest.len(),
                rest.len() - positive - negative
            );
        }
        if self.prune_rollouts > 0 || self.prune_removed > 0 {
            let _ = writeln!(
                text,
                "  occam prune: {} -> {} chars, {} clauses removed, {} rollouts, re-certified {:?}",
                self.prune_chars_before,
                self.prune_chars_after,
                self.prune_removed,
                self.prune_rollouts,
                self.prune_certified
            );
        }
        text.trim_end().to_string()
    }
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

    /// v6 HARVEST: how children are screened before entering the pool.
    /// [`ScreenRule::HeldOutB`] (default) preserves measured v2–v5
    /// behavior; [`ScreenRule::OnTargetA`] screens on the reflection chunk.
    #[builder(default = ScreenRule::HeldOutB)]
    pub screen_rule: ScreenRule,

    /// v6 HARVEST: restrict the operator portfolio to Rewrite. Patch,
    /// Simplify, Condense and Merge are content-loss or content-neutral
    /// channels; in the accretive regime they consumed ~2/3 of proposal
    /// slots (F4 D2) while pip-gepa is 100% accreting rewrites. `false`
    /// (default) keeps the measured bandit portfolio.
    #[builder(default = false)]
    pub rewrite_only: bool,

    /// v6 HARVEST: tell the Rewrite teacher to consider GLOBAL answer-policy
    /// edits (what every answer must include or cite, structure, what to
    /// omit, how to answer without evidence) alongside content edits, and
    /// to read the per-example scores and grader advice for patterns that
    /// repeat. Motivated by the DocsQA forensics: 100% of pip-gepa's holdout
    /// gain was one such policy (a `Sources:` citation line), which our
    /// content-steered operators never proposed. `false` (default) leaves
    /// the operator prompts as measured.
    #[builder(default = false)]
    pub policy_channel: bool,

    /// v6.1 HARVEST: share of proposal slots given to the dedicated
    /// [`RatchetOp::Policy`] operator (a bounded edit of the answer-policy
    /// layer) when `policy_channel` is on. The first live micro-arm showed
    /// that a hint inside the facts Rewrite loses to the facts-first task
    /// override: 8 of 12 hypotheses led with "add pinned facts" and
    /// citations appeared only as per-topic lines, never as the mandatory
    /// footer template that carries the measured gain. `0.0` (default)
    /// keeps the operator off.
    #[builder(default = 0.0)]
    pub policy_share: f64,

    /// Whether gate (screen) kills enter the ledger's falsified digest —
    /// "do NOT re-propose". A screen kill at n = 3–5 is mostly noise, and
    /// under the on-target rule it only says the edit did not help its own
    /// examples this time; feeding it back as falsified steered the
    /// teacher away from ~10 true fact bundles per run (F4 D7). `true`
    /// (default) preserves measured behavior.
    #[builder(default = true)]
    pub ledger_gate_kills: bool,

    /// v6 HARVEST: fold a child's screen rollouts into the FORAGE ledger
    /// only when the child is promoted, so a killed child's harvest never
    /// retires a column's open checks (F4 D8). `false` (default) preserves
    /// measured behavior (every paid rollout folds in immediately).
    #[builder(default = false)]
    pub forage_fold_on_promotion: bool,

    /// Validation columns in the shared scout slice under
    /// [`AcceptMode::Gate`] — the only val evaluation accepted children
    /// get, used for pool ranking and champion screening (every candidate
    /// shares the same slice, so ranks stay comparable).
    #[builder(default = 12)]
    pub scout_size: usize,

    /// Reduce P×N near the rollout limit while retaining whole reflection
    /// batches and complete shared validation coverage. Supported only for
    /// on-target, gate-accept search without repair probes or rewrite fan-out.
    /// Other configurations retain the original conservative step precheck.
    #[builder(default = false)]
    pub adaptive_step_size: bool,

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
    /// trainset columns at the end. This separates screening from search,
    /// but the screen's noisy argmax still has selection bias. `0` disables.
    /// Under `certify_ship` this stays the
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
    /// The claim concerns the population represented by the certification
    /// columns under the evidence process assumptions. Reserving training
    /// rows does not certify performance on an out-of-distribution target.
    #[builder(default = false)]
    pub certify_ship: bool,

    /// Certification columns reserved out of the trainset (never seen by
    /// any batch, race, or champion screen). The e-process stops early on
    /// big effects. Size this reserve using power at the smallest gain worth
    /// shipping and the observed paired variance; 150 columns can have low
    /// power for small gains. Only read when `certify_ship` is set.
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

    /// E5a band mixture for the certification wallets
    /// (`docs/ratchet-v5-escrow-design.md`): fixed Kelly bets for the
    /// anticipated effect band, mixed with the aGRAPA wallet, so growth
    /// starts at column 1 instead of after the plug-in warms up — the
    /// repair for the measured +.03-class conservative miss (forage4:
    /// argmax +.0315 SIG on holdout, certification e 1.07 at 90 cols,
    /// refused). Validity exact (a convex mixture of test
    /// supermartingales is a test supermartingale). `None` (default) =
    /// the v4 aGRAPA-only wallet, byte-identical.
    pub certify_band: Option<BandBets>,

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

    /// Permit training examples in proposals without declaring the task a
    /// corpus-fact task or encouraging fact accumulation. The final evaluation
    /// must remain disjoint. Unlike `allow_memorization`, this is task-neutral.
    #[builder(default = false)]
    pub allow_training_examples: bool,

    /// FORAGE (v5): how the Rewrite teacher sees the run's accumulated
    /// evidence. [`TeacherContext::Minibatch`] (default) preserves measured
    /// v2–v4 behavior — one reflection chunk of feedback per proposal.
    /// [`TeacherContext::Digest`] appends the bounded contrastive digest of
    /// EVERY paid rollout (open checks + flip contrasts) to the teacher's
    /// feedback; [`TeacherContext::Tools`] additionally hands the Rewrite
    /// call three read-only drill-down tools over the step's ledger
    /// snapshot. Motivation and literature: `optimizer/forage.rs` header.
    #[builder(default = TeacherContext::Minibatch)]
    pub teacher_context: TeacherContext,

    /// FORAGE digest size cap in characters. The digest is deterministic
    /// engine-computed context, so the cap is a token-budget knob, not a
    /// validity knob.
    #[builder(default = 15_000)]
    pub forage_chars: usize,

    /// Max open-check lines rendered in the FORAGE digest.
    #[builder(default = 60)]
    pub forage_open_cap: usize,

    /// Max flip-contrast lines rendered in the FORAGE digest.
    #[builder(default = 12)]
    pub forage_contrast_cap: usize,

    /// Harness-supplied verifiable-check extractor: parses one rollout's
    /// metric feedback into the discrete checks it failed (the task's
    /// feedback format is harness knowledge — same seam as `certify_cols`).
    /// `None` degrades the digest to weakest-column feedback excerpts.
    pub check_extractor: Option<CheckExtractor>,

    /// FORAGE v5.1 anchored accretion: every proposal must reproduce the
    /// leaf's compile-time base instruction VERBATIM as its opening —
    /// additions go after it, so style edits to the base are structurally
    /// impossible (checked in the proposal retry loop, like leaks).
    /// Simplify is ineligible under anchoring (Occam pressure stripping
    /// fact bundles is a measured refusal mechanism on corpus-fact tasks).
    /// Motivation: the first live FORAGE arm harvested substantively
    /// correct facts but its teacher bundled them with conciseness
    /// meta-rules; on a coverage-dominant judge the bundle certified
    /// NEGATIVE (−.0265, e .33) and transferred +.013 n.s. — style
    /// contamination, not content, was the regression channel. Note: the
    /// clause-ablation finale is anchor-unaware; leave `ablate_budget` at
    /// 0 when anchoring. `false` (default) preserves measured behavior.
    #[builder(default = false)]
    pub anchored_accretion: bool,

    /// FORAGE v5.2: under `anchored_accretion`, also allow the appended
    /// section to carry a small set of high-value BEHAVIORAL rules
    /// (evidence-verification discipline, honest-refusal templates,
    /// avoiding unsupported claims) alongside the FACTS. Motivation: the
    /// gepa2000 DocsQA winner's edge is two-channel — broad behavior rules
    /// PLUS dense facts — and the plain anchored override bans the first
    /// channel entirely (its round-2 arm transferred −.0016 n.s. while
    /// gepa held +.065). Conciseness/scope-limiting rules stay banned
    /// (measured to tax a coverage-dominant judge). `false` (default)
    /// preserves round-2 facts-only behavior.
    #[builder(default = false)]
    pub anchor_allow_rules: bool,

    /// ASSAY Stage 1 (v5.3) re-audit: fraction of each step's rollout spend
    /// re-invested, at the head of the next generation, in re-evaluating
    /// standing pool members on rotating never-yet-drawn validation columns
    /// — the cheapest faithful copy of the measured cause of the gepa gap
    /// (per-sentence evidence: gepa re-evaluates every accepted candidate
    /// on full val every accepted iteration, ~50× the evidence our one
    /// 3–5-column gate provides; five preregistered DocsQA arms eliminated
    /// content bandwidth, style, and content class as explanations —
    /// docs/handoff-assay-verified-accretion.md). Each audited member keeps
    /// a cumulative paired-vs-seed [`PairedEvidence`]; pool ranking and
    /// champion screening switch from scout-slice means to the cumulative
    /// evidence mean, and a member whose HARM certifies at `race_alpha` is
    /// evicted mid-run (recorded as [`TrialOutcome::Audited`], so the
    /// ledger teaches the teacher). `0.0` (default) preserves measured
    /// behavior byte-for-byte.
    #[builder(default = 0.0)]
    pub reaudit_budget_frac: f64,

    /// Validation columns per re-audit block (one batched member+seed
    /// evaluation per block; seed cells cache-hit after the first visit).
    #[builder(default = 8)]
    pub reaudit_block: usize,

    /// ASSAY Stage 2 (v5.3): per-pin verified accretion. At endgame, the
    /// champion's appended section is parsed into PINS (one teaching line
    /// each); every pin is ablation-probed — champion vs champion−pin,
    /// paired, on a draw of the columns whose recorded checks motivated it
    /// (the ForageLedger knows) — feeding a per-pin [`PairedEvidence`]. The
    /// bundle is then composed from the pins whose e-processes pass stopped
    /// e-BH at `pin_fdr_alpha` (Wang & Ramdas, FDR under arbitrary
    /// dependence), and the frozen certification still takes the ship
    /// decision on the composed bundle. This is the removal channel
    /// anchored accretion gave up: probing a pin on its OWN columns turns a
    /// +.01-pooled effect into a +.5-conditional one, so true pins certify
    /// in a handful of targeted rollouts while wrong/stale AND inert pins
    /// fail to earn their place. Requires `certify_ship` and a
    /// `check_extractor`; `false` (default) preserves measured behavior.
    #[builder(default = false)]
    pub pin_assay: bool,

    /// FDR level for the stopped e-BH selection over per-pin e-values.
    #[builder(default = 0.1)]
    pub pin_fdr_alpha: f64,

    /// Max provenance columns probed per pin (a pin with zero attributable
    /// columns is dropped as unfundable — no evidence, no place).
    #[builder(default = 20)]
    pub pin_probe_cols: usize,

    /// Rollout reserve for the pin assay (held with the certification
    /// reserve once a child has been promoted).
    #[builder(default = 400)]
    pub pin_probe_budget: usize,

    /// Fresh paired draws per provenance column. Live rollouts are
    /// stochastic (student temperature, judge), so repeated FRESH rollouts
    /// on the same column are independent paired observations — the power
    /// fix for tasks whose checks are near-unique per question (a pin may
    /// have only 2–6 attributable columns). `1` (default) keeps the
    /// batched cache-friendly path; above 1, every draw forces an uncached
    /// rollout via the probe path. Leave at 1 in a deterministic world:
    /// repeats there would pseudo-replicate identical deltas.
    #[builder(default = 1)]
    pub pin_probe_repeats: usize,

    /// E3 SUBSET SCREEN (`docs/ratchet-v5-escrow-design.md`, ContextCite —
    /// Cohen-Wang et al., NeurIPS 2024, arXiv:2409.00729): replaces the
    /// per-pin ablation probes with JOINT credit assignment. Draw this many
    /// random pin-subsets (each pin kept w.p. ½), evaluate each composed
    /// variant once on one shared provenance-weighted column slice, and fit
    /// a lasso of mean score on the inclusion vectors — every pin's
    /// marginal-effect estimate from one shared pool of rollouts, instead
    /// of one starved e-process per pin (forage4 measured the starvation:
    /// 10 of 46 pins drew 0 columns, most 1–2, and e-BH at m=46 needed
    /// e ≥ 230). The composed bundle keeps positive-coefficient pins; the
    /// FROZEN certification still takes the ship decision, so the screen
    /// needs no statistical guarantee of its own — it only has to point
    /// the certifier at a better bundle. `0` (default) = stage-2 per-pin
    /// probes, byte-identical. Requires `pin_assay`.
    #[builder(default = 0)]
    pub pin_screen_draws: usize,

    /// Columns in the shared screen slice (provenance-weighted: columns
    /// named by the most pins first, topped up at random). Screen cost =
    /// `pin_screen_draws × pin_screen_cols` rollouts, capped by
    /// `pin_probe_budget`.
    #[builder(default = 20)]
    pub pin_screen_cols: usize,

    /// Rank fallback for the screen: when the noise floors keep fewer
    /// pins than this, top up by coefficient rank among positive, funded
    /// pins. The floors are honest about per-pin certainty, but the
    /// composed challenger only needs ENRICHMENT — the frozen
    /// certification takes the ship decision — so at live noise (where
    /// a pin's pooled effect ≈ conditional/slice sits near the floor)
    /// ranking recovers a true-dense bundle the floors would starve.
    /// `0` (default) = floors only.
    #[builder(default = 0)]
    pub pin_screen_min_keep: usize,

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

    /// Soft length PRIOR (v6.5, teacher-facing): the target instruction
    /// size in chars. When set, every Rewrite and Policy reflection input
    /// ends with a LENGTH BUDGET block (current size, budget, hard limit,
    /// and how to fold rules instead of dropping them), the shortening
    /// retry names the target, the hard `length_cap` path becomes a loose
    /// guard at `length_prior × length_prior_slack` (an explicit
    /// `length_cap` still wins when smaller), and the pool rankings on the
    /// shared slice break within-`selection_z` ties toward the shorter
    /// candidate. Live motivation (DocsQA, 2026-09-02): HARVEST winners
    /// grew to 8–13k chars while a 3.8k hand-written prompt carried the
    /// whole gain, and every score-0 rollout was a context-length overflow
    /// — size became a correctness cost. TEMPER's 1,500-char HARD cap was
    /// falsified because its size killed the best lineage's rewrites
    /// (showdown handoff §1); the prior steers the teacher instead of
    /// rejecting its proposals. `None` (default) changes nothing: the
    /// teacher sees byte-identical inputs and every ranking is unchanged.
    pub length_prior: Option<usize>,

    /// Loose hard guard above the prior: under a `length_prior` the
    /// effective `length_cap` is `prior × slack` (rounded). Only read when
    /// `length_prior` is set.
    #[builder(default = 1.25)]
    pub length_prior_slack: f64,

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

    /// Post-certification Occam pruning (v6.4, lineage-credit ablation;
    /// `docs/handoff-harvest-v6.md` §5 items 2 and 4): once the champion is
    /// certified vs the seed, its clauses whose lineage-credit EMA is ≤ 0
    /// (see [`ClauseCredit`]; seed clauses are never touched) are removed
    /// as ONE bundle and raced paired against the full champion on the
    /// fresh∪variant selection columns — the ablation finale's columns; the
    /// shared ranking slice when there are none — with one bisection on a
    /// loss. The surviving pruned prompt must then RE-CERTIFY vs the seed
    /// with whatever certification budget the first pass left; if it does
    /// not, the original certified champion ships and the attempt is
    /// reported (`prune_*` on the result). Requires `certify_ship`.
    #[builder(default = false)]
    pub prune_after_certify: bool,

    /// Rollout budget reserved for the pruning probes (only read when
    /// `prune_after_certify` is set; held back alongside the certification
    /// budget once a child is promoted). At most three probes run — the
    /// bundle and its two halves — so `3 × champion_fresh` covers it. The
    /// re-certification spends the first certification's leftover, not
    /// this.
    #[builder(default = 120)]
    pub prune_budget: usize,

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
        // Up to 200 fresh pairs; actual power depends on effect, variance,
        // available columns and the challenger alpha split. Big effects
        // stop early and refund the difference.
        .certify_budget(400)
        // e ≥ 10, anytime-valid — still a far stronger ship guarantee than
        // an unreplicated argmax; the strict 0.05 default suits big-win
        // tasks where evidence is cheap.
        .certify_alpha(0.1)
        .ledger_falsify_margin(0.02)
        .build()
}

/// Builds a RATCHET v6 "HARVEST" preset — the accretive search re-tuned by
/// the 2026-09-01 forensics (scratchpad reports F1/F2/F4; memory
/// `docsqa-cite-term-2026-09-01`): pip-gepa's DocsQA edge was one global
/// answer policy found by an un-steered rewrite teacher, verified on its
/// own reflection batch, ranked on the full shared validation slice, and
/// never killed. HARVEST = that loop plus RATCHET's pawl:
///
/// - `ScreenRule::OnTargetA` (child vs parent on its reflection chunk,
///   strict `>`), batch B off;
/// - `AcceptMode::Gate` with the FULL validation slice as the shared
///   ranking slice (`scout_size = usize::MAX`) — no race, no kills, no
///   fades; pool ranking and Pareto coverage on identical columns;
/// - Rewrite-only portfolio, policy channel on, gate kills kept out of the
///   falsified digest, reflection minibatch 5 (pip-gepa's);
/// - parent sampling 70% incumbent lineage (full-slice argmax) / 30%
///   Pareto coverage: the L2 selection-theory review (2026-09-01) measured
///   greedy full-slice arms accumulating 1.5–2× more transferable content
///   than pure coverage sampling across seven friction regimes, and the
///   shared ranking avoids comparing candidates on unequal columns, but
///   adaptive selection can still overfit that shared validation slice;
/// - sequential paired ship certification vs the seed on reserved or
///   harness-supplied columns (`certify_ship`, alpha 0.1) — the honesty.
///
/// Set `prompt_model`, `max_rollouts`, and for corpus-fact tasks
/// `allow_memorization = true`. Budget note: certification and the
/// champion screen are held back from the search budget once a child is
/// promoted; size `max_rollouts` = search budget + `certify_budget` +
/// `champion_k × champion_fresh`.
pub fn harvest() -> Ratchet {
    let mut optimizer = Ratchet::builder()
        .num_iterations(150)
        .minibatch_size(5)
        .parents_per_step(2)
        .proposals_per_parent(2)
        .screen_rule(ScreenRule::OnTargetA)
        .accept_mode(AcceptMode::Gate)
        .scout_size(usize::MAX)
        .rewrite_only(true)
        .policy_channel(true)
        .policy_share(0.5)
        .ledger_gate_kills(false)
        .forage_fold_on_promotion(true)
        .parent_greed(0.7)
        .race_rule(RaceRule::Eprocess)
        .champion_k(4)
        .certify_ship(true)
        .certify_budget(400)
        .certify_alpha(0.1)
        .ledger_falsify_margin(0.02)
        .build();
    optimizer.gate_size = Some(0);
    optimizer
}

/// Experimental HARVEST preset for validation-selected prompt search.
///
/// Returns the validation winner, with no independent certification claim.
/// All training rows and the rollout budget remain available to search, and
/// smaller complete proposal batches can use the budget tail. Training-text
/// inclusion is permitted, as in reference GEPA; evaluation examples must
/// remain outside the optimizer's train and validation sets.
pub fn harvest_search() -> Ratchet {
    let mut optimizer = harvest();
    optimizer.certify_ship = false;
    optimizer.certify_reserve = 0;
    optimizer.certify_budget = 0;
    optimizer.champion_fresh = 0;
    optimizer.selection_z = 0.0;
    optimizer.adaptive_step_size = true;
    optimizer.rewrite_fanout = 1;
    optimizer.allow_training_examples = true;
    optimizer
}

/// A JSON value as prompt text: bare strings unquoted, everything else
/// serialized.
/// Coordinate-descent lasso for the E3 subset screen: regresses draw-mean
/// score on standardized pin-inclusion vectors; λ picked by MSE on a 25%
/// holdout of draws, then refit on all draws at the winner. Returns, per
/// pin, `(coefficient, keep_floor)` on the raw score scale — the floor is
/// 1.5 residual standard errors, so keeping demands a pin beat its own
/// estimation noise. Empty when there are too few draws to fit (< 8).
fn lasso_screen(x_rows: &[Vec<f64>], y: &[f64], rng: &mut StdRng) -> Vec<(f64, f64)> {
    let k = x_rows.len();
    if k < 8 || y.len() != k || x_rows[0].is_empty() {
        return Vec::new();
    }
    let m = x_rows[0].len();
    let mut mean_x = vec![0.0f64; m];
    for row in x_rows {
        for j in 0..m {
            mean_x[j] += row[j];
        }
    }
    for v in &mut mean_x {
        *v /= k as f64;
    }
    let mut sd_x = vec![0.0f64; m];
    for row in x_rows {
        for j in 0..m {
            let d = row[j] - mean_x[j];
            sd_x[j] += d * d;
        }
    }
    for v in &mut sd_x {
        *v = (*v / k as f64).sqrt().max(1e-9);
    }
    let y_mean = y.iter().sum::<f64>() / k as f64;
    let xs: Vec<Vec<f64>> = x_rows
        .iter()
        .map(|row| (0..m).map(|j| (row[j] - mean_x[j]) / sd_x[j]).collect())
        .collect();
    let yc: Vec<f64> = y.iter().map(|v| v - y_mean).collect();

    let mut order: Vec<usize> = (0..k).collect();
    order.shuffle(rng);
    let held = (k / 4).max(2);
    let holdout: Vec<usize> = order[..held].to_vec();
    let fit_rows: Vec<usize> = order[held..].to_vec();

    let fit_lasso = |rows: &[usize], lambda: f64, warm: &[f64]| -> Vec<f64> {
        let n = rows.len() as f64;
        let mut beta = warm.to_vec();
        let mut resid: Vec<f64> = rows
            .iter()
            .map(|&i| yc[i] - (0..m).map(|j| xs[i][j] * beta[j]).sum::<f64>())
            .collect();
        for _ in 0..200 {
            let mut max_move = 0.0f64;
            for j in 0..m {
                let mut rho = 0.0;
                let mut z = 0.0;
                for (r, &i) in rows.iter().enumerate() {
                    rho += xs[i][j] * (resid[r] + xs[i][j] * beta[j]);
                    z += xs[i][j] * xs[i][j];
                }
                if z <= 0.0 {
                    continue;
                }
                let rho = rho / n;
                let z = z / n;
                let updated = (rho.abs() - lambda).max(0.0) * rho.signum() / z;
                let delta = updated - beta[j];
                if delta != 0.0 {
                    for (r, &i) in rows.iter().enumerate() {
                        resid[r] -= xs[i][j] * delta;
                    }
                    beta[j] = updated;
                    max_move = max_move.max(delta.abs());
                }
            }
            if max_move < 1e-7 {
                break;
            }
        }
        beta
    };

    let nf = fit_rows.len() as f64;
    let lambda_max = (0..m)
        .map(|j| {
            (fit_rows.iter().map(|&i| xs[i][j] * yc[i]).sum::<f64>() / nf).abs()
        })
        .fold(0.0f64, f64::max)
        .max(1e-9);
    let mut best_mse = f64::INFINITY;
    let mut best_beta = vec![0.0f64; m];
    let mut best_lambda = lambda_max;
    let mut warm = vec![0.0f64; m];
    for step in 1..=6 {
        let lambda = lambda_max * 0.5f64.powi(step);
        warm = fit_lasso(&fit_rows, lambda, &warm);
        let mse = holdout
            .iter()
            .map(|&i| {
                let pred: f64 = (0..m).map(|j| xs[i][j] * warm[j]).sum();
                let err = yc[i] - pred;
                err * err
            })
            .sum::<f64>()
            / holdout.len() as f64;
        if mse < best_mse {
            best_mse = mse;
            best_beta = warm.clone();
            best_lambda = lambda;
        }
    }
    let all: Vec<usize> = (0..k).collect();
    let beta = fit_lasso(&all, best_lambda, &best_beta);
    // Per-pin keep floors: 1.5 residual standard errors on the raw scale —
    // a pin must beat its own estimation noise, not a magic constant, so
    // the rule transfers across worlds with different score scales.
    let resid_sd = (all
        .iter()
        .map(|&i| {
            let pred: f64 = (0..m).map(|j| xs[i][j] * beta[j]).sum();
            let err = yc[i] - pred;
            err * err
        })
        .sum::<f64>()
        / k as f64)
        .sqrt();
    // 2.0 rather than the orthogonal-design 1.5: random binary designs at
    // the draws/pins ratios the budget affords (~1.5–2.5) inflate the true
    // standard errors by roughly n/(n−m), and the floor must cover it.
    (0..m)
        .map(|j| {
            (
                beta[j] / sd_x[j],
                2.0 * resid_sd / ((k as f64).sqrt() * sd_x[j]),
            )
        })
        .collect()
}

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

// ---------------------------------------------------------------------------
// Lineage-diff clause credit (v6.4)
// ---------------------------------------------------------------------------

/// EMA weight of the newest attribution event in [`ClauseCredit::ema`].
const CLAUSE_EMA_BETA: f64 = 0.5;

/// Folds a clause to its identity: leading list markers (`-`, `*`, `•`,
/// `1.`, `1)`, `(1)`, `a.`, `a)`) stripped, case and whitespace folded,
/// trailing `.,;:` dropped. Empty for a marker-only clause.
fn normalize_clause(clause: &str) -> String {
    let mut text = clause.trim();
    loop {
        let start = text.trim_start();
        let stripped = if let Some(rest) = start.strip_prefix(['-', '*', '•', '–', '—']) {
            Some(rest)
        } else if let Some(rest) = start.strip_prefix('(') {
            let digits = rest.chars().take_while(|c| c.is_ascii_digit()).count();
            match (digits, rest[digits..].strip_prefix(')')) {
                (1..=3, Some(tail)) => Some(tail),
                _ => None,
            }
        } else {
            let digits = start.chars().take_while(|c| c.is_ascii_digit()).count();
            let marker = if digits > 0 && digits <= 3 {
                digits
            } else if start
                .chars()
                .next()
                .map(|c| c.is_ascii_alphabetic())
                .unwrap_or(false)
                && start[1..].starts_with(['.', ')'])
            {
                1
            } else {
                0
            };
            if marker == 0 {
                None
            } else {
                let rest = &start[marker..];
                match rest.strip_prefix('.').or_else(|| rest.strip_prefix(')')) {
                    Some(tail) if tail.is_empty() || tail.starts_with(char::is_whitespace) => {
                        Some(tail)
                    }
                    _ => None,
                }
            }
        };
        match stripped {
            Some(rest) if rest != text => text = rest,
            _ => break,
        }
    }
    let folded = text
        .split_whitespace()
        .map(|word| word.to_lowercase())
        .collect::<Vec<_>>()
        .join(" ");
    folded.trim_end_matches(['.', ',', ';', ':']).trim().to_string()
}

/// Stable clause id: FNV-1a of the normalized text, 16 hex chars.
fn clause_id(clause: &str) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in normalize_clause(clause).as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

/// An instruction's clauses as `(id, text)` — [`split_clauses`] units
/// (newline blocks; sentences inside blocks over 200 chars), first
/// occurrence per id, marker-only lines dropped.
fn clause_units(text: &str) -> Vec<(String, String)> {
    let mut seen = std::collections::HashSet::new();
    let mut units = Vec::new();
    for clause in split_clauses(text) {
        if normalize_clause(&clause).is_empty() {
            continue;
        }
        let id = clause_id(&clause);
        if seen.insert(id.clone()) {
            units.push((id, clause));
        }
    }
    units
}

/// The run-level clause ledger behind [`ClauseCredit`].
#[derive(Default)]
struct ClauseLedger {
    credits: std::collections::BTreeMap<String, ClauseCredit>,
}

impl ClauseLedger {
    fn entry(&mut self, id: &str, text: &str, generation: usize) -> &mut ClauseCredit {
        self.credits
            .entry(id.to_string())
            .or_insert_with(|| ClauseCredit {
                id: id.to_string(),
                text: text.to_string(),
                added_at_gen: generation,
                times_present: 0,
                times_credited: 0,
                delta_sum: 0.0,
                cols_won: 0,
                cols_lost: 0,
                ema: 0.0,
                in_champion: false,
            })
    }

    /// Opens the ledger with the seed's clauses (generation 0, present).
    fn register_seed(&mut self, text: &str) {
        for (id, clause) in clause_units(text) {
            self.entry(&id, &clause, 0).times_present += 1;
        }
    }

    /// Credits one promoted child: every clause it carries is present once
    /// more; its per-column `deltas` vs the parent go to the clauses it
    /// added (evenly shared) and, sign-flipped, to the ones it removed.
    fn credit(&mut self, parent_text: &str, child_text: &str, deltas: &[f64], generation: usize) {
        let parent = clause_units(parent_text);
        let child = clause_units(child_text);
        for (id, clause) in &child {
            self.entry(id, clause, generation).times_present += 1;
        }
        if deltas.is_empty() {
            return;
        }
        let parent_ids: std::collections::HashSet<&str> =
            parent.iter().map(|(id, _)| id.as_str()).collect();
        let child_ids: std::collections::HashSet<&str> =
            child.iter().map(|(id, _)| id.as_str()).collect();
        let added: Vec<&(String, String)> = child
            .iter()
            .filter(|(id, _)| !parent_ids.contains(id.as_str()))
            .collect();
        let removed: Vec<&(String, String)> = parent
            .iter()
            .filter(|(id, _)| !child_ids.contains(id.as_str()))
            .collect();
        let sum: f64 = deltas.iter().sum();
        let mean = sum / deltas.len() as f64;
        let won = deltas.iter().filter(|delta| **delta > SCORE_EPS).count();
        let lost = deltas.iter().filter(|delta| **delta < -SCORE_EPS).count();
        if !added.is_empty() {
            let share = 1.0 / added.len() as f64;
            for (id, clause) in added {
                self.attribute(id, clause, generation, sum * share, mean * share, won, lost);
            }
        }
        if !removed.is_empty() {
            let share = 1.0 / removed.len() as f64;
            for (id, clause) in removed {
                self.attribute(id, clause, generation, -sum * share, -mean * share, lost, won);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn attribute(
        &mut self,
        id: &str,
        text: &str,
        generation: usize,
        delta_sum: f64,
        signal: f64,
        won: usize,
        lost: usize,
    ) {
        let credit = self.entry(id, text, generation);
        credit.times_credited += 1;
        credit.delta_sum += delta_sum;
        credit.cols_won += won;
        credit.cols_lost += lost;
        credit.ema = if credit.times_credited == 1 {
            signal
        } else {
            CLAUSE_EMA_BETA * signal + (1.0 - CLAUSE_EMA_BETA) * credit.ema
        };
    }

    /// Signed evidence for a clause id (`0.0` when never credited).
    fn ema_of(&self, id: &str) -> f64 {
        self.credits.get(id).map(|credit| credit.ema).unwrap_or(0.0)
    }

    /// Closes the ledger: the winner's clauses (document order across
    /// `champion_texts`) first, flagged `in_champion`, then everything
    /// else strongest evidence first.
    fn into_report(mut self, champion_texts: &[&str], champion_generation: usize) -> Vec<ClauseCredit> {
        let mut order: Vec<String> = Vec::new();
        for text in champion_texts {
            for (id, clause) in clause_units(text) {
                let credit = self.entry(&id, &clause, champion_generation);
                if !credit.in_champion {
                    credit.in_champion = true;
                    if credit.times_present == 0 {
                        // Rewritten past the last promotion (ablation or
                        // pruning): present in the winner only.
                        credit.times_present = 1;
                    }
                    order.push(id);
                }
            }
        }
        let mut report: Vec<ClauseCredit> =
            order.iter().filter_map(|id| self.credits.remove(id)).collect();
        let mut rest: Vec<ClauseCredit> = self.credits.into_values().collect();
        rest.sort_by(|a, b| b.ema.total_cmp(&a.ema).then_with(|| a.id.cmp(&b.id)));
        report.extend(rest);
        report
    }
}

/// Pick distinct, non-seed hypotheses BEFORE allocating the family alpha.
/// Candidate IDs describe lineage; engine rows describe the actual full
/// slot assignment. A composed seed or duplicate rewrite is not a second
/// hypothesis. The validation argmax remains eligible when the fresh
/// screen selects the seed.
fn certification_challengers(
    engine: &mut Engine,
    seed_row: usize,
    champion: &GEPACandidate,
    val_argmax: &GEPACandidate,
) -> Vec<(GEPACandidate, usize)> {
    let mut challengers = Vec::new();
    for candidate in [champion, val_argmax] {
        let row = engine.register(candidate.engine_candidate());
        if row != seed_row && !challengers.iter().any(|(_, seen)| *seen == row) {
            challengers.push((candidate.clone(), row));
        }
    }
    challengers
}

/// Certification wallet warm-started from a candidate's search-time
/// paired-vs-seed validation deltas (`prior_row`'s pool row): the val
/// estimate is selection-biased, but a bet prior only shapes λ — a
/// too-optimistic prior costs growth, never validity — and its
/// 25-pseudo-observation weight washes out quickly. Fewer than 5 paired
/// columns fall back to the plain wallet.
fn warm_paired_evidence(
    matrix: &ScoreMatrix,
    prior_row: Option<usize>,
    seed_row: usize,
    val_cols: &[usize],
    delta_scale: f64,
) -> PairedEvidence {
    let deltas: Vec<f64> = prior_row
        .map(|row| {
            val_cols
                .iter()
                .filter_map(|&col| Some(matrix.score(row, col)? - matrix.score(seed_row, col)?))
                .collect()
        })
        .unwrap_or_default();
    if deltas.len() >= 5 {
        let n = deltas.len() as f64;
        let mean = deltas.iter().sum::<f64>() / n;
        let var = deltas.iter().map(|d| (d - mean) * (d - mean)).sum::<f64>() / n;
        PairedEvidence::with_bet_prior(delta_scale, mean, var.sqrt().max(0.05 * delta_scale), 25.0)
    } else {
        PairedEvidence::new(delta_scale)
    }
}

struct Reflectors {
    rewrite: Predict<ProposeRewrite>,
    patch: Predict<ProposePatch>,
    simplify: Predict<ProposeSimplify>,
    merge: Predict<ProposeMerge>,
    condense: Predict<ProposeCondense>,
    policy: Predict<ProposePolicy>,
    diagnose: Predict<DiagnoseRepair>,
}

impl Reflectors {
    fn new(lm: &crate::LM) -> Self {
        Self {
            rewrite: Predict::<ProposeRewrite>::builder().lm(lm.clone()).build(),
            patch: Predict::<ProposePatch>::builder().lm(lm.clone()).build(),
            simplify: Predict::<ProposeSimplify>::builder().lm(lm.clone()).build(),
            merge: Predict::<ProposeMerge>::builder().lm(lm.clone()).build(),
            condense: Predict::<ProposeCondense>::builder().lm(lm.clone()).build(),
            policy: Predict::<ProposePolicy>::builder().lm(lm.clone()).build(),
            diagnose: Predict::<DiagnoseRepair>::builder().lm(lm.clone()).build(),
        }
    }
}

/// What an operator proposal produced.
struct OpProposal {
    /// `None` = abandon this trial (leak/length/anchor after retry, or
    /// reflection error).
    instruction: Option<String>,
    hypothesis: String,
    reflection_calls: usize,
    leak_retried: bool,
    leaked: bool,
    length_retried: bool,
    overlong: bool,
    anchor_retried: bool,
    unanchored: bool,
}

impl OpProposal {
    fn abandoned(
        reflection_calls: usize,
        leak_retried: bool,
        leaked: bool,
        length_retried: bool,
        overlong: bool,
        anchor_retried: bool,
        unanchored: bool,
    ) -> Self {
        Self {
            instruction: None,
            hypothesis: String::new(),
            reflection_calls,
            leak_retried,
            leaked,
            length_retried,
            overlong,
            anchor_retried,
            unanchored,
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
    Policy {
        feedback: String,
    },
    Patch {
        better: String,
        worse: String,
    },
    Simplify {
        validated: String,
    },
    Condense {
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
    /// The reflection chunk (trainset columns) this slot's teacher saw —
    /// the on-target screen columns under [`ScreenRule::OnTargetA`].
    chunk_cols: Vec<usize>,
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

fn anchor_addendum(anchor: &str) -> String {
    format!(
        "\n\nANCHOR VIOLATION: your previous proposal did not begin with the \
         required base instruction verbatim. Reproduce this base EXACTLY, \
         character for character, as the opening of your improved \
         instruction, then append your additions after it:\n---\n{anchor}\n---"
    )
}

fn leak_addendum(leaks: &[&str]) -> String {
    format!(
        "\n\nIMPORTANT: your previous proposal copied these example-specific strings \
         verbatim: {leaks:?}. Never copy answers, entities, or facts that appear only \
         in specific examples — the module will face different inputs. State the \
         general rule instead."
    )
}

fn length_addendum(len: usize, cap: usize, target: Option<usize>) -> String {
    let base = format!(
        "\n\nIMPORTANT: your previous proposal was {len} characters; the hard limit \
         is {cap}. Rewrite it to fit: keep the rules with measured value, cut \
         speculative clauses and redundant phrasing."
    );
    match target {
        None => base,
        Some(target) => format!(
            "{base} Aim for the LENGTH BUDGET of {target} characters: fold rules \
             (state each requirement once in the rule + response template + final \
             checklist form) rather than dropping any rule the feedback shows is \
             needed."
        ),
    }
}

/// The length guard one operator call runs under: the hard cap (`None` =
/// unlimited) and, under a `length_prior`, the target the shortening retry
/// is told to aim for.
#[derive(Clone, Copy, Debug, Default)]
struct LengthGuard {
    cap: Option<usize>,
    target: Option<usize>,
}

/// Total chars a candidate deploys: every slot of a joint candidate, or the
/// single instruction.
fn candidate_chars(candidate: &GEPACandidate) -> usize {
    if candidate.slots.is_empty() {
        candidate.instruction.len()
    } else {
        candidate.slots.values().map(String::len).sum()
    }
}

/// Runs one operator call with the leak and length guards: propose, check for
/// verbatim example strings and the length cap, retry once naming the
/// violations, abandon on a second violation.
macro_rules! guarded_propose {
    ($reflector:expr, $forbidden:expr, $cap:expr, $make_input:expr) => {
        guarded_propose!($reflector, $forbidden, $cap, None::<&str>, $make_input)
    };
    ($reflector:expr, $forbidden:expr, $cap:expr, $anchor:expr, $make_input:expr) => {{
        let mut leak_retried = false;
        let mut length_retried = false;
        let mut anchor_retried = false;
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
                            anchor_retried, false,
                        ));
                        break;
                    }
                    let leaks = find_leaks(&instruction, $forbidden);
                    let guard: LengthGuard = $cap;
                    let cap: Option<usize> = guard.cap;
                    let overlong = cap.is_some_and(|cap| instruction.len() > cap);
                    // Anchored accretion: the proposal must open with the
                    // frozen base verbatim (leading whitespace tolerated) —
                    // style edits to the base are structurally impossible.
                    let anchor: Option<&str> = $anchor;
                    let unanchored = anchor
                        .is_some_and(|a| !instruction.trim_start().starts_with(a.trim()));
                    if leaks.is_empty() && !overlong && !unanchored {
                        outcome = Some(OpProposal {
                            instruction: Some(instruction),
                            hypothesis,
                            reflection_calls: calls,
                            leak_retried,
                            leaked: false,
                            length_retried,
                            overlong: false,
                            anchor_retried,
                            unanchored: false,
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
                                guard.target,
                            ));
                        }
                        if unanchored {
                            anchor_retried = true;
                            extra.push_str(&anchor_addendum(
                                anchor.expect("unanchored implies an anchor"),
                            ));
                        }
                        continue;
                    }
                    tracing::warn!(
                        leaks = ?leaks,
                        overlong,
                        unanchored,
                        "proposal still violated guards after retry; abandoning trial"
                    );
                    outcome = Some(OpProposal::abandoned(
                        calls,
                        leak_retried,
                        !leaks.is_empty(),
                        length_retried,
                        overlong,
                        anchor_retried,
                        unanchored,
                    ));
                    break;
                }
                Err(err) => {
                    tracing::warn!(error = %err, "reflection call failed; abandoning trial");
                    outcome = Some(OpProposal::abandoned(
                        calls, leak_retried, false, length_retried, false,
                        anchor_retried, false,
                    ));
                    break;
                }
            }
        }
        outcome.expect("the proposal loop always sets an outcome")
    }};
}

impl Ratchet {
    /// The hard length cap actually in force: `length_cap`, or under a
    /// `length_prior` the loose guard `prior × length_prior_slack`
    /// (rounded) — the smaller of the two when both are set. `None` when
    /// neither knob is set.
    pub fn effective_length_cap(&self) -> Option<usize> {
        let prior_cap = self
            .length_prior
            .map(|prior| ((prior as f64) * self.length_prior_slack).round().max(1.0) as usize);
        match (self.length_cap, prior_cap) {
            (Some(cap), Some(guard)) => Some(cap.min(guard)),
            (Some(cap), None) => Some(cap),
            (None, guard) => guard,
        }
    }

    /// The teacher-facing LENGTH BUDGET block appended to every Rewrite and
    /// Policy reflection input under a `length_prior`; empty (so the input
    /// is byte-identical) when the prior is unset.
    fn length_prior_block(&self, current_len: usize) -> String {
        let Some(budget) = self.length_prior else {
            return String::new();
        };
        let hard = self.effective_length_cap().unwrap_or(budget);
        format!(
            "\n\nLENGTH BUDGET: the instruction is currently {current_len} chars; the \
             budget is {budget} chars (hard limit {hard} chars; over-limit proposals \
             are rejected). Stay within budget by folding: merge redundant rules, \
             state each requirement once in the rule + response template + final \
             checklist form, drop per-topic reminders the template already \
             enforces, and never drop a rule the feedback shows is needed. \
             Shorter is preferred when two versions would score the same."
        )
    }

    /// v6.5 Occam tie-break under a `length_prior`: among `entries`
    /// (candidate, engine row) returns the index of the shortest one whose
    /// paired scores on `cols` cannot be separated from the argmax at
    /// `best_idx` (within `selection_z`·SE, the [`within_noise`] rule).
    /// Without a prior, or with `selection_z` 0, `best_idx` stands.
    fn occam_shorter_idx(
        &self,
        matrix: &ScoreMatrix,
        entries: &[(&GEPACandidate, usize)],
        best_idx: usize,
        cols: &[usize],
    ) -> usize {
        if self.length_prior.is_none() || self.selection_z <= 0.0 || entries.is_empty() {
            return best_idx;
        }
        let best_scores = row_scores(matrix, entries[best_idx].1, cols);
        let mut pick = best_idx;
        for (idx, (candidate, row)) in entries.iter().enumerate() {
            if idx == best_idx || candidate_chars(candidate) >= candidate_chars(entries[pick].0) {
                continue;
            }
            if within_noise(&best_scores, &row_scores(matrix, *row, cols), self.selection_z) {
                pick = idx;
            }
        }
        if pick != best_idx {
            tracing::debug!(
                argmax_chars = candidate_chars(entries[best_idx].0),
                picked_chars = candidate_chars(entries[pick].0),
                "length prior: within-noise tie broken toward the shorter candidate"
            );
        }
        pick
    }

    /// The pool's frontier argmax by shared-slice mean, with the v6.5
    /// Occam tie-break applied under a `length_prior` (identical to
    /// `Lineage::best_by_average` otherwise).
    fn occam_best_by_average<'a>(
        &self,
        pool: &'a Lineage,
        pareto: &'a ParetoView,
        matrix: &ScoreMatrix,
        cols: &[usize],
    ) -> Option<&'a GEPACandidate> {
        let best = pool.best_by_average(pareto)?;
        if self.length_prior.is_none() || self.selection_z <= 0.0 {
            return Some(best);
        }
        let entries: Vec<(&'a GEPACandidate, usize)> = pool.frontier_entries(pareto).collect();
        let best_idx = entries
            .iter()
            .position(|(candidate, _)| candidate.id == best.id)?;
        Some(entries[self.occam_shorter_idx(matrix, &entries, best_idx, cols)].0)
    }

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

    fn supports_adaptive_step(&self) -> bool {
        self.adaptive_step_size
            && matches!(self.screen_rule, ScreenRule::OnTargetA)
            && matches!(self.accept_mode, AcceptMode::Gate)
            && !matches!(self.slot_credit, SlotCredit::Repair)
            && self.rewrite_fanout <= 1
    }

    /// Each proposal pays parent + child on its reflection batch, then the
    /// child on the whole scout slice. Every pool parent already has complete
    /// scout coverage. Cache hits and screen rejections only reduce this cost.
    fn affordable_step(&self, available: usize, scout_len: usize) -> Option<(usize, usize, usize)> {
        let per_proposal = self.minibatch_size.saturating_mul(2).saturating_add(scout_len).max(1);
        let mut best = None;
        for p in 1..=self.parents_per_step.max(1) {
            for n in 1..=self.proposals_per_parent.max(1) {
                let count = p.saturating_mul(n);
                let cost = count.saturating_mul(per_proposal);
                if cost <= available && best.is_none_or(|(bp, bn, _)| (count, p) > (bp * bn, bp)) {
                    best = Some((p, n, cost));
                }
            }
        }
        best
    }

    /// Sequential paired certification of one challenger row against the
    /// seed row: `certify_cols` block by block under the anytime-valid
    /// e-process in `evidence`, stopping the moment the gain (`Some(true)`)
    /// or the harm (`Some(false)`) is certified at `alpha`, or when
    /// `budget` rollouts past `spend_before` — or the columns — run out
    /// (`None`, unproven). Seed cells already in the matrix are
    /// cache-served, so a re-run on the same stream bills the challenger
    /// only.
    #[allow(clippy::too_many_arguments)]
    async fn certify_against_seed(
        &self,
        engine: &mut Engine,
        target: &mut OptimizeTarget<'_>,
        challenger_row: usize,
        seed_row: usize,
        certify_cols: &[usize],
        spend_before: usize,
        budget: usize,
        alpha: f64,
        evidence: &mut PairedEvidence,
    ) -> Result<Option<bool>> {
        let mut verdict: Option<bool> = None;
        for block in certify_cols.chunks(self.certify_block.max(1)) {
            if engine.spend().metric_calls - spend_before >= budget {
                break;
            }
            let evals = match engine
                .evaluate_many(target, &[challenger_row, seed_row], Some(block))
                .await?
            {
                BatchEvalOutcome::Complete(evals) => evals,
                BatchEvalOutcome::BudgetExhausted { .. } => break,
            };
            for (champ, seed) in evals[0].rollouts.iter().zip(&evals[1].rollouts) {
                evidence.observe(champ.eval.score - seed.eval.score);
            }
            if evidence.certified_gain(alpha) {
                verdict = Some(true);
                break;
            }
            if evidence.certified_harm(alpha) {
                verdict = Some(false);
                break;
            }
        }
        Ok(verdict)
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
    fn render_ledger(
        trials: &[TrialRecord],
        window: usize,
        falsify_margin: f64,
        gate_kills: bool,
    ) -> String {
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
                    } => gate_kills && parent_gate - child_gate >= falsify_margin,
                    TrialOutcome::RaceKilled { paired_delta, .. }
                    | TrialOutcome::Faded { paired_delta } => {
                        -paired_delta >= falsify_margin
                    }
                    // An audit eviction is certified harm — never a noise
                    // kill, so it is always decisive regardless of margin.
                    TrialOutcome::Audited { .. } => true,
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
                TrialOutcome::Audited { columns, paired_delta, .. } => {
                    format!(
                        "EVICTED by re-audit: certified worse than the seed over \
                         {columns} columns, vs seed {paired_delta:+.3} — its additions are harmful"
                    )
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
        // FORAGE (v5): the run-level contrastive ledger folds in every
        // rollout the search pays for; the tooled per-step Rewrite
        // predictor needs its own LM handle.
        let teacher_lm = self
            .prompt_model
            .clone()
            .expect("prompt_model presence checked above");
        let mut forage_ledger = ForageLedger::default();

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
        let seed_row = {
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
            forage_ledger.observe_eval(&eval);
            pool.add(seed, row, &eval);
            row
        };
        // v6.4 lineage-diff clause credit: the seed's clauses open the
        // ledger at generation 0 (present, never credited).
        let mut clause_ledger = ClauseLedger::default();
        for leaf in &leaves {
            clause_ledger.register_seed(&leaf.instruction);
        }

        // ASSAY Stage 1 re-audit state (`reaudit_budget_frac` > 0 only):
        // per-candidate cumulative paired-vs-seed evidence, an allowance
        // accumulated from each step's actual spend, and the marker it is
        // measured from. The evidence ranks the pool (in place of the noisy
        // scout-slice mean) and evicts members whose harm certifies.
        let auditing = self.reaudit_budget_frac > 0.0;
        let mut audit_evidence: HashMap<usize, PairedEvidence> = HashMap::new();
        let mut audit_allowance = 0.0f64;
        let mut audit_spend_marker = engine.spend().metric_calls;
        let (mut audit_evicted, mut audit_rollouts) = (0usize, 0usize);
        // Cumulative-evidence mean for ranking: the seed is the 0.0
        // baseline; a promoted member without evidence yet ranks at it.
        fn audit_mean(
            evidence: &HashMap<usize, PairedEvidence>,
            candidate: &GEPACandidate,
        ) -> f64 {
            if candidate.generation == 0 {
                0.0
            } else {
                evidence
                    .get(&candidate.id)
                    .map(|paired| paired.mean_delta())
                    .unwrap_or(0.0)
            }
        }

        // The hard length cap in force for the whole run: `length_cap`, or
        // under a `length_prior` the loose guard `prior × slack` (the
        // smaller wins when both are set).
        let hard_length_cap = self.effective_length_cap();
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
        let (mut anchor_retries, mut anchor_skipped) = (0usize, 0usize);
        // Anchored accretion: each leaf's frozen base = its compile-time
        // incumbent instruction.
        let anchors: Vec<String> = leaves
            .iter()
            .map(|leaf| leaf.instruction.clone())
            .collect();
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

        // Screen evaluations of children that have not (yet) been promoted:
        // the FORAGE ledger folds them in only on promotion, so a killed
        // child's harvest never retires a column's open checks (F4 D8).
        let mut gate_child_evals: HashMap<usize, crate::optimizer::engine::CandidateEval> =
            HashMap::new();
        // v6.2: identifiers the grader has asked to be cited so far — the
        // backticked tokens on feedback lines that mention citing — shown to
        // the Policy operator as examples of the required form, so a rule
        // learned from five examples is written for the whole input space.
        let mut cited_ids: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        'generations: for generation in 0..self.num_iterations {
            gate_child_evals.clear();
            // Keep enough budget in reserve for the fresh-data champion
            // phase — including this step's own worst-case spend (P parents
            // × (N reflection mini-batches + gate evaluations) + a full
            // race; see `step_worst_case`).
            // Promotion state gates the scheduled knobs (cap/lexicase after
            // promotion), the ablation reserve, and Simplify/Merge
            // eligibility further down.
            let has_promotion = pool.entries.len() > leaves.len();
            let mut p_count = self.parents_per_step.max(1);
            let mut n_count = self.proposals_per_parent.max(1);

            // Under Gate accept, a step's post-gate cost is the scout fan-out
            // instead of one full-val race.
            let step_val_need = match self.accept_mode {
                AcceptMode::Race => val_cols.len(),
                AcceptMode::Gate => {
                    (self.step_fanout() * p_count + 1) * scout_cols.len()
                }
            };
            let mut worst_step = self.step_worst_case(gate_size, step_val_need);
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
            // columns (40 of 80; prereg amendment 2).
            let certify_hold = if self.certify_ship && has_promotion && !certify_cols.is_empty()
            {
                self.certify_budget
                    + if self.pin_assay { self.pin_probe_budget } else { 0 }
                    + if self.prune_after_certify { self.prune_budget } else { 0 }
                    // The fan-out margin prices race overshoot; there is no
                    // race under the on-target screen (children pay the
                    // shared ranking slice, already in `step_val_need`).
                    // Capped at half the certification budget as in the
                    // main repo (prereg amendment 7: uncapped, it starves
                    // the search).
                    + if ablate_reserve == 0
                        && !matches!(self.screen_rule, ScreenRule::OnTargetA)
                    {
                        self.ablate_margin(val_cols.len())
                    } else {
                        0
                    }
            } else {
                0
            };
            if self.supports_adaptive_step() {
                let reserved = 2 * selection_cols.len() + ablate_reserve + certify_hold;
                let available = self.max_rollouts.unwrap_or(usize::MAX)
                    .saturating_sub(engine.spend().metric_calls)
                    .saturating_sub(reserved);
                let Some((p, n, cost)) = self.affordable_step(available, scout_cols.len()) else {
                    break;
                };
                p_count = p;
                n_count = n;
                worst_step = cost;
                tracing::debug!(generation, parents = p, proposals = n, available,
                    "adaptive complete proposal batch");
            }
            if !engine.budget_allows(
                2 * selection_cols.len() + worst_step + ablate_reserve + certify_hold,
            ) {
                break;
            }

            // ---- ASSAY Stage 1 re-audit: spend the allowance earned by the
            // previous step re-evaluating standing pool members on rotating
            // never-yet-drawn validation columns, paired against the seed.
            // Placed at the generation head so a step that ends early (gate
            // wipe-out, empty race) still buys its members audit exposure.
            // Blocks target uncovered columns only (`ScoreMatrix` row
            // coverage), so no delta is ever observed twice; seed cells
            // cache-hit after their first visit. A member whose cumulative
            // harm certifies at `race_alpha` is evicted and its fade enters
            // the falsified ledger, teaching the teacher mid-run.
            if auditing {
                let spent_now = engine.spend().metric_calls;
                audit_allowance +=
                    (spent_now - audit_spend_marker) as f64 * self.reaudit_budget_frac;
                audit_spend_marker = spent_now;
                let block_len = self.reaudit_block.max(1);
                // Never eat what the loop head just guaranteed the step,
                // the finale, and the certification.
                let reserve =
                    2 * selection_cols.len() + worst_step + ablate_reserve + certify_hold;
                while audit_allowance >= 1.0 {
                    if !engine.budget_allows(reserve + 2 * block_len) {
                        break;
                    }
                    // Least-audited member that still has uncovered columns.
                    let next = pool
                        .entries
                        .iter()
                        .filter(|(candidate, _)| candidate.generation > 0)
                        .map(|(candidate, row)| {
                            let evidence_count = audit_evidence
                                .get(&candidate.id)
                                .map(|paired| paired.count())
                                .unwrap_or(0);
                            (candidate.id, *row, evidence_count)
                        })
                        .filter(|&(_, row, _)| {
                            val_cols
                                .iter()
                                .any(|&col| engine.matrix().score(row, col).is_none())
                        })
                        .min_by_key(|&(_, _, evidence_count)| evidence_count);
                    let Some((audited_id, audited_row, _)) = next else {
                        // Every member fully covered: nothing to buy here.
                        audit_allowance = 0.0;
                        break;
                    };
                    let mut uncovered: Vec<usize> = val_cols
                        .iter()
                        .copied()
                        .filter(|&col| engine.matrix().score(audited_row, col).is_none())
                        .collect();
                    uncovered.shuffle(&mut rng);
                    uncovered.truncate(block_len);
                    let before = engine.spend().metric_calls;
                    let evals = match engine
                        .evaluate_many(target, &[audited_row, seed_row], Some(&uncovered))
                        .await?
                    {
                        BatchEvalOutcome::Complete(evals) => evals,
                        BatchEvalOutcome::BudgetExhausted { .. } => break,
                    };
                    for eval in &evals {
                        forage_ledger.observe_eval(eval);
                    }
                    let paid = engine.spend().metric_calls - before;
                    audit_rollouts += paid;
                    audit_allowance -= paid.max(1) as f64;
                    let evidence = audit_evidence
                        .entry(audited_id)
                        .or_insert_with(|| PairedEvidence::new(delta_scale));
                    {
                        let matrix = engine.matrix();
                        for &col in &uncovered {
                            if let (Some(child), Some(seed)) = (
                                matrix.score(audited_row, col),
                                matrix.score(seed_row, col),
                            ) {
                                evidence.observe(child - seed);
                            }
                        }
                    }
                    if evidence.certified_harm(self.race_alpha) {
                        let columns = evidence.count();
                        let paired_delta = evidence.mean_delta();
                        // Reuse the promotion record's operator/hypothesis so
                        // the ledger names the falsified idea, not the audit.
                        let promoted_record = trials.iter().find(|trial| {
                            matches!(
                                trial.outcome,
                                TrialOutcome::Promoted { candidate_id, .. }
                                    if candidate_id == audited_id
                            )
                        });
                        trials.push(TrialRecord {
                            generation,
                            operator: promoted_record
                                .map(|record| record.operator)
                                .unwrap_or(RatchetOp::Rewrite),
                            parent_id: pool
                                .entries
                                .iter()
                                .find(|(candidate, _)| candidate.id == audited_id)
                                .and_then(|(candidate, _)| candidate.parent_id)
                                .unwrap_or(0),
                            hypothesis: promoted_record
                                .map(|record| record.hypothesis.clone())
                                .unwrap_or_default(),
                            outcome: TrialOutcome::Audited {
                                candidate_id: audited_id,
                                columns,
                                paired_delta,
                            },
                        });
                        pool.entries.retain(|(candidate, _)| candidate.id != audited_id);
                        audit_evicted += 1;
                        tracing::info!(
                            candidate = audited_id,
                            columns,
                            paired_delta,
                            "re-audit evicted a pool member (harm certified)"
                        );
                    }
                }
            }

            let effective_cap = LengthGuard {
                cap: if self.cap_after_promotion && !has_promotion {
                    None
                } else {
                    hard_length_cap
                },
                target: self.length_prior,
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
                Self::render_ledger(
                    &trials,
                    self.ledger_window,
                    self.ledger_falsify_margin,
                    self.ledger_gate_kills,
                ),
                Self::render_repair_lessons(&repair_lessons)
            );

            // FORAGE: render the step's contrastive digest from everything
            // already paid for. Computed before the plan phase so the leak
            // guard can extend its forbidden set over every column the
            // digest exposes (state is one step stale by construction —
            // this step's parent evals feed the NEXT step's digest).
            // Deterministic; consumes no rng and no budget.
            let forage_digest: ForageDigest = if matches!(
                self.teacher_context,
                TeacherContext::Digest | TeacherContext::Tools
            ) {
                forage_ledger.render_digest(
                    self.check_extractor.as_ref(),
                    generation,
                    self.forage_open_cap,
                    self.forage_contrast_cap,
                    self.forage_chars,
                )
            } else {
                ForageDigest {
                    text: String::new(),
                    columns: Vec::new(),
                }
            };

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
                // Under the re-audit, "the head of the compounding chain" is
                // the cumulative-evidence argmax, not the scout-mean argmax
                // — the scout slice is exactly the under-evidenced estimate
                // the mechanism replaces.
                let greedy_pick = if !greedy {
                    None
                } else if auditing {
                    pool.entries
                        .iter()
                        .map(|(candidate, _)| candidate)
                        .max_by(|a, b| {
                            audit_mean(&audit_evidence, a)
                                .total_cmp(&audit_mean(&audit_evidence, b))
                        })
                } else {
                    self.occam_best_by_average(
                        &pool,
                        &pareto_snapshot,
                        engine.matrix(),
                        &val_cols,
                    )
                };
                let sampled_parent = if let Some(parent) = greedy_pick {
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
                forage_ledger.observe_eval(&parent_eval);

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
                if let Some(cap) = effective_cap.cap {
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
                if self.allow_training_examples && !self.allow_memorization {
                    task_description.push_str(
                        "\n\nTRAINING DATA POLICY: the examples and feedback supplied here \
                         are permitted training data. You may include an example or a \
                         concrete detail when it helps the task. Choose edits that will \
                         work on new inputs; including training content is optional. \
                         The task requirements, including any privacy requirements, \
                         still apply. This permission overrides generic prohibitions \
                         on copying training content in the proposal instructions.",
                    );
                } else if self.allow_memorization && self.policy_channel {
                    task_description.push_str(
                        "\n\nTASK-CONFIG: this is a corpus-fact task. Ignore any rule \
                         against copying example-specific content: concrete facts, \
                         names, values, and identifiers from the execution feedback \
                         may be written into the instruction when the evidence \
                         supports them.",
                    );
                } else if self.allow_memorization {
                    task_description.push_str(
                        "\n\nTASK-CONFIG OVERRIDE: this is a corpus-fact task. \
                         Ignore any rule against copying example-specific content: \
                         distilling CONCRETE facts, names, values, and definitions \
                         from the execution feedback INTO the instruction is the \
                         highest-value edit. Accumulate a growing, organized FACTS \
                         section; keep every previously pinned fact and add the new \
                         ones this feedback reveals.",
                    );
                }
                // v6 HARVEST policy channel: the winning edit on a judged
                // task is often a global answer policy, not a fact; the
                // teacher is told to look for both.
                if self.policy_channel {
                    task_description.push_str(
                        "\n\nPROPOSAL SCOPE: consider two kinds of edits and choose \
                         whichever the evidence supports best. (1) CONTENT the \
                         feedback names — facts, definitions, exact names and \
                         values the answers were missing. (2) GLOBAL ANSWER-POLICY \
                         changes that would raise the score on EVERY input: what \
                         every answer must always include or cite and in what \
                         exact form, how to structure the answer, what to leave \
                         out, and how to answer when the evidence is missing. Read \
                         the per-example scores and the grader's advice for \
                         patterns that repeat across examples — a policy that fixes \
                         a repeated pattern is worth more than any single fact. \
                         Keep everything in the current instruction that works. \
                         State policies as short explicit rules the module can \
                         follow on inputs it has never seen.",
                    );
                }
                if self.anchored_accretion {
                    task_description.push_str(
                        "\n\nANCHORED ACCRETION: your improved instruction MUST \
                         reproduce the current base instruction VERBATIM as its \
                         opening — every character unchanged — followed by your \
                         additions. Never edit, compress, or restyle the base; \
                         proposals that modify the base are rejected. Write each \
                         teaching as ONE dense, telegraphic line (symbol: fact); \
                         when a new fact overlaps an existing entry, extend that \
                         entry instead of appending a duplicate. Pin every \
                         distinct fact the evidence supports — density per \
                         character, not brevity, is the goal.",
                    );
                    if self.anchor_allow_rules {
                        task_description.push_str(
                            "\n\nThe appended section may ALSO carry a small \
                             RULES part before the FACTS: high-value behavioral \
                             rules the failures justify — evidence-verification \
                             discipline (search before claiming; verify exact \
                             symbols), an exact sentence template for honestly \
                             refusing when sources do not establish an answer, \
                             and rules against stating unsupported claims or \
                             converting missing coverage into a 'no'. NEVER add \
                             conciseness, brevity, scope-limiting, or formatting \
                             rules — they measurably hurt answer coverage.",
                        );
                    } else {
                        task_description.push_str(
                            "\n\nAdd facts only: never add behavioral or style \
                             rules (conciseness, scope, formatting directives).",
                        );
                    }
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
                        if a_spread {
                            eligible.push(RatchetOp::Patch);
                        }
                    }
                    // Simplify and Merge only earn a slot once the search
                    // has at least one promoted child — before that, forced
                    // exploration of them is wasted budget (measured 1 win
                    // in 25 early attempts).
                    if has_promotion && parent_text.len() > 500 && !self.anchored_accretion {
                        eligible.push(RatchetOp::Simplify);
                    }
                    // FORAGE v5.2: under anchoring, Condense takes
                    // Simplify's compression role once the appended
                    // teachings have real mass — representation shrinks,
                    // content provably stays (the density lever behind the
                    // measured ~transfer-per-character slope).
                    if has_promotion
                        && self.anchored_accretion
                        && parent_text.len() > anchors[leaf_idx].len() + 1500
                    {
                        eligible.push(RatchetOp::Condense);
                    }
                    if has_promotion && merge_pair.is_some() && !merge_taken {
                        eligible.push(RatchetOp::Merge);
                    }
                    if self.rewrite_only {
                        eligible.retain(|op| *op == RatchetOp::Rewrite);
                    }
                    if eligible.is_empty() {
                        skipped += 1;
                        tracing::debug!(generation, "no eligible operator; skipping slot");
                        continue;
                    }
                    let op = if self.policy_channel
                        && self.policy_share > 0.0
                        && !a_perfect
                        && rng.gen_range(0.0f64..1.0) < self.policy_share
                    {
                        RatchetOp::Policy
                    } else {
                        Self::choose_op(&eligible, &op_stats, &mut rng)
                    };

                    let mut forbidden = if self.allow_memorization || self.allow_training_examples {
                        Vec::new()
                    } else {
                        GEPA::leak_strings(target, chunk_cols)
                    };
                    // FORAGE: digest evidence spans columns beyond this
                    // chunk — extend the guard so the digest can't become
                    // an unguarded leak channel.
                    if !self.allow_memorization && !self.allow_training_examples && !forage_digest.columns.is_empty() {
                        for text in GEPA::leak_strings(target, &forage_digest.columns) {
                            if !forbidden.contains(&text) {
                                forbidden.push(text);
                            }
                        }
                    }
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
                    for rollout in &chunk_eval.rollouts {
                        if let Some(feedback) = rollout.eval.feedback.as_deref() {
                            for line in feedback.lines() {
                                if !line.to_ascii_lowercase().contains("cit") {
                                    continue;
                                }
                                let mut rest = line;
                                while let Some(start) = rest.find('`') {
                                    let after = &rest[start + 1..];
                                    let Some(end) = after.find('`') else { break };
                                    let token = &after[..end];
                                    if token.contains('/')
                                        && !token.contains(' ')
                                        && token.len() < 80
                                    {
                                        cited_ids.insert(token.to_string());
                                    }
                                    rest = &after[end + 1..];
                                }
                            }
                        }
                    }
                    let cited_examples = if cited_ids.is_empty() {
                        String::new()
                    } else {
                        format!(
                            "\n\nIdentifiers the grader has asked to be cited so far across \
                             the whole search (examples of the required FORM — the module \
                             must determine the right one for each input itself): {}",
                            cited_ids.iter().cloned().collect::<Vec<_>>().join(", ")
                        )
                    };
                    let rewrite_feedback = || {
                        let mut feedback = format!(
                            "Parent average score: {:.3}\n{}",
                            chunk_eval.mean(),
                            GEPA::summarize_feedback(&sampled_parent.module_name, &chunk_eval)
                        );
                        if let Some(context) = &repair_context {
                            feedback.push_str("\n\n");
                            feedback.push_str(context);
                        }
                        if !forage_digest.text.is_empty() {
                            feedback.push_str("\n\n");
                            feedback.push_str(&forage_digest.text);
                        }
                        feedback
                    };
                    // v6.5 length prior: the LENGTH BUDGET block closes both
                    // reflective operators' feedback (empty when the prior
                    // is unset — the teacher's input is byte-identical).
                    let length_block = self.length_prior_block(parent_text.len());
                    let input = match op {
                        RatchetOp::Rewrite => PlannedInput::Rewrite {
                            feedback: format!("{}{length_block}", rewrite_feedback()),
                        },
                        RatchetOp::Policy => PlannedInput::Policy {
                            feedback: format!(
                                "{}{cited_examples}{length_block}",
                                rewrite_feedback()
                            ),
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
                        RatchetOp::Condense => PlannedInput::Condense {
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
                        chunk_cols: chunk_cols.to_vec(),
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

            // FORAGE Tools mode: the step's Rewrite calls run through a
            // fresh predictor carrying three read-only tools over an
            // immutable ledger snapshot (depth-1 agentic reflection — the
            // teacher can drill into any column's contrastive record, and
            // cannot spend budget or mutate search state). Other operators
            // keep their plain predictors.
            let tooled_rewrite: Option<Predict<ProposeRewrite>> =
                if matches!(self.teacher_context, TeacherContext::Tools) {
                    let snapshot = std::sync::Arc::new(
                        forage_ledger.snapshot(self.check_extractor.as_ref()),
                    );
                    Some(
                        Predict::<ProposeRewrite>::builder()
                            .lm(teacher_lm.clone())
                            .with_tools(forage_tools(snapshot))
                            .build(),
                    )
                } else {
                    None
                };

            // ---- Propose phase: every planned slot's reflection call runs
            // concurrently (the blog's parallel dispatch). The leak and
            // length guards run inside each call's own retry loop.
            let proposals: Vec<OpProposal> = {
                let reflectors = &reflectors;
                let tooled_rewrite = &tooled_rewrite;
                let ledger = &ledger;
                let groups = &groups;
                let anchors = &anchors;
                let futures: Vec<_> = planned
                    .iter()
                    .map(|slot| async move {
                        let parent_text = &groups[slot.group].parent_text;
                        let task_description = &groups[slot.group].task_description;
                        let anchor: Option<&str> = self
                            .anchored_accretion
                            .then(|| anchors[groups[slot.group].leaf_idx].as_str());
                        match &slot.input {
                            PlannedInput::Rewrite { feedback } => {
                                let rewrite_ref =
                                    tooled_rewrite.as_ref().unwrap_or(&reflectors.rewrite);
                                guarded_propose!(
                                    rewrite_ref,
                                    &slot.forbidden,
                                    effective_cap,
                                    anchor,
                                    |extra: String| ProposeRewriteInput {
                                        task_description: task_description.clone(),
                                        current_instruction: parent_text.clone(),
                                        execution_feedback: format!("{feedback}{extra}"),
                                        trial_ledger: ledger.clone(),
                                    }
                                )
                            }
                            PlannedInput::Policy { feedback } => {
                                guarded_propose!(
                                    reflectors.policy,
                                    &slot.forbidden,
                                    effective_cap,
                                    anchor,
                                    |extra: String| ProposePolicyInput {
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
                                    anchor,
                                    |extra: String| ProposePatchInput {
                                        task_description: task_description.clone(),
                                        current_instruction: parent_text.clone(),
                                        better_rollout: better.clone(),
                                        worse_rollout: format!("{worse}{extra}"),
                                        trial_ledger: ledger.clone(),
                                    }
                                )
                            }
                            PlannedInput::Condense { validated } => {
                                guarded_propose!(
                                    reflectors.condense,
                                    &slot.forbidden,
                                    effective_cap,
                                    anchor,
                                    |extra: String| ProposeCondenseInput {
                                        task_description: task_description.clone(),
                                        current_instruction: parent_text.clone(),
                                        validated_lessons: format!("{validated}{extra}"),
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
                                    anchor,
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
                anchor_retries += proposal.anchor_retried as usize;
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
                    } else if proposal.unanchored {
                        anchor_skipped += 1;
                        trials.push(TrialRecord {
                            generation,
                            operator: slot.op,
                            parent_id: slot.lineage_parent.id,
                            hypothesis: proposal.hypothesis,
                            outcome: TrialOutcome::AnchorSkipped,
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
                // ---- v6 on-target screen (ScreenRule::OnTargetA): each
                // child is scored on the reflection chunk it was written
                // from, paired against its lineage parent (cache-served
                // there), and passes iff its summed score strictly beats
                // the parent's — pip-gepa's acceptance. Batch B is unused.
                if matches!(self.screen_rule, ScreenRule::OnTargetA) {
                    for pending in group_children {
                        let slot = &planned[pending.slot];
                        let cols: Vec<usize> = slot.chunk_cols.clone();
                        if cols.is_empty() {
                            race_entries.push(RaceEntry {
                                child: pending.child,
                                child_row: pending.child_row,
                                hypothesis: pending.hypothesis,
                                op: slot.op,
                                lineage_parent_id: slot.lineage_parent.id,
                                lineage_row: slot.lineage_row,
                                evidence: PairedEvidence::new(delta_scale),
                            });
                            continue;
                        }
                        let rows = vec![slot.lineage_row, pending.child_row];
                        let evals = match engine.evaluate_many(target, &rows, Some(&cols)).await? {
                            BatchEvalOutcome::Complete(evals) => evals,
                            BatchEvalOutcome::BudgetExhausted { .. } => break 'generations,
                        };
                        GEPA::require_feedback(
                            &evals[1],
                            &pending.child.module_name,
                            generation + 1,
                        )?;
                        let parent_scores = row_scores(engine.matrix(), slot.lineage_row, &cols);
                        let child_scores = row_scores(engine.matrix(), pending.child_row, &cols);
                        let parent_sum: f64 = parent_scores.iter().flatten().sum();
                        let child_sum: f64 = child_scores.iter().flatten().sum();
                        let n = cols.len() as f64;
                        if child_sum <= parent_sum + SCORE_EPS {
                            gate_rejected += 1;
                            trials.push(TrialRecord {
                                generation,
                                operator: slot.op,
                                parent_id: slot.lineage_parent.id,
                                hypothesis: pending.hypothesis.clone(),
                                outcome: TrialOutcome::GateKilled {
                                    parent_gate: parent_sum / n,
                                    child_gate: child_sum / n,
                                },
                            });
                            continue;
                        }
                        if self.forage_fold_on_promotion {
                            gate_child_evals.insert(pending.child_row, evals[1].clone());
                        } else {
                            forage_ledger.observe_eval(&evals[1]);
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
                    let fanout_rewrite =
                        tooled_rewrite.as_ref().unwrap_or(&reflectors.rewrite);
                    let fanout_anchor: Option<&str> = self
                        .anchored_accretion
                        .then(|| anchors[group.leaf_idx].as_str());
                    let second = guarded_propose!(
                        fanout_rewrite,
                        &slot.forbidden,
                        effective_cap,
                        fanout_anchor,
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
                    anchor_retries += second.anchor_retried as usize;
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
                for (row, eval) in rows.iter().zip(&evals) {
                    let is_child = group_children.iter().any(|pending| pending.child_row == *row)
                        || sibling.as_ref().is_some_and(|(_, sib_row, _, _)| sib_row == row);
                    if is_child && self.forage_fold_on_promotion {
                        gate_child_evals.insert(*row, eval.clone());
                    } else {
                        forage_ledger.observe_eval(eval);
                    }
                }
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
                        for eval in &evals {
                            forage_ledger.observe_eval(eval);
                        }
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
                    forage_ledger.observe_eval(eval);
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
                if let Some(eval) = gate_child_evals.remove(&entry.child_row) {
                    forage_ledger.observe_eval(&eval);
                }
                // v6.4 lineage-diff credit (SEPO-style): the child's
                // per-column deltas vs its lineage parent on the shared
                // ranking slice, attributed to the clauses the child added
                // (sign-flipped to the ones it removed). Zero extra
                // rollouts — both rows are already in the matrix.
                {
                    let parent_text = pool
                        .entries
                        .iter()
                        .find(|(candidate, _)| candidate.id == entry.lineage_parent_id)
                        .map(|(candidate, _)| {
                            candidate
                                .slots
                                .get(&entry.child.module_name)
                                .cloned()
                                .unwrap_or_else(|| candidate.instruction.clone())
                        });
                    if let Some(parent_text) = parent_text {
                        let matrix = engine.matrix();
                        let deltas: Vec<f64> = ranking_cols
                            .iter()
                            .filter_map(|&col| {
                                Some(
                                    matrix.score(entry.child_row, col)?
                                        - matrix.score(entry.lineage_row, col)?,
                                )
                            })
                            .collect();
                        clause_ledger.credit(
                            &parent_text,
                            &entry.child.instruction,
                            &deltas,
                            generation,
                        );
                    }
                }
                let promoted = pool
                    .add_scored(entry.child, entry.child_row, child_scores)
                    .clone();
                // ASSAY: open the member's cumulative paired-vs-seed wallet
                // with the ranking slice it just paid for; every later
                // re-audit block extends it on never-yet-drawn columns.
                if auditing {
                    let mut evidence = PairedEvidence::new(delta_scale);
                    let matrix = engine.matrix();
                    for &col in ranking_cols {
                        if let (Some(child), Some(seed)) = (
                            matrix.score(entry.child_row, col),
                            matrix.score(seed_row, col),
                        ) {
                            evidence.observe(child - seed);
                        }
                    }
                    audit_evidence.insert(promoted.id, evidence);
                }
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

        // Noise-aware Occam winner — same rule as GEPA. Under the re-audit
        // the argmax is taken over cumulative paired-vs-seed evidence means
        // (50–150 columns by endgame) instead of the scout-slice mean
        // (~12–16 columns): the switch the whole mechanism exists for.
        let pareto = engine.pareto_over(&val_cols);
        let mut best_by_mean_candidate = if auditing {
            pool.entries
                .iter()
                .map(|(candidate, _)| candidate)
                .max_by(|a, b| {
                    audit_mean(&audit_evidence, a).total_cmp(&audit_mean(&audit_evidence, b))
                })
                .cloned()
                .context("no candidates available in the pool")?
        } else {
            self.occam_best_by_average(&pool, &pareto, engine.matrix(), &val_cols)
                .cloned()
                .context("no candidates available on Pareto frontier")?
        };
        let val_argmax_candidate = best_by_mean_candidate.clone();

        // Fresh-data champion selection: race the top two candidates (by
        // validation mean) on never-touched trainset columns — plus, under
        // TEMPER, the variant-tempering columns (paraphrase twins of seen
        // examples, equally unseen by the search). This reduces training
        // overfit, but selecting the maximum of several fresh noisy means
        // still has a winner's curse. Independent certification below tests
        // improvement over the seed; it does not prove optimality among
        // finalists. A prompt that memorized seen phrasings can lose on the
        // variant columns.
        let mut fresh_selection_cols = 0usize;
        let mut variant_selection_cols = 0usize;
        if !selection_cols.is_empty() {
            let mut ranked: Vec<(GEPACandidate, usize)> = pool
                .entries
                .iter()
                .map(|(candidate, row)| (candidate.clone(), *row))
                .collect();
            if auditing {
                // Champion screening ranks by cumulative evidence too — the
                // scout-mean sort is the winner's-curse channel the audit
                // closes (mem4: a 57-member pool's true best sat at scout
                // rank 1 and the screen still lost it to a noise draw).
                ranked.sort_by(|a, b| {
                    audit_mean(&audit_evidence, &b.0)
                        .total_cmp(&audit_mean(&audit_evidence, &a.0))
                });
            } else {
                ranked.sort_by(|a, b| {
                    b.0.average_score()
                        .partial_cmp(&a.0.average_score())
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
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
                    // v6.5 length prior: among the racers the paired
                    // selection columns cannot separate from the argmax,
                    // the shortest wins (no-op without a prior).
                    let best_idx = {
                        let entries: Vec<(&GEPACandidate, usize)> = ranked[..k]
                            .iter()
                            .map(|(candidate, row)| (candidate, *row))
                            .collect();
                        self.occam_shorter_idx(engine.matrix(), &entries, best_idx, &selection_cols)
                    };
                    best_by_mean_candidate = ranked[best_idx].0.clone();
                }
            }
        }

        // ---- ASSAY Stage 2: per-pin evidence + stopped e-BH composition.
        // Parse the champion's appended section into pins, ablation-probe
        // each on the columns whose recorded checks motivated it, keep the
        // set that passes e-BH at `pin_fdr_alpha`, and hand the COMPOSED
        // bundle to certification as the champion challenger. The raw
        // argmax stays in the challenger list (Bonferroni), so an
        // over-stripped composition cannot lose a certifiable pile.
        let mut pins_tested = 0usize;
        let mut pins_kept = 0usize;
        let mut pin_probe_rollouts = 0usize;
        let mut pin_assay_report: Vec<(String, usize, f64, f64)> = Vec::new();
        let assaying = self.pin_assay
            && self.certify_ship
            && !certify_cols.is_empty()
            && best_by_mean_candidate.generation > 0;
        if assaying {
            match &self.check_extractor {
                None => tracing::warn!(
                    "pin_assay needs a check_extractor to attribute pins; skipping"
                ),
                Some(extractor) => {
                    let seed_base = pool
                        .entries
                        .iter()
                        .find(|(candidate, _)| candidate.generation == 0)
                        .map(|(candidate, _)| {
                            candidate
                                .slots
                                .get(&best_by_mean_candidate.module_name)
                                .cloned()
                                .unwrap_or_else(|| candidate.instruction.clone())
                        })
                        .unwrap_or_default();
                    let champ_text = best_by_mean_candidate.instruction.clone();
                    if !champ_text.starts_with(&seed_base) {
                        tracing::warn!(
                            "pin_assay: champion does not open with the seed base \
                             (anchoring off?); skipping composition"
                        );
                    } else {
                        let appended = &champ_text[seed_base.len()..];
                        let pins: Vec<String> = appended
                            .lines()
                            .map(str::trim)
                            .filter(|line| !line.is_empty())
                            .map(str::to_string)
                            .collect();
                        // Provenance: a ledger column funds a pin when any
                        // check it ever surfaced (worst rollout preserves
                        // the pre-harvest state) matches the pin's text.
                        let pin_matches = |pin: &str, check: &str| -> bool {
                            let pin_lc = pin.to_lowercase();
                            let check_lc = check.trim().to_lowercase();
                            if check_lc.is_empty() {
                                return false;
                            }
                            if pin_lc.contains(&check_lc) || check_lc.contains(pin_lc.trim()) {
                                return true;
                            }
                            let words: Vec<&str> = check_lc
                                .split_whitespace()
                                .filter(|word| word.len() > 3)
                                .collect();
                            !words.is_empty()
                                && words.iter().filter(|word| pin_lc.contains(**word)).count()
                                    * 2
                                    >= words.len()
                        };
                        let champ_row =
                            engine.register(best_by_mean_candidate.engine_candidate());
                        let pin_spend_before = engine.spend().metric_calls;
                        // Provenance sweep: once the chain covers a fact, no
                        // later rollout re-surfaces its check, so the ledger
                        // only names the columns drawn while the check was
                        // open. The SEED covers nothing — one sweep of it
                        // over the batch pool and validation columns records
                        // every column's open checks (cells already paid by
                        // the scout/audit cache-hit), giving each pin its
                        // full provenance map. Certification columns stay
                        // untouched.
                        {
                            let mut sweep: Vec<usize> = train_pool
                                .iter()
                                .chain(val_cols.iter())
                                .copied()
                                .collect();
                            sweep.shuffle(&mut rng);
                            // The subset screen's power lives in its DRAW
                            // count, so the provenance sweep takes a quarter
                            // of the budget there instead of half.
                            sweep.truncate(if self.pin_screen_draws > 0 {
                                self.pin_probe_budget / 4
                            } else {
                                self.pin_probe_budget / 2
                            });
                            if !sweep.is_empty() && engine.budget_allows(sweep.len()) {
                                if let EvalOutcome::Complete(eval) =
                                    engine.evaluate(target, seed_row, Some(&sweep)).await?
                                {
                                    forage_ledger.observe_eval(&eval);
                                }
                            }
                        }
                        let kept_idx: std::collections::BTreeSet<usize> = if self
                            .pin_screen_draws
                            > 0
                        {
                            // === E3 SUBSET SCREEN (ContextCite, NeurIPS 24,
                            // arXiv:2409.00729): joint credit assignment.
                            // Random pin-subsets, each evaluated once on ONE
                            // shared provenance-weighted slice; a lasso of
                            // draw-mean score on inclusion vectors estimates
                            // every pin's marginal effect from the shared
                            // rollout pool — no per-pin e-process to starve.
                            // The screen carries no statistical claim; the
                            // frozen certification still takes the ship
                            // decision on the composed bundle.
                            let m = pins.len();
                            // Stratified shared slice: round-robin one
                            // provenance column per pin (shuffled within
                            // each pin) until the cap — every pin with any
                            // attributable column gets slice coverage
                            // before any pin gets two. A pin with no
                            // column in the slice is invisible to the
                            // regression and cannot be kept; set
                            // `pin_screen_cols` ≥ the expected pin count
                            // to cover every fundable pin.
                            let per_pin: Vec<Vec<usize>> = pins
                                .iter()
                                .map(|pin| {
                                    let mut cols: Vec<usize> = forage_ledger
                                        .entries
                                        .iter()
                                        .filter(|(_, entry)| {
                                            extractor(&entry.worst_feedback)
                                                .iter()
                                                .chain(
                                                    extractor(&entry.best_feedback).iter(),
                                                )
                                                .any(|check| pin_matches(pin, check))
                                        })
                                        .map(|(&col, _)| col)
                                        .collect();
                                    cols.shuffle(&mut rng);
                                    cols
                                })
                                .collect();
                            let cap = self.pin_screen_cols.max(1);
                            let mut slice: Vec<usize> = Vec::new();
                            let mut seen: std::collections::BTreeSet<usize> =
                                Default::default();
                            let mut depth = 0usize;
                            'fill: loop {
                                let mut advanced = false;
                                for cols in &per_pin {
                                    if slice.len() >= cap {
                                        break 'fill;
                                    }
                                    if let Some(&col) = cols.get(depth) {
                                        advanced = true;
                                        if seen.insert(col) {
                                            slice.push(col);
                                        }
                                    }
                                }
                                if !advanced {
                                    break;
                                }
                                depth += 1;
                            }
                            if slice.len() < self.pin_screen_cols {
                                let mut extra: Vec<usize> = val_cols
                                    .iter()
                                    .copied()
                                    .filter(|col| !slice.contains(col))
                                    .collect();
                                extra.shuffle(&mut rng);
                                extra.truncate(self.pin_screen_cols - slice.len());
                                slice.extend(extra);
                            }
                            let mut x_rows: Vec<Vec<f64>> = Vec::new();
                            let mut y: Vec<f64> = Vec::new();
                            let mut included = vec![0usize; m];
                            for _ in 0..self.pin_screen_draws {
                                if slice.is_empty()
                                    || engine.spend().metric_calls - pin_spend_before
                                        + slice.len()
                                        > self.pin_probe_budget
                                    || !engine.budget_allows(slice.len())
                                {
                                    break;
                                }
                                let inclusion: Vec<bool> = (0..m)
                                    .map(|_| rng.gen_range(0.0f64..1.0) < 0.5)
                                    .collect();
                                let kept_lines: Vec<&str> = pins
                                    .iter()
                                    .enumerate()
                                    .filter(|(idx, _)| inclusion[*idx])
                                    .map(|(_, pin)| pin.as_str())
                                    .collect();
                                let variant_text = if kept_lines.is_empty() {
                                    seed_base.clone()
                                } else {
                                    format!("{seed_base}\n{}", kept_lines.join("\n"))
                                };
                                let mut variant = best_by_mean_candidate.clone();
                                variant.instruction = variant_text.clone();
                                variant
                                    .slots
                                    .insert(variant.module_name.clone(), variant_text);
                                let variant_row =
                                    engine.register(variant.engine_candidate());
                                if let EvalOutcome::Complete(eval) = engine
                                    .evaluate(target, variant_row, Some(&slice))
                                    .await?
                                {
                                    let scores: Vec<f64> = eval
                                        .rollouts
                                        .iter()
                                        .map(|rollout| rollout.eval.score)
                                        .collect();
                                    if !scores.is_empty() {
                                        y.push(
                                            scores.iter().sum::<f64>()
                                                / scores.len() as f64,
                                        );
                                        for (idx, &inc) in inclusion.iter().enumerate() {
                                            if inc {
                                                included[idx] += 1;
                                            }
                                        }
                                        x_rows.push(
                                            inclusion
                                                .iter()
                                                .map(|&inc| if inc { 1.0 } else { 0.0 })
                                                .collect(),
                                        );
                                    }
                                }
                            }
                            pins_tested = m;
                            pin_probe_rollouts =
                                engine.spend().metric_calls - pin_spend_before;
                            let coefs = lasso_screen(&x_rows, &y, &mut rng);
                            // Screen-failure fallback (too few draws to fit):
                            // keep everything — composition becomes a no-op
                            // and the champion certifies as it stood.
                            let kept: std::collections::BTreeSet<usize> =
                                if coefs.is_empty() {
                                    (0..m).collect()
                                } else {
                                    let mut kept: std::collections::BTreeSet<usize> = coefs
                                        .iter()
                                        .enumerate()
                                        .filter(|&(idx, &(coef, floor))| {
                                            coef > floor.max(1e-4) && included[idx] >= 3
                                        })
                                        .map(|(idx, _)| idx)
                                        .collect();
                                    if kept.len() < self.pin_screen_min_keep {
                                        let mut ranked: Vec<(usize, f64)> = coefs
                                            .iter()
                                            .enumerate()
                                            .filter(|&(idx, &(coef, _))| {
                                                coef > 1e-4 && included[idx] >= 3
                                            })
                                            .map(|(idx, &(coef, _))| (idx, coef))
                                            .collect();
                                        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
                                        for (idx, _) in
                                            ranked.into_iter().take(self.pin_screen_min_keep)
                                        {
                                            kept.insert(idx);
                                        }
                                    }
                                    kept
                                };
                            for (idx, pin) in pins.iter().enumerate() {
                                pin_assay_report.push((
                                    pin.clone(),
                                    included[idx],
                                    coefs.get(idx).map(|&(coef, _)| coef).unwrap_or(0.0),
                                    if kept.contains(&idx) { 1.0 } else { 0.0 },
                                ));
                            }
                            tracing::info!(
                                pins = m,
                                kept = kept.len(),
                                draws = x_rows.len(),
                                slice = slice.len(),
                                rollouts = pin_probe_rollouts,
                                "pin screen: subset-lasso composition"
                            );
                            kept
                        } else {
                        // Provenance per pin, then probe in provenance-count
                        // order (richest first): with a bounded probe budget
                        // the fundable pins get their full draw instead of
                        // file order starving the tail arbitrarily. Ordering
                        // by ancillary information (column counts, never the
                        // deltas) leaves e-process validity untouched.
                        let mut pin_cols: Vec<Vec<usize>> = pins
                            .iter()
                            .map(|pin| {
                                let mut cols: Vec<usize> = forage_ledger
                                    .entries
                                    .iter()
                                    .filter(|(_, entry)| {
                                        extractor(&entry.worst_feedback)
                                            .iter()
                                            .chain(extractor(&entry.best_feedback).iter())
                                            .any(|check| pin_matches(pin, check))
                                    })
                                    .map(|(&col, _)| col)
                                    .collect();
                                cols.shuffle(&mut rng);
                                cols.truncate(self.pin_probe_cols.max(1));
                                cols
                            })
                            .collect();
                        let mut order: Vec<usize> = (0..pins.len()).collect();
                        order.sort_by_key(|&idx| std::cmp::Reverse(pin_cols[idx].len()));
                        let mut evidences: Vec<PairedEvidence> = pins
                            .iter()
                            .map(|_| {
                                // Optimistic bet prior sized for the
                                // conditional (own-column) effect a real pin
                                // shows; it shapes growth only — a junk pin
                                // still cannot certify. Zero attributable
                                // columns → unfundable: e stays 1.
                                PairedEvidence::with_bet_prior(
                                    delta_scale,
                                    0.3 * delta_scale,
                                    0.25 * delta_scale,
                                    8.0,
                                )
                            })
                            .collect();
                        for &pin_idx in &order {
                            let pin = &pins[pin_idx];
                            let cols = std::mem::take(&mut pin_cols[pin_idx]);
                            let evidence = &mut evidences[pin_idx];
                            if !cols.is_empty()
                                && engine.spend().metric_calls - pin_spend_before
                                    < self.pin_probe_budget
                                && engine.budget_allows(2 * cols.len())
                            {
                                let ablated_lines: Vec<&str> = pins
                                    .iter()
                                    .filter(|other| other.as_str() != pin.as_str())
                                    .map(String::as_str)
                                    .collect();
                                let ablated_text = if ablated_lines.is_empty() {
                                    seed_base.clone()
                                } else {
                                    format!("{seed_base}\n{}", ablated_lines.join("\n"))
                                };
                                let mut ablated = best_by_mean_candidate.clone();
                                ablated.instruction = ablated_text.clone();
                                ablated
                                    .slots
                                    .insert(ablated.module_name.clone(), ablated_text);
                                let ablated_row = engine.register(ablated.engine_candidate());
                                if self.pin_probe_repeats <= 1 {
                                    let complete = matches!(
                                        engine
                                            .evaluate_many(
                                                target,
                                                &[champ_row, ablated_row],
                                                Some(&cols)
                                            )
                                            .await?,
                                        BatchEvalOutcome::Complete(_)
                                    );
                                    if complete {
                                        let matrix = engine.matrix();
                                        for &col in &cols {
                                            if let (Some(champ), Some(ablated)) = (
                                                matrix.score(champ_row, col),
                                                matrix.score(ablated_row, col),
                                            ) {
                                                evidence.observe(champ - ablated);
                                            }
                                        }
                                    }
                                } else {
                                    // Repeated fresh draws: every rollout is
                                    // forced (uncached), so each visit to a
                                    // column is a new paired observation.
                                    'repeats: for _ in 0..self.pin_probe_repeats {
                                        for &col in &cols {
                                            if engine.spend().metric_calls
                                                - pin_spend_before
                                                >= self.pin_probe_budget
                                            {
                                                break 'repeats;
                                            }
                                            let champ = engine
                                                .probe_rollout(target, champ_row, col)
                                                .await;
                                            let ablated_probe = engine
                                                .probe_rollout(target, ablated_row, col)
                                                .await;
                                            match (champ, ablated_probe) {
                                                (
                                                    Ok(Some((champ_eval, _))),
                                                    Ok(Some((ablated_eval, _))),
                                                ) => evidence.observe(
                                                    champ_eval.score - ablated_eval.score,
                                                ),
                                                (Err(err), _) | (_, Err(err)) => {
                                                    tracing::warn!(
                                                        error = %err,
                                                        "pin probe rollout failed; \
                                                         stopping this pin"
                                                    );
                                                    break 'repeats;
                                                }
                                                _ => break 'repeats,
                                            }
                                        }
                                    }
                                }
                            }
                            pins_tested += 1;
                        }
                        // Report in the champion's appended (pile) order.
                        for (pin, evidence) in pins.iter().zip(&evidences) {
                            pin_assay_report.push((
                                pin.clone(),
                                evidence.count(),
                                evidence.mean_delta(),
                                evidence.e_gain(),
                            ));
                        }
                        pin_probe_rollouts = engine.spend().metric_calls - pin_spend_before;
                        // Stopped e-BH at FDR `pin_fdr_alpha`: keep the k*
                        // largest e-values where e_(k) ≥ m / (α·k) — valid
                        // at any data-dependent stopping time, under
                        // arbitrary dependence between the pin streams.
                        let m = evidences.len();
                        let mut ranked: Vec<(usize, f64)> = evidences
                            .iter()
                            .enumerate()
                            .map(|(idx, evidence)| (idx, evidence.e_gain()))
                            .collect();
                        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
                        let mut kept_count = 0usize;
                        for (k, (_, e_value)) in ranked.iter().enumerate() {
                            let rank = k + 1;
                            if *e_value
                                >= m as f64 / (self.pin_fdr_alpha.max(f64::MIN_POSITIVE)
                                    * rank as f64)
                            {
                                kept_count = rank;
                            }
                        }
                        tracing::info!(
                            pins = m,
                            kept = kept_count,
                            rollouts = pin_probe_rollouts,
                            "pin assay: stopped e-BH composition"
                        );
                        ranked
                            .iter()
                            .take(kept_count)
                            .map(|(idx, _)| *idx)
                            .collect()
                        };
                        pins_kept = kept_idx.len();
                        if pins_kept < pins.len() {
                            let kept_lines: Vec<&str> = pins
                                .iter()
                                .enumerate()
                                .filter(|(idx, _)| kept_idx.contains(idx))
                                .map(|(_, pin)| pin.as_str())
                                .collect();
                            let composed_text = if kept_lines.is_empty() {
                                seed_base.clone()
                            } else {
                                format!("{seed_base}\n{}", kept_lines.join("\n"))
                            };
                            let mut composed = best_by_mean_candidate.clone();
                            composed.instruction = composed_text.clone();
                            composed
                                .slots
                                .insert(composed.module_name.clone(), composed_text);
                            let composed_row = engine.register(composed.engine_candidate());
                            // Pool membership keeps every downstream lookup
                            // (seed protection, ablation, certification
                            // priors) total; empty scores rank it nowhere.
                            let composed = pool
                                .add_scored(composed, composed_row, Vec::new())
                                .clone();
                            best_by_mean_candidate = composed;
                        }
                    }
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
        // What the certification consumed and the per-challenger level it
        // ran at — the Occam re-certification below inherits both.
        let mut certify_spent = 0usize;
        let mut certify_alpha_each = self.certify_alpha;
        if certifying {
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
            let challengers = certification_challengers(
                engine, seed_row, &best_candidate, &val_argmax_candidate,
            );
            let alpha_each = self.certify_alpha / challengers.len().max(1) as f64;
            let spend_before = engine.spend().metric_calls;
            let mut shipped: Option<GEPACandidate> = None;
            'challengers: for (challenger, champion_row) in &challengers {
                // Warm-start the bets from the search-time val delta
                // (challenger's pool row vs seed, paired). The val estimate
                // is selection-biased, but a bet prior only shapes λ — a
                // too-optimistic prior costs growth, never validity — and
                // its 25-pseudo-observation weight washes out quickly.
                let prior_row = pool
                    .entries
                    .iter()
                    .find(|(candidate, _)| candidate.id == challenger.id)
                    .map(|(_, row)| *row);
                let mut evidence = warm_paired_evidence(
                    engine.matrix(),
                    prior_row,
                    seed_row,
                    &val_cols,
                    delta_scale,
                );
                if let Some(band) = &self.certify_band {
                    evidence = evidence.banded(band);
                }
                let verdict = self
                    .certify_against_seed(
                        engine,
                        target,
                        *champion_row,
                        seed_row,
                        &certify_cols,
                        spend_before,
                        self.certify_budget,
                        alpha_each,
                        &mut evidence,
                    )
                    .await?;
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
            certify_spent = engine.spend().metric_calls - spend_before;
            certify_alpha_each = alpha_each;
            best_candidate = match shipped {
                Some(champion) => champion,
                None => seed_candidate,
            };
        }

        // v6.4 post-certification Occam pruning (handoff-harvest-v6 §5
        // items 2 and 4; L1 Design A step 4): the certified champion's
        // clauses whose lineage-credit EMA is ≤ 0 — added by children that
        // did not move the shared ranking slice, or never credited at all —
        // come out as ONE bundle, raced paired against the full champion on
        // the selection columns (bisected once on a loss; seed clauses are
        // never touched). Whatever survives must RE-CERTIFY vs the seed on
        // the certification stream with the budget the first pass left;
        // otherwise the original certified champion ships.
        let mut prune_chars_before = 0usize;
        let mut prune_chars_after = 0usize;
        let mut prune_removed = 0usize;
        let mut prune_rollouts = 0usize;
        let mut prune_certified: Option<bool> = None;
        if self.prune_after_certify && certified == Some(true) && best_candidate.generation > 0 {
            let spend_before = engine.spend().metric_calls;
            let prune_cols: Vec<usize> = if selection_cols.is_empty() {
                val_cols.clone()
            } else {
                selection_cols.clone()
            };
            let seed_text = pool
                .entries
                .iter()
                .find(|(candidate, _)| candidate.generation == 0)
                .map(|(candidate, _)| {
                    candidate
                        .slots
                        .get(&best_candidate.module_name)
                        .cloned()
                        .unwrap_or_else(|| candidate.instruction.clone())
                })
                .unwrap_or_default();
            let protected: std::collections::HashSet<String> = clause_units(&seed_text)
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            let clauses = split_clauses(&best_candidate.instruction);
            prune_chars_before = best_candidate.instruction.len();
            prune_chars_after = prune_chars_before;
            // Least evidence first: negative EMAs lead, never-credited
            // clauses (EMA 0) follow, document order breaks ties.
            let mut zero_set: Vec<(usize, f64)> = clauses
                .iter()
                .enumerate()
                .filter_map(|(idx, clause)| {
                    let id = clause_id(clause);
                    if protected.contains(&id) {
                        return None;
                    }
                    let ema = clause_ledger.ema_of(&id);
                    (ema <= 0.0).then_some((idx, ema))
                })
                .collect();
            zero_set.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
            let mut pruned: Option<(GEPACandidate, usize)> = None;
            if !zero_set.is_empty() && zero_set.len() < clauses.len() && !prune_cols.is_empty() {
                let champion_row = engine.register(best_candidate.engine_candidate());
                // Cache-served when the champion screen already ran here.
                if let EvalOutcome::Complete(_) = engine
                    .evaluate(target, champion_row, Some(&prune_cols))
                    .await?
                {
                    let baseline_scores = row_scores(engine.matrix(), champion_row, &prune_cols);
                    let baseline_evaluated: Vec<f64> =
                        baseline_scores.iter().filter_map(|score| *score).collect();
                    let baseline_mean = baseline_evaluated.iter().sum::<f64>()
                        / baseline_evaluated.len().max(1) as f64;
                    let prune_z = if self.selection_z > 0.0 { self.selection_z } else { 1.0 };
                    let whole: Vec<usize> = zero_set.iter().map(|(idx, _)| *idx).collect();
                    let mut bundles: Vec<Vec<usize>> = vec![whole.clone()];
                    if whole.len() >= 2 {
                        let (low, high) = whole.split_at(whole.len() / 2);
                        bundles.push(low.to_vec());
                        bundles.push(high.to_vec());
                    }
                    for bundle in bundles {
                        if engine.spend().metric_calls - spend_before >= self.prune_budget
                            || !engine.budget_allows(prune_cols.len())
                        {
                            break;
                        }
                        let text = clauses
                            .iter()
                            .enumerate()
                            .filter(|(idx, _)| !bundle.contains(idx))
                            .map(|(_, clause)| clause.as_str())
                            .collect::<Vec<_>>()
                            .join("\n");
                        let mut probe = best_candidate.clone();
                        probe.slots.insert(probe.module_name.clone(), text.clone());
                        probe.instruction = text;
                        let row = engine.register(probe.engine_candidate());
                        let eval = match engine.evaluate(target, row, Some(&prune_cols)).await? {
                            EvalOutcome::Complete(eval) => eval,
                            EvalOutcome::BudgetExhausted { .. } => break,
                        };
                        let scores = row_scores(engine.matrix(), row, &prune_cols);
                        let mean = eval.mean();
                        let survives =
                            mean >= baseline_mean || within_noise(&baseline_scores, &scores, prune_z);
                        tracing::info!(
                            clauses = bundle.len(),
                            mean,
                            baseline = baseline_mean,
                            survives,
                            "occam prune bundle probe"
                        );
                        if survives {
                            pruned = Some((probe, bundle.len()));
                            break;
                        }
                    }
                }
            }
            match pruned {
                Some((candidate, removed)) => {
                    let remaining = self.certify_budget.saturating_sub(certify_spent);
                    let pruned_row = engine.register(candidate.engine_candidate());
                    // Warm-start from the champion's search-time val delta:
                    // the pruned prompt is the champion minus clauses that
                    // measurably did nothing there.
                    let prior_row = pool
                        .entries
                        .iter()
                        .find(|(entry, _)| entry.id == best_candidate.id)
                        .map(|(_, row)| *row);
                    let mut evidence = warm_paired_evidence(
                        engine.matrix(),
                        prior_row,
                        seed_row,
                        &val_cols,
                        delta_scale,
                    );
                    if let Some(band) = &self.certify_band {
                        evidence = evidence.banded(band);
                    }
                    let recert_before = engine.spend().metric_calls;
                    let verdict = self
                        .certify_against_seed(
                            engine,
                            target,
                            pruned_row,
                            seed_row,
                            &certify_cols,
                            recert_before,
                            remaining,
                            certify_alpha_each,
                            &mut evidence,
                        )
                        .await?;
                    prune_certified = verdict;
                    tracing::info!(
                        removed,
                        chars_before = prune_chars_before,
                        chars_after = candidate.instruction.len(),
                        cols = evidence.count(),
                        mean_delta = evidence.mean_delta(),
                        e_gain = evidence.e_gain(),
                        verdict = ?verdict,
                        remaining_budget = remaining,
                        "occam prune re-certification"
                    );
                    if verdict == Some(true) {
                        prune_removed = removed;
                        prune_chars_after = candidate.instruction.len();
                        best_candidate = candidate;
                    }
                }
                None => tracing::info!(
                    candidates = zero_set.len(),
                    clauses = clauses.len(),
                    "occam prune: nothing removable survived the selection race"
                ),
            }
            prune_rollouts = engine.spend().metric_calls - spend_before;
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

        // Close the clause ledger on the shipped winner's text (every slot).
        let clause_credits = {
            let texts: Vec<&str> = if best_candidate.slots.is_empty() {
                vec![best_candidate.instruction.as_str()]
            } else {
                best_candidate.slots.values().map(String::as_str).collect()
            };
            clause_ledger.into_report(&texts, best_candidate.generation)
        };

        let operator_stats = op_stats
            .into_iter()
            .map(|(op, (attempts, wins))| (op.name().to_string(), attempts, wins))
            .collect();

        let result = RatchetResult {
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
            audit_evicted,
            audit_rollouts,
            pins_tested,
            pins_kept,
            pin_probe_rollouts,
            pin_assay_report,
            skipped,
            leak_retries,
            leak_skipped,
            length_retries,
            length_skipped,
            length_prior: self.length_prior,
            length_cap_in_force: hard_length_cap,
            anchor_retries,
            anchor_skipped,
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
            clause_credits,
            prune_chars_before,
            prune_chars_after,
            prune_removed,
            prune_rollouts,
            prune_certified,
        };
        if self.track_stats {
            let report = result.clause_credit_report();
            if !report.is_empty() {
                tracing::info!("run summary — {report}");
            }
        }
        Ok(Report::Ratchet(result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_preset_has_no_certificate_or_reserved_selection_data() {
        let search = harvest_search();
        assert!(!search.certify_ship);
        assert_eq!(search.certify_budget, 0);
        assert_eq!(search.certify_reserve, 0);
        assert_eq!(search.champion_fresh, 0);
        assert_eq!(search.selection_z, 0.0);
        assert!(!search.allow_memorization);
        assert!(search.allow_training_examples);
        assert!(search.supports_adaptive_step());
        assert!(harvest().certify_ship);
        assert!(!harvest().adaptive_step_size);
    }

    #[test]
    fn adaptive_batches_keep_full_evaluation_and_respect_tail_budget() {
        let search = harvest_search();
        assert_eq!(search.affordable_step(160, 30), Some((2, 2, 160)));
        assert_eq!(search.affordable_step(159, 30), Some((2, 1, 80)));
        assert_eq!(search.affordable_step(79, 30), Some((1, 1, 40)));
        assert_eq!(search.affordable_step(39, 30), None);
        assert_eq!(search.affordable_step(0, 30), None);
        // Changing the number of validation examples changes the whole cost;
        // the scheduler must not silently truncate evaluation to fit.
        assert_eq!(search.affordable_step(40, 31), None);
        let mut unsupported = search;
        unsupported.slot_credit = SlotCredit::Repair;
        assert!(!unsupported.supports_adaptive_step());
        unsupported.slot_credit = SlotCredit::RoundRobin;
        unsupported.screen_rule = ScreenRule::HeldOutB;
        assert!(!unsupported.supports_adaptive_step());
    }

    #[test]
    fn certification_excludes_seed_and_duplicate_hypotheses_before_alpha_split() {
        let seed = GEPACandidate {
            id: 0, instruction: "seed".into(), module_name: "answer".into(),
            example_scores: Vec::new(), parent_id: None, generation: 0,
            slots: Default::default(),
        };
        let mut engine = Engine::new(Default::default());
        let seed_row = engine.register(seed.engine_candidate());
        let mut child = seed.mutate("improved".into(), 1);
        child.id = 1;

        // A seed-winning screen must not strand the validation challenger.
        let contenders = certification_challengers(&mut engine, seed_row, &seed, &child);
        assert_eq!(contenders.len(), 1);
        assert_eq!(contenders[0].0.id, child.id);

        // A composed candidate may have a new lineage ID but be the seed.
        let mut composed_seed = seed.mutate("seed".into(), 2);
        composed_seed.id = 2;
        assert_eq!(certification_challengers(
            &mut engine, seed_row, &child, &composed_seed,
        ).len(), 1);

        // Duplicate content gets one wallet even with distinct IDs.
        let mut duplicate = child.clone();
        duplicate.id = 3;
        assert_eq!(certification_challengers(
            &mut engine, seed_row, &child, &duplicate,
        ).len(), 1);
        assert!(certification_challengers(
            &mut engine, seed_row, &seed, &composed_seed,
        ).is_empty());

        // Identity must include ALL slots, not just the rewritten text.
        duplicate.slots.insert("plan".into(), "a different plan".into());
        assert_eq!(certification_challengers(
            &mut engine, seed_row, &child, &duplicate,
        ).len(), 2);
    }

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
        let text = Ratchet::render_ledger(&trials, 5, 0.0, true);
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
        assert!(Ratchet::render_ledger(&neutral, 5, 0.0, true).contains("(none yet)"));
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
        let all = Ratchet::render_ledger(&trials, 12, 0.0, true);
        assert!(all.contains("a decisive failure"));
        assert!(all.contains("noise kill"));
        assert!(all.contains("marginal gate kill"));
        // A 0.02 margin keeps only the decisive kill.
        let filtered = Ratchet::render_ledger(&trials, 12, 0.02, true);
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
        let digest = Ratchet::render_ledger(&faded, 12, 0.02, true);
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
    fn length_prior_off_leaves_the_teacher_inputs_byte_identical() {
        for optimizer in [Ratchet::builder().build(), compound(), harvest(), temper(vec![])] {
            assert_eq!(optimizer.length_prior, None);
            assert_eq!(optimizer.length_prior_slack, 1.25);
            // The hard cap in force is exactly the explicit `length_cap`.
            assert_eq!(optimizer.effective_length_cap(), optimizer.length_cap);
            // The reflection feedback both operators see is unchanged.
            let feedback = "Parent average score: 0.500\nexample 3 (score 0.20): ...";
            let rendered = format!("{feedback}{}", optimizer.length_prior_block(4_200));
            assert_eq!(rendered, feedback);
            assert!(!rendered.contains("LENGTH BUDGET"));
        }
        // The shortening retry keeps its measured wording without a target.
        assert_eq!(
            length_addendum(1_800, 1_500, None),
            "\n\nIMPORTANT: your previous proposal was 1800 characters; the hard limit \
             is 1500. Rewrite it to fit: keep the rules with measured value, cut \
             speculative clauses and redundant phrasing."
        );
    }

    #[test]
    fn length_prior_sets_the_loose_guard_and_briefs_the_teacher() {
        let mut optimizer = harvest();
        optimizer.length_prior = Some(4_000);
        assert_eq!(optimizer.effective_length_cap(), Some(5_000));
        optimizer.length_prior_slack = 1.5;
        assert_eq!(optimizer.effective_length_cap(), Some(6_000));
        // An explicit cap wins when tighter, yields when looser.
        optimizer.length_cap = Some(3_000);
        assert_eq!(optimizer.effective_length_cap(), Some(3_000));
        optimizer.length_cap = Some(9_000);
        assert_eq!(optimizer.effective_length_cap(), Some(6_000));
        optimizer.length_cap = None;
        optimizer.length_prior_slack = 1.25;
        let block = optimizer.length_prior_block(4_200);
        assert!(block.starts_with(
            "\n\nLENGTH BUDGET: the instruction is currently 4200 chars; the budget is \
             4000 chars (hard limit 5000 chars; over-limit proposals are rejected)."
        ));
        assert!(block.contains("never drop a rule the feedback shows is needed"));
        assert!(block.ends_with("Shorter is preferred when two versions would score the same."));
        // The shortening retry names the target on top of its usual text.
        let retry = length_addendum(5_300, 5_000, Some(4_000));
        assert!(retry.starts_with(&length_addendum(5_300, 5_000, None)));
        assert!(retry.contains("LENGTH BUDGET of 4000 characters"));
    }

    #[test]
    fn occam_tie_break_prefers_the_shorter_candidate_only_under_a_prior() {
        let candidate = |id: usize, chars: usize| GEPACandidate {
            id,
            instruction: "x".repeat(chars),
            module_name: "m".into(),
            example_scores: Vec::new(),
            parent_id: None,
            generation: id,
            slots: Default::default(),
        };
        let long = candidate(0, 400);
        let short = candidate(1, 100);
        let worse = candidate(2, 50);
        let mut matrix = ScoreMatrix::new(6);
        for col in 0..6 {
            // long vs short: a ±.04 see-saw, mean gap .01 — inside 1 SE.
            matrix.record(0, col, if col % 2 == 0 { 0.55 } else { 0.50 });
            matrix.record(1, col, if col % 2 == 0 { 0.50 } else { 0.53 });
            // worse: clearly below on every column — never picked.
            matrix.record(2, col, 0.20);
        }
        let cols: Vec<usize> = (0..6).collect();
        let entries = vec![(&long, 0usize), (&short, 1usize), (&worse, 2usize)];
        let mut optimizer = harvest();
        assert_eq!(optimizer.occam_shorter_idx(&matrix, &entries, 0, &cols), 0);
        optimizer.length_prior = Some(200);
        assert_eq!(optimizer.occam_shorter_idx(&matrix, &entries, 0, &cols), 1);
        optimizer.selection_z = 0.0;
        assert_eq!(optimizer.occam_shorter_idx(&matrix, &entries, 0, &cols), 0);
    }

    #[test]
    fn occam_prune_knobs_stay_off_in_every_preset() {
        for optimizer in [Ratchet::builder().build(), compound(), harvest(), temper(vec![])] {
            assert!(!optimizer.prune_after_certify);
            assert_eq!(optimizer.prune_budget, 120);
        }
    }

    #[test]
    fn clause_ids_fold_list_markers_case_and_trailing_punctuation() {
        let base = clause_id("Cite sources");
        for variant in [
            "- Cite sources.",
            "* cite Sources",
            "2) Cite   sources;",
            "12. CITE SOURCES:",
            "(3) cite sources",
            "a. Cite sources.",
            "• cite sources",
        ] {
            assert_eq!(clause_id(variant), base, "{variant:?} should keep the id");
        }
        assert_ne!(clause_id("Cite sources"), clause_id("Cite two sources"));
        // A sentence that merely starts with a short word is not a marker.
        assert_eq!(normalize_clause("No. Then stop."), "no. then stop");
        assert_eq!(normalize_clause("1.5 is a number"), "1.5 is a number");
        assert_eq!(normalize_clause("- "), "");
        // Units: blank and marker-only lines drop, duplicates fold.
        let units = clause_units("Answer.\n\n- \nCite sources.\n1. cite sources");
        assert_eq!(units.len(), 2);
        assert_eq!(units[1].1, "Cite sources.");
    }

    #[test]
    fn clause_ledger_credits_added_clauses_and_flips_removed_ones() {
        let mut ledger = ClauseLedger::default();
        let seed = "Answer the question.\nBe brief.";
        ledger.register_seed(seed);
        // Child adds one clause; deltas won 2 / lost 1 / tied 1.
        let child = "Answer the question.\nBe brief.\nCite sources.";
        ledger.credit(seed, child, &[0.1, 0.1, 0.0, -0.05], 3);
        let cite = ledger.credits.get(&clause_id("Cite sources.")).unwrap();
        assert_eq!(cite.added_at_gen, 3);
        assert_eq!((cite.times_present, cite.times_credited), (1, 1));
        assert!((cite.delta_sum - 0.15).abs() < 1e-12);
        assert_eq!((cite.cols_won, cite.cols_lost), (2, 1));
        assert!((cite.ema - 0.0375).abs() < 1e-12);
        let brief = ledger.credits.get(&clause_id("Be brief.")).unwrap();
        assert_eq!((brief.times_present, brief.times_credited), (2, 0));
        assert_eq!(brief.added_at_gen, 0);
        // Grandchild removes "Be brief." and gains: the removed clause is
        // credited with the sign flipped (its absence helped).
        let grandchild = "Answer the question.\nCite sources.";
        ledger.credit(child, grandchild, &[0.02, 0.02, 0.02, 0.02], 5);
        let brief = ledger.credits.get(&clause_id("Be brief.")).unwrap();
        assert_eq!(brief.times_credited, 1);
        assert!((brief.ema + 0.02).abs() < 1e-12);
        assert_eq!((brief.cols_won, brief.cols_lost), (0, 4));
        // Two clauses added together share the delta sum; the re-added
        // "cite sources" (renumbered) keeps its id and its EMA blends.
        let bundle = "Answer the question.\n1. Cite sources.\n2. Answer in English.";
        ledger.credit(seed, bundle, &[0.2, 0.2], 7);
        let cite = ledger.credits.get(&clause_id("Cite sources.")).unwrap();
        assert_eq!(cite.times_credited, 2);
        assert!((cite.delta_sum - 0.35).abs() < 1e-12);
        assert!((cite.ema - (0.5 * 0.1 + 0.5 * 0.0375)).abs() < 1e-12);
        let english = ledger.credits.get(&clause_id("Answer in English.")).unwrap();
        assert!((english.ema - 0.1).abs() < 1e-12);
        assert_eq!(english.added_at_gen, 7);
        // Closing on the winner flags document order first, then the rest
        // by evidence.
        let report = ledger.into_report(&[grandchild], 5);
        assert!(report[0].in_champion && report[0].added_at_gen == 0);
        assert!(report[1].in_champion && report[1].text == "Cite sources.");
        assert!(report[2..].iter().all(|credit| !credit.in_champion));
        assert!(report[2].ema >= report[3].ema);
        // An empty delta vector records presence only. (A lone letter plus
        // a period, "B.", would fold to an empty marker-only clause.)
        let mut quiet = ClauseLedger::default();
        quiet.credit("Alpha.", "Alpha.\nBeta.", &[], 1);
        let beta = quiet.credits.get(&clause_id("Beta.")).unwrap();
        assert_eq!((beta.times_present, beta.times_credited), (1, 0));
        assert_eq!(quiet.ema_of(&clause_id("Beta.")), 0.0);
        assert!(clause_units("B.").is_empty());
    }

    #[test]
    fn clause_credit_report_lists_winner_clauses_with_evidence_tags() {
        let mut ledger = ClauseLedger::default();
        ledger.register_seed("Answer.");
        ledger.credit("Answer.", "Answer.\nCite sources.", &[0.1, 0.1], 1);
        ledger.credit("Answer.\nCite sources.", "Answer.\nCite sources.\nNote J1.", &[0.0, 0.0], 2);
        let credits = ledger.into_report(&["Answer.\nCite sources.\nNote J1."], 2);
        let result = RatchetResult {
            best_candidate: GEPACandidate {
                id: 2,
                instruction: "Answer.\nCite sources.\nNote J1.".into(),
                module_name: "m".into(),
                example_scores: Vec::new(),
                parent_id: Some(1),
                generation: 2,
                slots: Default::default(),
            },
            best_by_mean: None,
            all_candidates: Vec::new(),
            trials: Vec::new(),
            operator_stats: Vec::new(),
            total_rollouts: 0,
            total_lm_calls: 0,
            evolution_history: Vec::new(),
            gate_rejected: 0,
            race_killed: 0,
            faded: 0,
            audit_evicted: 0,
            audit_rollouts: 0,
            pins_tested: 0,
            pins_kept: 0,
            pin_probe_rollouts: 0,
            pin_assay_report: Vec::new(),
            skipped: 0,
            leak_retries: 0,
            leak_skipped: 0,
            length_retries: 0,
            length_skipped: 0,
            length_prior: None,
            length_cap_in_force: None,
            anchor_retries: 0,
            anchor_skipped: 0,
            fresh_selection_cols: 0,
            variant_selection_cols: 0,
            ablation_removed: 0,
            ablation_rollouts: 0,
            repair_probes_run: 0,
            repair_recovered: 0,
            repair_slot_stats: Vec::new(),
            repair_lessons: Vec::new(),
            certified: Some(true),
            certify_cols_used: 0,
            certify_mean_delta: 0.0,
            certify_e_value: 0.0,
            pins: Vec::new(),
            clause_credits: credits,
            prune_chars_before: 30,
            prune_chars_after: 22,
            prune_removed: 1,
            prune_rollouts: 40,
            prune_certified: Some(true),
        };
        let report = result.clause_credit_report();
        let lines: Vec<&str> = report.lines().collect();
        assert!(lines[0].contains("3 clauses seen, 3 in the shipped winner"));
        assert!(lines[1].contains("[seed") && lines[1].ends_with("Answer."));
        assert!(lines[2].contains("[   +") && lines[2].contains("Cite sources."));
        assert!(lines[3].contains("[   0") && lines[3].contains("Note J1."));
        assert!(lines[4].contains("occam prune: 30 -> 22 chars, 1 clauses removed, 40 rollouts, re-certified Some(true)"));
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
