//! The shared evaluation engine (vision §5.4): every optimizer is a thin
//! strategy over this core.
//!
//! # Why this lives in `optimizer/`, not `evaluate/`
//!
//! The `evaluate/` module owns the *strategy-free* primitives: the
//! [`TypedMetric`] trait and the traced rollout loop
//! (`evaluate_examples_traced`). This engine composes those primitives with
//! optimizer-side vocabulary — candidates, budgets, rollout caching, Pareto
//! bookkeeping, minibatch gating — and applies candidates through the
//! crate-internal mutation seam (`with_named_predictor` /
//! `DynPredictor::apply_update`) that already lives in `optimizer/`. Putting it
//! here keeps the dependency arrow one-way: `optimizer` → `evaluate`, never the
//! reverse.
//!
//! # The pieces
//!
//! - **[`Candidate`]** — a named set of parameter overlays (predictor name →
//!   instruction/demos) plus a stable content hash. This is the §5.4 overlay
//!   contract in its pre-IR form: cheap to clone, trivially serializable, and
//!   applied/restored in exactly one place ([`apply_candidate`] /
//!   [`restore_candidate`]) through the single mutation seam.
//! - **[`EvalEngine`]** — bounded-concurrency async fan-out over
//!   (candidate × examples) with per-rollout trace capture, a rollout cache,
//!   budget metering, minibatch gating, and a per-instance score matrix.
//! - **[`ScoreMatrix`] / [`ParetoView`]** — (candidates × examples) score
//!   bookkeeping generalizing what GEPA's `ParetoFrontier` does, usable by any
//!   strategy.
//!
//! # Concurrency model (and the candidate-parallelism seam)
//!
//! Candidates mutate shared module state through `apply_update`, so the engine
//! **serializes candidate application** and parallelizes across *examples*
//! within one candidate (`buffer_unordered`, bounded by
//! [`EngineConfig::concurrency`]). This is correct today because a module is
//! immutable (`&M`) for the duration of one candidate's fan-out.
//! Candidate-level parallelism — evaluating many candidates over one skeleton
//! simultaneously — requires overlays applied at render time (the IR `Overlay`
//! of vision §5.2) or module cloning; that seam is deliberately left open here:
//! when render-time overlays land, only [`EvalEngine::evaluate`] changes, not
//! its callers.
//!
//! # Cache keying
//!
//! Rollouts are cached on `(baseline hash, candidate hash, example uid,
//! cache salt)`:
//!
//! - the *baseline hash* is the module's [`ModuleState`] before the overlay is
//!   applied, so permanently installing a winner mid-run (COPRO between
//!   rounds, MIPRO after demo bootstrap) correctly invalidates stale entries;
//! - the *cache salt* ([`EngineConfig::cache_salt`]) is the sampling-params
//!   seam: today sampling params live on LM configs outside the candidate, so
//!   callers that change them must bump the salt. When model refs become
//!   overlay parameters, they fold into the candidate hash and the salt can
//!   retire.

use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use anyhow::{Result, anyhow};
use futures::stream::{self, StreamExt, TryStreamExt};
use serde::{Deserialize, Serialize};

use crate::core::{ModuleState, PredictState, StateUpdate};
use crate::evaluate::{DEFAULT_EVAL_CONCURRENCY, Eval, TypedMetric};
use crate::optimizer::pareto::ParetoStatistics;
use crate::optimizer::with_named_predictor;
use crate::predictors::Example;
use crate::trace::{JsonMap, Trace, TraceMeta, TraceOutcome, capture_with_meta};
use crate::utils::hash::StableHasher;
use crate::{Facet, LmUsage, Module, Signature};

/// Score tolerance for Pareto win/tie comparisons (matches the historical
/// `ParetoFrontier` tolerance).
pub(crate) const SCORE_EPS: f64 = 1e-6;

// ---------------------------------------------------------------------------
// Canonical hashing
// ---------------------------------------------------------------------------

/// Hashes a JSON value structurally with object keys visited in sorted order,
/// so values that differ only in map iteration order (e.g. `HashMap`-backed
/// demo rows under serde_json's `preserve_order`) hash identically.
fn hash_json_canonical(value: &serde_json::Value, hasher: &mut StableHasher) {
    use serde_json::Value;
    use std::hash::Hasher as _;

    fn write_str(hasher: &mut StableHasher, s: &str) {
        use std::hash::Hasher as _;
        hasher.write(&(s.len() as u64).to_le_bytes());
        hasher.write(s.as_bytes());
    }

    match value {
        Value::Null => hasher.write(b"n"),
        Value::Bool(b) => {
            hasher.write(b"b");
            hasher.write(&[*b as u8]);
        }
        Value::Number(n) => {
            hasher.write(b"#");
            write_str(hasher, &n.to_string());
        }
        Value::String(s) => {
            hasher.write(b"s");
            write_str(hasher, s);
        }
        Value::Array(items) => {
            hasher.write(b"[");
            for item in items {
                hash_json_canonical(item, hasher);
            }
            hasher.write(b"]");
        }
        Value::Object(map) => {
            hasher.write(b"{");
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            for key in keys {
                hasher.write(b"k");
                write_str(hasher, key);
                hash_json_canonical(&map[key.as_str()], hasher);
            }
            hasher.write(b"}");
        }
    }
}

/// Stable content hash of any serializable value, insensitive to map ordering.
pub(crate) fn canonical_hash<T: Serialize>(value: &T) -> u64 {
    use std::hash::Hasher as _;
    let mut hasher = StableHasher::new();
    match serde_json::to_value(value) {
        Ok(json) => hash_json_canonical(&json, &mut hasher),
        Err(_) => hasher.write(b"<unserializable>"),
    }
    hasher.finish()
}

fn json_object(value: Result<serde_json::Value, serde_json::Error>) -> Option<JsonMap> {
    match value {
        Ok(serde_json::Value::Object(map)) => Some(map),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Candidate as data
// ---------------------------------------------------------------------------

/// A per-predictor parameter overlay: which optimizable values to install.
///
/// `None` fields leave the predictor's current value untouched — an overlay is
/// a *partial* update, resolved against whatever module state is live when the
/// candidate is applied.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Overlay {
    /// `Some` installs this instruction override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instruction: Option<String>,
    /// `Some` replaces the demo set. Rows are flat JSON objects (field name →
    /// value, input and output fields merged), the same shape as
    /// [`PredictState::demos`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub demos: Option<Vec<JsonMap>>,
}

impl Overlay {
    fn to_update(&self) -> StateUpdate {
        StateUpdate {
            instruction: self.instruction.clone().map(Some),
            demos: self.demos.clone(),
        }
    }
}

/// A candidate is *data*: a named set of [`Overlay`]s (predictor name →
/// instruction/demos) plus a stable hash ([`Candidate::stable_hash`]).
///
/// This is the §5.4 overlay contract in its pre-IR form — cheap to clone,
/// serializable, applied and restored in exactly one place
/// ([`apply_candidate`] / [`restore_candidate`]). The empty candidate
/// (`Candidate::default()`) is the baseline: the module exactly as it is.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Candidate {
    /// Predictor name (fx slot name / facet dotted path) → overlay.
    pub overlays: BTreeMap<String, Overlay>,
}

impl Candidate {
    /// The empty candidate: no overlays, the module as-is.
    pub fn new() -> Self {
        Self::default()
    }

    /// Single-predictor instruction candidate — the trivial (COPRO/MIPRO) case.
    pub fn with_instruction(name: impl Into<String>, instruction: impl Into<String>) -> Self {
        let mut candidate = Self::new();
        candidate.set_instruction(name, instruction);
        candidate
    }

    /// Sets the instruction overlay for a named predictor.
    pub fn set_instruction(
        &mut self,
        name: impl Into<String>,
        instruction: impl Into<String>,
    ) -> &mut Self {
        self.overlays.entry(name.into()).or_default().instruction = Some(instruction.into());
        self
    }

    /// Sets the demo overlay for a named predictor. Each row is a flat JSON
    /// object with the signature's input and output fields merged.
    pub fn set_demos(&mut self, name: impl Into<String>, demos: Vec<JsonMap>) -> &mut Self {
        self.overlays.entry(name.into()).or_default().demos = Some(demos);
        self
    }

    pub fn is_empty(&self) -> bool {
        self.overlays.is_empty()
    }

    /// Stable content hash: identical overlay content hashes identically across
    /// processes and map orderings — the cache and checkpoint identity.
    pub fn stable_hash(&self) -> u64 {
        canonical_hash(self)
    }
}

/// Saved pre-overlay state for the predictors a candidate touched. Produced by
/// [`apply_candidate`], consumed by [`restore_candidate`].
#[derive(Clone, Debug)]
pub struct CandidateUndo {
    saved: BTreeMap<String, PredictState>,
}

/// Applies a candidate's overlays to a module through the single mutation seam
/// (`DynPredictor::apply_update`), returning the undo snapshot.
///
/// This is the **one** place candidate state is written. If any overlay fails
/// to apply (unknown predictor name, demo schema mismatch), the overlays
/// applied so far are rolled back before the error returns.
pub fn apply_candidate<M>(module: &mut M, candidate: &Candidate) -> Result<CandidateUndo>
where
    M: for<'a> Facet<'a>,
{
    let mut undo = CandidateUndo {
        saved: BTreeMap::new(),
    };
    for (name, overlay) in &candidate.overlays {
        let applied = with_named_predictor(module, name, |predictor| {
            let prior = predictor.dump_state();
            predictor.apply_update(overlay.to_update())?;
            Ok(prior)
        });
        match applied {
            Ok(prior) => {
                undo.saved.insert(name.clone(), prior);
            }
            Err(err) => {
                return match restore_candidate(module, undo) {
                    Ok(()) => Err(err),
                    Err(restore_err) => Err(anyhow!(
                        "failed to apply candidate: {err}; and failed to roll back partial application: {restore_err}"
                    )),
                };
            }
        }
    }
    Ok(undo)
}

/// Restores the pre-candidate state captured by [`apply_candidate`].
///
/// Attempts every predictor even if one fails, then reports the first error.
pub fn restore_candidate<M>(module: &mut M, undo: CandidateUndo) -> Result<()>
where
    M: for<'a> Facet<'a>,
{
    let mut first_error = None;
    for (name, state) in undo.saved {
        if let Err(err) = with_named_predictor(module, &name, |predictor| predictor.load_state(state.clone()))
            && first_error.is_none()
        {
            first_error = Some(anyhow!("failed to restore `{name}`: {err}"));
        }
    }
    match first_error {
        Some(err) => Err(err),
        None => Ok(()),
    }
}

/// Content hash of a module's full optimizable state ([`ModuleState`]) —
/// the "skeleton" a candidate overlays. Part of the rollout-cache key.
fn baseline_hash<M>(module: &mut M) -> Result<u64>
where
    M: for<'a> Facet<'a>,
{
    Ok(canonical_hash(&ModuleState::from_module(module)?))
}

// ---------------------------------------------------------------------------
// Budget metering
// ---------------------------------------------------------------------------

/// Hard caps on evaluation spend. `None` = unlimited.
///
/// `max_metric_calls` and `max_lm_calls` are metered at rollout granularity
/// (one module execution = one metric call = one LM call unit; auxiliary LM
/// spend like reflection calls is charged via [`EvalEngine::charge`]).
/// Exact per-span counts and token totals are tracked in [`Spend`]
/// (`lm_spans`, `tokens`) from the captured traces; `max_tokens` stops the
/// engine once recorded token usage reaches the cap.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Budget {
    pub max_metric_calls: Option<usize>,
    pub max_lm_calls: Option<usize>,
    pub max_tokens: Option<u64>,
}

impl Budget {
    pub fn unlimited() -> Self {
        Self::default()
    }

    /// Whether `upcoming_rollouts` more rollouts fit within the caps.
    pub fn allows(&self, spend: &Spend, upcoming_rollouts: usize) -> bool {
        if upcoming_rollouts == 0 {
            return true;
        }
        if self
            .max_metric_calls
            .is_some_and(|max| spend.metric_calls.saturating_add(upcoming_rollouts) > max)
        {
            return false;
        }
        if self
            .max_lm_calls
            .is_some_and(|max| spend.lm_calls.saturating_add(upcoming_rollouts) > max)
        {
            return false;
        }
        if self
            .max_tokens
            .is_some_and(|max| spend.tokens.total_tokens >= max)
        {
            return false;
        }
        true
    }
}

/// What the engine has consumed so far. Serialized into checkpoints and
/// reported to strategies.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Spend {
    /// Metric evaluations executed (cache hits don't re-run the metric).
    pub metric_calls: usize,
    /// LM call units: one per executed rollout plus auxiliary charges
    /// ([`EvalEngine::charge`]).
    pub lm_calls: usize,
    /// Exact `Predict` spans observed across captured rollout traces.
    pub lm_spans: usize,
    /// Rollouts served from the cache instead of executed.
    pub cache_hits: usize,
    /// Token totals summed from captured span usage.
    pub tokens: LmUsage,
}

// ---------------------------------------------------------------------------
// Rollout cache
// ---------------------------------------------------------------------------

/// In-memory rollout cache: `(baseline, candidate, example, salt)` → [`Eval`].
///
/// A candidate re-evaluated on a seen example returns the cached `Eval` with
/// no LM call and no metric call. Serialized into checkpoints so a resumed run
/// skips completed rollouts. (Disk-backed storage can layer on later — the key
/// is already a stable string.)
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RolloutCache {
    entries: BTreeMap<String, Eval>,
}

impl RolloutCache {
    fn key(baseline: u64, candidate: u64, example: u64, salt: u64) -> String {
        format!("{baseline:016x}:{candidate:016x}:{example:016x}:{salt:016x}")
    }

    pub fn get(&self, baseline: u64, candidate: u64, example: u64, salt: u64) -> Option<&Eval> {
        self.entries.get(&Self::key(baseline, candidate, example, salt))
    }

    pub fn insert(&mut self, baseline: u64, candidate: u64, example: u64, salt: u64, eval: Eval) {
        self.entries.insert(Self::key(baseline, candidate, example, salt), eval);
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Score matrix + Pareto bookkeeping
// ---------------------------------------------------------------------------

/// Per-instance score matrix: candidates × examples.
///
/// Rows are candidate indices (in registration order), columns are example
/// indices. Cells are `None` until a candidate has been evaluated on that
/// example. This generalizes what GEPA's `ParetoFrontier` tracked so any
/// strategy can use it; dominance queries go through [`ScoreMatrix::pareto`].
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ScoreMatrix {
    columns: usize,
    rows: Vec<Vec<Option<f64>>>,
}

impl ScoreMatrix {
    pub fn new(columns: usize) -> Self {
        Self {
            columns,
            rows: Vec::new(),
        }
    }

    /// Number of candidate rows.
    pub fn candidates(&self) -> usize {
        self.rows.len()
    }

    /// Number of example columns.
    pub fn examples(&self) -> usize {
        self.columns
    }

    /// Grows the matrix to at least `candidates` rows. Rows are stored
    /// sparsely — cells past a row's recorded extent read as `None`.
    pub fn ensure_rows(&mut self, candidates: usize) {
        while self.rows.len() < candidates {
            self.rows.push(Vec::new());
        }
    }

    /// Records a score, growing rows and columns as needed.
    pub fn record(&mut self, candidate: usize, example: usize, score: f64) {
        self.ensure_rows(candidate + 1);
        if example >= self.columns {
            self.columns = example + 1;
        }
        let row = &mut self.rows[candidate];
        if row.len() <= example {
            row.resize(example + 1, None);
        }
        row[example] = Some(score);
    }

    pub fn score(&self, candidate: usize, example: usize) -> Option<f64> {
        self.rows.get(candidate)?.get(example).copied().flatten()
    }

    /// A candidate's recorded row. May be shorter than
    /// [`examples`](Self::examples); missing cells mean "not evaluated".
    pub fn row(&self, candidate: usize) -> &[Option<f64>] {
        &self.rows[candidate]
    }

    /// Mean over the cells this candidate has been scored on (`None` if none).
    pub fn mean(&self, candidate: usize) -> Option<f64> {
        let filled: Vec<f64> = self.rows.get(candidate)?.iter().flatten().copied().collect();
        if filled.is_empty() {
            None
        } else {
            Some(filled.iter().sum::<f64>() / filled.len() as f64)
        }
    }

    /// The fully-scored candidate with the highest mean.
    pub fn best_by_mean(&self) -> Option<usize> {
        (0..self.rows.len())
            .filter_map(|idx| self.mean(idx).map(|mean| (idx, mean)))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
    }

    /// Pareto view over all example columns.
    pub fn pareto(&self) -> ParetoView {
        self.pareto_over(&(0..self.columns).collect::<Vec<_>>())
    }

    /// Pareto view restricted to a subset of example columns (e.g. GEPA's
    /// validation columns when train and validation examples share a matrix).
    pub fn pareto_over(&self, columns: &[usize]) -> ParetoView {
        let best: Vec<Option<f64>> = columns
            .iter()
            .map(|&col| {
                self.rows
                    .iter()
                    .filter_map(|row| row.get(col).copied().flatten())
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            })
            .collect();
        let wins: Vec<usize> = self
            .rows
            .iter()
            .map(|row| {
                columns
                    .iter()
                    .zip(&best)
                    .filter(|(col, best)| {
                        matches!(
                            (row.get(**col).copied().flatten(), best),
                            (Some(score), Some(best)) if score + SCORE_EPS >= *best
                        )
                    })
                    .count()
            })
            .collect();
        ParetoView { best, wins }
    }
}

/// Dominance snapshot computed from a [`ScoreMatrix`]: which candidates win
/// (or tie, within tolerance) on at least one example.
///
/// A candidate that is best on *zero* examples is dominated — the historical
/// `ParetoFrontier` pruning falls out of `wins == 0`.
#[derive(Clone, Debug)]
pub struct ParetoView {
    best: Vec<Option<f64>>,
    wins: Vec<usize>,
}

impl ParetoView {
    /// Best score achieved per (viewed) example column, `None` if unscored.
    pub fn best_scores(&self) -> &[Option<f64>] {
        &self.best
    }

    /// How many viewed examples this candidate wins or ties on.
    pub fn wins(&self, candidate: usize) -> usize {
        self.wins.get(candidate).copied().unwrap_or(0)
    }

    /// Candidate indices on the frontier (winning on ≥ 1 example), ascending.
    pub fn frontier(&self) -> Vec<usize> {
        (0..self.wins.len()).filter(|&idx| self.wins[idx] > 0).collect()
    }

    /// Frontier statistics in the shape GEPA reports.
    pub fn statistics(&self) -> ParetoStatistics {
        let frontier = self.frontier();
        let coverage: Vec<usize> = frontier.iter().map(|&idx| self.wins[idx]).collect();
        ParetoStatistics {
            num_candidates: frontier.len(),
            num_examples_covered: self.best.iter().filter(|best| best.is_some()).count(),
            avg_coverage: if coverage.is_empty() {
                0.0
            } else {
                coverage.iter().sum::<usize>() as f32 / coverage.len() as f32
            },
            max_coverage: coverage.iter().copied().max().unwrap_or(0),
            min_coverage: coverage.iter().copied().min().unwrap_or(0),
        }
    }
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// Engine tuning knobs.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct EngineConfig {
    /// Rollouts in flight at once within one candidate's fan-out.
    pub concurrency: usize,
    /// Hard spend caps; the engine stops cleanly when a batch wouldn't fit.
    pub budget: Budget,
    /// Sampling-params seam: folded into every cache key. Bump it when
    /// changing LM sampling settings outside the candidate (see module docs).
    pub cache_salt: u64,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            concurrency: DEFAULT_EVAL_CONCURRENCY,
            budget: Budget::unlimited(),
            cache_salt: 0,
        }
    }
}

/// One evaluated (or cache-served) rollout.
#[derive(Clone, Debug)]
pub struct RolloutOutcome {
    /// Index into the engine's example set.
    pub example: usize,
    pub eval: Eval,
    /// The captured execution trace; `None` when served from the cache.
    pub trace: Option<Trace>,
}

/// A candidate's results over one evaluation batch, in request order.
#[derive(Clone, Debug)]
pub struct CandidateEval {
    /// Engine candidate index.
    pub candidate: usize,
    pub rollouts: Vec<RolloutOutcome>,
}

impl CandidateEval {
    /// Arithmetic mean score over the batch (`0.0` for an empty batch).
    pub fn mean(&self) -> f64 {
        if self.rollouts.is_empty() {
            return 0.0;
        }
        self.rollouts.iter().map(|r| r.eval.score).sum::<f64>() / self.rollouts.len() as f64
    }

    pub fn scores(&self) -> Vec<f64> {
        self.rollouts.iter().map(|r| r.eval.score).collect()
    }
}

/// Result of [`EvalEngine::evaluate`].
#[derive(Clone, Debug)]
pub enum EvalOutcome {
    Complete(CandidateEval),
    /// The batch's uncached portion didn't fit in the remaining budget;
    /// nothing ran and spend is unchanged. `needed` is the number of rollouts
    /// that would have executed.
    BudgetExhausted { needed: usize },
}

impl EvalOutcome {
    /// The completed evaluation, if the budget allowed it.
    pub fn completed(self) -> Option<CandidateEval> {
        match self {
            Self::Complete(eval) => Some(eval),
            Self::BudgetExhausted { .. } => None,
        }
    }
}

/// Result of [`EvalEngine::evaluate_gated`].
#[derive(Clone, Debug)]
pub enum GateOutcome {
    /// Minibatch (or promotion) evaluation didn't fit the remaining budget.
    BudgetExhausted { needed: usize },
    /// Minibatch mean did not beat the threshold; no full evaluation ran.
    Rejected { minibatch: CandidateEval },
    /// Minibatch mean beat the threshold and the full evaluation ran.
    Promoted {
        minibatch: CandidateEval,
        full: CandidateEval,
    },
}

/// Serialized engine state: candidates, score matrix, spend, and the rollout
/// cache. A resumed run skips completed rollouts via the cache.
#[derive(Serialize, Deserialize)]
struct EngineCheckpoint {
    version: u32,
    example_uids: Vec<u64>,
    candidates: Vec<Candidate>,
    matrix: ScoreMatrix,
    cache: RolloutCache,
    spend: Spend,
}

/// The shared evaluation core (vision §5.4).
///
/// Owns the example set, the candidate registry, the score matrix, the rollout
/// cache, and the budget meter. Strategies (GEPA, COPRO, MIPRO, bootstrap)
/// register [`Candidate`]s and call [`evaluate`](Self::evaluate) /
/// [`evaluate_gated`](Self::evaluate_gated); the engine handles application,
/// fan-out, caching, accounting, and bookkeeping.
pub struct EvalEngine<'m, S: Signature, MT> {
    examples: Vec<Example<S>>,
    example_uids: Vec<u64>,
    metric: &'m MT,
    config: EngineConfig,
    candidates: Vec<Candidate>,
    candidate_hashes: Vec<u64>,
    matrix: ScoreMatrix,
    cache: RolloutCache,
    spend: Spend,
}

impl<'m, S, MT> EvalEngine<'m, S, MT>
where
    S: Signature,
{
    pub fn new(examples: Vec<Example<S>>, metric: &'m MT, config: EngineConfig) -> Self {
        let example_uids = examples
            .iter()
            .map(|example| canonical_hash(&(&example.input, &example.output)))
            .collect();
        let matrix = ScoreMatrix::new(examples.len());
        Self {
            examples,
            example_uids,
            metric,
            config,
            candidates: Vec::new(),
            candidate_hashes: Vec::new(),
            matrix,
            cache: RolloutCache::default(),
            spend: Spend::default(),
        }
    }

    /// Rebuilds an engine from a [`checkpoint`](Self::checkpoint), validating
    /// that `examples` matches the checkpointed set. Completed rollouts are
    /// served from the restored cache instead of re-executing.
    pub fn resume(
        examples: Vec<Example<S>>,
        metric: &'m MT,
        config: EngineConfig,
        checkpoint: &str,
    ) -> Result<Self> {
        let checkpoint: EngineCheckpoint =
            serde_json::from_str(checkpoint).map_err(|err| anyhow!("invalid engine checkpoint: {err}"))?;
        if checkpoint.version != 1 {
            return Err(anyhow!(
                "unsupported engine checkpoint version {}",
                checkpoint.version
            ));
        }
        let mut engine = Self::new(examples, metric, config);
        if engine.example_uids != checkpoint.example_uids {
            return Err(anyhow!(
                "engine checkpoint does not match the provided example set"
            ));
        }
        engine.candidate_hashes = checkpoint.candidates.iter().map(Candidate::stable_hash).collect();
        engine.candidates = checkpoint.candidates;
        engine.matrix = checkpoint.matrix;
        engine.cache = checkpoint.cache;
        engine.spend = checkpoint.spend;
        engine.matrix.ensure_rows(engine.candidates.len());
        Ok(engine)
    }

    /// Serializes engine state (candidates, matrix, spend, cache) to JSON.
    pub fn checkpoint(&self) -> Result<String> {
        serde_json::to_string(&EngineCheckpoint {
            version: 1,
            example_uids: self.example_uids.clone(),
            candidates: self.candidates.clone(),
            matrix: self.matrix.clone(),
            cache: self.cache.clone(),
            spend: self.spend,
        })
        .map_err(|err| anyhow!("failed to serialize engine checkpoint: {err}"))
    }

    pub fn examples(&self) -> &[Example<S>] {
        &self.examples
    }

    pub fn num_examples(&self) -> usize {
        self.examples.len()
    }

    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    pub fn spend(&self) -> &Spend {
        &self.spend
    }

    pub fn matrix(&self) -> &ScoreMatrix {
        &self.matrix
    }

    /// Pareto view over all example columns (see [`ScoreMatrix::pareto`]).
    pub fn pareto(&self) -> ParetoView {
        self.matrix.pareto()
    }

    /// Pareto view over a column subset (see [`ScoreMatrix::pareto_over`]).
    pub fn pareto_over(&self, columns: &[usize]) -> ParetoView {
        self.matrix.pareto_over(columns)
    }

    /// Registers a candidate, deduplicating by content hash. Returns its index.
    pub fn register(&mut self, candidate: Candidate) -> usize {
        let hash = candidate.stable_hash();
        if let Some(existing) = self.candidate_hashes.iter().position(|&h| h == hash) {
            return existing;
        }
        self.candidates.push(candidate);
        self.candidate_hashes.push(hash);
        self.matrix.ensure_rows(self.candidates.len());
        self.candidates.len() - 1
    }

    pub fn candidate(&self, index: usize) -> &Candidate {
        &self.candidates[index]
    }

    pub fn candidate_hash(&self, index: usize) -> u64 {
        self.candidate_hashes[index]
    }

    pub fn num_candidates(&self) -> usize {
        self.candidates.len()
    }

    /// Whether `upcoming_rollouts` more rollouts fit the remaining budget.
    pub fn budget_allows(&self, upcoming_rollouts: usize) -> bool {
        self.config.budget.allows(&self.spend, upcoming_rollouts)
    }

    /// Charges auxiliary spend against the budget — reflection LM calls,
    /// teacher passes, or any strategy-side consumption the engine didn't run.
    pub fn charge(&mut self, metric_calls: usize, lm_calls: usize) {
        self.spend.metric_calls = self.spend.metric_calls.saturating_add(metric_calls);
        self.spend.lm_calls = self.spend.lm_calls.saturating_add(lm_calls);
    }

    /// Evaluates a registered candidate over `subset` example indices (`None`
    /// = the full set): applies the overlay through the mutation seam, fans
    /// out uncached rollouts with bounded concurrency under per-rollout trace
    /// capture, restores the module, and records scores into the matrix and
    /// cache.
    ///
    /// Cached rollouts return their `Eval` with `trace: None` and consume no
    /// budget. If the uncached portion doesn't fit the remaining budget the
    /// engine runs nothing and returns [`EvalOutcome::BudgetExhausted`].
    pub async fn evaluate<M>(
        &mut self,
        module: &mut M,
        candidate: usize,
        subset: Option<&[usize]>,
    ) -> Result<EvalOutcome>
    where
        S::Input: Clone,
        M: Module<Input = S::Input> + for<'a> Facet<'a>,
        MT: TypedMetric<S, M>,
    {
        let candidate_hash = *self
            .candidate_hashes
            .get(candidate)
            .ok_or_else(|| anyhow!("candidate index {candidate} is not registered"))?;
        let indices: Vec<usize> = match subset {
            Some(subset) => subset.to_vec(),
            None => (0..self.examples.len()).collect(),
        };
        if let Some(&bad) = indices.iter().find(|&&idx| idx >= self.examples.len()) {
            return Err(anyhow!(
                "example index {bad} out of range ({} examples)",
                self.examples.len()
            ));
        }

        let baseline = baseline_hash(module)?;
        let salt = self.config.cache_salt;

        let mut cached: HashMap<usize, Eval> = HashMap::new();
        let mut pending: Vec<usize> = Vec::new();
        for &idx in &indices {
            match self.cache.get(baseline, candidate_hash, self.example_uids[idx], salt) {
                Some(eval) => {
                    cached.insert(idx, eval.clone());
                }
                None => {
                    if !cached.contains_key(&idx) && !pending.contains(&idx) {
                        pending.push(idx);
                    }
                }
            }
        }

        if !self.budget_allows(pending.len()) {
            return Ok(EvalOutcome::BudgetExhausted {
                needed: pending.len(),
            });
        }

        let fresh: Vec<(usize, Eval, Trace)> = if pending.is_empty() {
            Vec::new()
        } else {
            let undo = apply_candidate(module, &self.candidates[candidate])?;
            let ran = self.run_rollouts(&*module, &pending, candidate_hash).await;
            let restored = restore_candidate(module, undo);
            match (ran, restored) {
                (Ok(fresh), Ok(())) => fresh,
                (Ok(_), Err(restore_err)) => return Err(restore_err),
                (Err(eval_err), Ok(())) => return Err(eval_err),
                (Err(eval_err), Err(restore_err)) => {
                    return Err(anyhow!(
                        "candidate evaluation failed: {eval_err}; failed to restore module state: {restore_err}"
                    ));
                }
            }
        };

        // Accounting: fresh rollouts consume budget, cached hits are free.
        self.spend.metric_calls += fresh.len();
        self.spend.lm_calls += fresh.len();
        self.spend.cache_hits += cached.len();
        for (_, _, trace) in &fresh {
            self.spend.lm_spans += trace.spans.len();
            for span in &trace.spans {
                self.spend.tokens = self.spend.tokens + span.usage;
            }
        }

        // Bookkeeping: cache inserts + matrix records.
        let mut fresh_by_idx: HashMap<usize, (Eval, Trace)> = HashMap::with_capacity(fresh.len());
        for (idx, eval, trace) in fresh {
            self.cache
                .insert(baseline, candidate_hash, self.example_uids[idx], salt, eval.clone());
            self.matrix.record(candidate, idx, eval.score);
            fresh_by_idx.insert(idx, (eval, trace));
        }
        for (&idx, eval) in &cached {
            self.matrix.record(candidate, idx, eval.score);
        }

        let rollouts = indices
            .iter()
            .map(|&idx| {
                if let Some((eval, trace)) = fresh_by_idx.remove(&idx) {
                    RolloutOutcome {
                        example: idx,
                        eval,
                        trace: Some(trace),
                    }
                } else {
                    let eval = cached
                        .get(&idx)
                        .cloned()
                        .expect("every requested index is either fresh or cached");
                    RolloutOutcome {
                        example: idx,
                        eval,
                        trace: None,
                    }
                }
            })
            .collect();

        Ok(EvalOutcome::Complete(CandidateEval {
            candidate,
            rollouts,
        }))
    }

    /// Minibatch gating (the GEPA acceptance pattern): evaluates the candidate
    /// on `minibatch`; only if the minibatch mean strictly beats `threshold`
    /// does it promote to a full-set evaluation.
    pub async fn evaluate_gated<M>(
        &mut self,
        module: &mut M,
        candidate: usize,
        minibatch: &[usize],
        threshold: f64,
    ) -> Result<GateOutcome>
    where
        S::Input: Clone,
        M: Module<Input = S::Input> + for<'a> Facet<'a>,
        MT: TypedMetric<S, M>,
    {
        let minibatch_eval = match self.evaluate(module, candidate, Some(minibatch)).await? {
            EvalOutcome::Complete(eval) => eval,
            EvalOutcome::BudgetExhausted { needed } => {
                return Ok(GateOutcome::BudgetExhausted { needed });
            }
        };

        if minibatch_eval.mean() <= threshold {
            return Ok(GateOutcome::Rejected {
                minibatch: minibatch_eval,
            });
        }

        match self.evaluate(module, candidate, None).await? {
            EvalOutcome::Complete(full) => Ok(GateOutcome::Promoted {
                minibatch: minibatch_eval,
                full,
            }),
            EvalOutcome::BudgetExhausted { needed } => Ok(GateOutcome::BudgetExhausted { needed }),
        }
    }

    /// Bounded-concurrency fan-out over uncached examples for one applied
    /// candidate. `module` is immutable here — the overlay was installed by
    /// the caller — so example-level parallelism is safe.
    async fn run_rollouts<M>(
        &self,
        module: &M,
        pending: &[usize],
        candidate_hash: u64,
    ) -> Result<Vec<(usize, Eval, Trace)>>
    where
        S::Input: Clone,
        M: Module<Input = S::Input>,
        MT: TypedMetric<S, M>,
    {
        let metric = self.metric;
        stream::iter(pending.iter().map(|&idx| {
            let example = &self.examples[idx];
            async move {
                let input = example.input.clone();
                let meta = TraceMeta {
                    candidate_hash: Some(candidate_hash),
                    input: json_object(serde_json::to_value(&example.input)),
                    ..TraceMeta::default()
                };
                let started = Instant::now();
                let (result, mut trace) = capture_with_meta(meta, || module.call(input)).await;
                let predicted = result.map_err(|err| anyhow!("{err}"))?;
                // Metric runs outside the capture scope so LM-as-judge metrics
                // don't pollute the execution trace.
                let eval = metric.evaluate(example, &predicted, Some(&trace)).await?;
                trace.outcome = Some(TraceOutcome {
                    output: json_object(serde_json::to_value(&*predicted)),
                    error: None,
                    eval: Some(eval.clone()),
                    duration_us: started.elapsed().as_micros() as u64,
                });
                Ok::<_, anyhow::Error>((idx, eval, trace))
            }
        }))
        .buffer_unordered(self.config.concurrency.max(1))
        .try_collect()
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn demo(pairs: &[(&str, &str)]) -> JsonMap {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), serde_json::Value::String(v.to_string())))
            .collect()
    }

    #[test]
    fn candidate_hash_is_stable_across_orderings() {
        let mut a = Candidate::new();
        a.set_instruction("drafter", "be brief");
        a.set_demos("refiner", vec![demo(&[("q", "one"), ("a", "1"), ("b", "2")])]);

        let mut b = Candidate::new();
        b.set_demos("refiner", vec![demo(&[("q", "one"), ("a", "1"), ("b", "2")])]);
        b.set_instruction("drafter", "be brief");

        assert_eq!(a.stable_hash(), b.stable_hash());

        let mut c = a.clone();
        c.set_instruction("drafter", "be thorough");
        assert_ne!(a.stable_hash(), c.stable_hash());

        let empty = Candidate::new();
        assert_ne!(a.stable_hash(), empty.stable_hash());
        assert_eq!(empty.stable_hash(), Candidate::default().stable_hash());
    }

    #[test]
    fn candidate_round_trips_through_json() {
        let mut candidate = Candidate::new();
        candidate.set_instruction("drafter", "be brief");
        candidate.set_demos("drafter", vec![demo(&[("q", "one"), ("a", "1")])]);

        let json = serde_json::to_string(&candidate).unwrap();
        let restored: Candidate = serde_json::from_str(&json).unwrap();
        assert_eq!(candidate.stable_hash(), restored.stable_hash());
    }

    #[test]
    fn score_matrix_pareto_wins_frontier_and_stats() {
        let mut matrix = ScoreMatrix::new(3);
        // c0 wins example 0; c1 wins examples 1 and 2; c2 dominated everywhere.
        matrix.record(0, 0, 0.9);
        matrix.record(0, 1, 0.5);
        matrix.record(0, 2, 0.5);
        matrix.record(1, 0, 0.5);
        matrix.record(1, 1, 0.9);
        matrix.record(1, 2, 0.9);
        matrix.record(2, 0, 0.3);
        matrix.record(2, 1, 0.3);
        matrix.record(2, 2, 0.3);

        let pareto = matrix.pareto();
        assert_eq!(pareto.wins(0), 1);
        assert_eq!(pareto.wins(1), 2);
        assert_eq!(pareto.wins(2), 0);
        assert_eq!(pareto.frontier(), vec![0, 1]);
        assert_eq!(pareto.best_scores(), &[Some(0.9), Some(0.9), Some(0.9)]);

        let stats = pareto.statistics();
        assert_eq!(stats.num_candidates, 2);
        assert_eq!(stats.num_examples_covered, 3);
        assert_eq!(stats.max_coverage, 2);
        assert_eq!(stats.min_coverage, 1);

        // Ties within tolerance count as wins.
        matrix.record(2, 0, 0.9);
        assert_eq!(matrix.pareto().wins(2), 1);

        // Column-restricted view: over example 0 only, c1 no longer wins.
        let restricted = matrix.pareto_over(&[0]);
        assert_eq!(restricted.wins(1), 0);
        assert_eq!(restricted.frontier(), vec![0, 2]);

        assert_eq!(matrix.best_by_mean(), Some(1));
        assert_eq!(matrix.mean(0), Some((0.9 + 0.5 + 0.5) / 3.0));
    }

    #[test]
    fn budget_allows_enforces_caps() {
        let budget = Budget {
            max_metric_calls: Some(4),
            max_lm_calls: None,
            max_tokens: None,
        };
        let mut spend = Spend::default();
        assert!(budget.allows(&spend, 4));
        assert!(!budget.allows(&spend, 5));
        spend.metric_calls = 3;
        assert!(budget.allows(&spend, 1));
        assert!(!budget.allows(&spend, 2));
        // Zero upcoming rollouts always fit (cache-only batches).
        spend.metric_calls = 10;
        assert!(budget.allows(&spend, 0));

        let token_budget = Budget {
            max_metric_calls: None,
            max_lm_calls: None,
            max_tokens: Some(100),
        };
        let mut spend = Spend::default();
        assert!(token_budget.allows(&spend, 1));
        spend.tokens.total_tokens = 100;
        assert!(!token_budget.allows(&spend, 1));
    }
}
