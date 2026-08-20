//! The shared evaluation engine (vision §5.4): every optimizer is a thin
//! strategy over this core.
//!
//! # One engine, two lanes
//!
//! [`Engine`] owns the strategy-independent bookkeeping — candidate registry,
//! rollout cache, budget metering, score matrix / Pareto views, minibatch
//! gating — and evaluates candidates against an
//! [`OptimizeTarget`](crate::optimizer::OptimizeTarget), which is one of two
//! lanes:
//!
//! - **module lane** — a typed module + trainset + [`TypedMetric`]. Candidates
//!   are name-keyed [`Candidate`]s injected *ambiently* per rollout via
//!   [`fx::with_params`](crate::fx::with_params); each
//!   [`Predict`](crate::Predict) leaf binds its own entry at call time.
//!   Nothing is ever mutated during evaluation, so candidates fan out
//!   concurrently exactly like the program lane.
//! - **program lane** — an interpreter-loaded [`Program`](crate::ir::Program)
//!   + labeled [`DemoRow`](crate::ir::DemoRow)s + a
//!   [`ProgramMetric`](crate::optimizer::ProgramMetric). Candidates are
//!   [`ir::Overlay`](crate::ir::Overlay)s (or [`Candidate`]s bound through
//!   [`fx::Params::bind`](crate::fx::Params::bind)) read through at render
//!   time.
//!
//! The traced-rollout loop itself is owned by `evaluate/` —
//! the engine composes `rollout_traced`, it does not reimplement it.
//!
//! # Cache keying
//!
//! Rollouts are cached on `(baseline identity, candidate hash, example uid,
//! cache salt)`:
//!
//! - the *baseline identity* is computed once per run when the target is
//!   constructed (module lane: hash of the predictors() state snapshot;
//!   program lane: the program hash). Installing a winner and building a new
//!   target yields a new baseline, invalidating stale entries;
//! - the *cache salt* ([`EngineConfig::cache_salt`]) is the sampling-params
//!   seam: sampling params live on LM configs outside the candidate, so
//!   callers that change them must bump the salt.

use std::collections::{BTreeMap, HashMap};

use anyhow::{Result, anyhow};
use futures::stream::{self, StreamExt, TryStreamExt};
use serde::{Deserialize, Serialize};

use crate::evaluate::{DEFAULT_EVAL_CONCURRENCY, Eval};
use crate::optimizer::target::OptimizeTarget;
use crate::trace::{JsonMap, Trace};
use crate::utils::hash::StableHasher;
use crate::LmUsage;

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

// ---------------------------------------------------------------------------
// Candidate as data
// ---------------------------------------------------------------------------

/// One leaf's slice of a [`Candidate`]: which optimizable values to inject.
///
/// Unset fields leave the leaf's incumbent value untouched — a candidate is a
/// *partial* configuration, resolved per slot at render time (ambient entries
/// win over instance state, exactly the precedence the old mutation seam had).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CandidateSlot {
    /// `Some` injects this instruction override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instruction: Option<String>,
    /// Explicitly reset the instruction to the signature default, winning
    /// over any instance override. Mutually exclusive with `instruction`.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub clear_instruction: bool,
    /// `Some` replaces the demo set (an empty vec clears it). Rows are flat
    /// JSON objects (field name → value, input and output fields merged), the
    /// same shape as [`PredictState::demos`](crate::core::PredictState).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub demos: Option<Vec<JsonMap>>,
}

/// A candidate is *data*: name-keyed per-leaf overlays plus a stable content
/// hash ([`Candidate::stable_hash`]). The candidate currency of the module
/// lane, and — via [`to_params`](Candidate::to_params) +
/// [`fx::Params::bind`](crate::fx::Params::bind) — of the program lane too.
///
/// Cheap to clone, serializable, and **never applied by mutation**: the
/// engine scopes it ambiently around each rollout
/// ([`fx::with_params`](crate::fx::with_params)); the single mutating step is
/// the caller-driven final install
/// ([`OptimizeTarget::install`](crate::optimizer::OptimizeTarget::install)).
/// The empty candidate (`Candidate::default()`) is the baseline: the module
/// exactly as it is.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Candidate {
    /// Leaf name (the [`Predictors`](crate::Predictors) contract name) → slot.
    pub slots: BTreeMap<String, CandidateSlot>,
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
        let slot = self.slots.entry(name.into()).or_default();
        slot.instruction = Some(instruction.into());
        slot.clear_instruction = false;
        self
    }

    /// Explicitly resets a named predictor's instruction to its signature
    /// default (winning over any instance override).
    pub fn clear_instruction(&mut self, name: impl Into<String>) -> &mut Self {
        let slot = self.slots.entry(name.into()).or_default();
        slot.instruction = None;
        slot.clear_instruction = true;
        self
    }

    /// Sets the demo overlay for a named predictor. Each row is a flat JSON
    /// object with the signature's input and output fields merged; an empty
    /// vec clears the demo set.
    pub fn set_demos(&mut self, name: impl Into<String>, demos: Vec<JsonMap>) -> &mut Self {
        self.slots.entry(name.into()).or_default().demos = Some(demos);
        self
    }

    /// The instruction this candidate injects for `name`, if any.
    pub fn instruction_of(&self, name: &str) -> Option<&str> {
        self.slots.get(name)?.instruction.as_deref()
    }

    /// The demo set this candidate injects for `name`, if any.
    pub fn demos_of(&self, name: &str) -> Option<&[JsonMap]> {
        self.slots.get(name)?.demos.as_deref()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Stable content hash: identical content hashes identically across
    /// processes and map orderings — the cache identity.
    pub fn stable_hash(&self) -> u64 {
        canonical_hash(self)
    }

    /// Converts to the ambient-injection currency: name-keyed
    /// [`fx::Params`](crate::fx::Params) with explicit clears preserved.
    pub fn to_params(&self) -> crate::fx::Params {
        let mut params = crate::fx::Params::new();
        for (name, slot) in &self.slots {
            if slot.clear_instruction {
                params.clear_instruction(name.clone());
            } else if let Some(text) = &slot.instruction {
                params.set_instruction(name.clone(), text.clone());
            }
            if let Some(demos) = &slot.demos {
                params.set_demos(name.clone(), demos.clone());
            }
        }
        params
    }
}

// ---------------------------------------------------------------------------
// Budget metering
// ---------------------------------------------------------------------------

/// Hard caps on evaluation spend. `None` = unlimited.
///
/// `max_metric_calls` and `max_lm_calls` are metered at rollout granularity
/// (one module execution = one metric call = one LM call unit; auxiliary LM
/// spend like reflection calls is charged via [`Engine::charge`]).
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

/// What the engine has consumed so far. Reported to strategies.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Spend {
    /// Metric evaluations executed (cache hits don't re-run the metric).
    pub metric_calls: usize,
    /// LM call units: one per executed rollout plus auxiliary charges
    /// ([`Engine::charge`]).
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
/// no LM call and no metric call. (Disk-backed storage can layer on later —
/// the key is already a stable string.)
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

/// Snapshot of the Pareto frontier at a point in the search.
///
/// Useful for plotting convergence. A healthy search has `num_candidates` growing
/// slowly (diversity is maintained) while `avg_coverage` increases (candidates are
/// getting more robust). If `num_candidates` is 1, the search has collapsed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoStatistics {
    /// Candidates currently on the frontier. 1 means the search has converged
    /// (or collapsed) to a single instruction.
    pub num_candidates: usize,
    /// Examples where at least one frontier candidate is the best. Should approach
    /// total eval set size as the search progresses.
    pub num_examples_covered: usize,
    /// Mean examples won per candidate. Higher means candidates are more robust;
    /// lower means more specialization.
    pub avg_coverage: f32,
    /// Most examples won by any single candidate.
    pub max_coverage: usize,
    /// Fewest examples won by any frontier candidate (always >= 1 by construction).
    pub min_coverage: usize,
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// Engine tuning knobs.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct EngineConfig {
    /// Rollouts in flight at once within one evaluation batch.
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
    /// Index into the target's example set.
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

/// Result of [`Engine::evaluate`].
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

/// Result of [`Engine::evaluate_many`].
#[derive(Clone, Debug)]
pub enum BatchEvalOutcome {
    /// One [`CandidateEval`] per requested candidate, in request order.
    Complete(Vec<CandidateEval>),
    /// The uncached portion of the batch didn't fit the remaining budget;
    /// nothing ran and spend is unchanged.
    BudgetExhausted { needed: usize },
}

impl BatchEvalOutcome {
    /// The completed evaluations, if the budget allowed the batch.
    pub fn completed(self) -> Option<Vec<CandidateEval>> {
        match self {
            Self::Complete(evals) => Some(evals),
            Self::BudgetExhausted { .. } => None,
        }
    }
}

/// Result of [`Engine::evaluate_gated`].
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

/// A registered candidate: its cache-identity hash plus its payload.
pub(crate) enum CandidatePayload {
    /// Module-lane (name-keyed) candidate, pre-converted to the ambient
    /// injection currency. Also evaluable on a program target via
    /// [`fx::Params::bind`](crate::fx::Params::bind).
    Params {
        candidate: Candidate,
        params: std::sync::Arc<crate::fx::Params>,
    },
    /// Program-lane native candidate (can carry non-Params kinds: model refs,
    /// context policies, code). Only evaluable on a program target.
    Overlay(std::sync::Arc<crate::ir::Overlay>),
}

/// A candidate bound against a concrete target, ready to inject per rollout.
#[derive(Clone)]
pub(crate) enum BoundCandidate {
    Params(std::sync::Arc<crate::fx::Params>),
    Overlay(std::sync::Arc<crate::ir::Overlay>),
}

/// The shared evaluation core (vision §5.4).
///
/// Owns the candidate registry, the score matrix, the rollout cache, and the
/// budget meter. Strategies (GEPA, COPRO, MIPRO, SIMBA, bootstrap) register
/// [`Candidate`]s and call [`evaluate`](Self::evaluate) /
/// [`evaluate_many`](Self::evaluate_many) /
/// [`evaluate_gated`](Self::evaluate_gated) against an
/// [`OptimizeTarget`](crate::optimizer::OptimizeTarget); the engine handles
/// binding, fan-out, caching, accounting, and bookkeeping. Because candidate
/// injection is ambient (never mutation), rollouts for *different candidates*
/// share one bounded-concurrency fan-out in both lanes.
pub struct Engine {
    config: EngineConfig,
    candidates: Vec<(u64, CandidatePayload)>,
    matrix: ScoreMatrix,
    cache: RolloutCache,
    spend: Spend,
    /// High-water mark of *distinct candidates* with rollouts in flight at
    /// the same instant (see [`peak_candidate_concurrency`](Self::peak_candidate_concurrency)).
    peak_candidates_in_flight: usize,
}

impl Engine {
    pub fn new(config: EngineConfig) -> Self {
        Self {
            config,
            candidates: Vec::new(),
            matrix: ScoreMatrix::new(0),
            cache: RolloutCache::default(),
            spend: Spend::default(),
            peak_candidates_in_flight: 0,
        }
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

    /// The rollout cache. Keys are `(baseline identity, candidate hash,
    /// example uid, salt)` — candidate identity is in the key, so two
    /// candidates on the same example occupy distinct entries.
    pub fn cache(&self) -> &RolloutCache {
        &self.cache
    }

    /// Pareto view over all example columns (see [`ScoreMatrix::pareto`]).
    pub fn pareto(&self) -> ParetoView {
        self.matrix.pareto()
    }

    /// Pareto view over a column subset (see [`ScoreMatrix::pareto_over`]).
    pub fn pareto_over(&self, columns: &[usize]) -> ParetoView {
        self.matrix.pareto_over(columns)
    }

    /// The parallelism gauge: the maximum number of **distinct candidates**
    /// that have had rollouts in flight simultaneously across all batches so
    /// far.
    pub fn peak_candidate_concurrency(&self) -> usize {
        self.peak_candidates_in_flight
    }

    /// Registers a module-lane candidate, deduplicating by content hash.
    /// Returns its index.
    pub fn register(&mut self, candidate: Candidate) -> usize {
        let hash = candidate.stable_hash();
        if let Some(existing) = self.candidates.iter().position(
            |(h, payload)| *h == hash && matches!(payload, CandidatePayload::Params { .. }),
        ) {
            return existing;
        }
        let params = std::sync::Arc::new(candidate.to_params());
        self.candidates
            .push((hash, CandidatePayload::Params { candidate, params }));
        self.matrix.ensure_rows(self.candidates.len());
        self.candidates.len() - 1
    }

    /// Registers a program-lane [`ir::Overlay`](crate::ir::Overlay) candidate,
    /// deduplicating by [`Overlay::hash`](crate::ir::Overlay::hash). Returns
    /// its index.
    pub fn register_overlay(&mut self, overlay: crate::ir::Overlay) -> usize {
        let hash = overlay.hash();
        if let Some(existing) = self.candidates.iter().position(
            |(h, payload)| *h == hash && matches!(payload, CandidatePayload::Overlay(_)),
        ) {
            return existing;
        }
        self.candidates
            .push((hash, CandidatePayload::Overlay(std::sync::Arc::new(overlay))));
        self.matrix.ensure_rows(self.candidates.len());
        self.candidates.len() - 1
    }

    /// The registered module-lane candidate at `index`, if it is one.
    pub fn candidate(&self, index: usize) -> Option<&Candidate> {
        match &self.candidates.get(index)?.1 {
            CandidatePayload::Params { candidate, .. } => Some(candidate),
            CandidatePayload::Overlay(_) => None,
        }
    }

    pub fn candidate_hash(&self, index: usize) -> u64 {
        self.candidates[index].0
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

    /// Evaluates N registered candidates over `subset` example indices
    /// (`None` = the target's full set) in **one** bounded-concurrency
    /// fan-out — candidate-level parallelism in both lanes, since candidates
    /// are injected per rollout, never applied to shared state.
    ///
    /// Cached rollouts return their `Eval` with `trace: None` and consume no
    /// budget. If the uncached portion doesn't fit the remaining budget the
    /// engine runs nothing and returns [`BatchEvalOutcome::BudgetExhausted`].
    pub async fn evaluate_many(
        &mut self,
        target: &OptimizeTarget<'_>,
        candidates: &[usize],
        subset: Option<&[usize]>,
    ) -> Result<BatchEvalOutcome> {
        for &candidate in candidates {
            if candidate >= self.candidates.len() {
                return Err(anyhow!("candidate index {candidate} is not registered"));
            }
        }
        let num_examples = target.num_examples();
        let indices: Vec<usize> = match subset {
            Some(subset) => subset.to_vec(),
            None => (0..num_examples).collect(),
        };
        if let Some(&bad) = indices.iter().find(|&&idx| idx >= num_examples) {
            return Err(anyhow!(
                "example index {bad} out of range ({num_examples} examples)"
            ));
        }

        let baseline = target.baseline();
        let salt = self.config.cache_salt;

        // Partition the (candidate × example) grid into cached and pending.
        let mut cached: HashMap<(usize, usize), Eval> = HashMap::new();
        let mut pending: Vec<(usize, usize)> = Vec::new();
        for &candidate in candidates {
            let candidate_hash = self.candidates[candidate].0;
            for &idx in &indices {
                let key = (candidate, idx);
                if cached.contains_key(&key) || pending.contains(&key) {
                    continue;
                }
                match self
                    .cache
                    .get(baseline, candidate_hash, target.example_uid(idx), salt)
                {
                    Some(eval) => {
                        cached.insert(key, eval.clone());
                    }
                    None => pending.push(key),
                }
            }
        }

        if !self.budget_allows(pending.len()) {
            return Ok(BatchEvalOutcome::BudgetExhausted {
                needed: pending.len(),
            });
        }

        // Bind each requested candidate against the target once.
        let mut bound: HashMap<usize, BoundCandidate> = HashMap::new();
        for &candidate in candidates {
            bound.insert(candidate, target.bind(&self.candidates[candidate].1)?);
        }

        let (fresh, batch_peak) = self.run_rollouts(target, &pending, &bound).await?;
        self.peak_candidates_in_flight = self.peak_candidates_in_flight.max(batch_peak);

        // Accounting: fresh rollouts consume budget, cached hits are free.
        self.spend.metric_calls += fresh.len();
        self.spend.lm_calls += fresh.len();
        self.spend.cache_hits += cached.len();
        for (_, _, _, trace) in &fresh {
            self.spend.lm_spans += trace.spans.len();
            for span in &trace.spans {
                self.spend.tokens = self.spend.tokens + span.usage;
            }
        }

        // Bookkeeping: cache inserts + matrix records.
        let mut fresh_by_key: HashMap<(usize, usize), (Eval, Trace)> =
            HashMap::with_capacity(fresh.len());
        for (candidate, idx, eval, trace) in fresh {
            self.cache.insert(
                baseline,
                self.candidates[candidate].0,
                target.example_uid(idx),
                salt,
                eval.clone(),
            );
            self.matrix.record(candidate, idx, eval.score);
            fresh_by_key.insert((candidate, idx), (eval, trace));
        }
        for (&(candidate, idx), eval) in &cached {
            self.matrix.record(candidate, idx, eval.score);
        }

        let evals = candidates
            .iter()
            .map(|&candidate| CandidateEval {
                candidate,
                rollouts: indices
                    .iter()
                    .map(|&idx| {
                        if let Some((eval, trace)) = fresh_by_key.remove(&(candidate, idx)) {
                            RolloutOutcome {
                                example: idx,
                                eval,
                                trace: Some(trace),
                            }
                        } else {
                            let eval = cached
                                .get(&(candidate, idx))
                                .cloned()
                                .expect("every requested pair is either fresh or cached");
                            RolloutOutcome {
                                example: idx,
                                eval,
                                trace: None,
                            }
                        }
                    })
                    .collect(),
            })
            .collect();

        Ok(BatchEvalOutcome::Complete(evals))
    }

    /// Single-candidate convenience over [`evaluate_many`](Self::evaluate_many).
    pub async fn evaluate(
        &mut self,
        target: &OptimizeTarget<'_>,
        candidate: usize,
        subset: Option<&[usize]>,
    ) -> Result<EvalOutcome> {
        match self.evaluate_many(target, &[candidate], subset).await? {
            BatchEvalOutcome::Complete(mut evals) => Ok(EvalOutcome::Complete(evals.remove(0))),
            BatchEvalOutcome::BudgetExhausted { needed } => {
                Ok(EvalOutcome::BudgetExhausted { needed })
            }
        }
    }

    /// Minibatch gating (the GEPA acceptance pattern): evaluates the candidate
    /// on `minibatch`; only if the minibatch mean strictly beats `threshold`
    /// does it promote to a full-set evaluation.
    pub async fn evaluate_gated(
        &mut self,
        target: &OptimizeTarget<'_>,
        candidate: usize,
        minibatch: &[usize],
        threshold: f64,
    ) -> Result<GateOutcome> {
        let minibatch_eval = match self.evaluate(target, candidate, Some(minibatch)).await? {
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

        match self.evaluate(target, candidate, None).await? {
            EvalOutcome::Complete(full) => Ok(GateOutcome::Promoted {
                minibatch: minibatch_eval,
                full,
            }),
            EvalOutcome::BudgetExhausted { needed } => Ok(GateOutcome::BudgetExhausted { needed }),
        }
    }

    /// The one shared fan-out: every pending `(candidate, example)` pair —
    /// across all candidates, in both lanes — in a single `buffer_unordered`
    /// stream. Returns the fresh rollouts and the batch's distinct-candidate
    /// concurrency high-water mark.
    async fn run_rollouts(
        &self,
        target: &OptimizeTarget<'_>,
        pending: &[(usize, usize)],
        bound: &HashMap<usize, BoundCandidate>,
    ) -> Result<(Vec<(usize, usize, Eval, Trace)>, usize)> {
        let gauge = Gauge::default();

        let fresh: Vec<(usize, usize, Eval, Trace)> =
            stream::iter(pending.iter().map(|&(candidate, idx)| {
                let bound = bound[&candidate].clone();
                let candidate_hash = self.candidates[candidate].0;
                let gauge = &gauge;
                async move {
                    let _in_flight = gauge.enter(candidate);
                    let (eval, trace) = target.run(idx, bound, candidate_hash).await?;
                    Ok::<_, anyhow::Error>((candidate, idx, eval, trace))
                }
            }))
            .buffer_unordered(self.config.concurrency.max(1))
            .try_collect()
            .await?;

        Ok((fresh, gauge.peak()))
    }
}

/// Counts distinct candidates with rollouts in flight; records the peak.
#[derive(Default)]
struct Gauge {
    in_flight: std::sync::Mutex<HashMap<usize, usize>>,
    peak: std::sync::atomic::AtomicUsize,
}

impl Gauge {
    fn enter(&self, candidate: usize) -> GaugeGuard<'_> {
        let mut in_flight = self.in_flight.lock().unwrap();
        *in_flight.entry(candidate).or_insert(0) += 1;
        self.peak
            .fetch_max(in_flight.len(), std::sync::atomic::Ordering::Relaxed);
        GaugeGuard {
            gauge: self,
            candidate,
        }
    }

    fn peak(&self) -> usize {
        self.peak.load(std::sync::atomic::Ordering::Relaxed)
    }
}

struct GaugeGuard<'a> {
    gauge: &'a Gauge,
    candidate: usize,
}

impl Drop for GaugeGuard<'_> {
    fn drop(&mut self) {
        let mut in_flight = self.gauge.in_flight.lock().unwrap();
        if let Some(count) = in_flight.get_mut(&self.candidate) {
            *count -= 1;
            if *count == 0 {
                in_flight.remove(&self.candidate);
            }
        }
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
    fn explicit_clears_are_distinct_candidate_content() {
        let unset = Candidate::with_instruction("drafter", "x");
        let mut cleared = Candidate::with_instruction("drafter", "x");
        cleared.clear_instruction("drafter");
        assert_ne!(unset.stable_hash(), cleared.stable_hash());

        // Clearing demos (empty set) differs from leaving them unset.
        let mut no_demos = Candidate::new();
        no_demos.set_demos("drafter", Vec::new());
        assert_ne!(no_demos.stable_hash(), Candidate::new().stable_hash());

        // to_params preserves the explicit clear.
        let params = cleared.to_params();
        assert!(params.get("drafter").is_some());
        assert_eq!(params.get("drafter").unwrap().instruction_override, None);
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
