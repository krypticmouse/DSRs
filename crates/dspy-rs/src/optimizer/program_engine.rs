//! The IR-native evaluation path (RFC 0002 IR-6): candidate [`Overlay`]s
//! evaluated over one shared `Arc<Program>` through the [`Interpreter`] with
//! **candidate-level parallelism**.
//!
//! This is the seam `engine.rs` deliberately left open: the module lane must
//! serialize candidate application because candidates mutate shared predictor
//! state through `apply_update`. The dynamic lane has no mutation at all —
//! the interpreter reads instruction/demos/model/context/code through the
//! overlay at render time — so N candidates × M examples fan out in **one**
//! bounded-concurrency stream over one program instance. Nothing is applied,
//! nothing is restored.
//!
//! [`ProgramEvalEngine`] mirrors the module-lane [`EvalEngine`] machinery
//! piece for piece — same [`EngineConfig`] (concurrency/budget/salt), same
//! [`RolloutCache`] (keys are `(program hash, overlay hash, example uid,
//! salt)`, so per-candidate hits are exact), same [`ScoreMatrix`]/Pareto
//! bookkeeping, same [`Spend`] accounting, same minibatch gate — but speaks
//! JSON at the boundary: examples are labeled [`DemoRow`]s and the metric is
//! a [`ProgramMetric`] over output `JsonMap`s. The module-lane path is
//! untouched; this is an *additional* entry point
//! ([`evaluate_program_candidates`](ProgramEvalEngine::evaluate_program_candidates)),
//! not a rewrite.
//!
//! [`EvalEngine`]: crate::optimizer::engine::EvalEngine

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Result, anyhow};
use futures::stream::{self, StreamExt, TryStreamExt};

use crate::evaluate::Eval;
use crate::ir::interp::{Budget as RunBudget, Interpreter};
use crate::ir::params::{DemoRow, Overlay};
use crate::optimizer::engine::{
    CandidateEval, EngineConfig, GateOutcome, RolloutCache, RolloutOutcome, ScoreMatrix, Spend,
    canonical_hash,
};
use crate::trace::{JsonMap, Trace, TraceMeta, TraceOutcome, capture_with_meta};

/// How a program-lane strategy tells the engine what "good" means: score one
/// interpreter output (`JsonMap` of the program's output signature fields)
/// against a labeled example.
///
/// The JSON-native sibling of [`TypedMetric`](crate::evaluate::TypedMetric) —
/// loaded programs have no static output type, so the metric sees the same
/// value model the interpreter produces. The rollout's captured [`Trace`] is
/// always provided; LM-as-judge metrics run outside the capture scope exactly
/// as in the module lane.
#[allow(async_fn_in_trait)]
pub trait ProgramMetric: Send + Sync {
    async fn evaluate(
        &self,
        example: &DemoRow,
        output: &JsonMap,
        trace: Option<&Trace>,
    ) -> Result<Eval>;
}

/// Result of [`ProgramEvalEngine::evaluate_program_candidates`].
#[derive(Clone, Debug)]
pub enum ProgramEvalOutcome {
    /// One [`CandidateEval`] per requested candidate, in request order.
    Complete(Vec<CandidateEval>),
    /// The uncached portion of the batch didn't fit the remaining budget;
    /// nothing ran and spend is unchanged.
    BudgetExhausted { needed: usize },
}

impl ProgramEvalOutcome {
    /// The completed evaluations, if the budget allowed the batch.
    pub fn completed(self) -> Option<Vec<CandidateEval>> {
        match self {
            Self::Complete(evals) => Some(evals),
            Self::BudgetExhausted { .. } => None,
        }
    }
}

/// The shared evaluation core for the dynamic lane: candidate overlays over
/// one interpreter-loaded program.
///
/// Owns the labeled example set, the candidate registry (deduplicated by
/// [`Overlay::hash`]), the score matrix, the rollout cache, and the spend
/// meter. Strategies register overlays and call
/// [`evaluate_program_candidates`](Self::evaluate_program_candidates) /
/// [`evaluate_gated`](Self::evaluate_gated).
pub struct ProgramEvalEngine<'m, MT> {
    examples: Vec<DemoRow>,
    example_uids: Vec<u64>,
    metric: &'m MT,
    config: EngineConfig,
    candidates: Vec<Arc<Overlay>>,
    candidate_hashes: Vec<u64>,
    matrix: ScoreMatrix,
    cache: RolloutCache,
    spend: Spend,
    /// High-water mark of *distinct candidates* with rollouts in flight at
    /// the same instant — the parallelism gauge (see
    /// [`peak_candidate_concurrency`](Self::peak_candidate_concurrency)).
    peak_candidates_in_flight: usize,
}

impl<'m, MT: ProgramMetric> ProgramEvalEngine<'m, MT> {
    pub fn new(examples: Vec<DemoRow>, metric: &'m MT, config: EngineConfig) -> Self {
        let example_uids = examples.iter().map(canonical_hash).collect();
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
            peak_candidates_in_flight: 0,
        }
    }

    pub fn examples(&self) -> &[DemoRow] {
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

    /// The rollout cache. Keys are `(program hash, overlay hash, example uid,
    /// salt)` — candidate identity is the overlay hash, so two candidates on
    /// the same example occupy distinct entries.
    pub fn cache(&self) -> &RolloutCache {
        &self.cache
    }

    /// Pareto view over all example columns (see [`ScoreMatrix::pareto`]).
    pub fn pareto(&self) -> crate::optimizer::engine::ParetoView {
        self.matrix.pareto()
    }

    /// Pareto view over a column subset (see [`ScoreMatrix::pareto_over`]).
    pub fn pareto_over(&self, columns: &[usize]) -> crate::optimizer::engine::ParetoView {
        self.matrix.pareto_over(columns)
    }

    /// The parallelism gauge: the maximum number of **distinct candidates**
    /// that have had rollouts in flight simultaneously across all batches so
    /// far. A value ≥ 2 is positive evidence that candidate-level parallelism
    /// actually happened (the module lane is structurally pinned to 1).
    pub fn peak_candidate_concurrency(&self) -> usize {
        self.peak_candidates_in_flight
    }

    /// Registers a candidate overlay, deduplicating by [`Overlay::hash`].
    /// Returns its index.
    pub fn register(&mut self, overlay: Overlay) -> usize {
        let hash = overlay.hash();
        if let Some(existing) = self.candidate_hashes.iter().position(|&h| h == hash) {
            return existing;
        }
        self.candidates.push(Arc::new(overlay));
        self.candidate_hashes.push(hash);
        self.matrix.ensure_rows(self.candidates.len());
        self.candidates.len() - 1
    }

    pub fn candidate(&self, index: usize) -> &Arc<Overlay> {
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

    /// Charges auxiliary spend (reflection calls, teacher passes) against the
    /// budget, mirroring the module lane.
    pub fn charge(&mut self, metric_calls: usize, lm_calls: usize) {
        self.spend.metric_calls = self.spend.metric_calls.saturating_add(metric_calls);
        self.spend.lm_calls = self.spend.lm_calls.saturating_add(lm_calls);
    }

    /// **The IR-native entry point**: evaluates N registered candidates over
    /// `subset` example indices (`None` = the full set) through `interp`,
    /// with TRUE candidate-level parallelism.
    ///
    /// Every uncached `(candidate, example)` pair across *all* requested
    /// candidates joins one bounded-concurrency fan-out
    /// ([`EngineConfig::concurrency`]); each rollout runs
    /// `interp.run(input, Some(overlay), …)` under its own capture scope with
    /// `TraceMeta.candidate_hash = overlay.hash()` and
    /// `tags["program"] = hex(program_hash)` (RFC 0002 §3.3). There is no
    /// apply/restore: the overlays read through at render over the one shared
    /// `Arc<Program>`.
    ///
    /// Cached rollouts return their `Eval` with `trace: None` and consume no
    /// budget. If the uncached portion doesn't fit the remaining budget the
    /// engine runs nothing and returns
    /// [`ProgramEvalOutcome::BudgetExhausted`].
    pub async fn evaluate_program_candidates(
        &mut self,
        interp: &Interpreter,
        candidates: &[usize],
        subset: Option<&[usize]>,
    ) -> Result<ProgramEvalOutcome> {
        for &candidate in candidates {
            if candidate >= self.candidates.len() {
                return Err(anyhow!("candidate index {candidate} is not registered"));
            }
        }
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

        let baseline = interp.program().meta.program_hash;
        let salt = self.config.cache_salt;

        // Partition the full (candidate × example) grid into cached and
        // pending pairs.
        let mut cached: HashMap<(usize, usize), Eval> = HashMap::new();
        let mut pending: Vec<(usize, usize)> = Vec::new();
        for &candidate in candidates {
            let candidate_hash = self.candidate_hashes[candidate];
            for &idx in &indices {
                let key = (candidate, idx);
                if cached.contains_key(&key) || pending.contains(&key) {
                    continue;
                }
                match self
                    .cache
                    .get(baseline, candidate_hash, self.example_uids[idx], salt)
                {
                    Some(eval) => {
                        cached.insert(key, eval.clone());
                    }
                    None => pending.push(key),
                }
            }
        }

        if !self.budget_allows(pending.len()) {
            return Ok(ProgramEvalOutcome::BudgetExhausted {
                needed: pending.len(),
            });
        }

        let (fresh, batch_peak) = self.run_rollouts(interp, &pending).await?;
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
                self.candidate_hashes[candidate],
                self.example_uids[idx],
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

        Ok(ProgramEvalOutcome::Complete(evals))
    }

    /// Single-candidate convenience over
    /// [`evaluate_program_candidates`](Self::evaluate_program_candidates).
    pub async fn evaluate(
        &mut self,
        interp: &Interpreter,
        candidate: usize,
        subset: Option<&[usize]>,
    ) -> Result<crate::optimizer::engine::EvalOutcome> {
        use crate::optimizer::engine::EvalOutcome;
        match self
            .evaluate_program_candidates(interp, &[candidate], subset)
            .await?
        {
            ProgramEvalOutcome::Complete(mut evals) => {
                Ok(EvalOutcome::Complete(evals.remove(0)))
            }
            ProgramEvalOutcome::BudgetExhausted { needed } => {
                Ok(EvalOutcome::BudgetExhausted { needed })
            }
        }
    }

    /// Minibatch gating (the GEPA acceptance pattern), program-lane edition:
    /// evaluates the candidate on `minibatch`; only if the minibatch mean
    /// strictly beats `threshold` does it promote to a full-set evaluation.
    pub async fn evaluate_gated(
        &mut self,
        interp: &Interpreter,
        candidate: usize,
        minibatch: &[usize],
        threshold: f64,
    ) -> Result<GateOutcome> {
        use crate::optimizer::engine::EvalOutcome;
        let minibatch_eval = match self.evaluate(interp, candidate, Some(minibatch)).await? {
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

        match self.evaluate(interp, candidate, None).await? {
            EvalOutcome::Complete(full) => Ok(GateOutcome::Promoted {
                minibatch: minibatch_eval,
                full,
            }),
            EvalOutcome::BudgetExhausted { needed } => Ok(GateOutcome::BudgetExhausted { needed }),
        }
    }

    /// The one shared fan-out: every pending `(candidate, example)` pair —
    /// across all candidates — in a single `buffer_unordered` stream. Returns
    /// the fresh rollouts and the batch's distinct-candidate concurrency
    /// high-water mark.
    async fn run_rollouts(
        &self,
        interp: &Interpreter,
        pending: &[(usize, usize)],
    ) -> Result<(Vec<(usize, usize, Eval, Trace)>, usize)> {
        let metric = self.metric;
        let program_tag = format!("{:016x}", interp.program().meta.program_hash);
        let gauge = Gauge::default();

        let fresh: Vec<(usize, usize, Eval, Trace)> = stream::iter(pending.iter().map(
            |&(candidate, idx)| {
                let overlay = Arc::clone(&self.candidates[candidate]);
                let candidate_hash = self.candidate_hashes[candidate];
                let example = &self.examples[idx];
                let program_tag = &program_tag;
                let gauge = &gauge;
                async move {
                    let _in_flight = gauge.enter(candidate);
                    let meta = TraceMeta {
                        candidate_hash: Some(candidate_hash),
                        input: Some(example.input.clone()),
                        tags: [("program".to_string(), program_tag.clone())]
                            .into_iter()
                            .collect(),
                        ..TraceMeta::default()
                    };
                    let started = Instant::now();
                    let (result, mut trace) = capture_with_meta(meta, || {
                        interp.run(
                            example.input.clone(),
                            Some(overlay),
                            RunBudget::unlimited(),
                        )
                    })
                    .await;
                    let output = result.map_err(|err| {
                        anyhow!("candidate {candidate} failed on example {idx}: {err}")
                    })?;
                    // Metric runs outside the capture scope so LM-as-judge
                    // metrics don't pollute the execution trace.
                    let eval = metric.evaluate(example, &output, Some(&trace)).await?;
                    trace.outcome = Some(TraceOutcome {
                        output: Some(output),
                        error: None,
                        eval: Some(eval.clone()),
                        duration_us: started.elapsed().as_micros() as u64,
                    });
                    Ok::<_, anyhow::Error>((candidate, idx, eval, trace))
                }
            },
        ))
        .buffer_unordered(self.config.concurrency.max(1))
        .try_collect()
        .await?;

        Ok((fresh, gauge.peak()))
    }
}

/// Counts distinct candidates with rollouts in flight; records the peak.
#[derive(Default)]
struct Gauge {
    in_flight: Mutex<HashMap<usize, usize>>,
    peak: AtomicUsize,
}

impl Gauge {
    fn enter(&self, candidate: usize) -> GaugeGuard<'_> {
        let mut in_flight = self.in_flight.lock().unwrap();
        *in_flight.entry(candidate).or_insert(0) += 1;
        self.peak.fetch_max(in_flight.len(), Ordering::Relaxed);
        GaugeGuard {
            gauge: self,
            candidate,
        }
    }

    fn peak(&self) -> usize {
        self.peak.load(Ordering::Relaxed)
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
