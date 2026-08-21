//! BootstrapFewShot: the minimal end-to-end exercise of the optimizer
//! contract (vision §5.4) — teacher pass under trace capture, demo harvesting
//! by trace name-join, candidate evaluation on the shared engine.

use std::collections::BTreeMap;

use anyhow::{Result, anyhow};
use bon::Builder;
use serde::{Deserialize, Serialize};

use crate::core::ToInput;
use crate::evaluate::{DEFAULT_EVAL_CONCURRENCY, TypedMetric};
use crate::optimizer::engine::{Candidate, Engine, EngineConfig, EvalOutcome, Spend};
use crate::optimizer::harvest::{collect_demo_candidates, select_demos};
use crate::optimizer::{OptimizeTarget, Optimizer, OptimizerCommon, Report};
use crate::trace::Trace;
use crate::{Module, Predictors};

/// Few-shot demo bootstrapper — the simplest complete optimizer.
///
/// One teacher pass, one candidate, one comparison:
///
/// 1. **Teacher pass** — runs the target over the trainset under trace
///    capture (via the shared [`Engine`]), scoring each rollout with the
///    metric.
/// 2. **Harvest** — successful `Predict` spans scoring at least
///    `min_demo_score` become few-shot demo rows, joined to their predictor
///    purely by trace component name (the [`Predictors`] contract name). A
///    span scores as the rollout does unless the metric attached a span-level
///    eval ([`TypedMetric::evaluate_spans`]), which then takes precedence.
/// 3. **Candidate eval** — the harvested demos form a demo [`Candidate`],
///    evaluated ambiently on the same engine (teacher rollouts already sit in
///    the rollout cache, so the baseline never re-runs).
/// 4. **Keep if better** — the demos are installed (once, at the end) only
///    when the candidate's mean score beats the baseline.
///
/// ```ignore
/// let bootstrap = BootstrapFewShot::builder().max_demos(4).build();
/// let report = bootstrap.compile_module(&mut module, &trainset, &metric).await?;
/// if report.adopted {
///     println!("{:.3} -> {:.3}", report.baseline_score, report.candidate_score.unwrap());
/// }
/// ```
#[derive(Builder)]
pub struct BootstrapFewShot {
    /// Maximum demos installed per predictor.
    #[builder(default = 4)]
    pub max_demos: usize,

    /// Minimum score for a span to qualify as a demo: the span's own eval
    /// when the metric attached one ([`TypedMetric::evaluate_spans`]), the
    /// whole-rollout metric score otherwise. Defaults to `1.0` — full-credit
    /// only, assuming a 0–1 metric; lower it for graded metrics.
    #[builder(default = 1.0)]
    pub min_demo_score: f64,

    /// Concurrent rollouts in flight during evaluation.
    #[builder(default = DEFAULT_EVAL_CONCURRENCY)]
    pub eval_concurrency: usize,

    /// Hard cap on metric calls (rollouts). `None` = unlimited.
    pub max_metric_calls: Option<usize>,
    /// Hard cap on LM call units. `None` = unlimited.
    pub max_lm_calls: Option<usize>,
}

/// What a [`BootstrapFewShot`] run did.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BootstrapReport {
    /// Mean metric score of the unmodified module over the trainset.
    pub baseline_score: f64,
    /// Mean score with demos attached; `None` if no demos were harvested or
    /// the budget stopped before the candidate evaluation.
    pub candidate_score: Option<f64>,
    /// Whether the demo candidate beat the baseline and was installed.
    pub adopted: bool,
    /// Demos harvested per predictor (leaf name -> count).
    pub demos_per_predictor: BTreeMap<String, usize>,
    /// Engine spend for the whole run.
    pub spend: Spend,
}

impl BootstrapFewShot {
    fn common(&self) -> OptimizerCommon {
        OptimizerCommon {
            eval_concurrency: self.eval_concurrency,
            max_metric_calls: self.max_metric_calls,
            max_lm_calls: self.max_lm_calls,
            ..OptimizerCommon::default()
        }
    }

    /// Convenience: bootstraps a typed module over a trainset with this
    /// optimizer's default engine.
    pub async fn compile_module<E, M, MT>(
        &self,
        module: &mut M,
        trainset: &[E],
        metric: &MT,
    ) -> Result<BootstrapReport>
    where
        E: ToInput<M::Input> + serde::Serialize + Send + Sync,
        M: Module + Predictors,
        MT: TypedMetric<E, M>,
    {
        let mut target = OptimizeTarget::module(module, trainset, metric);
        let mut engine = Engine::new(Optimizer::engine_config(self));
        let report = Optimizer::compile(self, &mut target, &mut engine).await?;
        report
            .into_bootstrap()
            .ok_or_else(|| anyhow!("BootstrapFewShot must return a bootstrap report"))
    }
}

#[async_trait::async_trait(?Send)]
impl Optimizer for BootstrapFewShot {
    fn engine_config(&self) -> EngineConfig {
        self.common().engine_config()
    }

    async fn compile(
        &self,
        target: &mut OptimizeTarget<'_>,
        engine: &mut Engine,
    ) -> Result<Report> {
        if target.leaves().is_empty() {
            return Err(anyhow!("no optimizable predictors found"));
        }

        // 1. Teacher pass: baseline candidate over the full trainset, traced.
        let baseline = engine.register(Candidate::new());
        let baseline_eval = match engine.evaluate(target, baseline, None).await? {
            EvalOutcome::Complete(eval) => eval,
            EvalOutcome::BudgetExhausted { needed } => {
                return Err(anyhow!(
                    "budget too small for the teacher pass ({needed} rollouts needed)"
                ));
            }
        };
        let baseline_score = baseline_eval.mean();

        // 2. Harvest demos from successful spans of well-scored rollouts.
        let scored_traces: Vec<(f64, &Trace)> = baseline_eval
            .rollouts
            .iter()
            .filter_map(|rollout| {
                rollout
                    .trace
                    .as_ref()
                    .map(|trace| (rollout.eval.score, trace))
            })
            .collect();
        let demos = select_demos(
            collect_demo_candidates(scored_traces, self.min_demo_score),
            self.max_demos,
        );
        let demos_per_predictor: BTreeMap<String, usize> = demos
            .iter()
            .map(|(name, demos)| (name.clone(), demos.len()))
            .collect();

        if demos.is_empty() {
            return Ok(Report::Bootstrap(BootstrapReport {
                baseline_score,
                candidate_score: None,
                adopted: false,
                demos_per_predictor,
                spend: *engine.spend(),
            }));
        }

        // 3. Demos become a candidate, evaluated ambiently on the same engine.
        let mut candidate = Candidate::new();
        for (name, demo_set) in demos {
            candidate.set_demos(name, demo_set);
        }
        let candidate_idx = engine.register(candidate.clone());
        let candidate_eval = match engine.evaluate(target, candidate_idx, None).await? {
            EvalOutcome::Complete(eval) => eval,
            EvalOutcome::BudgetExhausted { .. } => {
                return Ok(Report::Bootstrap(BootstrapReport {
                    baseline_score,
                    candidate_score: None,
                    adopted: false,
                    demos_per_predictor,
                    spend: *engine.spend(),
                }));
            }
        };
        let candidate_score = candidate_eval.mean();

        // 4. Keep if better: the run's one mutation, at the end.
        let adopted = candidate_score > baseline_score;
        if adopted {
            target.install(&candidate)?;
        }

        Ok(Report::Bootstrap(BootstrapReport {
            baseline_score,
            candidate_score: Some(candidate_score),
            adopted,
            demos_per_predictor,
            spend: *engine.spend(),
        }))
    }
}
