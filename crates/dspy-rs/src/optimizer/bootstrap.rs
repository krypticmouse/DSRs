//! BootstrapFewShot: the minimal end-to-end exercise of the optimizer
//! contract (vision §5.4) — teacher pass under trace capture, demo harvesting
//! by trace name-join, candidate evaluation on the shared engine.

use std::collections::BTreeMap;

use anyhow::{Result, anyhow};
use bon::Builder;
use serde::{Deserialize, Serialize};

use crate::evaluate::{DEFAULT_EVAL_CONCURRENCY, TypedMetric};
use crate::optimizer::engine::{
    Budget, Candidate, EngineConfig, EvalEngine, EvalOutcome, Spend, apply_candidate,
};
use crate::optimizer::harvest::{collect_demo_candidates, select_demos};
use crate::optimizer::{Optimizer, predictor_names};
use crate::predictors::Example;
use crate::trace::Trace;
use crate::{Facet, Module, Signature};

/// Few-shot demo bootstrapper — the simplest complete optimizer.
///
/// One teacher pass, one candidate, one comparison:
///
/// 1. **Teacher pass** — runs the module over the trainset under trace
///    capture (via the shared [`EvalEngine`]), scoring each rollout with the
///    metric.
/// 2. **Harvest** — successful `Predict` spans from rollouts scoring at least
///    `min_demo_score` become few-shot demo rows, joined to their predictor
///    purely by trace component name.
/// 3. **Candidate eval** — the harvested demos form a demo-overlay
///    [`Candidate`], evaluated on the same engine (teacher rollouts already
///    sit in the rollout cache, so the baseline never re-runs).
/// 4. **Keep if better** — the demos are installed permanently only when the
///    candidate's mean score beats the baseline.
///
/// ```ignore
/// let bootstrap = BootstrapFewShot::builder().max_demos(4).build();
/// let report = bootstrap.compile(&mut module, trainset, &metric).await?;
/// if report.adopted {
///     println!("{:.3} -> {:.3}", report.baseline_score, report.candidate_score.unwrap());
/// }
/// ```
#[derive(Builder)]
pub struct BootstrapFewShot {
    /// Maximum demos installed per predictor.
    #[builder(default = 4)]
    pub max_demos: usize,

    /// Minimum whole-rollout metric score for a rollout's spans to qualify as
    /// demos. Defaults to `1.0` — full-credit rollouts only, assuming a 0–1
    /// metric; lower it for graded metrics.
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
    /// Demos harvested per predictor (dotted path -> count).
    pub demos_per_predictor: BTreeMap<String, usize>,
    /// Engine spend for the whole run.
    pub spend: Spend,
}

impl Optimizer for BootstrapFewShot {
    type Report = BootstrapReport;

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

        // 1. Teacher pass: baseline candidate over the full trainset, traced.
        let baseline = engine.register(Candidate::new());
        let baseline_eval = match engine.evaluate(module, baseline, None).await? {
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
            return Ok(BootstrapReport {
                baseline_score,
                candidate_score: None,
                adopted: false,
                demos_per_predictor,
                spend: *engine.spend(),
            });
        }

        // 3. Demos become a candidate, evaluated on the same engine.
        let mut candidate = Candidate::new();
        for (name, demo_set) in demos {
            candidate.set_demos(name, demo_set);
        }
        let candidate_idx = engine.register(candidate.clone());
        let candidate_eval = match engine.evaluate(module, candidate_idx, None).await? {
            EvalOutcome::Complete(eval) => eval,
            EvalOutcome::BudgetExhausted { .. } => {
                return Ok(BootstrapReport {
                    baseline_score,
                    candidate_score: None,
                    adopted: false,
                    demos_per_predictor,
                    spend: *engine.spend(),
                });
            }
        };
        let candidate_score = candidate_eval.mean();

        // 4. Keep if better: permanent install through the one candidate seam.
        let adopted = candidate_score > baseline_score;
        if adopted {
            let _undo = apply_candidate(module, &candidate)?;
        }

        Ok(BootstrapReport {
            baseline_score,
            candidate_score: Some(candidate_score),
            adopted,
            demos_per_predictor,
            spend: *engine.spend(),
        })
    }
}
