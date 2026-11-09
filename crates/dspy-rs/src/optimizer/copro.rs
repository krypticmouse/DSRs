use anyhow::{Result, anyhow};
use bon::Builder;
use std::collections::BTreeMap;

use crate::core::ToInput;
use crate::evaluate::TypedMetric;
use crate::optimizer::engine::{Candidate, Engine, EngineConfig, EvalOutcome};
use crate::optimizer::target::LeafInfo;
use crate::optimizer::{OptimizeTarget, Optimizer, OptimizerCommon, Report};
use crate::{Module, Predictors};

fn set_instruction_if_changed(candidate: &mut Candidate, leaf: &LeafInfo, instruction: &str) {
    if instruction == leaf.instruction {
        let remove = if let Some(slot) = candidate.slots.get_mut(&leaf.name) {
            slot.instruction = None;
            slot.demos.is_none() && !slot.clear_instruction
        } else {
            false
        };
        if remove {
            candidate.slots.remove(&leaf.name);
        }
    } else {
        candidate.set_instruction(&leaf.name, instruction);
    }
}

/// Breadth-first instruction optimizer.
///
/// COPRO (Collaborative Prompt Optimization) generates `breadth` candidate instructions
/// per predictor, evaluates each on the trainset, keeps the best, then repeats for
/// `depth` rounds. Simple and predictable — good for quick iteration when you want
/// better instructions without complex search.
///
/// COPRO is a thin strategy over the shared [`Engine`]: each candidate
/// instruction is a name-keyed [`Candidate`] layered on the winners
/// accumulated so far, evaluated through the engine's cached
/// bounded-concurrency ambient-injection fan-out. Nothing mutates the module
/// during the search; the accumulated winner is installed once at the end
/// through [`OptimizeTarget::install`]. Repeated candidates (the base
/// instruction always competes) deduplicate by content hash and are served
/// from the rollout cache.
///
/// Metric feedback is included in training-only proposal evidence when available;
/// numerical scores select round winners. If you want feedback-driven acceptance
/// and evolution strategies, use [`GEPA`](crate::GEPA) instead.
///
/// # Hyperparameters
///
/// - **`breadth`** (default: 10) — candidates per round per predictor. Higher = more
///   exploration but proportionally more LM calls. Must be > 1.
/// - **`depth`** (default: 3) — optimization rounds. Each round refines the previous
///   best instruction. Diminishing returns beyond ~5.
/// - **`init_temperature`** (default: 1.4) — proposal LM temperature.
/// - **`prompt_model`** — optional separate LM for generating candidate instructions.
///   Falls back to the global LM if unset.
///
/// # Cost
///
/// Total LM calls ≈ `breadth × depth × num_predictors × trainset_size`, minus
/// rollout-cache hits (previous winners re-compete for free). For a module
/// with 2 predictors, breadth=10, depth=3, and 50 training examples: ~3000 calls.
///
/// ```ignore
/// let copro = COPRO::builder().breadth(10).depth(3).build();
/// copro.compile_module(&mut module, &trainset, &metric).await?;
/// ```
#[derive(Builder)]
pub struct COPRO {
    /// Candidate instructions generated per round (must be > 1).
    #[builder(default = 10)]
    pub breadth: usize,
    /// Optimization rounds — each refines the previous best.
    #[builder(default = 3)]
    pub depth: usize,
    /// Proposal LM temperature.
    #[builder(default = 1.4)]
    pub init_temperature: f32,
    /// Whether to track per-round statistics.
    #[builder(default = false)]
    pub track_stats: bool,
    /// Optional separate LM for generating candidate instructions.
    pub prompt_model: Option<crate::LM>,
    /// Concurrent LM calls in flight during candidate evaluation.
    #[builder(default = crate::evaluate::DEFAULT_EVAL_CONCURRENCY)]
    pub eval_concurrency: usize,
}

impl COPRO {
    fn common(&self) -> OptimizerCommon {
        OptimizerCommon {
            eval_concurrency: self.eval_concurrency,
            ..OptimizerCommon::default()
        }
    }

    async fn candidate_instructions(
        &self,
        base_instruction: &str,
        leaf: &LeafInfo,
        observations: String,
        engine: &mut Engine,
    ) -> Result<Vec<String>> {
        let mut candidates = Vec::with_capacity(self.breadth.max(1));
        candidates.push(base_instruction.to_string());

        candidates.extend(
            super::propose::propose(
                self.prompt_model.as_ref(),
                self.init_temperature,
                leaf,
                base_instruction,
                observations,
                self.breadth - 1,
                engine,
            )
            .await?,
        );
        Ok(candidates)
    }

    /// Convenience: optimizes a typed module over a trainset with this
    /// optimizer's default engine, installing the winning instructions.
    pub async fn compile_module<E, M, MT>(
        &self,
        module: &mut M,
        trainset: &[E],
        metric: &MT,
    ) -> Result<()>
    where
        E: ToInput<M::Input> + serde::Serialize + Send + Sync,
        M: Module + Predictors,
        MT: TypedMetric<E, M>,
    {
        let mut target = OptimizeTarget::module(module, trainset, metric);
        let mut engine = Engine::new(Optimizer::engine_config(self));
        Optimizer::compile(self, &mut target, &mut engine).await?;
        Ok(())
    }
}

#[async_trait::async_trait(?Send)]
impl Optimizer for COPRO {
    fn engine_config(&self) -> EngineConfig {
        self.common().engine_config()
    }

    async fn compile(
        &self,
        target: &mut OptimizeTarget<'_>,
        engine: &mut Engine,
    ) -> Result<Report> {
        if self.breadth <= 1 {
            return Err(anyhow!("breadth must be greater than 1"));
        }

        let leaves = target.leaves().to_vec();
        if leaves.is_empty() {
            return Err(anyhow!("no optimizable predictors found"));
        }

        // Winners accumulate here; the module itself is never touched until
        // the final install.
        let train = target.train_columns();
        let validation = target.val_columns();
        anyhow::ensure!(
            !train.is_empty() && !validation.is_empty(),
            "training and validation sets must be nonempty"
        );
        let mut current = Candidate::new();
        let baseline = engine.register(current.clone());
        let baseline_train = match engine.evaluate(target, baseline, Some(&train)).await? {
            EvalOutcome::Complete(eval) => eval,
            EvalOutcome::BudgetExhausted { needed } => {
                return Err(anyhow!(
                    "budget exhausted ({needed} rollouts needed) during baseline training evaluation"
                ));
            }
        };
        let baseline_validation = match engine.evaluate(target, baseline, Some(&validation)).await?
        {
            EvalOutcome::Complete(eval) => eval.mean(),
            EvalOutcome::BudgetExhausted { needed } => {
                return Err(anyhow!(
                    "budget exhausted ({needed} rollouts needed) during baseline validation evaluation"
                ));
            }
        };
        // Keep a small, train-only evidence set separate from validation scoring.
        let baseline_observations = baseline_train
            .rollouts
            .iter()
            .filter_map(|rollout| {
                rollout.trace.as_ref().map(|trace| {
                    Ok(serde_json::json!({
                        "score": rollout.eval.score,
                        "feedback": rollout.eval.feedback.as_deref(),
                        "trace": trace.to_jsonl()?,
                    }))
                })
            })
            .take(8)
            .collect::<Result<Vec<_>>>()?;
        let baseline_observations = serde_json::to_string(&baseline_observations)?;
        let baseline_observations = crate::utils::truncate(&baseline_observations, 24000);
        let mut best_validation = baseline_validation;
        let mut winner = current.clone();
        let mut history: BTreeMap<String, Vec<(String, f64)>> = BTreeMap::new();
        let mut current_instructions: BTreeMap<String, String> = leaves
            .iter()
            .map(|leaf| (leaf.name.clone(), leaf.instruction.clone()))
            .collect();

        for depth in 0..self.depth {
            for leaf in &leaves {
                let base_instruction = current_instructions[&leaf.name].clone();
                let observations = format!(
                    "Round {}. Training traces with metric scores and feedback: {}\nPreviously evaluated instructions and training scores: {}",
                    depth + 1,
                    baseline_observations,
                    serde_json::to_string(history.entry(leaf.name.clone()).or_default())?
                );
                let instructions = self
                    .candidate_instructions(&base_instruction, leaf, observations, engine)
                    .await?;

                let mut best: Option<(f64, String)> = None;
                for instruction in instructions {
                    let mut candidate = current.clone();
                    set_instruction_if_changed(&mut candidate, leaf, &instruction);
                    let row = engine.register(candidate);
                    let eval = match engine.evaluate(target, row, Some(&train)).await? {
                        EvalOutcome::Complete(eval) => eval,
                        EvalOutcome::BudgetExhausted { needed } => {
                            return Err(anyhow!(
                                "budget exhausted ({needed} rollouts needed) during COPRO round"
                            ));
                        }
                    };
                    let score = eval.mean();
                    history
                        .entry(leaf.name.clone())
                        .or_default()
                        .push((instruction.clone(), score));
                    if best.as_ref().is_none_or(|(top, _)| score > *top) {
                        best = Some((score, instruction));
                    }
                }

                let (_, best_instruction) =
                    best.expect("breadth > 1 guarantees at least one candidate");
                set_instruction_if_changed(&mut current, leaf, &best_instruction);
                current_instructions.insert(leaf.name.clone(), best_instruction);
            }
            let row = engine.register(current.clone());
            let validation_score = match engine.evaluate(target, row, Some(&validation)).await? {
                EvalOutcome::Complete(eval) => eval.mean(),
                EvalOutcome::BudgetExhausted { needed } => {
                    return Err(anyhow!(
                        "budget exhausted ({needed} rollouts needed) during COPRO validation"
                    ));
                }
            };
            if validation_score > best_validation {
                best_validation = validation_score;
                winner = current.clone();
            }
        }

        // The one mutation of the run: install the accumulated winner.
        target.install(&winner)?;

        Ok(Report::None)
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
    struct CoproStateSig {
        #[input]
        prompt: String,

        #[output]
        answer: String,
    }

    struct CoproStateModule {
        predictor: Predict<CoproStateSig>,
    }

    crate::predictors!(CoproStateModule { predictor });

    impl Module for CoproStateModule {
        type Input = CoproStateSigInput;
        type Output = CoproStateSigOutput;

        async fn forward(
            &self,
            input: CoproStateSigInput,
        ) -> Result<Predicted<CoproStateSigOutput>, PredictError> {
            Ok(Predicted::new(
                CoproStateSigOutput {
                    answer: input.prompt,
                },
                CallMetadata::default(),
            ))
        }
    }

    struct AlwaysFailMetric;

    type CoproRow = (CoproStateSigInput, CoproStateSigOutput);

    impl TypedMetric<CoproRow, CoproStateModule> for AlwaysFailMetric {
        async fn evaluate(
            &self,
            _example: &CoproRow,
            _prediction: &Predicted<CoproStateSigOutput>,
            _trace: Option<&Trace>,
        ) -> Result<Eval> {
            Err(anyhow!("metric failure"))
        }
    }

    fn trainset() -> Vec<CoproRow> {
        vec![(
            CoproStateSigInput {
                prompt: "one".to_string(),
            },
            CoproStateSigOutput {
                answer: "one".to_string(),
            },
        )]
    }

    #[tokio::test]
    async fn compile_leaves_state_untouched_when_metric_errors() {
        let optimizer = COPRO::builder().breadth(2).depth(1).build();
        let mut module = CoproStateModule {
            predictor: Predict::<CoproStateSig>::builder()
                .instruction("seed-instruction")
                .build(),
        };

        let err = optimizer
            .compile_module(&mut module, &trainset(), &AlwaysFailMetric)
            .await
            .expect_err("candidate scoring should propagate metric failure");
        assert!(err.to_string().contains("metric failure"));

        // Candidates are ambient — a failed run can't have leaked state.
        assert_eq!(
            PredictorInfo::instruction(&module.predictor),
            "seed-instruction"
        );
    }
}
