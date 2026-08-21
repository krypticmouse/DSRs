use anyhow::{Result, anyhow};
use bon::Builder;
use std::collections::BTreeMap;

use crate::core::ToInput;
use crate::evaluate::TypedMetric;
use crate::optimizer::engine::{Candidate, Engine, EngineConfig, EvalOutcome};
use crate::optimizer::target::LeafInfo;
use crate::optimizer::{OptimizeTarget, Optimizer, OptimizerCommon, Report};
use crate::{Module, Predictors};

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
/// Does not use feedback from the metric — only the numerical score matters. If you
/// have rich textual feedback, use [`GEPA`](crate::GEPA) instead.
///
/// # Hyperparameters
///
/// - **`breadth`** (default: 10) — candidates per round per predictor. Higher = more
///   exploration but proportionally more LM calls. Must be > 1.
/// - **`depth`** (default: 3) — optimization rounds. Each round refines the previous
///   best instruction. Diminishing returns beyond ~5.
/// - **`init_temperature`** (default: 1.4) — **currently unused.** Reserved for LM-generated
///   candidate diversity. Setting this has no effect.
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
    /// **Currently unused.** Reserved for controlling LM-generated candidate diversity.
    /// Setting this has no effect.
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

    fn candidate_instructions(
        &self,
        base_instruction: &str,
        leaf: &LeafInfo,
        depth: usize,
    ) -> Vec<String> {
        let mut candidates = Vec::with_capacity(self.breadth.max(1));
        candidates.push(base_instruction.to_string());

        let output_hint = leaf
            .output_fields
            .last()
            .map(|(name, _)| name.as_str())
            .unwrap_or("output");

        for idx in 0..self.breadth.saturating_sub(1) {
            candidates.push(format!(
                "{base_instruction}\n\nOptimization hint (d{} c{}): Be explicit and concise for `{}`.",
                depth + 1,
                idx + 1,
                output_hint,
            ));
        }

        candidates
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
        let mut current = Candidate::new();
        let mut current_instructions: BTreeMap<String, String> = leaves
            .iter()
            .map(|leaf| (leaf.name.clone(), leaf.instruction.clone()))
            .collect();

        for depth in 0..self.depth {
            for leaf in &leaves {
                let base_instruction = current_instructions[&leaf.name].clone();
                let instructions = self.candidate_instructions(&base_instruction, leaf, depth);

                let mut best: Option<(f64, String)> = None;
                for instruction in instructions {
                    let mut candidate = current.clone();
                    candidate.set_instruction(&leaf.name, &instruction);
                    let row = engine.register(candidate);
                    let eval = match engine.evaluate(target, row, None).await? {
                        EvalOutcome::Complete(eval) => eval,
                        EvalOutcome::BudgetExhausted { needed } => {
                            return Err(anyhow!(
                                "budget exhausted ({needed} rollouts needed) during COPRO round"
                            ));
                        }
                    };
                    let score = eval.mean();
                    if best.as_ref().is_none_or(|(top, _)| score > *top) {
                        best = Some((score, instruction));
                    }
                }

                let (_, best_instruction) =
                    best.expect("breadth > 1 guarantees at least one candidate");
                current.set_instruction(&leaf.name, &best_instruction);
                current_instructions.insert(leaf.name.clone(), best_instruction);
            }
        }

        // The one mutation of the run: install the accumulated winner.
        target.install(&current)?;

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
