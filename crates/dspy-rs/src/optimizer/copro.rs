use anyhow::{Result, anyhow};
use bon::Builder;

use crate::core::DynPredictor;
use crate::evaluate::TypedMetric;
use crate::optimizer::engine::{
    Budget, Candidate, EngineConfig, EvalEngine, EvalOutcome, apply_candidate,
};
use crate::optimizer::{Optimizer, predictor_names, with_named_predictor};
use crate::core::ToInput;
use crate::{Facet, Module};

/// Breadth-first instruction optimizer.
///
/// COPRO (Collaborative Prompt Optimization) generates `breadth` candidate instructions
/// per predictor, evaluates each on the trainset, keeps the best, then repeats for
/// `depth` rounds. Simple and predictable — good for quick iteration when you want
/// better instructions without complex search.
///
/// COPRO is a thin strategy over the shared [`EvalEngine`]: each candidate
/// instruction is an overlay [`Candidate`] evaluated through the engine's
/// cached bounded-concurrency fan-out, and each round's winner is installed
/// permanently through the one candidate seam ([`apply_candidate`]). Repeated
/// candidates within a round (the base instruction always competes) are
/// deduplicated by content hash and served from the rollout cache.
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
/// Total LM calls ≈ `breadth × depth × num_predictors × trainset_size`. For a module
/// with 2 predictors, breadth=10, depth=3, and 50 training examples: ~3000 calls.
///
/// ```ignore
/// let copro = COPRO::builder().breadth(10).depth(3).build();
/// copro.compile(&mut module, trainset, &metric).await?;
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
    fn current_instruction<M>(module: &mut M, predictor_name: &str) -> Result<String>
    where
        M: for<'a> Facet<'a>,
    {
        with_named_predictor(module, predictor_name, |predictor| {
            Ok(predictor.instruction())
        })
    }

    fn candidate_instructions(
        &self,
        base_instruction: &str,
        predictor: &dyn DynPredictor,
        depth: usize,
    ) -> Vec<String> {
        let mut candidates = Vec::with_capacity(self.breadth.max(1));
        candidates.push(base_instruction.to_string());

        let output_hint = predictor
            .schema()
            .output_fields()
            .last()
            .map(|field| field.lm_name)
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
}

impl Optimizer for COPRO {
    type Report = ();

    async fn compile<E, M, MT>(
        &self,
        module: &mut M,
        trainset: Vec<E>,
        metric: &MT,
    ) -> Result<Self::Report>
    where
        E: ToInput<M::Input> + serde::Serialize + Send + Sync,
        M: Module + for<'a> Facet<'a>,
        MT: TypedMetric<E, M>,
    {
        if self.breadth <= 1 {
            return Err(anyhow!("breadth must be greater than 1"));
        }

        let predictor_names = predictor_names(module)?;

        if predictor_names.is_empty() {
            return Err(anyhow!("no optimizable predictors found"));
        }

        let mut engine = EvalEngine::new(
            trainset,
            metric,
            EngineConfig {
                concurrency: self.eval_concurrency,
                budget: Budget::unlimited(),
                cache_salt: 0,
            },
        );

        for depth in 0..self.depth {
            for predictor_name in &predictor_names {
                let base_instruction = Self::current_instruction(module, predictor_name)?;

                let candidates = with_named_predictor(module, predictor_name, |predictor| {
                    Ok(self.candidate_instructions(&base_instruction, predictor, depth))
                })?;

                let mut best: Option<(f64, String)> = None;
                for instruction in candidates {
                    let row =
                        engine.register(Candidate::with_instruction(predictor_name, &instruction));
                    let eval = match engine.evaluate(module, row, None).await? {
                        EvalOutcome::Complete(eval) => eval,
                        EvalOutcome::BudgetExhausted { needed } => {
                            return Err(anyhow!(
                                "unexpected budget exhaustion ({needed} rollouts) with an unlimited budget"
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
                // Permanent install through the one candidate seam. The engine's
                // baseline hash changes with it, correctly invalidating cached
                // rollouts recorded against the previous round's skeleton.
                let _undo = apply_candidate(
                    module,
                    &Candidate::with_instruction(predictor_name, best_instruction),
                )?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use anyhow::{Result, anyhow};

    use super::*;
    use crate::evaluate::{Eval, TypedMetric};
    use crate::trace::Trace;
    use crate::{CallMetadata, Predict, PredictError, Predicted, Signature};

    #[derive(Signature, Clone, Debug)]
    struct CoproStateSig {
        #[input]
        prompt: String,

        #[output]
        answer: String,
    }

    #[derive(facet::Facet)]
    #[facet(crate = facet)]
    struct CoproStateModule {
        predictor: Predict<CoproStateSig>,
    }

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
    async fn compile_restores_state_when_metric_errors() {
        let optimizer = COPRO::builder().breadth(2).depth(1).build();
        let mut module = CoproStateModule {
            predictor: Predict::<CoproStateSig>::builder()
                .instruction("seed-instruction")
                .build(),
        };

        let err = optimizer
            .compile(&mut module, trainset(), &AlwaysFailMetric)
            .await
            .expect_err("candidate scoring should propagate metric failure");
        assert!(err.to_string().contains("metric failure"));

        let instruction = with_named_predictor(&mut module, "predictor", |predictor| {
            Ok(predictor.instruction())
        })
        .expect("predictor lookup should succeed");
        assert_eq!(instruction, "seed-instruction");
    }
}
