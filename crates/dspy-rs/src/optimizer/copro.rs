use anyhow::{Result, anyhow};
use bon::Builder;

use crate::core::DynPredictor;
use crate::evaluate::{TypedMetric, average_score};
use crate::optimizer::{
    Optimizer, evaluate_module_with_metric, predictor_names, with_named_predictor,
};
use crate::predictors::Example;
use crate::{Facet, Module, Signature};

/// Breadth-first instruction optimizer.
///
/// COPRO (Collaborative Prompt Optimization) generates `breadth` candidate instructions
/// per predictor, evaluates each on the trainset, keeps the best, then repeats for
/// `depth` rounds. Simple and predictable — good for quick iteration when you want
/// better instructions without complex search.
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

    fn set_instruction<M>(module: &mut M, predictor_name: &str, instruction: String) -> Result<()>
    where
        M: for<'a> Facet<'a>,
    {
        with_named_predictor(module, predictor_name, |predictor| {
            predictor.set_instruction(instruction);
            Ok(())
        })
    }

    async fn score_candidate<S, M, MT>(
        &self,
        module: &mut M,
        predictor_name: &str,
        candidate_instruction: &str,
        trainset: &[Example<S>],
        metric: &MT,
    ) -> Result<f32>
    where
        S: Signature,
        S::Input: Clone,
        M: Module<Input = S::Input> + for<'a> Facet<'a>,
        MT: TypedMetric<S, M>,
    {
        let original_state = with_named_predictor(module, predictor_name, |predictor| {
            Ok(predictor.dump_state())
        })?;

        Self::set_instruction(module, predictor_name, candidate_instruction.to_string())?;
        let evaluation = evaluate_module_with_metric(&*module, trainset, metric).await;

        match evaluation {
            Ok(outcomes) => {
                with_named_predictor(module, predictor_name, |predictor| {
                    predictor.load_state(original_state.clone())
                })?;
                Ok(average_score(&outcomes))
            }
            Err(eval_err) => {
                if let Err(restore_err) =
                    with_named_predictor(module, predictor_name, |predictor| {
                        predictor.load_state(original_state)
                    })
                {
                    return Err(anyhow!(
                        "candidate evaluation failed: {eval_err}; failed to restore predictor state: {restore_err}"
                    ));
                }
                Err(eval_err)
            }
        }
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
        if self.breadth <= 1 {
            return Err(anyhow!("breadth must be greater than 1"));
        }

        let predictor_names = predictor_names(module)?;

        if predictor_names.is_empty() {
            return Err(anyhow!("no optimizable predictors found"));
        }

        for depth in 0..self.depth {
            for predictor_name in &predictor_names {
                let base_instruction = Self::current_instruction(module, predictor_name)?;

                let candidates = with_named_predictor(module, predictor_name, |predictor| {
                    Ok(self.candidate_instructions(&base_instruction, predictor, depth))
                })?;

                let mut best_instruction = base_instruction.clone();
                let mut best_score = f32::MIN;

                for candidate in candidates {
                    let score = self
                        .score_candidate::<S, _, _>(
                            module,
                            predictor_name,
                            &candidate,
                            &trainset,
                            metric,
                        )
                        .await?;
                    if score > best_score {
                        best_score = score;
                        best_instruction = candidate;
                    }
                }

                Self::set_instruction(module, predictor_name, best_instruction)?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use anyhow::{Result, anyhow};

    use super::*;
    use crate::evaluate::{MetricOutcome, TypedMetric};
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

    impl TypedMetric<CoproStateSig, CoproStateModule> for AlwaysFailMetric {
        async fn evaluate(
            &self,
            _example: &Example<CoproStateSig>,
            _prediction: &Predicted<CoproStateSigOutput>,
        ) -> Result<MetricOutcome> {
            Err(anyhow!("metric failure"))
        }
    }

    fn trainset() -> Vec<Example<CoproStateSig>> {
        vec![Example::new(
            CoproStateSigInput {
                prompt: "one".to_string(),
            },
            CoproStateSigOutput {
                answer: "one".to_string(),
            },
        )]
    }

    #[tokio::test]
    async fn score_candidate_restores_state_when_metric_errors() {
        let optimizer = COPRO::builder().breadth(2).depth(1).build();
        let mut module = CoproStateModule {
            predictor: Predict::<CoproStateSig>::builder()
                .instruction("seed-instruction")
                .build(),
        };

        let err = optimizer
            .score_candidate::<CoproStateSig, _, _>(
                &mut module,
                "predictor",
                "candidate instruction",
                &trainset(),
                &AlwaysFailMetric,
            )
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
