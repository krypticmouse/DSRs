use anyhow::{Result, anyhow};
use bon::Builder;

use crate::core::DynPredictor;
use crate::evaluate::{TypedMetric, average_score};
use crate::optimizer::{
    Optimizer, evaluate_module_with_metric, predictor_names, with_named_predictor,
};
use crate::{Example, Facet, Module};

#[derive(Builder)]
pub struct COPRO {
    #[builder(default = 10)]
    pub breadth: usize,
    #[builder(default = 3)]
    pub depth: usize,
    #[builder(default = 1.4)]
    pub init_temperature: f32,
    #[builder(default = false)]
    pub track_stats: bool,
    pub prompt_model: Option<crate::LM>,
}

impl COPRO {
    fn current_instruction<M>(module: &mut M, predictor_name: &str) -> Result<String>
    where
        M: for<'a> Facet<'a>,
    {
        with_named_predictor(module, predictor_name, |predictor| Ok(predictor.instruction()))
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

    async fn score_candidate<M, MT>(
        &self,
        module: &mut M,
        predictor_name: &str,
        candidate_instruction: &str,
        trainset: &[Example],
        metric: &MT,
    ) -> Result<f32>
    where
        M: Module + for<'a> Facet<'a>,
        MT: TypedMetric<M>,
    {
        Self::set_instruction(module, predictor_name, candidate_instruction.to_string())?;
        let outcomes = evaluate_module_with_metric(&*module, trainset, metric).await?;
        Ok(average_score(&outcomes))
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

    async fn compile<M, MT>(
        &self,
        module: &mut M,
        trainset: Vec<Example>,
        metric: &MT,
    ) -> Result<Self::Report>
    where
        M: Module + for<'a> Facet<'a>,
        MT: TypedMetric<M>,
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
                        .score_candidate(module, predictor_name, &candidate, &trainset, metric)
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
