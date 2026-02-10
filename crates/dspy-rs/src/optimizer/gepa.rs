/// GEPA (Genetic-Pareto) Optimizer Implementation on typed metric path.
use anyhow::{Context, Result, anyhow};
use bon::Builder;
use serde::{Deserialize, Serialize};

use crate::evaluate::{MetricOutcome, TypedMetric, average_score, input_from_example};
use crate::optimizer::{
    Optimizer, evaluate_module_with_metric, predictor_names, with_named_predictor,
};
use crate::{BamlType, BamlValue, Example, Facet, Module};

use super::pareto::ParetoFrontier;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GEPACandidate {
    pub id: usize,
    pub instruction: String,
    pub module_name: String,
    pub example_scores: Vec<f32>,
    pub parent_id: Option<usize>,
    pub generation: usize,
}

impl GEPACandidate {
    pub fn average_score(&self) -> f32 {
        if self.example_scores.is_empty() {
            return 0.0;
        }
        self.example_scores.iter().sum::<f32>() / self.example_scores.len() as f32
    }

    pub fn mutate(&self, new_instruction: String, generation: usize) -> Self {
        Self {
            id: 0,
            instruction: new_instruction,
            module_name: self.module_name.clone(),
            example_scores: Vec::new(),
            parent_id: Some(self.id),
            generation,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GEPAResult {
    pub best_candidate: GEPACandidate,
    pub all_candidates: Vec<GEPACandidate>,
    pub total_rollouts: usize,
    pub total_lm_calls: usize,
    pub evolution_history: Vec<(usize, f32)>,
    pub highest_score_achieved_per_val_task: Vec<f32>,
    pub best_outputs_valset: Option<Vec<BamlValue>>,
    pub frontier_history: Vec<ParetoStatistics>,
}

pub use super::pareto::ParetoStatistics;

#[derive(Builder)]
pub struct GEPA {
    #[builder(default = 20)]
    pub num_iterations: usize,

    #[builder(default = 25)]
    pub minibatch_size: usize,

    #[builder(default = 10)]
    pub num_trials: usize,

    #[builder(default = 1.0)]
    pub temperature: f32,

    #[builder(default = true)]
    pub track_stats: bool,

    #[builder(default = false)]
    pub track_best_outputs: bool,

    pub max_rollouts: Option<usize>,
    pub max_lm_calls: Option<usize>,
    pub prompt_model: Option<crate::LM>,
    pub valset: Option<Vec<Example>>,
}

impl GEPA {
    fn set_instruction<M>(module: &mut M, module_name: &str, instruction: String) -> Result<()>
    where
        M: for<'a> Facet<'a>,
    {
        with_named_predictor(module, module_name, |predictor| {
            predictor.set_instruction(instruction);
            Ok(())
        })
    }

    async fn evaluate_candidate<M, MT>(
        &self,
        module: &mut M,
        module_name: &str,
        instruction: &str,
        examples: &[Example],
        metric: &MT,
    ) -> Result<Vec<MetricOutcome>>
    where
        M: Module + for<'a> Facet<'a>,
        MT: TypedMetric<M>,
    {
        Self::set_instruction(module, module_name, instruction.to_string())?;
        evaluate_module_with_metric(&*module, examples, metric).await
    }

    fn require_feedback(
        outcomes: &[MetricOutcome],
        module_name: &str,
        generation: usize,
    ) -> Result<()> {
        if outcomes.iter().any(|o| o.feedback.is_none()) {
            return Err(anyhow!(
                "GEPA requires feedback for every evaluated example (module=`{module_name}`, generation={generation})"
            ));
        }
        Ok(())
    }

    fn summarize_feedback(outcomes: &[MetricOutcome]) -> String {
        let mut lines = Vec::new();
        for (idx, outcome) in outcomes.iter().enumerate() {
            if let Some(feedback) = &outcome.feedback {
                lines.push(format!(
                    "{}: score={:.3}; {}",
                    idx + 1,
                    outcome.score,
                    feedback.feedback
                ));
            }
        }
        lines.join("\n")
    }

    async fn collect_best_outputs<M>(module: &M, eval_set: &[Example]) -> Result<Vec<BamlValue>>
    where
        M: Module,
        M::Output: BamlType,
    {
        let mut outputs = Vec::with_capacity(eval_set.len());
        for example in eval_set {
            let input = input_from_example::<M::Input>(example)?;
            let predicted = module.call(input).await.map_err(|err| anyhow!("{err}"))?;
            outputs.push(predicted.into_inner().to_baml_value());
        }
        Ok(outputs)
    }
}

impl Optimizer for GEPA {
    type Report = GEPAResult;

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
        let eval_set = self.valset.as_ref().unwrap_or(&trainset);

        let predictor_names = predictor_names(module)?;

        if predictor_names.is_empty() {
            return Err(anyhow!("no optimizable predictors found"));
        }

        let mut frontier = ParetoFrontier::new();

        for module_name in &predictor_names {
            let instruction = {
                with_named_predictor(module, module_name, |predictor| Ok(predictor.instruction()))?
            };

            let outcomes = self
                .evaluate_candidate(module, module_name, &instruction, eval_set, metric)
                .await?;
            Self::require_feedback(&outcomes, module_name, 0)?;

            let scores: Vec<f32> = outcomes.iter().map(|o| o.score).collect();
            let candidate = GEPACandidate {
                id: 0,
                instruction,
                module_name: module_name.clone(),
                example_scores: scores.clone(),
                parent_id: None,
                generation: 0,
            };
            frontier.add_candidate(candidate, &scores);
        }

        let mut all_candidates = Vec::new();
        let mut evolution_history = Vec::new();
        let mut frontier_history = Vec::new();
        let mut total_rollouts = 0usize;
        let mut total_lm_calls = 0usize;

        for generation in 0..self.num_iterations {
            if let Some(max_rollouts) = self.max_rollouts
                && total_rollouts >= max_rollouts
            {
                break;
            }

            if let Some(max_lm_calls) = self.max_lm_calls
                && total_lm_calls >= max_lm_calls
            {
                break;
            }

            let parent = frontier
                .sample_proportional_to_coverage()
                .context("failed to sample from frontier")?
                .clone();

            let minibatch: Vec<Example> = trainset
                .iter()
                .take(self.minibatch_size.max(1))
                .cloned()
                .collect();

            let parent_outcomes = self
                .evaluate_candidate(
                    module,
                    &parent.module_name,
                    &parent.instruction,
                    &minibatch,
                    metric,
                )
                .await?;
            Self::require_feedback(&parent_outcomes, &parent.module_name, generation)?;

            let feedback_summary = Self::summarize_feedback(&parent_outcomes);
            let parent_score = average_score(&parent_outcomes);
            total_rollouts += parent_outcomes.len();

            let child_instruction = format!(
                "{}\n\n[GEPA gen {}] Improve based on feedback:\n{}\n(Parent score {:.3})",
                parent.instruction,
                generation + 1,
                feedback_summary,
                parent_score,
            );

            let child = parent.mutate(child_instruction, generation + 1);

            let child_outcomes = self
                .evaluate_candidate(
                    module,
                    &child.module_name,
                    &child.instruction,
                    eval_set,
                    metric,
                )
                .await?;
            Self::require_feedback(&child_outcomes, &child.module_name, generation + 1)?;

            let child_scores: Vec<f32> = child_outcomes.iter().map(|o| o.score).collect();
            total_rollouts += child_scores.len();
            total_lm_calls += 1;

            let mut child = child;
            child.example_scores = child_scores.clone();
            let _added = frontier.add_candidate(child.clone(), &child_scores);

            if self.track_stats {
                all_candidates.push(child);
                let best_avg = frontier
                    .best_by_average()
                    .map(|c| c.average_score())
                    .unwrap_or(0.0);
                evolution_history.push((generation, best_avg));
                frontier_history.push(frontier.statistics());
            }
        }

        let best_candidate = frontier
            .best_by_average()
            .cloned()
            .context("no candidates available on Pareto frontier")?;

        Self::set_instruction(
            module,
            &best_candidate.module_name,
            best_candidate.instruction.clone(),
        )?;

        let highest_score_achieved_per_val_task = if frontier.is_empty() {
            Vec::new()
        } else {
            let mut highs = vec![f32::MIN; eval_set.len()];
            for candidate in frontier.candidates() {
                for (idx, score) in candidate.example_scores.iter().enumerate() {
                    if idx < highs.len() {
                        highs[idx] = highs[idx].max(*score);
                    }
                }
            }
            highs
        };

        let best_outputs_valset = if self.track_best_outputs {
            Some(Self::collect_best_outputs(module, eval_set).await?)
        } else {
            None
        };

        Ok(GEPAResult {
            best_candidate,
            all_candidates,
            total_rollouts,
            total_lm_calls,
            evolution_history,
            highest_score_achieved_per_val_task,
            best_outputs_valset,
            frontier_history,
        })
    }
}
