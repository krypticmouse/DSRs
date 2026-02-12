use anyhow::{Context, Result, anyhow};
use bon::Builder;
use serde::{Deserialize, Serialize};

use crate::evaluate::{MetricOutcome, TypedMetric, average_score};
use crate::optimizer::{
    Optimizer, evaluate_module_with_metric, predictor_names, with_named_predictor,
};
use crate::predictors::Example;
use crate::{BamlType, BamlValue, Facet, Module, Signature};

use super::pareto::ParetoFrontier;

/// A single instruction candidate tracked through GEPA's evolutionary search.
///
/// Carries the instruction text, per-example scores, lineage (parent_id), and
/// generation number. The Pareto frontier selects candidates that aren't dominated
/// on any individual example — not just by average score.
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

/// Full report from a [`GEPA`] optimization run.
///
/// Contains the winning candidate, the complete candidate history (if `track_stats`
/// was enabled), budget usage, and optionally the best outputs on the validation set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GEPAResult {
    /// The candidate with the best average score on the Pareto frontier.
    pub best_candidate: GEPACandidate,
    /// All candidates evaluated (empty unless `track_stats` is enabled).
    pub all_candidates: Vec<GEPACandidate>,
    /// Total evaluation rollouts consumed.
    pub total_rollouts: usize,
    /// Total LM calls consumed (rollouts + candidate generation).
    pub total_lm_calls: usize,
    /// (generation, best_average_score) pairs for plotting convergence.
    pub evolution_history: Vec<(usize, f32)>,
    /// Highest score achieved per validation example across all candidates.
    pub highest_score_achieved_per_val_task: Vec<f32>,
    /// Best outputs on the validation set (only if `track_best_outputs` is enabled).
    pub best_outputs_valset: Option<Vec<BamlValue>>,
    /// Pareto frontier statistics per generation.
    pub frontier_history: Vec<ParetoStatistics>,
}

pub use super::pareto::ParetoStatistics;

/// Genetic-Pareto instruction optimizer with feedback-driven evolution.
///
/// GEPA uses an evolutionary search guided by per-example feedback from your metric.
/// Unlike [`COPRO`](crate::COPRO) which only uses numerical scores, GEPA requires your
/// [`TypedMetric`] to return [`MetricOutcome::with_feedback`] — textual feedback
/// explaining *why* each example scored the way it did. This feedback gets appended
/// to the instruction as a mutation prompt for the next generation, so the quality
/// of your feedback directly determines the quality of GEPA's search.
///
/// The Pareto frontier tracks candidates that aren't dominated on any individual
/// training example, not just by average score. This means GEPA finds instructions
/// that are robust across diverse inputs rather than overfitting to easy examples.
///
/// Only searches instruction space — no demo mutation, no crossover between candidates.
/// Each child has exactly one parent.
///
/// # Hyperparameters
///
/// - **`num_iterations`** (default: 20) — evolutionary generations. More = deeper search.
/// - **`minibatch_size`** (default: 25) — examples per parent evaluation within each
///   generation. Controls exploration vs cost.
/// - **`num_trials`** (default: 10) — **currently unused.** Reserved for multi-child
///   evolution (one child per generation right now). Setting this has no effect.
/// - **`temperature`** (default: 1.0) — **currently unused.** Reserved for mutation
///   diversity control. Setting this has no effect.
/// - **`max_rollouts`** / **`max_lm_calls`** — hard budget caps. Optimization stops
///   when either limit would be exceeded by the next batch.
/// - **`track_stats`** (default: true) — record all candidates and frontier history.
/// - **`track_best_outputs`** (default: false) — re-run the best instruction on the
///   eval set and record outputs.
/// - **`prompt_model`** — optional separate LM for candidate generation.
///
/// # Requires feedback
///
/// GEPA will error if any [`MetricOutcome`] from your metric has `feedback: None`.
/// Use [`MetricOutcome::with_feedback`] or provide a [`FeedbackMetric`](crate::FeedbackMetric).
///
/// # Cost
///
/// Roughly `num_iterations × (minibatch_size + eval_set_size) + initial_eval` LM calls.
/// Budget caps (`max_rollouts`, `max_lm_calls`) prevent runaway costs.
///
/// ```ignore
/// let gepa = GEPA::builder()
///     .num_iterations(20)
///     .max_lm_calls(Some(500))
///     .build();
/// let report = gepa.compile(&mut module, trainset, &feedback_metric).await?;
/// println!("Best score: {:.3}", report.best_candidate.average_score());
/// ```
#[derive(Builder)]
pub struct GEPA {
    /// Evolutionary generations to run.
    #[builder(default = 20)]
    pub num_iterations: usize,

    /// Examples per parent evaluation within each generation.
    #[builder(default = 25)]
    pub minibatch_size: usize,

    /// **Currently unused.** Reserved for multi-child evolution (one child per
    /// generation right now). Setting this has no effect.
    #[builder(default = 10)]
    pub num_trials: usize,

    /// **Currently unused.** Reserved for mutation diversity control.
    /// Setting this has no effect.
    #[builder(default = 1.0)]
    pub temperature: f32,

    /// Record all candidates and frontier history in the report.
    #[builder(default = true)]
    pub track_stats: bool,

    /// Re-run the best instruction on the eval set and record outputs.
    #[builder(default = false)]
    pub track_best_outputs: bool,

    /// Hard cap on total evaluation rollouts.
    pub max_rollouts: Option<usize>,
    /// Hard cap on total LM calls (rollouts + generation).
    pub max_lm_calls: Option<usize>,
    /// Optional separate LM for candidate generation.
    pub prompt_model: Option<crate::LM>,
}

impl GEPA {
    fn would_exceed_budget(current: usize, batch_cost: usize, max_budget: Option<usize>) -> bool {
        max_budget.is_some_and(|max| current.saturating_add(batch_cost) > max)
    }

    fn set_instruction<M>(module: &mut M, module_name: &str, instruction: String) -> Result<()>
    where
        M: for<'a> Facet<'a>,
    {
        with_named_predictor(module, module_name, |predictor| {
            predictor.set_instruction(instruction);
            Ok(())
        })
    }

    async fn evaluate_candidate<S, M, MT>(
        &self,
        module: &mut M,
        module_name: &str,
        instruction: &str,
        examples: &[Example<S>],
        metric: &MT,
    ) -> Result<Vec<MetricOutcome>>
    where
        S: Signature,
        S::Input: Clone,
        M: Module<Input = S::Input> + for<'a> Facet<'a>,
        MT: TypedMetric<S, M>,
    {
        let original_state =
            with_named_predictor(module, module_name, |predictor| Ok(predictor.dump_state()))?;

        Self::set_instruction(module, module_name, instruction.to_string())?;
        let evaluation = evaluate_module_with_metric(&*module, examples, metric).await;

        match evaluation {
            Ok(outcomes) => {
                with_named_predictor(module, module_name, |predictor| {
                    predictor.load_state(original_state.clone())
                })?;
                Ok(outcomes)
            }
            Err(eval_err) => {
                if let Err(restore_err) = with_named_predictor(module, module_name, |predictor| {
                    predictor.load_state(original_state)
                }) {
                    return Err(anyhow!(
                        "candidate evaluation failed: {eval_err}; failed to restore predictor state: {restore_err}"
                    ));
                }
                Err(eval_err)
            }
        }
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

    async fn collect_best_outputs<S, M>(
        module: &M,
        eval_set: &[Example<S>],
    ) -> Result<Vec<BamlValue>>
    where
        S: Signature,
        S::Input: Clone,
        M: Module<Input = S::Input>,
        M::Output: BamlType,
    {
        let mut outputs = Vec::with_capacity(eval_set.len());
        for example in eval_set {
            let input = example.input.clone();
            let predicted = module.call(input).await.map_err(|err| anyhow!("{err}"))?;
            outputs.push(predicted.into_inner().to_baml_value());
        }
        Ok(outputs)
    }

    /// Runs GEPA with an explicit validation set separate from the trainset.
    ///
    /// When `valset` is `Some`, initial evaluation and child scoring use the validation
    /// set, while parent re-evaluation uses the trainset minibatch. When `None`, the
    /// trainset serves both roles.
    ///
    /// # Errors
    ///
    /// - No optimizable predictors found
    /// - Any metric evaluation returns `feedback: None`
    /// - LM call failure during evaluation
    pub async fn compile_with_valset<S, M, MT>(
        &self,
        module: &mut M,
        trainset: Vec<Example<S>>,
        valset: Option<Vec<Example<S>>>,
        metric: &MT,
    ) -> Result<GEPAResult>
    where
        S: Signature,
        S::Input: Clone,
        M: Module<Input = S::Input> + for<'a> Facet<'a>,
        MT: TypedMetric<S, M>,
    {
        let eval_set = valset.as_deref().unwrap_or(&trainset);

        let predictor_names = predictor_names(module)?;

        if predictor_names.is_empty() {
            return Err(anyhow!("no optimizable predictors found"));
        }

        let mut frontier = ParetoFrontier::new();
        let mut total_lm_calls = 0usize;
        let mut total_rollouts = 0usize;

        for module_name in &predictor_names {
            if Self::would_exceed_budget(total_lm_calls, eval_set.len(), self.max_lm_calls)
                || Self::would_exceed_budget(total_rollouts, eval_set.len(), self.max_rollouts)
            {
                break;
            }

            let instruction = {
                with_named_predictor(module, module_name, |predictor| Ok(predictor.instruction()))?
            };

            let outcomes = self
                .evaluate_candidate::<S, _, _>(module, module_name, &instruction, eval_set, metric)
                .await?;
            total_lm_calls = total_lm_calls.saturating_add(outcomes.len());
            total_rollouts = total_rollouts.saturating_add(outcomes.len());
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

            let minibatch_end = trainset.len().min(self.minibatch_size.max(1));
            let minibatch = &trainset[..minibatch_end];

            if Self::would_exceed_budget(total_lm_calls, minibatch.len(), self.max_lm_calls)
                || Self::would_exceed_budget(total_rollouts, minibatch.len(), self.max_rollouts)
            {
                break;
            }

            let parent_outcomes = self
                .evaluate_candidate::<S, _, _>(
                    module,
                    &parent.module_name,
                    &parent.instruction,
                    minibatch,
                    metric,
                )
                .await?;
            total_lm_calls = total_lm_calls.saturating_add(parent_outcomes.len());
            Self::require_feedback(&parent_outcomes, &parent.module_name, generation)?;

            let feedback_summary = Self::summarize_feedback(&parent_outcomes);
            let parent_score = average_score(&parent_outcomes);
            total_rollouts += parent_outcomes.len();

            if Self::would_exceed_budget(total_lm_calls, eval_set.len(), self.max_lm_calls)
                || Self::would_exceed_budget(total_rollouts, eval_set.len(), self.max_rollouts)
            {
                break;
            }

            let child_instruction = format!(
                "{}\n\n[GEPA gen {}] Improve based on feedback:\n{}\n(Parent score {:.3})",
                parent.instruction,
                generation + 1,
                feedback_summary,
                parent_score,
            );

            let child = parent.mutate(child_instruction, generation + 1);

            let child_outcomes = self
                .evaluate_candidate::<S, _, _>(
                    module,
                    &child.module_name,
                    &child.instruction,
                    eval_set,
                    metric,
                )
                .await?;
            total_lm_calls = total_lm_calls.saturating_add(child_outcomes.len());
            Self::require_feedback(&child_outcomes, &child.module_name, generation + 1)?;

            let child_scores: Vec<f32> = child_outcomes.iter().map(|o| o.score).collect();
            total_rollouts += child_scores.len();

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
            if Self::would_exceed_budget(total_lm_calls, eval_set.len(), self.max_lm_calls)
                || Self::would_exceed_budget(total_rollouts, eval_set.len(), self.max_rollouts)
            {
                tracing::debug!(
                    eval_examples = eval_set.len(),
                    total_lm_calls,
                    total_rollouts,
                    max_lm_calls = ?self.max_lm_calls,
                    max_rollouts = ?self.max_rollouts,
                    "skipping best output collection because budget would be exceeded"
                );
                None
            } else {
                let outputs = Self::collect_best_outputs::<S, _>(module, eval_set).await?;
                total_lm_calls = total_lm_calls.saturating_add(eval_set.len());
                total_rollouts = total_rollouts.saturating_add(eval_set.len());
                Some(outputs)
            }
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

impl Optimizer for GEPA {
    type Report = GEPAResult;

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
        self.compile_with_valset::<S, _, _>(module, trainset, None, metric)
            .await
    }
}

#[cfg(test)]
mod tests {
    use anyhow::{Result, anyhow};

    use super::*;
    use crate::evaluate::{MetricOutcome, TypedMetric};
    use crate::{CallMetadata, Predict, PredictError, Predicted, Signature};

    #[derive(Signature, Clone, Debug)]
    struct GepaStateSig {
        #[input]
        prompt: String,

        #[output]
        answer: String,
    }

    #[derive(facet::Facet)]
    #[facet(crate = facet)]
    struct GepaStateModule {
        predictor: Predict<GepaStateSig>,
    }

    impl Module for GepaStateModule {
        type Input = GepaStateSigInput;
        type Output = GepaStateSigOutput;

        async fn forward(
            &self,
            input: GepaStateSigInput,
        ) -> Result<Predicted<GepaStateSigOutput>, PredictError> {
            Ok(Predicted::new(
                GepaStateSigOutput {
                    answer: input.prompt,
                },
                CallMetadata::default(),
            ))
        }
    }

    struct AlwaysFailMetric;

    impl TypedMetric<GepaStateSig, GepaStateModule> for AlwaysFailMetric {
        async fn evaluate(
            &self,
            _example: &Example<GepaStateSig>,
            _prediction: &Predicted<GepaStateSigOutput>,
        ) -> Result<MetricOutcome> {
            Err(anyhow!("metric failure"))
        }
    }

    fn eval_set() -> Vec<Example<GepaStateSig>> {
        vec![Example::new(
            GepaStateSigInput {
                prompt: "one".to_string(),
            },
            GepaStateSigOutput {
                answer: "one".to_string(),
            },
        )]
    }

    #[tokio::test]
    async fn evaluate_candidate_restores_state_when_metric_errors() {
        let optimizer = GEPA::builder().num_iterations(1).minibatch_size(1).build();
        let mut module = GepaStateModule {
            predictor: Predict::<GepaStateSig>::builder()
                .instruction("seed-instruction")
                .build(),
        };

        let err = optimizer
            .evaluate_candidate::<GepaStateSig, _, _>(
                &mut module,
                "predictor",
                "candidate instruction",
                &eval_set(),
                &AlwaysFailMetric,
            )
            .await
            .expect_err("candidate evaluation should propagate metric failure");
        assert!(err.to_string().contains("metric failure"));

        let instruction = with_named_predictor(&mut module, "predictor", |predictor| {
            Ok(predictor.instruction())
        })
        .expect("predictor lookup should succeed");
        assert_eq!(instruction, "seed-instruction");
    }
}
