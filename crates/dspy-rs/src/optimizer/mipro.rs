/// MIPROv2 Optimizer (typed metric path).
use anyhow::{Result, anyhow};
use bon::Builder;

use crate::evaluate::{TypedMetric, average_score};
use crate::optimizer::{
    Optimizer, evaluate_module_with_metric, predictor_names, with_named_predictor,
};
use crate::{BamlType, BamlValue, Example, Facet, Module, SignatureSchema};

/// Represents a single execution trace of the program.
#[derive(Clone, Debug)]
pub struct Trace {
    pub inputs: Example,
    pub outputs: BamlValue,
    pub score: Option<f32>,
}

impl Trace {
    pub fn new(inputs: Example, outputs: BamlValue, score: Option<f32>) -> Self {
        Self {
            inputs,
            outputs,
            score,
        }
    }

    pub fn format_for_prompt(&self) -> String {
        let mut result = String::new();
        result.push_str("Input:\n");

        for (key, value) in &self.inputs.data {
            result.push_str(&format!("  {}: {}\n", key, value));
        }

        result.push_str("Output:\n");
        result.push_str(&format!("  {}\n", self.outputs));

        if let Some(score) = self.score {
            result.push_str(&format!("Score: {:.3}\n", score));
        }

        result
    }
}

/// Represents a candidate prompt with its associated examples and score.
#[derive(Clone, Debug)]
pub struct PromptCandidate {
    pub instruction: String,
    pub demos: Vec<Example>,
    pub score: f32,
}

impl PromptCandidate {
    pub fn new(instruction: String, demos: Vec<Example>) -> Self {
        Self {
            instruction,
            demos,
            score: 0.0,
        }
    }

    pub fn with_score(mut self, score: f32) -> Self {
        self.score = score;
        self
    }
}

/// Library of prompting tips and best practices.
pub struct PromptingTips {
    pub tips: Vec<String>,
}

impl PromptingTips {
    pub fn default_tips() -> Self {
        Self {
            tips: vec![
                "Use clear and specific language".to_string(),
                "Provide context about the task domain".to_string(),
                "Specify the desired output format".to_string(),
                "Use chain-of-thought reasoning for complex tasks".to_string(),
                "Include few-shot examples when helpful".to_string(),
                "Break down complex instructions into steps".to_string(),
                "Use role-playing (e.g., 'You are an expert...') when appropriate".to_string(),
                "Specify constraints and edge cases".to_string(),
                "Request explanations or reasoning when needed".to_string(),
                "Use structured output formats (JSON, lists, etc.) when applicable".to_string(),
                "Consider the model's strengths and limitations".to_string(),
                "Be explicit about what to avoid or exclude".to_string(),
                "Use positive framing (what to do vs. what not to do)".to_string(),
                "Provide examples of both correct and incorrect outputs when useful".to_string(),
                "Use delimiters or markers to separate different sections".to_string(),
            ],
        }
    }

    pub fn format_for_prompt(&self) -> String {
        self.tips
            .iter()
            .enumerate()
            .map(|(i, tip)| format!("{}. {}", i + 1, tip))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[derive(Builder)]
pub struct MIPROv2 {
    #[builder(default = 10)]
    pub num_candidates: usize,

    #[builder(default = 3)]
    pub max_bootstrapped_demos: usize,

    #[builder(default = 3)]
    pub max_labeled_demos: usize,

    #[builder(default = 20)]
    pub num_trials: usize,

    #[builder(default = 25)]
    pub minibatch_size: usize,

    #[builder(default = 1.0)]
    pub temperature: f32,

    pub prompt_model: Option<crate::LM>,

    #[builder(default = true)]
    pub track_stats: bool,

    pub seed: Option<u64>,
}

impl MIPROv2 {
    async fn generate_traces<M, MT>(
        &self,
        module: &M,
        examples: &[Example],
        metric: &MT,
    ) -> Result<Vec<Trace>>
    where
        M: Module,
        MT: TypedMetric<M>,
    {
        let mut traces = Vec::with_capacity(examples.len());
        for example in examples {
            let input = crate::evaluate::input_from_example::<M::Input>(example)?;
            let predicted = module.call(input).await.map_err(|err| anyhow!("{err}"))?;
            let outcome = metric.evaluate(example, &predicted).await?;
            let (output, _) = predicted.into_parts();
            traces.push(Trace::new(example.clone(), output.to_baml_value(), Some(outcome.score)));
        }

        Ok(traces)
    }

    pub fn select_best_traces(&self, traces: &[Trace], num_select: usize) -> Vec<Trace> {
        let mut scored_traces: Vec<_> = traces
            .iter()
            .filter(|t| t.score.is_some())
            .cloned()
            .collect();

        scored_traces.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        scored_traces.into_iter().take(num_select).collect()
    }

    fn generate_candidate_instructions(
        &self,
        program_description: &str,
        traces: &[Trace],
        num_candidates: usize,
    ) -> Vec<String> {
        let tips = PromptingTips::default_tips();
        let score_hint = traces
            .iter()
            .filter_map(|t| t.score)
            .fold(0.0f32, f32::max);

        (0..num_candidates)
            .map(|idx| {
                let tip = &tips.tips[idx % tips.tips.len()];
                format!(
                    "{program_description}\n\nOptimization candidate {}:\n- {}\n- Target score >= {:.3}",
                    idx + 1,
                    tip,
                    score_hint
                )
            })
            .collect()
    }

    pub fn create_prompt_candidates(
        &self,
        instructions: Vec<String>,
        traces: &[Trace],
    ) -> Vec<PromptCandidate> {
        let best_traces = self.select_best_traces(traces, self.max_labeled_demos);
        let demo_examples: Vec<Example> = best_traces.into_iter().map(|t| t.inputs).collect();

        instructions
            .into_iter()
            .map(|inst| PromptCandidate::new(inst, demo_examples.clone()))
            .collect()
    }

    async fn evaluate_candidate<M, MT>(
        &self,
        module: &mut M,
        candidate: &PromptCandidate,
        eval_examples: &[Example],
        predictor_name: &str,
        metric: &MT,
    ) -> Result<f32>
    where
        M: Module + for<'a> Facet<'a>,
        MT: TypedMetric<M>,
    {
        with_named_predictor(module, predictor_name, |predictor| {
            predictor.set_instruction(candidate.instruction.clone());
            predictor.set_demos_from_examples(candidate.demos.clone())?;
            Ok(())
        })?;

        let minibatch: Vec<Example> = eval_examples
            .iter()
            .take(self.minibatch_size)
            .cloned()
            .collect();

        let outcomes = evaluate_module_with_metric(&*module, &minibatch, metric).await?;
        Ok(average_score(&outcomes))
    }

    async fn evaluate_and_select_best<M, MT>(
        &self,
        module: &mut M,
        candidates: Vec<PromptCandidate>,
        eval_examples: &[Example],
        predictor_name: &str,
        metric: &MT,
    ) -> Result<PromptCandidate>
    where
        M: Module + for<'a> Facet<'a>,
        MT: TypedMetric<M>,
    {
        let mut evaluated = Vec::new();

        for candidate in candidates {
            let score = self
                .evaluate_candidate(module, &candidate, eval_examples, predictor_name, metric)
                .await?;
            evaluated.push(candidate.with_score(score));
        }

        evaluated
            .into_iter()
            .max_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or_else(|| anyhow!("no candidates to evaluate"))
    }

    pub fn format_schema_fields(&self, signature: &SignatureSchema) -> String {
        let mut result = String::new();

        result.push_str("Input Fields:\n");
        for field in signature.input_fields() {
            let desc = if field.docs.is_empty() {
                "No description"
            } else {
                field.docs.as_str()
            };
            result.push_str(&format!("  - {}: {}\n", field.lm_name, desc));
        }

        result.push_str("\nOutput Fields:\n");
        for field in signature.output_fields() {
            let desc = if field.docs.is_empty() {
                "No description"
            } else {
                field.docs.as_str()
            };
            result.push_str(&format!("  - {}: {}\n", field.lm_name, desc));
        }

        result
    }
}

impl Optimizer for MIPROv2 {
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
        let predictor_names = predictor_names(module)?;

        if predictor_names.is_empty() {
            return Err(anyhow!("no optimizable predictors found"));
        }

        for predictor_name in predictor_names {
            let signature_desc = {
                with_named_predictor(module, &predictor_name, |predictor| {
                    Ok(self.format_schema_fields(predictor.schema()))
                })?
            };

            let traces = self.generate_traces(module, &trainset, metric).await?;
            let instructions = self.generate_candidate_instructions(
                &signature_desc,
                &traces,
                self.num_candidates,
            );
            let candidates = self.create_prompt_candidates(instructions, &traces);
            let best_candidate = self
                .evaluate_and_select_best(module, candidates, &trainset, &predictor_name, metric)
                .await?;

            with_named_predictor(module, &predictor_name, |predictor| {
                predictor.set_instruction(best_candidate.instruction.clone());
                predictor.set_demos_from_examples(best_candidate.demos)?;
                Ok(())
            })?;
        }

        Ok(())
    }
}
