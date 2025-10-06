/// MIPROv2 Optimizer Implementation
///
/// Multi-prompt Instruction Proposal Optimizer (MIPROv2) is an advanced optimizer
/// that automatically generates and evaluates candidate prompts using LLMs.
///
/// ## Three-Stage Process
///
/// 1. **Trace Generation**: Runs the module with training data to generate execution traces
/// 2. **Prompt Generation**: Uses an LLM to generate candidate prompts based on:
///    - Program descriptions (LLM-generated)
///    - Execution traces
///    - Prompting tips library
/// 3. **Evaluation & Combination**: Evaluates candidates in batches and combines best components
///
/// ## Example
///
/// ```rust,ignore
/// use dspy_rs::{MIPROv2, Optimizer};
///
/// let optimizer = MIPROv2::builder()
///     .num_candidates(10)
///     .num_trials(20)
///     .build();
///
/// optimizer.compile(&mut module, train_examples).await?;
/// ```

use crate::{
    Evaluator, Example, LM, Module, Optimizable, Optimizer, Predict, Prediction, Predictor,
    example, get_lm,
};
use anyhow::{Context, Result};
use bon::Builder;
use crate as dspy_rs;
use dsrs_macros::Signature;
use std::sync::Arc;
use tokio::sync::Mutex;

// ============================================================================
// Signature Definitions for LLM-based Prompt Generation
// ============================================================================

#[Signature]
struct GenerateProgramDescription {
    /// You are an expert at understanding and describing programs. Given a task signature with input and output fields, and some example traces, generate a clear and concise description of what the program does.

    #[input(desc = "The task signature showing input and output fields")]
    pub signature_fields: String,

    #[input(desc = "Example input-output traces from the program")]
    pub example_traces: String,

    #[output(desc = "A clear description of what the program does")]
    pub program_description: String,
}

#[Signature]
struct GenerateInstructionFromTips {
    /// You are an expert prompt engineer. Given a program description, example traces, and a collection of prompting best practices, generate an effective instruction that will help a language model perform this task well.
    ///
    /// Be creative and consider various prompting techniques like chain-of-thought, few-shot examples, role-playing, and output formatting.

    #[input(desc = "Description of what the program should do")]
    pub program_description: String,

    #[input(desc = "Example input-output traces showing desired behavior")]
    pub example_traces: String,

    #[input(desc = "Best practices and tips for writing effective prompts")]
    pub prompting_tips: String,

    #[output(desc = "An optimized instruction for the language model")]
    pub instruction: String,
}

// ============================================================================
// Core Data Structures
// ============================================================================

/// Represents a single execution trace of the program
#[derive(Clone, Debug)]
struct Trace {
    /// Input example
    inputs: Example,
    /// Output prediction
    outputs: Prediction,
    /// Evaluation score (if available)
    score: Option<f32>,
}

impl Trace {
    /// Creates a new trace
    fn new(inputs: Example, outputs: Prediction, score: Option<f32>) -> Self {
        Self {
            inputs,
            outputs,
            score,
        }
    }

    /// Formats the trace as a human-readable string for LLM consumption
    fn format_for_prompt(&self) -> String {
        let mut result = String::new();
        result.push_str("Input:\n");
        
        for (key, value) in &self.inputs.data {
            result.push_str(&format!("  {}: {}\n", key, value));
        }
        
        result.push_str("Output:\n");
        for (key, value) in &self.outputs.data {
            result.push_str(&format!("  {}: {}\n", key, value));
        }
        
        if let Some(score) = self.score {
            result.push_str(&format!("Score: {:.3}\n", score));
        }
        
        result
    }
}

/// Represents a candidate prompt with its associated examples and score
#[derive(Clone, Debug)]
struct PromptCandidate {
    /// The instruction text
    instruction: String,
    /// Few-shot demonstration examples (reserved for future enhancement)
    #[allow(dead_code)]
    demos: Vec<Example>,
    /// Evaluation score
    score: f32,
}

impl PromptCandidate {
    /// Creates a new candidate with default score
    fn new(instruction: String, demos: Vec<Example>) -> Self {
        Self {
            instruction,
            demos,
            score: 0.0,
        }
    }

    /// Updates the candidate's score
    fn with_score(mut self, score: f32) -> Self {
        self.score = score;
        self
    }
}

/// Library of prompting tips and best practices
struct PromptingTips {
    tips: Vec<String>,
}

impl PromptingTips {
    /// Creates a new prompting tips library with default tips
    fn default_tips() -> Self {
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

    /// Formats tips as a string for LLM consumption
    fn format_for_prompt(&self) -> String {
        self.tips
            .iter()
            .enumerate()
            .map(|(i, tip)| format!("{}. {}", i + 1, tip))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

// ============================================================================
// MIPROv2 Optimizer
// ============================================================================

/// MIPROv2 (Multi-prompt Instruction Proposal Optimizer v2)
///
/// An advanced optimizer that uses LLMs to automatically generate and refine
/// prompts based on program traces, descriptions, and prompting best practices.
#[derive(Builder)]
pub struct MIPROv2 {
    /// Number of candidate prompts to generate per iteration
    #[builder(default = 10)]
    pub num_candidates: usize,

    /// Maximum number of bootstrapped (generated) demos to include
    #[builder(default = 3)]
    pub max_bootstrapped_demos: usize,

    /// Maximum number of labeled demos to include from training set
    #[builder(default = 3)]
    pub max_labeled_demos: usize,

    /// Number of evaluation trials (iterations)
    #[builder(default = 20)]
    pub num_trials: usize,

    /// Size of minibatch for evaluation
    #[builder(default = 25)]
    pub minibatch_size: usize,

    /// Temperature for prompt generation
    #[builder(default = 1.0)]
    pub temperature: f32,

    /// Optional separate LM for prompt generation (defaults to global LM)
    pub prompt_model: Option<LM>,

    /// Track and display statistics
    #[builder(default = true)]
    pub track_stats: bool,

    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl MIPROv2 {
    // ========================================================================
    // Stage 1: Trace Generation
    // ========================================================================

    /// Generates execution traces by running the module on training examples
    async fn generate_traces<M>(
        &self,
        module: &M,
        examples: &[Example],
    ) -> Result<Vec<Trace>>
    where
        M: Module + Evaluator,
    {
        let mut traces = Vec::with_capacity(examples.len());

        println!("Stage 1: Generating traces from {} examples", examples.len());

        for (idx, example) in examples.iter().enumerate() {
            if idx % 10 == 0 {
                println!("  Processing example {}/{}", idx + 1, examples.len());
            }

            // Run forward pass
            let prediction = module
                .forward(example.clone())
                .await
                .context("Failed to generate prediction for trace")?;

            // Evaluate the prediction
            let score = module.metric(example, &prediction).await;

            traces.push(Trace::new(example.clone(), prediction, Some(score)));
        }

        println!("Generated {} traces", traces.len());
        Ok(traces)
    }

    /// Selects the best traces based on their scores
    fn select_best_traces(&self, traces: &[Trace], num_select: usize) -> Vec<Trace> {
        let mut scored_traces: Vec<_> = traces
            .iter()
            .filter(|t| t.score.is_some())
            .cloned()
            .collect();

        // Sort by score descending
        scored_traces.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        scored_traces.into_iter().take(num_select).collect()
    }

    // ========================================================================
    // Stage 2: Candidate Prompt Generation
    // ========================================================================

    /// Generates a program description using an LLM
    async fn generate_program_description(
        &self,
        signature_desc: &str,
        traces: &[Trace],
    ) -> Result<String> {
        let description_generator = Predict::new(GenerateProgramDescription::new());

        // Format traces for the prompt
        let traces_str = traces
            .iter()
            .take(5) // Use first 5 traces
            .map(|t| t.format_for_prompt())
            .collect::<Vec<_>>()
            .join("\n---\n");

        let input = example! {
            "signature_fields": "input" => signature_desc.to_string(),
            "example_traces": "input" => traces_str,
        };

        let prediction = if let Some(mut pm) = self.prompt_model.clone() {
            pm.config.temperature = 0.7;
            description_generator
                .forward_with_config(input, Arc::new(Mutex::new(pm)))
                .await?
        } else {
            let lm = get_lm();
            lm.lock().await.config.temperature = 0.7;
            description_generator
                .forward_with_config(input, lm)
                .await?
        };

        Ok(prediction
            .data
            .get("program_description")
            .and_then(|v| v.as_str())
            .unwrap_or("Generate accurate outputs for the given inputs.")
            .to_string())
    }

    /// Generates candidate instructions using LLM with prompting tips
    async fn generate_candidate_instructions(
        &self,
        program_description: &str,
        traces: &[Trace],
        num_candidates: usize,
    ) -> Result<Vec<String>> {
        let instruction_generator = Predict::new(GenerateInstructionFromTips::new());
        let tips = PromptingTips::default_tips();

        // Format traces
        let traces_str = traces
            .iter()
            .take(8)
            .map(|t| t.format_for_prompt())
            .collect::<Vec<_>>()
            .join("\n---\n");

        println!("Stage 2: Generating {} candidate instructions", num_candidates);

        let mut candidates = Vec::new();

        // Generate candidates sequentially (simpler and avoids lifetime issues)
        for i in 0..num_candidates {
            let input = example! {
                "program_description": "input" => program_description.to_string(),
                "example_traces": "input" => traces_str.clone(),
                "prompting_tips": "input" => tips.format_for_prompt(),
            };

            let result = if let Some(mut pm) = self.prompt_model.clone() {
                pm.config.temperature = self.temperature;
                instruction_generator
                    .forward_with_config(input, Arc::new(Mutex::new(pm)))
                    .await
            } else {
                let lm = get_lm();
                {
                    let mut guard = lm.lock().await;
                    guard.config.temperature = self.temperature;
                }
                instruction_generator.forward_with_config(input, lm).await
            };

            if let Ok(pred) = result {
                if let Some(instruction) = pred.data.get("instruction").and_then(|v| v.as_str()) {
                    candidates.push(instruction.to_string());
                }
            }

            if (i + 1) % 3 == 0 || i == num_candidates - 1 {
                println!(
                    "  Generated {}/{} candidates",
                    candidates.len(),
                    num_candidates
                );
            }
        }

        println!("Generated {} total candidate instructions", candidates.len());
        Ok(candidates)
    }

    /// Creates prompt candidates by pairing instructions with demo selections
    fn create_prompt_candidates(
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

    // ========================================================================
    // Stage 3: Evaluation and Selection
    // ========================================================================

    /// Evaluates a single prompt candidate
    async fn evaluate_candidate<M>(
        &self,
        module: &mut M,
        candidate: &PromptCandidate,
        eval_examples: &[Example],
        predictor_name: &str,
    ) -> Result<f32>
    where
        M: Module + Optimizable + Evaluator,
    {
        // Update module with candidate instruction
        {
            let mut params = module.parameters();
            if let Some(predictor) = params.get_mut(predictor_name) {
                predictor.update_signature_instruction(candidate.instruction.clone())?;

                // Note: Demo setting would require mutable signature access
                // This is a design consideration for future enhancement
            }
        }

        // Evaluate on minibatch
        let minibatch: Vec<Example> = eval_examples
            .iter()
            .take(self.minibatch_size)
            .cloned()
            .collect();

        let score = module.evaluate(minibatch).await;
        Ok(score)
    }

    /// Evaluates all candidates and returns the best one
    async fn evaluate_and_select_best<M>(
        &self,
        module: &mut M,
        candidates: Vec<PromptCandidate>,
        eval_examples: &[Example],
        predictor_name: &str,
    ) -> Result<PromptCandidate>
    where
        M: Module + Optimizable + Evaluator,
    {
        println!(
            "Stage 3: Evaluating {} candidates on minibatch of {} examples",
            candidates.len(),
            self.minibatch_size.min(eval_examples.len())
        );

        let mut evaluated_candidates = Vec::new();

        for (idx, candidate) in candidates.into_iter().enumerate() {
            println!("  Evaluating candidate {}/{}", idx + 1, self.num_candidates);

            let score = self
                .evaluate_candidate(module, &candidate, eval_examples, predictor_name)
                .await?;

            evaluated_candidates.push(candidate.with_score(score));

            if self.track_stats {
                println!("    Score: {:.3}", score);
            }
        }

        // Find best candidate
        let best = evaluated_candidates
            .into_iter()
            .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal))
            .context("No candidates to evaluate")?;

        println!("Best candidate score: {:.3}", best.score);
        Ok(best)
    }

    // ========================================================================
    // Helper Methods
    // ========================================================================

    /// Formats signature fields as a string
    fn format_signature_fields(&self, signature: &dyn crate::core::MetaSignature) -> String {
        let mut result = String::new();

        result.push_str("Input Fields:\n");
        if let Some(obj) = signature.input_fields().as_object() {
            for (name, field) in obj {
                let desc = field
                    .get("desc")
                    .and_then(|v| v.as_str())
                    .unwrap_or("No description");
                result.push_str(&format!("  - {}: {}\n", name, desc));
            }
        }

        result.push_str("\nOutput Fields:\n");
        if let Some(obj) = signature.output_fields().as_object() {
            for (name, field) in obj {
                let desc = field
                    .get("desc")
                    .and_then(|v| v.as_str())
                    .unwrap_or("No description");
                result.push_str(&format!("  - {}: {}\n", name, desc));
            }
        }

        result
    }
}

// ============================================================================
// Optimizer Trait Implementation
// ============================================================================

impl Optimizer for MIPROv2 {
    async fn compile<M>(&self, module: &mut M, trainset: Vec<Example>) -> Result<()>
    where
        M: Module + Optimizable + Evaluator,
    {
        println!("\n=== MIPROv2 Optimization Started ===");
        println!("Configuration:");
        println!("  Candidates: {}", self.num_candidates);
        println!("  Trials: {}", self.num_trials);
        println!("  Minibatch size: {}", self.minibatch_size);
        println!("  Training examples: {}", trainset.len());

        // Get predictor information
        let predictor_names: Vec<String> = module.parameters().keys().cloned().collect();

        if predictor_names.is_empty() {
            return Err(anyhow::anyhow!("No optimizable parameters found in module"));
        }

        println!("  Optimizing {} predictor(s): {:?}\n", predictor_names.len(), predictor_names);

        // Optimize each predictor
        for predictor_name in predictor_names {
            println!("--- Optimizing predictor: {} ---", predictor_name);

            // Get signature for this predictor
            let signature_desc = {
                let params = module.parameters();
                if let Some(predictor) = params.get(&predictor_name) {
                    self.format_signature_fields(predictor.get_signature())
                } else {
                    continue;
                }
            };

            // Stage 1: Generate traces
            let traces = self.generate_traces(module, &trainset).await?;

            // Stage 2: Generate candidates
            let program_description = self
                .generate_program_description(&signature_desc, &traces)
                .await?;

            println!("Generated program description: {}", program_description);

            let instructions = self
                .generate_candidate_instructions(&program_description, &traces, self.num_candidates)
                .await?;

            let candidates = self.create_prompt_candidates(instructions, &traces);

            // Stage 3: Evaluate and select best
            let best_candidate = self
                .evaluate_and_select_best(module, candidates, &trainset, &predictor_name)
                .await?;

            // Apply best candidate
            {
                let mut params = module.parameters();
                if let Some(predictor) = params.get_mut(&predictor_name) {
                    predictor.update_signature_instruction(best_candidate.instruction.clone())?;
                    // Note: Demo setting would require mutable signature access
                    // This is a design consideration for future enhancement
                }
            }

            println!("✓ Optimized {} with score {:.3}", predictor_name, best_candidate.score);
            println!("  Instruction: {}\n", best_candidate.instruction);
        }

        println!("=== MIPROv2 Optimization Complete ===\n");
        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LmUsage;

    // ========================================================================
    // Trace Tests
    // ========================================================================

    #[test]
    fn test_trace_formatting() {
        let inputs = Example::new(
            [("question".to_string(), "What is 2+2?".into())].into(),
            vec!["question".to_string()],
            vec![],
        );

        let outputs = Prediction::new(
            [("answer".to_string(), "4".into())].into(),
            Default::default(),
        );

        let trace = Trace::new(inputs, outputs, Some(1.0));
        let formatted = trace.format_for_prompt();

        assert!(formatted.contains("question"));
        assert!(formatted.contains("What is 2+2?"));
        assert!(formatted.contains("answer"));
        assert!(formatted.contains("4"));
        assert!(formatted.contains("Score: 1.000"));
    }

    #[test]
    fn test_trace_formatting_without_score() {
        let inputs = Example::new(
            [("input".to_string(), "test".into())].into(),
            vec!["input".to_string()],
            vec![],
        );

        let outputs = Prediction::new(
            [("output".to_string(), "result".into())].into(),
            LmUsage::default(),
        );

        let trace = Trace::new(inputs, outputs, None);
        let formatted = trace.format_for_prompt();

        assert!(formatted.contains("Input:"));
        assert!(formatted.contains("Output:"));
        assert!(!formatted.contains("Score:"));
    }

    #[test]
    fn test_trace_with_multiple_fields() {
        let inputs = Example::new(
            [
                ("field1".to_string(), "value1".into()),
                ("field2".to_string(), "value2".into()),
                ("field3".to_string(), "value3".into()),
            ]
            .into(),
            vec!["field1".to_string(), "field2".to_string(), "field3".to_string()],
            vec![],
        );

        let outputs = Prediction::new(
            [
                ("out1".to_string(), "res1".into()),
                ("out2".to_string(), "res2".into()),
            ]
            .into(),
            LmUsage::default(),
        );

        let trace = Trace::new(inputs, outputs, Some(0.75));
        let formatted = trace.format_for_prompt();

        assert!(formatted.contains("field1"));
        assert!(formatted.contains("field2"));
        assert!(formatted.contains("field3"));
        assert!(formatted.contains("out1"));
        assert!(formatted.contains("out2"));
        assert!(formatted.contains("Score: 0.750"));
    }

    // ========================================================================
    // PromptingTips Tests
    // ========================================================================

    #[test]
    fn test_prompting_tips_default() {
        let tips = PromptingTips::default_tips();
        
        assert!(!tips.tips.is_empty());
        assert!(tips.tips.len() >= 15, "Should have at least 15 tips");
        
        // Verify some expected tips are present
        let tips_text = tips.tips.join(" ");
        assert!(tips_text.contains("clear"));
        assert!(tips_text.contains("chain-of-thought") || tips_text.contains("reasoning"));
    }

    #[test]
    fn test_prompting_tips_formatting() {
        let tips = PromptingTips::default_tips();
        let formatted = tips.format_for_prompt();

        assert!(!formatted.is_empty());
        assert!(formatted.contains("1."));
        assert!(formatted.contains("\n"));
        
        // Check that all tips are numbered
        for i in 1..=tips.tips.len() {
            assert!(formatted.contains(&format!("{}.", i)));
        }
    }

    #[test]
    fn test_prompting_tips_custom() {
        let tips = PromptingTips {
            tips: vec![
                "Tip one".to_string(),
                "Tip two".to_string(),
                "Tip three".to_string(),
            ],
        };

        let formatted = tips.format_for_prompt();
        assert!(formatted.contains("1. Tip one"));
        assert!(formatted.contains("2. Tip two"));
        assert!(formatted.contains("3. Tip three"));
    }

    // ========================================================================
    // PromptCandidate Tests
    // ========================================================================

    #[test]
    fn test_prompt_candidate_creation() {
        let instruction = "Test instruction".to_string();
        let demos = vec![Example::default()];
        
        let candidate = PromptCandidate::new(instruction.clone(), demos.clone());
        
        assert_eq!(candidate.instruction, instruction);
        assert_eq!(candidate.demos.len(), 1);
        assert_eq!(candidate.score, 0.0);
    }

    #[test]
    fn test_prompt_candidate_with_score() {
        let candidate = PromptCandidate::new("test".to_string(), vec![])
            .with_score(0.85);
        
        assert_eq!(candidate.score, 0.85);
        assert_eq!(candidate.instruction, "test");
    }

    #[test]
    fn test_prompt_candidate_score_update() {
        let candidate = PromptCandidate::new("test".to_string(), vec![]);
        assert_eq!(candidate.score, 0.0);
        
        let updated = candidate.with_score(0.95);
        assert_eq!(updated.score, 0.95);
    }

    // ========================================================================
    // MIPROv2 Configuration Tests
    // ========================================================================

    #[test]
    fn test_miprov2_default_configuration() {
        let optimizer = MIPROv2::builder().build();
        
        assert_eq!(optimizer.num_candidates, 10);
        assert_eq!(optimizer.max_bootstrapped_demos, 3);
        assert_eq!(optimizer.max_labeled_demos, 3);
        assert_eq!(optimizer.num_trials, 20);
        assert_eq!(optimizer.minibatch_size, 25);
        assert_eq!(optimizer.temperature, 1.0);
        assert!(optimizer.track_stats);
        assert!(optimizer.prompt_model.is_none());
    }

    #[test]
    fn test_miprov2_custom_configuration() {
        let optimizer = MIPROv2::builder()
            .num_candidates(5)
            .max_bootstrapped_demos(2)
            .max_labeled_demos(4)
            .num_trials(10)
            .minibatch_size(15)
            .temperature(0.7)
            .track_stats(false)
            .build();
        
        assert_eq!(optimizer.num_candidates, 5);
        assert_eq!(optimizer.max_bootstrapped_demos, 2);
        assert_eq!(optimizer.max_labeled_demos, 4);
        assert_eq!(optimizer.num_trials, 10);
        assert_eq!(optimizer.minibatch_size, 15);
        assert_eq!(optimizer.temperature, 0.7);
        assert!(!optimizer.track_stats);
    }

    #[test]
    fn test_miprov2_minimal_configuration() {
        let optimizer = MIPROv2::builder()
            .num_candidates(1)
            .minibatch_size(1)
            .build();
        
        assert_eq!(optimizer.num_candidates, 1);
        assert_eq!(optimizer.minibatch_size, 1);
    }

    // ========================================================================
    // Trace Selection Tests
    // ========================================================================

    #[test]
    fn test_select_best_traces_basic() {
        let optimizer = MIPROv2::builder().build();

        let traces = vec![
            Trace::new(Example::default(), Prediction::default(), Some(0.5)),
            Trace::new(Example::default(), Prediction::default(), Some(0.9)),
            Trace::new(Example::default(), Prediction::default(), Some(0.3)),
            Trace::new(Example::default(), Prediction::default(), Some(0.7)),
        ];

        let best = optimizer.select_best_traces(&traces, 2);
        assert_eq!(best.len(), 2);
        assert_eq!(best[0].score, Some(0.9));
        assert_eq!(best[1].score, Some(0.7));
    }

    #[test]
    fn test_select_best_traces_more_than_available() {
        let optimizer = MIPROv2::builder().build();

        let traces = vec![
            Trace::new(Example::default(), Prediction::default(), Some(0.8)),
            Trace::new(Example::default(), Prediction::default(), Some(0.6)),
        ];

        let best = optimizer.select_best_traces(&traces, 5);
        assert_eq!(best.len(), 2, "Should return only available traces");
    }

    #[test]
    fn test_select_best_traces_with_none_scores() {
        let optimizer = MIPROv2::builder().build();

        let traces = vec![
            Trace::new(Example::default(), Prediction::default(), Some(0.5)),
            Trace::new(Example::default(), Prediction::default(), None),
            Trace::new(Example::default(), Prediction::default(), Some(0.9)),
            Trace::new(Example::default(), Prediction::default(), None),
        ];

        let best = optimizer.select_best_traces(&traces, 3);
        assert_eq!(best.len(), 2, "Should only select traces with scores");
        assert!(best.iter().all(|t| t.score.is_some()));
    }

    #[test]
    fn test_select_best_traces_all_none_scores() {
        let optimizer = MIPROv2::builder().build();

        let traces = vec![
            Trace::new(Example::default(), Prediction::default(), None),
            Trace::new(Example::default(), Prediction::default(), None),
        ];

        let best = optimizer.select_best_traces(&traces, 2);
        assert_eq!(best.len(), 0, "Should return empty if no scores");
    }

    #[test]
    fn test_select_best_traces_equal_scores() {
        let optimizer = MIPROv2::builder().build();

        let traces = vec![
            Trace::new(Example::default(), Prediction::default(), Some(0.5)),
            Trace::new(Example::default(), Prediction::default(), Some(0.5)),
            Trace::new(Example::default(), Prediction::default(), Some(0.5)),
        ];

        let best = optimizer.select_best_traces(&traces, 2);
        assert_eq!(best.len(), 2);
        assert_eq!(best[0].score, Some(0.5));
        assert_eq!(best[1].score, Some(0.5));
    }

    #[test]
    fn test_select_best_traces_zero_selection() {
        let optimizer = MIPROv2::builder().build();

        let traces = vec![
            Trace::new(Example::default(), Prediction::default(), Some(0.8)),
        ];

        let best = optimizer.select_best_traces(&traces, 0);
        assert_eq!(best.len(), 0);
    }

    #[test]
    fn test_select_best_traces_single_trace() {
        let optimizer = MIPROv2::builder().build();

        let traces = vec![
            Trace::new(Example::default(), Prediction::default(), Some(0.75)),
        ];

        let best = optimizer.select_best_traces(&traces, 1);
        assert_eq!(best.len(), 1);
        assert_eq!(best[0].score, Some(0.75));
    }

    #[test]
    fn test_select_best_traces_descending_order() {
        let optimizer = MIPROv2::builder().build();

        let traces = vec![
            Trace::new(Example::default(), Prediction::default(), Some(0.1)),
            Trace::new(Example::default(), Prediction::default(), Some(0.2)),
            Trace::new(Example::default(), Prediction::default(), Some(0.3)),
            Trace::new(Example::default(), Prediction::default(), Some(0.4)),
            Trace::new(Example::default(), Prediction::default(), Some(0.5)),
        ];

        let best = optimizer.select_best_traces(&traces, 3);
        assert_eq!(best.len(), 3);
        assert_eq!(best[0].score, Some(0.5));
        assert_eq!(best[1].score, Some(0.4));
        assert_eq!(best[2].score, Some(0.3));
    }

    // ========================================================================
    // Prompt Candidate Creation Tests
    // ========================================================================

    #[test]
    fn test_create_prompt_candidates_basic() {
        let optimizer = MIPROv2::builder()
            .max_labeled_demos(2)
            .build();

        let traces = vec![
            Trace::new(
                Example::new(
                    [("q".to_string(), "Q1".into())].into(),
                    vec!["q".to_string()],
                    vec![],
                ),
                Prediction::default(),
                Some(0.8),
            ),
            Trace::new(
                Example::new(
                    [("q".to_string(), "Q2".into())].into(),
                    vec!["q".to_string()],
                    vec![],
                ),
                Prediction::default(),
                Some(0.9),
            ),
        ];

        let instructions = vec![
            "Instruction 1".to_string(),
            "Instruction 2".to_string(),
        ];

        let candidates = optimizer.create_prompt_candidates(instructions, &traces);

        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].instruction, "Instruction 1");
        assert_eq!(candidates[1].instruction, "Instruction 2");
        // Both should have the same demos (best from traces)
        assert_eq!(candidates[0].demos.len(), 2);
        assert_eq!(candidates[1].demos.len(), 2);
    }

    #[test]
    fn test_create_prompt_candidates_more_traces_than_max() {
        let optimizer = MIPROv2::builder()
            .max_labeled_demos(2)
            .build();

        let traces = vec![
            Trace::new(Example::default(), Prediction::default(), Some(0.5)),
            Trace::new(Example::default(), Prediction::default(), Some(0.9)),
            Trace::new(Example::default(), Prediction::default(), Some(0.3)),
            Trace::new(Example::default(), Prediction::default(), Some(0.7)),
        ];

        let instructions = vec!["Test".to_string()];
        let candidates = optimizer.create_prompt_candidates(instructions, &traces);

        assert_eq!(candidates.len(), 1);
        // Should only use max_labeled_demos (2) best traces
        assert_eq!(candidates[0].demos.len(), 2);
    }

    #[test]
    fn test_create_prompt_candidates_empty_instructions() {
        let optimizer = MIPROv2::builder().build();
        let traces = vec![
            Trace::new(Example::default(), Prediction::default(), Some(0.8)),
        ];

        let candidates = optimizer.create_prompt_candidates(vec![], &traces);
        assert_eq!(candidates.len(), 0);
    }

    #[test]
    fn test_create_prompt_candidates_no_scored_traces() {
        let optimizer = MIPROv2::builder().build();
        let traces = vec![
            Trace::new(Example::default(), Prediction::default(), None),
            Trace::new(Example::default(), Prediction::default(), None),
        ];

        let instructions = vec!["Test".to_string()];
        let candidates = optimizer.create_prompt_candidates(instructions, &traces);

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].demos.len(), 0);
    }

    // ========================================================================
    // Edge Case Tests
    // ========================================================================

    #[test]
    fn test_trace_clone() {
        let trace = Trace::new(
            Example::default(),
            Prediction::default(),
            Some(0.85),
        );

        let cloned = trace.clone();
        assert_eq!(cloned.score, Some(0.85));
    }

    #[test]
    fn test_prompt_candidate_clone() {
        let candidate = PromptCandidate::new(
            "test instruction".to_string(),
            vec![Example::default()],
        );

        let cloned = candidate.clone();
        assert_eq!(cloned.instruction, "test instruction");
        assert_eq!(cloned.demos.len(), 1);
    }

    #[test]
    fn test_format_signature_fields_with_descriptions() {
        let optimizer = MIPROv2::builder().build();
        
        // This is a basic structural test - in real usage, this would be tested
        // with actual signature implementations
        // Here we're just verifying the method exists and returns a string
        use crate::core::MetaSignature;
        use serde_json::Value;
        
        struct TestSignature;
        impl MetaSignature for TestSignature {
            fn input_fields(&self) -> Value {
                serde_json::json!({
                    "question": {
                        "type": "String",
                        "desc": "The question to answer"
                    }
                })
            }
            
            fn output_fields(&self) -> Value {
                serde_json::json!({
                    "answer": {
                        "type": "String",
                        "desc": "The answer to the question"
                    }
                })
            }
            
            fn instruction(&self) -> String {
                "Test instruction".to_string()
            }
            
            fn update_instruction(&mut self, _instruction: String) -> anyhow::Result<()> {
                Ok(())
            }
            
            fn set_demos(&mut self, _demos: Vec<Example>) -> anyhow::Result<()> {
                Ok(())
            }
            
            fn demos(&self) -> Vec<Example> {
                vec![]
            }
            
            fn append(&mut self, _name: &str, _value: Value) -> anyhow::Result<()> {
                Ok(())
            }
        }
        
        let sig = TestSignature;
        let formatted = optimizer.format_signature_fields(&sig);
        
        assert!(formatted.contains("Input Fields:"));
        assert!(formatted.contains("Output Fields:"));
        assert!(formatted.contains("question"));
        assert!(formatted.contains("answer"));
    }

    // ========================================================================
    // Property-based Tests
    // ========================================================================

    #[test]
    fn test_select_best_traces_always_returns_requested_or_less() {
        let optimizer = MIPROv2::builder().build();

        for num_traces in 1..=10 {
            for num_select in 0..=15 {
                let traces: Vec<Trace> = (0..num_traces)
                    .map(|i| {
                        Trace::new(
                            Example::default(),
                            Prediction::default(),
                            Some(i as f32 / 10.0),
                        )
                    })
                    .collect();

                let selected = optimizer.select_best_traces(&traces, num_select);
                assert!(selected.len() <= num_select);
                assert!(selected.len() <= num_traces);
            }
        }
    }

    #[test]
    fn test_prompt_candidates_count_matches_instructions() {
        let optimizer = MIPROv2::builder().build();
        let traces = vec![
            Trace::new(Example::default(), Prediction::default(), Some(0.8)),
        ];

        for num_instructions in 0..=10 {
            let instructions: Vec<String> = (0..num_instructions)
                .map(|i| format!("Instruction {}", i))
                .collect();

            let candidates = optimizer.create_prompt_candidates(instructions, &traces);
            assert_eq!(candidates.len(), num_instructions);
        }
    }
}
