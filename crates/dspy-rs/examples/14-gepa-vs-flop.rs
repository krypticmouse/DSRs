/*
Example: Comparison between GEPA and FLOP Optimizers

This example demonstrates how to use both GEPA (reflective prompt optimizer)
and FLOP (structural DAG optimizer) on the same task to compare their approaches.

GEPA optimizes the *instructions* of existing predictors using feedback.
FLOP optimizes the *structure* of the pipeline (splitting/fusing nodes) and instructions.

Scenario: A simplified RAG pipeline that needs to answer questions based on context.
We will see how GEPA improves the prompts and how FLOP restructures the graph.
*/

use anyhow::Result;
use bon::Builder;
use dspy_rs::evaluate::FeedbackEvaluator;
use dspy_rs::optimizer::FLOP;
use dspy_rs::trace::Executor;
use dspy_rs::{
    ChatAdapter, DataLoader, Example, FeedbackMetric, LM, Module, Optimizable, Predict, Prediction,
    Predictor, Signature, configure, example,
};
use futures::stream::{self, StreamExt};
use indexmap::IndexMap;
use kdam::{BarExt, tqdm};
use rand::seq::SliceRandom;
use std::sync::Arc;

// ============================================================================
// 1. Define the Module and Signatures
// ============================================================================

#[Signature]
struct RetrieveSignature {
    #[input]
    query: String,
    #[output]
    context: String,
}

#[Signature]
struct AnswerSignature {
    #[input]
    context: String,
    #[input]
    question: String,
    #[output]
    answer: String,
}

#[derive(Builder)]
struct RAGModule {
    #[builder(default = Predict::new(RetrieveSignature::new()))]
    pub retriever: Predict,
    #[builder(default = Predict::new(AnswerSignature::new()))]
    pub answerer: Predict,
}

impl Module for RAGModule {
    async fn forward(&self, inputs: Example) -> Result<Prediction> {
        let query = inputs
            .get("question", Some(""))
            .as_str()
            .unwrap()
            .to_string();

        // Step 1: Retrieve context (mocked for this example by the LLM)
        let mut retrieve_input = example! { "query": "input" => query.clone() };
        // Propagate node_id from inputs for proper tracing
        retrieve_input.node_id = inputs.node_id;

        let context_pred = self.retriever.forward(retrieve_input).await?;
        let context = context_pred
            .get("context", Some(""))
            .as_str()
            .unwrap()
            .to_string();

        // Step 2: Answer question
        let mut answer_input = example! {
            "context": "input" => context,
            "question": "input" => query
        };
        // IMPORTANT: Propagate node_id from context_pred to establish data flow
        // This tells the tracer that this input depends on the context retrieval output
        answer_input.node_id = context_pred.node_id;

        let answer_pred = self.answerer.forward(answer_input).await?;

        Ok(answer_pred)
    }
}

impl Optimizable for RAGModule {
    fn get_signature(&self) -> &dyn dspy_rs::core::MetaSignature {
        self.answerer.get_signature()
    }

    fn parameters(&mut self) -> IndexMap<String, &mut dyn Optimizable> {
        let mut params = IndexMap::new();
        params.insert(
            "retriever".to_string(),
            &mut self.retriever as &mut dyn Optimizable,
        );
        params.insert(
            "answerer".to_string(),
            &mut self.answerer as &mut dyn Optimizable,
        );
        params
    }
}

impl dspy_rs::evaluate::Evaluator for RAGModule {
    const MAX_CONCURRENCY: usize = 4;
    const DISPLAY_PROGRESS: bool = true;

    async fn metric(&self, example: &Example, prediction: &Prediction) -> f32 {
        let expected = example
            .data
            .get("answer")
            .map(|v| v.to_string().to_lowercase())
            .unwrap_or_default();
        let actual = prediction
            .get("answer", Some(""))
            .as_str()
            .unwrap()
            .to_lowercase();
        if actual.contains(&expected) || expected.contains(&actual) {
            1.0
        } else {
            0.0
        }
    }
}

// Implement FeedbackEvaluator for GEPA
impl FeedbackEvaluator for RAGModule {
    async fn feedback_metric(&self, _input: &Example, prediction: &Prediction) -> FeedbackMetric {
        // In a real scenario, this would compare against ground truth or use an LLM judge.
        // Here we simulate a simple metric: check if answer is not empty.
        let answer = prediction
            .get("answer", Some(""))
            .as_str()
            .unwrap()
            .to_string();
        let score = if answer.len() > 10 { 1.0 } else { 0.0 };

        FeedbackMetric {
            score,
            feedback: if score > 0.5 {
                "Good answer length.".to_string()
            } else {
                "Answer too short.".to_string()
            },
            metadata: std::collections::HashMap::new(),
        }
    }
}

// ============================================================================
// 2. Main Execution
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    // Setup LM
    let lm = LM::builder()
        .model("openai:gpt-4o-mini".to_string())
        .build()
        .await?;
    configure(lm.clone(), ChatAdapter);

    // Create Trainset
    let trainset = DataLoader::load_hf(
        "hotpotqa/hotpot_qa",
        vec!["question".to_string()],
        vec!["answer".to_string()],
        "fullwiki",
        "train",
        true,
    )?[..10]
        .to_vec();

    let mut testset = DataLoader::load_hf(
        "hotpotqa/hotpot_qa",
        vec!["question".to_string()],
        vec!["answer".to_string()],
        "fullwiki",
        "validation",
        true,
    )?;
    testset.shuffle(&mut rand::thread_rng());
    let testset: Vec<_> = testset.into_iter().take(1000).collect();

    // ------------------------------------------------------------------------
    // Run FLOP (Functional LLM Optimizer for Pipelines)
    // ------------------------------------------------------------------------
    println!("\n--- Running FLOP ---");
    println!("Goal: Optimize DAG structure and instructions based on task description.");

    let flop_module = RAGModule::builder().build();

    let flop = FLOP::builder()
        .lm(lm.clone())
        .task_description("Answer questions by first retrieving relevant context and then synthesizing an answer based on that context.".to_string())
        .max_steps(5)  // More steps to allow structural changes
        .build();

    // FLOP compiles by generating a new optimized graph
    let optimized_graph = flop
        .compile_and_get_graph(&flop_module, trainset.clone())
        .await?;

    // Evaluate the optimized graph on testset using Executor
    println!("\nEvaluating FLOP-optimized graph on testset...");
    let executor = Arc::new(Executor::new(optimized_graph));

    let total = testset.len();
    let mut pb = tqdm!(total = total, desc = "Evaluating FLOP");

    const MAX_CONCURRENCY: usize = 4;

    // Run evaluations concurrently with progress tracking
    let results: Vec<(Example, Option<Prediction>)> = stream::iter(testset.into_iter())
        .map(|example| {
            let executor = Arc::clone(&executor);
            async move {
                let predictions = executor.execute(example.clone()).await.ok();
                let last_pred = predictions.and_then(|p| p.into_iter().last());
                (example, last_pred)
            }
        })
        .buffer_unordered(MAX_CONCURRENCY)
        .inspect(|_| {
            let _ = pb.update(1);
        })
        .collect()
        .await;

    // Calculate score
    let correct = results
        .iter()
        .filter(|(example, prediction)| {
            if let Some(pred) = prediction {
                let expected = example
                    .data
                    .get("answer")
                    .map(|v| v.to_string().to_lowercase())
                    .unwrap_or_default();
                let actual = pred
                    .get("answer", Some(""))
                    .as_str()
                    .unwrap()
                    .to_lowercase();
                actual.contains(&expected) || expected.contains(&actual)
            } else {
                false
            }
        })
        .count();

    let flop_test_score = correct as f32 / total as f32;
    println!("\n  FLOP Test Score: {:.2}%", flop_test_score * 100.0);

    // ------------------------------------------------------------------------
    // Summary
    // ------------------------------------------------------------------------
    println!("\n=== Summary ===");
    println!("  FLOP Test Score: {:.2}%", flop_test_score * 100.0);

    Ok(())
}
