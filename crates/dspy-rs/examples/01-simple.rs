/*
Script to run a simple pipeline demonstrating the typed API.

This example shows:
1. Typed signatures with `#[derive(Signature)]`
2. Chain-of-thought via explicit `reasoning` output field
3. Module composition with typed predictors
4. Both typed direct calls AND Module trait compatibility

Run with:
```
cargo run --example 01-simple
```
*/

use anyhow::Result;
use bon::Builder;
use dspy_rs::{ChatAdapter, Example, LM, Module, Predict, Prediction, configure, init_tracing};

const QA_INSTRUCTION: &str = "Answer the question step by step.";
const RATE_INSTRUCTION: &str = "Rate the answer on a scale of 1 (very bad) to 10 (very good).";

#[derive(dspy_rs::Signature, Clone, Debug)]
pub struct QA {
    #[input]
    pub question: String,

    #[output(desc = "Think step by step before answering")]
    pub reasoning: String,

    #[output]
    pub answer: String,
}

#[derive(dspy_rs::Signature, Clone, Debug)]
pub struct Rate {
    #[input]
    pub question: String,

    #[input]
    pub answer: String,

    #[output]
    pub rating: i8,
}

/// A composed module that answers a question and then rates the answer.
/// Demonstrates how typed predictors work with the Module trait for composition.
#[derive(Builder)]
pub struct QARater {
    #[builder(default = Predict::<QA>::builder().instruction(QA_INSTRUCTION).build())]
    pub answerer: Predict<QA>,
    #[builder(default = Predict::<Rate>::builder().instruction(RATE_INSTRUCTION).build())]
    pub rater: Predict<Rate>,
}

impl Module for QARater {
    async fn forward(&self, inputs: Example) -> Result<Prediction> {
        // Step 1: Get the answer using the typed predictor
        // Module::forward converts Example -> typed input automatically
        let answerer_prediction = self.answerer.forward(inputs.clone()).await?;

        // Extract values from the prediction
        let question = inputs.data.get("question").unwrap().clone();
        let answer = answerer_prediction.data.get("answer").unwrap().clone();
        let reasoning = answerer_prediction.data.get("reasoning").unwrap().clone();

        // Step 2: Create input for the rater
        // We can use the typed input struct directly with call() for cleaner code
        let rate_input = RateInput {
            question: question.to_string(),
            answer: answer.to_string(),
        };

        // Use call() for typed access to the result
        let rate_result = self.rater.call(rate_input).await?;

        // Step 3: Compose the final prediction with all fields
        let mut combined = Prediction {
            lm_usage: answerer_prediction.lm_usage.clone(),
            ..Prediction::default()
        };
        combined.data.insert("question".into(), question);
        combined.data.insert("reasoning".into(), reasoning);
        combined.data.insert("answer".into(), answer);
        combined
            .data
            .insert("rating".into(), rate_result.rating.into());

        Ok(combined)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing()?;

    configure(
        LM::builder()
            .model("openai:gpt-4o-mini".to_string())
            .build()
            .await?,
        ChatAdapter,
    );

    // =========================================================================
    // Example 1: Direct typed API usage (recommended for simple cases)
    // =========================================================================
    println!("=== Example 1: Direct Typed API ===\n");

    let predict = Predict::<QA>::builder().instruction(QA_INSTRUCTION).build();
    let input = QAInput {
        question: "What is the capital of France?".to_string(),
    };

    // call() returns the typed output struct
    let output: QA = predict.call(input.clone()).await?;
    println!("Question: {}", output.question);
    println!("Reasoning: {}", output.reasoning);
    println!("Answer: {}", output.answer);

    // call_with_meta() returns CallResult with metadata
    let result = predict.call_with_meta(input).await?;
    println!("\nWith metadata:");
    println!("  Raw 'answer' field: {:?}", result.field_raw("answer"));
    println!("  Token usage: {:?}", result.lm_usage);

    // =========================================================================
    // Example 2: Module composition (for complex pipelines)
    // =========================================================================
    println!("\n=== Example 2: Module Composition ===\n");

    let qa_rater = QARater::builder().build();

    // Create an Example for Module::forward()
    let mut example = Example::default();
    example
        .data
        .insert("question".into(), "Why is the sky blue?".into());

    let prediction = qa_rater.forward(example).await?;
    println!("Composed pipeline result:");
    println!("  Question: {}", prediction.data.get("question").unwrap());
    println!("  Reasoning: {}", prediction.data.get("reasoning").unwrap());
    println!("  Answer: {}", prediction.data.get("answer").unwrap());
    println!("  Rating: {}", prediction.data.get("rating").unwrap());

    // =========================================================================
    // Example 3: Using demos (few-shot examples)
    // =========================================================================
    println!("\n=== Example 3: With Demos ===\n");

    let predict_with_demos = Predict::<QA>::builder()
        .instruction(QA_INSTRUCTION)
        .demo(QA {
            question: "What is 2+2?".to_string(),
            reasoning: "2+2 is a basic arithmetic operation. Adding 2 to 2 gives 4.".to_string(),
            answer: "4".to_string(),
        })
        .demo(QA {
            question: "What color is grass?".to_string(),
            reasoning: "Grass contains chlorophyll which reflects green light.".to_string(),
            answer: "Green".to_string(),
        })
        .build();

    let output = predict_with_demos
        .call(QAInput {
            question: "What is the largest planet in our solar system?".to_string(),
        })
        .await?;

    println!("With few-shot demos:");
    println!("  Question: {}", output.question);
    println!("  Reasoning: {}", output.reasoning);
    println!("  Answer: {}", output.answer);

    Ok(())
}
