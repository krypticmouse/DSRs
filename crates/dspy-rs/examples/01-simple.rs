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
use dspy_rs::data::RawExample;
use dspy_rs::{
    CallMetadata, Chat, ChatAdapter, Example, LM, LmError, Module, Predict, PredictError,
    Predicted, Prediction, configure, init_tracing,
};

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
    type Input = RawExample;
    type Output = Prediction;

    async fn forward(&self, inputs: RawExample) -> Result<Predicted<Prediction>, PredictError> {
        // Step 1: Convert module input into typed predictor input.
        let question = match inputs.data.get("question").and_then(|value| value.as_str()) {
            Some(question) => question.to_string(),
            None => {
                return Err(PredictError::Lm {
                    source: LmError::Provider {
                        provider: "QARater".to_string(),
                        message: "missing required string field `question`".to_string(),
                        source: None,
                    },
                });
            }
        };

        let answer_predicted = self
            .answerer
            .call(QAInput {
                question: question.clone(),
            })
            .await?;
        let answer_usage = answer_predicted.metadata().lm_usage.clone();
        let answerer_prediction = answer_predicted.into_inner();

        // Step 2: Rate the generated answer.
        let rate_predicted = self
            .rater
            .call(RateInput {
                question: question.clone(),
                answer: answerer_prediction.answer.clone(),
            })
            .await?;
        let rate_usage = rate_predicted.metadata().lm_usage.clone();
        let rate_result = rate_predicted.into_inner();

        // Step 3: Compose the final untyped prediction for module consumers.
        let mut combined = Prediction {
            lm_usage: answer_usage + rate_usage,
            ..Prediction::default()
        };
        combined
            .data
            .insert("question".into(), question.clone().into());
        combined
            .data
            .insert("reasoning".into(), answerer_prediction.reasoning.into());
        combined
            .data
            .insert("answer".into(), answerer_prediction.answer.into());
        combined
            .data
            .insert("rating".into(), rate_result.rating.into());

        Ok(Predicted::new(
            combined,
            CallMetadata::default(),
            Chat::new(vec![]),
        ))
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

    // forward() returns Predicted<Output>; access the typed output directly.
    let output = predict.call(input.clone()).await?.into_inner();
    println!("Question: {}", input.question);
    println!("Reasoning: {}", output.reasoning);
    println!("Answer: {}", output.answer);

    // Predicted carries typed output, metadata, and chat history.
    let result = predict.call(input).await?;
    println!("\nWith metadata:");
    println!(
        "  Raw 'answer' field: {:?}",
        result.metadata().field_raw("answer")
    );
    println!("  Token usage: {:?}", result.metadata().lm_usage);

    // =========================================================================
    // Example 2: Module composition (for complex pipelines)
    // =========================================================================
    println!("\n=== Example 2: Module Composition ===\n");

    let qa_rater = QARater::builder().build();

    // Create an untyped row for Module::forward()
    let mut example = RawExample::default();
    example
        .data
        .insert("question".into(), "Why is the sky blue?".into());

    let prediction = qa_rater.call(example).await?.into_inner();
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
        .demo(Example::new(
            QAInput {
                question: "What is 2+2?".to_string(),
            },
            QAOutput {
                reasoning: "2+2 is a basic arithmetic operation. Adding 2 to 2 gives 4."
                    .to_string(),
                answer: "4".to_string(),
            },
        ))
        .demo(Example::new(
            QAInput {
                question: "What color is grass?".to_string(),
            },
            QAOutput {
                reasoning: "Grass contains chlorophyll which reflects green light.".to_string(),
                answer: "Green".to_string(),
            },
        ))
        .build();

    let demo_question = "What is the largest planet in our solar system?".to_string();
    let output = predict_with_demos
        .call(QAInput {
            question: demo_question.clone(),
        })
        .await?
        .into_inner();

    println!("With few-shot demos:");
    println!("  Question: {}", demo_question);
    println!("  Reasoning: {}", output.reasoning);
    println!("  Answer: {}", output.answer);

    Ok(())
}
