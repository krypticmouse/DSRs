/*
Example showing typed tracing for a composed module.

Run with:
```
cargo run --example 12-tracing
```
*/

use anyhow::Result;
use bon::Builder;
use dspy_rs::{
    CallMetadata, ChatAdapter, Example, LM, LmUsage, Module, Predict, PredictError, Predicted,
    Prediction, Signature, configure, init_tracing,
    trace::{self, Executor},
};
use serde_json::json;
use std::collections::HashMap;

#[derive(Signature, Clone, Debug)]
struct QASignature {
    #[input]
    question: String,

    #[output]
    answer: String,
}

#[derive(Signature, Clone, Debug)]
struct RateSignature {
    #[input]
    question: String,

    #[input]
    answer: String,

    #[output]
    rating: i8,
}

#[derive(Builder)]
struct QARater {
    #[builder(default = Predict::<QASignature>::new())]
    answerer: Predict<QASignature>,

    #[builder(default = Predict::<RateSignature>::new())]
    rater: Predict<RateSignature>,
}

impl Module for QARater {
    type Input = QASignatureInput;
    type Output = Prediction;

    async fn forward(&self, input: QASignatureInput) -> Result<Predicted<Prediction>, PredictError> {
        let answer_predicted = self.answerer.call(input.clone()).await?;
        let answer_usage = answer_predicted.metadata().lm_usage.clone();
        let answer_output = answer_predicted.into_inner();

        let rating_predicted = self
            .rater
            .call(RateSignatureInput {
                question: input.question.clone(),
                answer: answer_output.answer.clone(),
            })
            .await?;
        let rating_usage = rating_predicted.metadata().lm_usage.clone();
        let rating_output = rating_predicted.into_inner();

        let prediction = Prediction::new(
            HashMap::from([
                ("question".to_string(), json!(input.question)),
                ("answer".to_string(), json!(answer_output.answer)),
                ("rating".to_string(), json!(rating_output.rating)),
            ]),
            LmUsage {
                prompt_tokens: answer_usage.prompt_tokens + rating_usage.prompt_tokens,
                completion_tokens: answer_usage.completion_tokens + rating_usage.completion_tokens,
                total_tokens: answer_usage.total_tokens + rating_usage.total_tokens,
            },
        );

        Ok(Predicted::new(prediction, CallMetadata::default()))
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

    let module = QARater::builder().build();

    println!("Starting trace...");
    let (result, graph) = trace::trace(|| async {
        module
            .call(QASignatureInput {
                question: "Hello".to_string(),
            })
            .await
    })
    .await;

    match result {
        Ok(predicted) => println!("Prediction keys: {:?}", predicted.into_inner().keys()),
        Err(err) => println!("Error (expected without credentials/network): {err}"),
    }

    println!("Graph nodes: {}", graph.nodes.len());
    for node in &graph.nodes {
        println!("Node {}: type={:?}, inputs={:?}", node.id, node.node_type, node.inputs);
    }

    println!("\nExecuting graph replay...");
    let executor = Executor::new(graph);
    let replay_input = Example::new(
        HashMap::from([(
            "question".to_string(),
            json!("What is the capital of Germany?"),
        )]),
        vec!["question".to_string()],
        vec![],
    );

    match executor.execute(replay_input).await {
        Ok(predictions) => println!("Replay outputs: {}", predictions.len()),
        Err(err) => println!("Replay failed (expected for Predict nodes): {err}"),
    }

    Ok(())
}
