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
    CallMetadata, LM, LmUsage, Module, Predict, PredictError, Predicted, Prediction,
    Signature, configure, init_tracing, trace,
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

    async fn forward(
        &self,
        input: QASignatureInput,
    ) -> Result<Predicted<Prediction>, PredictError> {
        let answer_predicted = self.answerer.call(input.clone()).await?;
        let answer_usage = answer_predicted.metadata().lm_usage;
        let answer_output = answer_predicted.into_inner();

        let rating_predicted = self
            .rater
            .call(RateSignatureInput {
                question: input.question.clone(),
                answer: answer_output.answer.clone(),
            })
            .await?;
        let rating_usage = rating_predicted.metadata().lm_usage;
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

    configure(LM::builder()
            .model("openai:gpt-4o-mini".to_string())
            .build()
            .await?);

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

    // Each Predict call records its typed inputs, its parsed output, an edge to
    // the previously recorded node, and an `instance_key` that optimizers can
    // join back to named predictor paths (see `predictor_instance_keys`).
    println!("Graph nodes: {}", graph.nodes.len());
    for node in &graph.nodes {
        println!(
            "Node {}: type={:?}, inputs={:?}",
            node.id, node.node_type, node.inputs
        );
        if let Some(input_data) = &node.input_data {
            println!("  recorded input: {:?}", input_data.data);
        }
        if let Some(output) = &node.output {
            println!("  recorded output: {:?}", output.data);
        }
    }

    Ok(())
}
