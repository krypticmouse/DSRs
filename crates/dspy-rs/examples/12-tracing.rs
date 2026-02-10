#![allow(deprecated)]

use anyhow::Result;
use bon::Builder;
use dspy_rs::{
    CallMetadata, ChatAdapter, LM, LegacyPredict, LegacySignature, LmError, Module, PredictError,
    Predicted, Prediction, Predictor, configure, example, init_tracing, prediction,
    trace::{self, IntoTracked},
};

#[LegacySignature]
struct QASignature {
    #[input]
    pub question: String,
    #[output]
    pub answer: String,
}

#[LegacySignature]
struct RateSignature {
    #[input]
    pub question: String,
    #[input]
    pub answer: String,
    #[output]
    pub rating: i8,
}

#[derive(Builder)]
pub struct QARater {
    #[builder(default = LegacyPredict::new(QASignature::new()))]
    pub answerer: LegacyPredict,
    #[builder(default = LegacyPredict::new(RateSignature::new()))]
    pub rater: LegacyPredict,
}

impl Module for QARater {
    type Input = dspy_rs::Example;
    type Output = Prediction;

    async fn forward(
        &self,
        inputs: dspy_rs::Example,
    ) -> Result<Predicted<Prediction>, PredictError> {
        let answerer_prediction = match self.answerer.forward(inputs.clone()).await {
            Ok(prediction) => prediction,
            Err(err) => {
                return Err(PredictError::Lm {
                    source: LmError::Provider {
                        provider: "legacy_predict".to_string(),
                        message: err.to_string(),
                        source: None,
                    },
                });
            }
        };

        // We use .get_tracked() to preserve lineage info
        let question = inputs.data.get("question").unwrap().clone().into_tracked(); // Input passed through
        let answer = answerer_prediction.get_tracked("answer");

        // The example! macro will now detect the tracked values and record a Map node.
        // We don't need .linked_to() anymore if we use tracked values.
        let inputs = example! {
            "question": "input" => question.clone(),
            "answer": "input" => answer.clone()
        };

        let rating_prediction = match self.rater.forward(inputs).await {
            Ok(prediction) => prediction,
            Err(err) => {
                return Err(PredictError::Lm {
                    source: LmError::Provider {
                        provider: "legacy_predict".to_string(),
                        message: err.to_string(),
                        source: None,
                    },
                });
            }
        };

        // Final output
        Ok(Predicted::new(
            prediction! {
                "answer"=> answer.value,
                "question"=> question.value,
                "rating"=> rating_prediction.data.get("rating").unwrap().clone(),
            }
            .set_lm_usage(rating_prediction.lm_usage),
            CallMetadata::default(),
        ))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing()?;

    // Configure with a dummy model string
    configure(
        LM::builder()
            .model("openai:gpt-4o-mini".to_string())
            .build()
            .await
            .unwrap(),
        ChatAdapter,
    );

    let module = QARater::builder().build();
    let example = example! {
        "question": "input" => "Hello",
    };

    println!("Starting trace...");
    let (result, graph) = trace::trace(|| async { module.call(example).await }).await;

    match result {
        Ok(predicted) => println!("Prediction keys: {:?}", predicted.into_inner().data.keys()),
        Err(e) => println!("Error (expected if no API key/network): {}", e),
    }

    println!("Graph Nodes: {}", graph.nodes.len());
    for node in &graph.nodes {
        println!(
            "Node {}: Type={:?}, Inputs={:?}",
            node.id, node.node_type, node.inputs
        );
    }

    // Check if the graph is connected:
    // Expected:
    // Node 0: Root (Initial input)
    // Node 1: LegacyPredict (Answerer) -> Inputs: [0]
    // Node 2: Map (Data Transform) -> Inputs: [0, 1]
    // Node 3: LegacyPredict (Rater)    -> Inputs: [2]

    // Execute the graph with new input
    println!("\nExecuting Graph with new input...");
    let executor = dspy_rs::trace::Executor::new(graph);
    let new_input = example! {
        "question": "input" => "What is the capital of Germany?",
    };

    match executor.execute(new_input).await {
        Ok(preds) => {
            if let Some(final_pred) = preds.first() {
                println!("Final Prediction from Graph: {:?}", final_pred);
            }
        }
        Err(e) => println!("Graph Execution Error: {}", e),
    }

    Ok(())
}
