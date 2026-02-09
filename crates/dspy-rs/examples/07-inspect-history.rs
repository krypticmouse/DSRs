/*
Script to inspect the history of an LM.

Run with:
```
cargo run --example 07-inspect-history
```
*/

#![allow(deprecated)]

use bon::Builder;
use dspy_rs::{
    CallMetadata, CallOutcome, CallOutcomeErrorKind, ChatAdapter, Example, LM, LegacyPredict,
    LegacySignature, LmError, Module, Prediction, Predictor, configure, example, get_lm,
    init_tracing,
};

#[LegacySignature]
struct QASignature {
    #[input]
    pub question: String,
    #[output]
    pub answer: String,
}

#[derive(Builder)]
pub struct QARater {
    #[builder(default = LegacyPredict::new(QASignature::new()))]
    pub answerer: LegacyPredict,
}

impl Module for QARater {
    type Input = Example;
    type Output = Prediction;

    async fn forward(&self, inputs: Example) -> CallOutcome<Prediction> {
        match self.answerer.forward(inputs).await {
            Ok(prediction) => CallOutcome::ok(prediction, CallMetadata::default()),
            Err(err) => CallOutcome::err(
                CallOutcomeErrorKind::Lm(LmError::Provider {
                    provider: "legacy_predict".to_string(),
                    message: err.to_string(),
                    source: None,
                }),
                CallMetadata::default(),
            ),
        }
    }
}

#[tokio::main]
async fn main() {
    init_tracing().expect("failed to initialize tracing");

    let lm = LM::builder()
        .model("openai:gpt-4o-mini".to_string())
        .build()
        .await
        .unwrap();
    configure(lm, ChatAdapter);

    let example = example! {
        "question": "input" => "What is the capital of France?",
    };

    let qa_rater = QARater::builder().build();
    let prediction = qa_rater.forward(example.clone()).await.into_result().unwrap();
    println!("Prediction: {prediction:?}");

    let history = get_lm().inspect_history(1).await;
    println!("History: {history:?}");
}
