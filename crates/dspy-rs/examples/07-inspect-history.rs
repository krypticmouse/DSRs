/*
Script to inspect the history of an LM.

Run with:
```
cargo run --example 07-inspect-history
```
*/

#![allow(deprecated)]

use anyhow::Result;
use bon::Builder;
use dspy_rs::{
    ChatAdapter, Example, LM, Module, LegacyPredict, Prediction, Predictor, LegacySignature,
    configure, example, get_lm,
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
    async fn forward(&self, inputs: Example) -> Result<Prediction> {
        return self.answerer.forward(inputs.clone()).await;
    }
}

#[tokio::main]
async fn main() {
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
    let prediction = qa_rater.forward(example.clone()).await.unwrap();
    println!("Prediction: {prediction:?}");

    let history = get_lm().inspect_history(1).await;
    println!("History: {history:?}");
}
