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
    CallMetadata, ChatAdapter, Example, LM, LegacyPredict, LegacySignature, LmError, Module,
    PredictError, Predicted, Prediction, Predictor, configure, example, get_lm, init_tracing,
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

    async fn forward(&self, inputs: Example) -> Result<Predicted<Prediction>, PredictError> {
        match self.answerer.forward(inputs).await {
            Ok(prediction) => Ok(Predicted::new(prediction, CallMetadata::default())),
            Err(err) => Err(PredictError::Lm {
                source: LmError::Provider {
                    provider: "legacy_predict".to_string(),
                    message: err.to_string(),
                    source: None,
                },
            }),
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
    let prediction = qa_rater.call(example.clone()).await.unwrap().into_inner();
    println!("Prediction: {prediction:?}");

    let history = get_lm().inspect_history(1).await;
    println!("History: {history:?}");
}
