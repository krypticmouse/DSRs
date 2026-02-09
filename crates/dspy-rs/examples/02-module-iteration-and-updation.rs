/*
Script to iterate and update the parameters of a module.

Run with:
```
cargo run --example 02-module-iteration-and-updation
```
*/

#![allow(deprecated)]

use bon::Builder;
use dspy_rs::{
    CallMetadata, CallOutcome, CallOutcomeErrorKind, Example, LegacyPredict, LegacySignature,
    LmError, Module, Optimizable, Prediction, Predictor, hashmap, init_tracing, prediction,
};

#[LegacySignature(cot)]
struct QASignature {
    #[input]
    pub question: String,

    #[output]
    pub answer: String,
}

#[LegacySignature]
struct RateSignature {
    /// Rate the answer on a scale of 1(very bad) to 10(very good)

    #[input]
    pub question: String,

    #[input]
    pub answer: String,

    #[output]
    pub rating: i8,
}

#[derive(Builder, Optimizable)]
pub struct QARater {
    #[parameter]
    #[builder(default = LegacyPredict::new(QASignature::new()))]
    pub answerer: LegacyPredict,

    #[parameter]
    #[builder(default = LegacyPredict::new(RateSignature::new()))]
    pub rater: LegacyPredict,
}

#[derive(Builder, Optimizable)]
pub struct NestedModule {
    #[parameter]
    #[builder(default = QARater::builder().build())]
    pub qa_outer: QARater,

    #[parameter]
    #[builder(default = QARater::builder().build())]
    pub qa_inner: QARater,

    #[parameter]
    #[builder(default = LegacyPredict::new(QASignature::new()))]
    pub extra: LegacyPredict,
}

impl Module for QARater {
    type Input = Example;
    type Output = Prediction;

    async fn forward(&self, inputs: Example) -> CallOutcome<Prediction> {
        let answerer_prediction = match self.answerer.forward(inputs.clone()).await {
            Ok(prediction) => prediction,
            Err(err) => {
                return CallOutcome::err(
                    CallOutcomeErrorKind::Lm(LmError::Provider {
                        provider: "legacy_predict".to_string(),
                        message: err.to_string(),
                        source: None,
                    }),
                    CallMetadata::default(),
                );
            }
        };

        let question = inputs.data.get("question").unwrap().clone();
        let answer = answerer_prediction.data.get("answer").unwrap().clone();

        let inputs = Example::new(
            hashmap! {
                "answer".to_string() => answer.clone(),
                "question".to_string() => question.clone()
            },
            vec!["answer".to_string(), "question".to_string()],
            vec![],
        );
        let rating_prediction = match self.rater.forward(inputs).await {
            Ok(prediction) => prediction,
            Err(err) => {
                return CallOutcome::err(
                    CallOutcomeErrorKind::Lm(LmError::Provider {
                        provider: "legacy_predict".to_string(),
                        message: err.to_string(),
                        source: None,
                    }),
                    CallMetadata::default(),
                );
            }
        };
        CallOutcome::ok(
            prediction! {
            "answer"=> answer,
            "question"=> question,
            "rating"=> rating_prediction.data.get("rating").unwrap().clone(),
        }
            .set_lm_usage(rating_prediction.lm_usage),
            CallMetadata::default(),
        )
    }
}

#[tokio::main]
async fn main() {
    init_tracing().expect("failed to initialize tracing");

    // Single module test
    let mut qa_rater = QARater::builder().build();
    for (name, param) in qa_rater.parameters() {
        param
            .update_signature_instruction("Updated instruction for ".to_string() + &name)
            .unwrap();
    }
    println!(
        "single.answerer -> {}",
        qa_rater.answerer.signature.instruction()
    );
    println!(
        "single.rater    -> {}",
        qa_rater.rater.signature.instruction()
    );

    // Nested module test
    let mut nested = NestedModule::builder().build();
    for (name, param) in nested.parameters() {
        param
            .update_signature_instruction("Deep updated: ".to_string() + &name)
            .unwrap();
    }

    // Show nested updates (module-in-module)
    println!(
        "nested.qa_outer.answerer -> {}",
        nested.qa_outer.answerer.signature.instruction()
    );
    println!(
        "nested.qa_outer.rater    -> {}",
        nested.qa_outer.rater.signature.instruction()
    );
    println!(
        "nested.qa_inner.answerer -> {}",
        nested.qa_inner.answerer.signature.instruction()
    );
    println!(
        "nested.qa_inner.rater    -> {}",
        nested.qa_inner.rater.signature.instruction()
    );
    println!(
        "nested.extra    -> {}",
        nested.extra.signature.instruction()
    );
}
