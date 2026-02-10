/*
Script to run a simple pipeline.

Run with:
```
cargo run --example 01-simple
```
*/

#![allow(deprecated)]

use anyhow::Result;
use bon::Builder;
use dspy_rs::{
    CallMetadata, ChatAdapter, Example, LM, LegacyPredict, LegacySignature, LmError, Module,
    PredictError, Predicted, Prediction, Predictor, configure, example, forward_all, hashmap,
    init_tracing, prediction,
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

#[derive(Builder)]
pub struct QARater {
    #[builder(default = LegacyPredict::new(QASignature::new()))]
    pub answerer: LegacyPredict,
    #[builder(default = LegacyPredict::new(RateSignature::new()))]
    pub rater: LegacyPredict,
}

impl Module for QARater {
    type Input = Example;
    type Output = Prediction;

    async fn forward(&self, inputs: Example) -> Result<Predicted<Prediction>, PredictError> {
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

        let question = inputs.data.get("question").unwrap().clone();
        let answer = answerer_prediction.data.get("answer").unwrap().clone();
        let answer_lm_usage = answerer_prediction.lm_usage;

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
                return Err(PredictError::Lm {
                    source: LmError::Provider {
                        provider: "legacy_predict".to_string(),
                        message: err.to_string(),
                        source: None,
                    },
                });
            }
        };
        let rating_lm_usage = rating_prediction.lm_usage;

        Ok(Predicted::new(
            prediction! {
                "answer"=> answer,
                "question"=> question,
                "rating"=> rating_prediction.data.get("rating").unwrap().clone(),
            }
            .set_lm_usage(answer_lm_usage + rating_lm_usage),
            CallMetadata::default(),
        ))
    }
}

#[tokio::main]
async fn main() {
    init_tracing().expect("failed to initialize tracing");

    // Anthropic
    configure(
        LM::builder()
            .model("anthropic:claude-sonnet-4-5-20250929".to_string())
            .build()
            .await
            .unwrap(),
        ChatAdapter,
    );

    let example = vec![
        example! {
            "question": "input" => "What is the capital of France?",
        },
        example! {
            "question": "input" => "What is the capital of Germany?",
        },
        example! {
            "question": "input" => "What is the capital of Italy?",
        },
    ];

    let qa_rater = QARater::builder().build();
    let prediction = forward_all(&qa_rater, example.clone(), 2)
        .await
        .into_iter()
        .map(|outcome| outcome.map(|predicted| predicted.into_inner()))
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    println!("Anthropic: {prediction:?}");

    // Gemini
    configure(
        LM::builder()
            .model("gemini:gemini-2.0-flash".to_string())
            .build()
            .await
            .unwrap(),
        ChatAdapter,
    );

    let prediction = forward_all(&qa_rater, example, 2)
        .await
        .into_iter()
        .map(|outcome| outcome.map(|predicted| predicted.into_inner()))
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    println!("Gemini: {prediction:?}");
}
