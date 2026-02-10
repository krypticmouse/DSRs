/*
Script to evaluate the answerer of the QARater module for a tiny sample of the HotpotQA dataset.

Run with:
```
cargo run --example 03-evaluate-hotpotqa --features dataloaders
```

Note: The `dataloaders` feature is required for loading datasets.
*/

use bon::Builder;
use dspy_rs::{
    CallMetadata, ChatAdapter, Evaluator, Example, LM, LegacyPredict, LegacySignature, LmError,
    Module, Optimizable, PredictError, Predicted, Prediction, Predictor, configure, init_tracing,
};

use dspy_rs::DataLoader;

#[LegacySignature(cot)]
struct QASignature {
    /// Concisely answer the question but be accurate. If it's a yes no question, answer with yes or no.

    #[input]
    pub question: String,

    #[output(desc = "Answer in less than 5 words.")]
    pub answer: String,
}

#[derive(Builder, Optimizable)]
pub struct QARater {
    #[parameter]
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

impl Evaluator for QARater {
    const MAX_CONCURRENCY: usize = 16;
    const DISPLAY_PROGRESS: bool = true;

    async fn metric(&self, example: &Example, prediction: &Prediction) -> f32 {
        let answer = example.data.get("answer").unwrap().clone();
        let prediction = prediction.data.get("answer").unwrap().clone();

        if answer.to_string().to_lowercase() == prediction.to_string().to_lowercase() {
            1.0
        } else {
            0.0
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_tracing()?;

    configure(
        LM::builder()
            .model("openai:gpt-4o-mini".to_string())
            .build()
            .await?,
        ChatAdapter {},
    );

    let examples = DataLoader::load_hf(
        "hotpotqa/hotpot_qa",
        vec!["question".to_string()],
        vec!["answer".to_string()],
        "fullwiki",
        "validation",
        true,
    )?[..128]
        .to_vec();

    let evaluator = QARater::builder().build();
    let metric = evaluator.evaluate(examples).await;

    println!("Metric: {metric}");
    Ok(())
}
