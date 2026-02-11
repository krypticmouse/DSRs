/*
Script to optimize a module via the typed optimizer API.

Run with:
```
cargo run --example 02-module-iteration-and-updation
```
*/

use anyhow::Result;
use bon::Builder;
use facet;
use dspy_rs::{
    COPRO, ChatAdapter, Example, LM, MetricOutcome, Module, Optimizer, Predict, PredictError,
    Predicted, Signature, TypedMetric, average_score, configure, evaluate_trainset, init_tracing,
};

#[derive(Signature, Clone, Debug)]
struct QA {
    #[input]
    question: String,

    #[output]
    answer: String,
}

#[derive(Builder, facet::Facet)]
#[facet(crate = facet)]
struct QAModule {
    #[builder(default = Predict::<QA>::builder().instruction("Answer clearly.").build())]
    answerer: Predict<QA>,
}

impl Module for QAModule {
    type Input = QAInput;
    type Output = QAOutput;

    async fn forward(&self, input: QAInput) -> Result<Predicted<QAOutput>, PredictError> {
        self.answerer.call(input).await
    }
}

struct ExactMatch;

impl TypedMetric<QA, QAModule> for ExactMatch {
    async fn evaluate(&self, example: &Example<QA>, prediction: &Predicted<QAOutput>) -> Result<MetricOutcome> {
        let expected = example.output.answer.trim().to_lowercase();
        let actual = prediction.answer.trim().to_lowercase();
        Ok(MetricOutcome::score((expected == actual) as u8 as f32))
    }
}

fn trainset() -> Vec<Example<QA>> {
    vec![
        Example::new(
            QAInput {
                question: "What is 2+2?".to_string(),
            },
            QAOutput {
                answer: "4".to_string(),
            },
        ),
        Example::new(
            QAInput {
                question: "Capital of France?".to_string(),
            },
            QAOutput {
                answer: "Paris".to_string(),
            },
        ),
    ]
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

    let metric = ExactMatch;
    let mut module = QAModule::builder().build();
    let trainset = trainset();

    let baseline = average_score(&evaluate_trainset(&module, &trainset, &metric).await?);
    println!("baseline score: {baseline:.3}");

    let optimizer = COPRO::builder().breadth(4).depth(1).build();
    optimizer.compile(&mut module, trainset.clone(), &metric).await?;

    let optimized = average_score(&evaluate_trainset(&module, &trainset, &metric).await?);
    println!("optimized score: {optimized:.3}");

    Ok(())
}
