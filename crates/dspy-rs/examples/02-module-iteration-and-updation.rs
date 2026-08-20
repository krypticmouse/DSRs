/*
Script to optimize a module via the typed optimizer API.

Run with:
```
cargo run --example 02-module-iteration-and-updation
```
*/

use anyhow::Result;
use bon::Builder;
use dspy_rs::prelude::*;

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

dspy_rs::predictors!(QAModule { answerer });

impl Module for QAModule {
    type Input = QAInput;
    type Output = QAOutput;

    async fn forward(&self, input: QAInput) -> Result<Predicted<QAOutput>, PredictError> {
        self.answerer.call(input).await
    }
}

struct ExactMatch;

// The metric's first parameter is the trainset row type. For a small inline
// trainset, an `(input, output)` tuple is row enough.
impl TypedMetric<(QAInput, QAOutput), QAModule> for ExactMatch {
    async fn evaluate(
        &self,
        example: &(QAInput, QAOutput),
        prediction: &Predicted<QAOutput>,
        _trace: Option<&dspy_rs::Trace>,
    ) -> Result<Eval> {
        let (_, gold) = example;
        let expected = gold.answer.trim().to_lowercase();
        let actual = prediction.answer.trim().to_lowercase();
        Ok(Eval::score((expected == actual) as u8 as f64))
    }
}

fn trainset() -> Vec<(QAInput, QAOutput)> {
    vec![
        (
            QAInput {
                question: "What is 2+2?".to_string(),
            },
            QAOutput {
                answer: "4".to_string(),
            },
        ),
        (
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

    configure(LM::builder()
            .model("openai:gpt-4o-mini".to_string())
            .build()
            .await?);

    let metric = ExactMatch;
    let mut module = QAModule::builder().build();
    let trainset = trainset();

    let baseline = average_score(&evaluate_trainset(&module, &trainset, &metric).await?);
    println!("baseline score: {baseline:.3}");

    let optimizer = COPRO::builder().breadth(4).depth(1).build();
    optimizer
        .compile_module(&mut module, &trainset, &metric)
        .await?;

    let optimized = average_score(&evaluate_trainset(&module, &trainset, &metric).await?);
    println!("optimized score: {optimized:.3}");

    Ok(())
}
