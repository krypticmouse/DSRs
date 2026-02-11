/*
Script to optimize a typed QA module for a HotpotQA subset with COPRO.

Run with:
```
cargo run --example 04-optimize-hotpotqa --features dataloaders
```
*/

use anyhow::Result;
use bon::Builder;
use facet;
use dspy_rs::{
    COPRO, ChatAdapter, DataLoader, Example, LM, MetricOutcome, Module, Optimizer, Predict,
    PredictError, Predicted, Signature, TypedMetric, average_score, configure,
    evaluate_trainset, init_tracing,
};
use dspy_rs::data::RawExample;

#[derive(Signature, Clone, Debug)]
struct QA {
    /// Concisely answer the question, but be accurate.

    #[input]
    question: String,

    #[output(desc = "Answer in less than 5 words.")]
    answer: String,
}

#[derive(Builder, facet::Facet)]
#[facet(crate = facet)]
struct QAModule {
    #[builder(default = Predict::<QA>::builder().instruction("Answer clearly and briefly.").build())]
    answerer: Predict<QA>,
}

impl Module for QAModule {
    type Input = QAInput;
    type Output = QAOutput;

    async fn forward(&self, input: QAInput) -> Result<Predicted<QAOutput>, PredictError> {
        self.answerer.call(input).await
    }
}

struct ExactMatchMetric;

impl TypedMetric<QA, QAModule> for ExactMatchMetric {
    async fn evaluate(
        &self,
        example: &Example<QA>,
        prediction: &Predicted<QAOutput>,
    ) -> Result<MetricOutcome> {
        let expected = example.output.answer.trim().to_lowercase();
        let actual = prediction.answer.trim().to_lowercase();
        Ok(MetricOutcome::score((expected == actual) as u8 as f32))
    }
}

fn typed_hotpot_examples(raw_examples: Vec<RawExample>) -> Vec<Example<QA>> {
    raw_examples
        .into_iter()
        .filter_map(|example| {
            let question = example
                .data
                .get("question")
                .and_then(|value| value.as_str())?
                .to_string();
            let answer = example
                .data
                .get("answer")
                .and_then(|value| value.as_str())?
                .to_string();
            Some(Example::new(
                QAInput { question },
                QAOutput { answer },
            ))
        })
        .collect()
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

    let raw_examples = DataLoader::load_hf(
        "hotpotqa/hotpot_qa",
        vec!["question".to_string()],
        vec!["answer".to_string()],
        "fullwiki",
        "validation",
        true,
    )?[..10]
        .to_vec();
    let examples = typed_hotpot_examples(raw_examples);

    let metric = ExactMatchMetric;
    let mut module = QAModule::builder().build();

    let baseline = average_score(&evaluate_trainset(&module, &examples, &metric).await?);
    println!("baseline score: {baseline:.3}");

    let optimizer = COPRO::builder().breadth(10).depth(1).build();
    optimizer
        .compile(&mut module, examples.clone(), &metric)
        .await?;

    let optimized = average_score(&evaluate_trainset(&module, &examples, &metric).await?);
    println!("optimized score: {optimized:.3}");

    Ok(())
}
