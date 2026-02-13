/*
Script to evaluate a typed QA predictor on a HotpotQA sample.

Run with:
```
cargo run --example 03-evaluate-hotpotqa --features dataloaders
```
*/

use anyhow::Result;
use dspy_rs::{
    ChatAdapter, DataLoader, Example, LM, MetricOutcome, Predict, Predicted, Signature,
    TypedLoadOptions, TypedMetric, average_score, configure, evaluate_trainset, init_tracing,
};

#[derive(Signature, Clone, Debug)]
struct QA {
    /// Concisely answer the question, but be accurate.

    #[input]
    question: String,

    #[output(desc = "Answer in less than 5 words.")]
    answer: String,
}

struct ExactMatchMetric;

impl TypedMetric<QA, Predict<QA>> for ExactMatchMetric {
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

    let examples = DataLoader::load_hf::<QA>(
        "hotpotqa/hotpot_qa",
        "fullwiki",
        "validation",
        true,
        TypedLoadOptions::default(),
    )?[..64]
        .to_vec();

    let module = Predict::<QA>::builder()
        .instruction("Answer with a short, factual response.")
        .build();
    let metric = ExactMatchMetric;

    let outcomes = evaluate_trainset(&module, &examples, &metric).await?;
    let score = average_score(&outcomes);

    println!("evaluated {} examples", outcomes.len());
    println!("average exact-match score: {score:.3}");
    Ok(())
}
