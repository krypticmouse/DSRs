/*
Script to evaluate a typed QA predictor on a HotpotQA sample.

Run with:
```
cargo run --example 03-evaluate-hotpotqa --features dataloaders
```
*/

use anyhow::Result;
use dspy_rs::{
    DataLoader, Example, LM, Eval, Predict, Predicted, Signature,
    TypedLoadOptions, TypedMetric, average_score, configure, evaluate_trainset_with_concurrency,
    init_tracing,
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
        _trace: Option<&dspy_rs::Trace>,
    ) -> Result<Eval> {
        let expected = example.output.answer.trim().to_lowercase();
        let actual = prediction.answer.trim().to_lowercase();

        Ok(Eval::score((expected == actual) as u8 as f64))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing()?;

    configure(LM::builder()
            .model("openai:gpt-4o-mini".to_string())
            .build()
            .await?);

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

    // Evaluation runs concurrently — 32 LM calls in flight, results in trainset
    // order. `evaluate_trainset` uses a default of 16; tune per provider limits.
    let outcomes = evaluate_trainset_with_concurrency(&module, &examples, &metric, 32).await?;
    let score = average_score(&outcomes);

    println!("evaluated {} examples", outcomes.len());
    println!("average exact-match score: {score:.3}");
    Ok(())
}
