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

/// A trainset row is a plain struct shaped like the *dataset*, not the
/// signature. `#[derive(Example)]` wires it to `QA` by field name:
/// `#[input]` fields form `QAInput`, `#[output]` fields form `QAOutput`, and
/// unmarked fields are metric-only gold data the module never sees.
#[derive(Example, facet::Facet, serde::Deserialize, serde::Serialize, Clone, Debug)]
#[facet(crate = facet)]
#[example(QA)]
struct HotpotRow {
    #[input]
    question: String,

    #[output]
    answer: String,

    /// HotpotQA's difficulty label — metric-only: it rides along in the row,
    /// invisible to the LM, and the metric reads it directly.
    level: String,
}

struct ExactMatchMetric;

impl TypedMetric<HotpotRow, Predict<QA>> for ExactMatchMetric {
    async fn evaluate(
        &self,
        example: &HotpotRow,
        prediction: &Predicted<QAOutput>,
        _trace: Option<&dspy_rs::Trace>,
    ) -> Result<Eval> {
        let expected = example.answer.trim().to_lowercase();
        let actual = prediction.answer.trim().to_lowercase();

        // The row carries gold fields the module never saw: tag each score
        // with the question's difficulty so misses can be sliced by level.
        Ok(Eval::with_feedback(
            (expected == actual) as u8 as f64,
            format!("level={}", example.level),
        ))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing()?;

    configure(LM::builder()
            .model("openai:gpt-4o-mini".to_string())
            .build()
            .await?);

    // Loaders are generic over the row struct: extra dataset columns land in
    // the row's metric-only fields instead of being thrown away.
    let examples = DataLoader::load_hf::<HotpotRow>(
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
