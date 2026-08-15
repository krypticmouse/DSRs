/*
Front Desk, chapter 11 ("What Does Better Even Mean?"): labeled tickets become
a dataset, judgment becomes a metric, and the desk gets its first honest score.

Two objects, neither exotic:
- a dataset: `Vec<Example<Triage>>`, inline here, or loaded from JSONL with
  `DataLoader::load_json` (same three arguments either way: path, lines, opts)
- a metric: a `TypedMetric` impl returning an `Eval` (a score, and optionally
  the feedback string that chapter 13's GEPA will feed on)

Run with:
```
cargo run --example 25-frontdesk-evaluate
```
(needs OPENAI_API_KEY)
*/

use anyhow::Result;
use dspy_rs::{
    DataLoader, Eval, Example, LM, Predict, Predicted, Signature, Trace, TypedLoadOptions,
    TypedMetric, average_score, configure, evaluate_trainset, init_tracing,
};

#[dspy_rs::Schema]
#[derive(Clone, Debug, PartialEq)]
enum Category {
    Bug,
    Billing,
    HowTo,
    FeatureRequest,
}

/// Classify a support ticket for the Sourdough app.
#[derive(Signature, Clone, Debug)]
struct Triage {
    /// The full text of the customer's ticket
    #[input]
    ticket: String,

    #[output]
    category: Category,
}

/// The predicted category either matches the label or it does not.
struct TriageAccuracy;

impl TypedMetric<Triage, Predict<Triage>> for TriageAccuracy {
    async fn evaluate(
        &self,
        example: &Example<Triage>,
        prediction: &Predicted<TriageOutput>,
        _trace: Option<&Trace>,
    ) -> Result<Eval> {
        let hit = example.output.category == prediction.category;
        Ok(Eval::score(if hit { 1.0 } else { 0.0 }))
    }
}

/// The labeled cases, inline. A demo and a test case were always the same
/// thing: an input plus the output you'd accept.
fn inline_trainset() -> Vec<Example<Triage>> {
    [
        (
            "Charged twice after the update crashed mid-payment",
            Category::Billing,
        ),
        ("How do I export my feeding log as CSV?", Category::HowTo),
        (
            "PLEASE add a dark mode, my starter and I feed at 2am",
            Category::FeatureRequest,
        ),
        (
            "App crashes every time I open the photo tab",
            Category::Bug,
        ),
        (
            "Can I move my subscription to a new email address?",
            Category::Billing,
        ),
        (
            "Where do I see my starter's rise history?",
            Category::HowTo,
        ),
    ]
    .into_iter()
    .map(|(ticket, category)| {
        Example::new(
            TriageInput {
                ticket: ticket.to_string(),
            },
            TriageOutput { category },
        )
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
    );

    // When the labels live in a JSONL file ({"ticket": ..., "category": ...},
    // one object per line), loading is one call, typed against the signature
    // it will test. `true` means JSONL.
    let jsonl_path = "examples/data/tickets.jsonl";
    let trainset: Vec<Example<Triage>> = if std::path::Path::new(jsonl_path).exists() {
        DataLoader::load_json::<Triage>(jsonl_path, true, TypedLoadOptions::default())?
    } else {
        inline_trainset()
    };
    println!("{} labeled tickets", trainset.len());

    let triage = Predict::<Triage>::builder().named("triage").build();

    // Fan the trainset through the predictor (16 concurrent by default),
    // meet each answer with the metric, bring the scores home.
    let evals = evaluate_trainset(&triage, &trainset, &TriageAccuracy).await?;
    println!("triage accuracy: {:.0}%", average_score(&evals) * 100.0);

    Ok(())
}
