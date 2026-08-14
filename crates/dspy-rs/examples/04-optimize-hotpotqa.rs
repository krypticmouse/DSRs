/*
Script to optimize a typed QA module for a HotpotQA subset with COPRO.

Run with:
```
cargo run --example 04-optimize-hotpotqa --features dataloaders
```
*/

use anyhow::Result;
use bon::Builder;
use dspy_rs::{
    COPRO, DataLoader, Example, LM, Eval, Module, ModuleState, Optimizer,
    Predict, PredictError, Predicted, Signature, TypedLoadOptions, TypedMetric, average_score,
    configure, evaluate_trainset, init_tracing,
};

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
    )?[..10]
        .to_vec();

    let metric = ExactMatchMetric;
    let mut module = QAModule::builder().build();

    let baseline = average_score(&evaluate_trainset(&module, &examples, &metric).await?);
    println!("baseline score: {baseline:.3}");

    let optimizer = COPRO::builder()
        .breadth(10)
        .depth(1)
        .eval_concurrency(16) // candidate evaluations fan out 16 LM calls at a time
        .build();
    optimizer
        .compile(&mut module, examples.clone(), &metric)
        .await?;

    let optimized = average_score(&evaluate_trainset(&module, &examples, &metric).await?);
    println!("optimized score: {optimized:.3}");

    // Persist the tuned instructions for later `ModuleState::load(...).apply(...)`.
    ModuleState::from_module(&mut module)?.save("optimized-hotpotqa.json")?;
    println!("saved optimized module state to optimized-hotpotqa.json");

    Ok(())
}
