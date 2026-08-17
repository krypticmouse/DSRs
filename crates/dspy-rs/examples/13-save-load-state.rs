/*
Example: persisting and reloading optimized module state.

After an optimizer tunes a module, the improved instructions and demos live only
in memory. `ModuleState` snapshots every `Predict` leaf (keyed by the dotted path
the optimizer walker discovers) into JSON, so production code can reload an
optimized program without re-running optimization.

Runs fully offline — no API key needed.

Run with:
```
cargo run --example 13-save-load-state
```
*/

use anyhow::Result;
use bon::Builder;
use dspy_rs::{Demo, Module, ModuleState, Predict, PredictError, Predicted, Signature};

#[derive(Signature, Clone, Debug)]
struct QA {
    /// Answer the question accurately and concisely.

    #[input]
    question: String,

    #[output]
    answer: String,
}

#[derive(Builder, facet::Facet)]
#[facet(crate = facet)]
struct QAPipeline {
    #[builder(default = Predict::<QA>::new())]
    answerer: Predict<QA>,
}

impl Module for QAPipeline {
    type Input = QAInput;
    type Output = QAOutput;

    async fn forward(&self, input: QAInput) -> Result<Predicted<QAOutput>, PredictError> {
        self.answerer.call(input).await
    }
}

fn main() -> Result<()> {
    // Stand-in for an optimizer run: a module with a tuned instruction and a
    // bootstrapped demo. (`optimizer.compile(...)` leaves the module in exactly
    // this kind of state.)
    let mut tuned = QAPipeline::builder()
        .answerer(
            Predict::<QA>::builder()
                .instruction("Answer in one short, factual sentence.")
                .demo(Demo::new(
                    QAInput {
                        question: "What is the capital of France?".to_string(),
                    },
                    QAOutput {
                        answer: "Paris".to_string(),
                    },
                ))
                .build(),
        )
        .build();

    // --- Save -------------------------------------------------------------
    let state = ModuleState::from_module(&mut tuned)?;
    println!("Snapshot of {} predictor(s):", state.predictors.len());
    for (path, predictor_state) in &state.predictors {
        println!(
            "  `{path}`: instruction={:?}, demos={}",
            predictor_state.instruction_override,
            predictor_state.demos.len()
        );
    }

    let json = state.to_json()?;
    println!("\nSerialized state:\n{json}\n");
    state.save("qa-pipeline-state.json")?;

    // --- Load into a fresh, untuned module --------------------------------
    let mut fresh = QAPipeline::builder().build();
    ModuleState::load("qa-pipeline-state.json")?.apply(&mut fresh)?;

    let restored = ModuleState::from_module(&mut fresh)?;
    let answerer = &restored.predictors["answerer"];
    println!(
        "Restored `answerer`: instruction={:?}, demos={}",
        answerer.instruction_override,
        answerer.demos.len()
    );

    // `apply` is strict: state referring to predictors the module doesn't have
    // is an error, so drifted save files fail loudly instead of half-loading.
    std::fs::remove_file("qa-pipeline-state.json").ok();
    Ok(())
}
