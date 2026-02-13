/*
Script to inspect LM history after a typed predictor call.

Run with:
```
cargo run --example 07-inspect-history
```
*/

use anyhow::Result;
use dspy_rs::{ChatAdapter, LM, Predict, Signature, configure, get_lm, init_tracing};

#[derive(Signature, Clone, Debug)]
struct QA {
    #[input]
    question: String,

    #[output]
    answer: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing()?;

    let lm = LM::builder()
        .model("openai:gpt-4o-mini".to_string())
        .build()
        .await?;
    configure(lm, ChatAdapter);

    let predictor = Predict::<QA>::new();
    let output = predictor
        .call(QAInput {
            question: "What is the capital of France?".to_string(),
        })
        .await?
        .into_inner();
    println!("prediction: {:?}", output.answer);

    let history = get_lm().inspect_history(1).await;
    println!("history: {history:?}");

    Ok(())
}
