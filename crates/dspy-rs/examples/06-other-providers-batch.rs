/*
Script to run typed batch inference against multiple providers.

Run with:
```
cargo run --example 06-other-providers-batch
```
*/

use anyhow::Result;
use dspy_rs::{
    ChatAdapter, LM, Predict, Signature, configure, forward_all, init_tracing,
};

#[derive(Signature, Clone, Debug)]
struct QA {
    #[input]
    question: String,

    #[output(desc = "Think step by step before answering")]
    reasoning: String,

    #[output]
    answer: String,
}

fn prompts() -> Vec<QAInput> {
    vec![
        QAInput {
            question: "What is the capital of France?".to_string(),
        },
        QAInput {
            question: "What is the capital of Germany?".to_string(),
        },
        QAInput {
            question: "What is the capital of Italy?".to_string(),
        },
    ]
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing()?;

    let predictor = Predict::<QA>::builder()
        .instruction("Answer with concise factual outputs.")
        .build();

    configure(
        LM::builder()
            .model("anthropic:claude-sonnet-4-5-20250929".to_string())
            .build()
            .await?,
        ChatAdapter,
    );

    let anthropic = forward_all(&predictor, prompts(), 2)
        .await
        .into_iter()
        .map(|outcome| outcome.map(|predicted| predicted.into_inner().answer))
        .collect::<Result<Vec<_>, _>>()?;
    println!("Anthropic: {anthropic:?}");

    configure(
        LM::builder()
            .model("gemini:gemini-2.0-flash".to_string())
            .build()
            .await?,
        ChatAdapter,
    );

    let gemini = forward_all(&predictor, prompts(), 2)
        .await
        .into_iter()
        .map(|outcome| outcome.map(|predicted| predicted.into_inner().answer))
        .collect::<Result<Vec<_>, _>>()?;
    println!("Gemini: {gemini:?}");

    Ok(())
}
