//! Run a typed prediction against local Ollama and print the prompt before sending it.
//!
//! First run `ollama pull llama3.2`, then:
//! `cargo run -p dspy-rs --no-default-features --example 19-ollama-structured-output`
//! Override OLLAMA_BASE_URL or OLLAMA_MODEL to use another local configuration.
//! Pass `--inspect-only` to print the prompt without sending a request.
//! Pass `--scalar-output` to opt into the verified root-string schema optimization.

use anyhow::Result;
use dspy_rs::{LM, Predict, Signature, init_tracing};

#[derive(Signature, Clone, Debug)]
/// Answer geography questions accurately.
struct QA {
    #[input]
    /// The geography question to answer.
    question: String,

    #[output]
    /// The city name, without extra commentary.
    answer: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing()?;
    let lm = LM::builder()
        .base_url(
            std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434/v1".into()),
        )
        .model(std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "llama3.2".into()))
        .temperature(0.0)
        .build()
        .await?
        .with_custom_output_schema(true)
        .with_scalar_output_schema(std::env::args().any(|arg| arg == "--scalar-output"));
    let predictor = Predict::<QA>::builder().lm(lm).build();
    let input = QAInput {
        question: "What is the capital of France?".into(),
    };
    let chat = predictor.build_chat(&input).await?;
    println!("Prompt: {}", serde_json::to_string_pretty(&chat.to_json())?);
    if std::env::args().any(|arg| arg == "--inspect-only") {
        return Ok(());
    }
    let (prediction, conversation) = predictor.call_and_parse(chat).await.map_err(|error| {
        eprintln!("Prediction failed: {error:#?}");
        error
    })?;
    println!("Answer: {}", prediction.answer);
    println!(
        "Conversation: {}",
        serde_json::to_string_pretty(&conversation.to_json())?
    );
    Ok(())
}
