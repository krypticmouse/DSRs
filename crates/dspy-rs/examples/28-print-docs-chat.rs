//! Print the exact ChatAdapter-rendered chat for the DocsQA signature, for
//! the scaffolding-isolation experiment (see docs-chat-worker/gepa/).
//!
//!     set -a; source .env; set +a
//!     cargo run --example 28-print-docs-chat

use anyhow::Result;
use dspy_rs::{LM, Predict, Signature, configure};

/// You are the DSRs docs assistant. DSRs (dspy-rs) is a Rust framework for
/// building and optimizing LM pipelines. Answer only from the attached
/// stores: dsrs-docs (published documentation) and dsrs-code (the Rust
/// sources under crates/). Search before answering; use grep for exact
/// symbol or signature lookups. When an answer touches implementation,
/// quote the real code with its file path. If the stores don't cover the
/// question, say so instead of guessing. Answer in concise markdown.
#[derive(Signature, Clone, Debug)]
struct DocsQA {
    /// The user's question about DSRs.
    #[input]
    question: String,

    #[output]
    answer: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    configure(
        LM::builder()
            .model("openrouter:openai/gpt-5.6-sol".to_string())
            .build()
            .await?,
    );
    let predict = Predict::<DocsQA>::new();
    let chat = predict
        .build_chat(&DocsQAInput {
            question: "___QUESTION___".to_string(),
        })
        .await?;
    for m in &chat.messages {
        println!("=== ROLE: {:?} ===", m.role);
        println!("{}", m.text_content());
    }
    Ok(())
}
