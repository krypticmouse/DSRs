use anyhow::Result;
use dspy_rs::{Chat, DummyLM, Example, Message, hashmap, init_tracing};

#[tokio::main]
async fn main() -> Result<()> {
    // Turn on human-readable tracing output with a sensible default filter.
    init_tracing()?;

    let lm = DummyLM::new().await;
    let example = Example::new(
        hashmap! {
            "problem".to_string() => "What is 2 + 2?".to_string().into(),
        },
        vec!["problem".to_string()],
        vec!["answer".to_string()],
    );
    let chat = Chat::new(vec![
        Message::system("You are a precise math assistant."),
        Message::user("What is 2 + 2?"),
    ]);

    let response = lm
        .call(
            example,
            chat,
            "[[ ## answer ## ]]\n4\n\n[[ ## completed ## ]]".to_string(),
        )
        .await?;

    println!("assistant response: {}", response.output.content());
    Ok(())
}
