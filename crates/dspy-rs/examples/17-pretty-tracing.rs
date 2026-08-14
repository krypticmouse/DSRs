use anyhow::Result;
use dspy_rs::{Chat, LM, LMClient, Message, TestCompletionModel, init_tracing};
use rig::completion::AssistantContent;
use rig::message::Text;

#[tokio::main]
async fn main() -> Result<()> {
    // Turn on human-readable tracing output with a sensible default filter.
    init_tracing()?;

    // Offline LM: a canned response instead of a live provider.
    let client = TestCompletionModel::new([AssistantContent::Text(Text {
        text: "[[ ## answer ## ]]\n4\n\n[[ ## completed ## ]]".to_string(),
    })]);
    let lm = temp_env::async_with_vars(
        [("OPENAI_API_KEY", Some("offline"))],
        LM::builder().model("openai:gpt-4o-mini".to_string()).build(),
    )
    .await?
    .with_client(LMClient::Test(client))
    .await?;

    let chat = Chat::new(vec![
        Message::system("You are a precise math assistant."),
        Message::user("What is 2 + 2?"),
    ]);

    let response = lm.call(chat, vec![]).await?;

    println!("assistant response: {}", response.output.content());
    Ok(())
}
