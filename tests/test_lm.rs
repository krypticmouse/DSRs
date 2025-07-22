use dspy_rs::clients::{chat::Chat, dummy_lm::DummyLM};
use openrouter_rs::{api::chat::Message, types::Role};

#[tokio::test]
#[cfg_attr(miri, ignore)] // Miri doesn't support async runtime
async fn test_dummy_lm() {
    let mut dummy_lm = DummyLM::default();

    assert_eq!(dummy_lm.history.len(), 0);

    let chat = Chat::new(vec![
        Message {
            role: Role::System,
            content: "You are a helpful assistant.".to_string(),
        },
        Message {
            role: Role::User,
            content: "Hello, world!".to_string(),
        },
    ]);

    let output = dummy_lm.call(&chat, "Hello, world!", "test").await.unwrap();
    let choice = &output.choices[0];
    if let openrouter_rs::types::Choice::NonStreaming(non_streaming) = choice {
        assert_eq!(
            non_streaming.message.content,
            Some("Hello, world!".to_string())
        );
    } else {
        panic!("Expected non-streaming choice");
    }
    assert_eq!(dummy_lm.history.len(), 1);

    // Check that the chat was stored in history
    let stored_history = &dummy_lm.history[0];
    assert_eq!(stored_history.input.messages.len(), 2);
    assert_eq!(stored_history.input.messages[0].role.to_string(), "system");
    assert_eq!(stored_history.input.messages[1].role.to_string(), "user");
    assert_eq!(stored_history.input.messages[1].content, "Hello, world!");

    let history = dummy_lm.inspect_history(1);
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].input.messages.len(), 2);
    assert_eq!(history[0].input.messages[0].role.to_string(), "system");
    assert_eq!(history[0].input.messages[1].role.to_string(), "user");
    assert_eq!(history[0].input.messages[1].content, "Hello, world!");
}
