use async_openai::types::{
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
    ChatCompletionRequestUserMessageArgs,
};
use dspy_rs::clients::{chat::Chat, dummy_lm::DummyLM};

#[cfg_attr(miri, ignore)] // Miri doesn't support tokio's I/O driver
#[tokio::test]
async fn test_dummy_lm() {
    let mut dummy_lm = DummyLM::default();

    assert_eq!(dummy_lm.history.len(), 0);

    let chat = Chat::new(vec![
        ChatCompletionRequestMessage::System(
            ChatCompletionRequestSystemMessageArgs::default()
                .content("You are a helpful assistant.".to_string())
                .build()
                .unwrap(),
        ),
        ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessageArgs::default()
                .content("Hello, world!".to_string())
                .build()
                .unwrap(),
        ),
    ]);

    let output = dummy_lm.call(&chat, "Hello, world!", "test").await.unwrap();
    let choice = &output.choices[0];
    assert_eq!(choice.message.content, Some("Hello, world!".to_string()));
    assert_eq!(dummy_lm.history.len(), 1);

    // Check that the chat was stored in history
    let stored_history = &dummy_lm.history[0];
    assert_eq!(stored_history.input.messages.len(), 2);
    assert_eq!(
        stored_history.input.messages[0],
        ChatCompletionRequestMessage::System(
            ChatCompletionRequestSystemMessageArgs::default()
                .content("You are a helpful assistant.".to_string())
                .build()
                .unwrap()
        )
    );
    assert_eq!(
        stored_history.input.messages[1],
        ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessageArgs::default()
                .content("Hello, world!".to_string())
                .build()
                .unwrap()
        )
    );

    let history = dummy_lm.inspect_history(1);
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].input.messages.len(), 2);
    assert_eq!(
        history[0].input.messages[0],
        ChatCompletionRequestMessage::System(
            ChatCompletionRequestSystemMessageArgs::default()
                .content("You are a helpful assistant.".to_string())
                .build()
                .unwrap()
        )
    );
    assert_eq!(
        history[0].input.messages[1],
        ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessageArgs::default()
                .content("Hello, world!".to_string())
                .build()
                .unwrap()
        )
    );
}
