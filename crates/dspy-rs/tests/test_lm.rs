use dspy_rs::{Chat, DummyLM, Message};

#[cfg_attr(miri, ignore)] // Miri doesn't support tokio's I/O driver
#[tokio::test]
async fn test_dummy_lm() {
    let mut dummy_lm = DummyLM::default();

    assert_eq!(dummy_lm.history.len(), 0);

    let chat = Chat::new(vec![
        Message::system("You are a helpful assistant."),
        Message::user("Hello, world!"),
    ]);

    let output = dummy_lm
        .call(chat, "DummySignature", "Hello, world!".to_string())
        .await
        .unwrap();
    let choice = &output.0.content();
    assert_eq!(choice, "Hello, world!");
    assert_eq!(dummy_lm.history.len(), 1);

    // Check that the chat was stored in history
    let stored_history = &dummy_lm.history[0];
    assert_eq!(stored_history.chat.len(), 2);
    assert_eq!(
        stored_history.chat.messages[0].content(),
        "You are a helpful assistant.".to_string(),
    );
    assert_eq!(
        stored_history.chat.messages[1].content(),
        "Hello, world!".to_string(),
    );

    let history = dummy_lm.inspect_history(1);
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].chat.len(), 2);
    assert_eq!(
        history[0].chat.messages[0].content(),
        "You are a helpful assistant.".to_string(),
    );
    assert_eq!(
        history[0].chat.messages[1].content(),
        "Hello, world!".to_string(),
    );
}
