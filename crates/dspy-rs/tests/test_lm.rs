use dspy_rs::{Chat, DummyLM, LmUsage, Message};
use rstest::*;

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

#[rstest]
fn test_lm_usage_add() {
    let usage1 = LmUsage {
        prompt_tokens: 10,
        completion_tokens: 20,
        total_tokens: 30,
        reasoning_tokens: Some(10),
    };
    let usage2 = LmUsage {
        prompt_tokens: 10,
        completion_tokens: 20,
        total_tokens: 30,
        reasoning_tokens: Some(10),
    };

    let usage3 = usage1.clone() + usage2.clone();

    assert_eq!(
        usage3.prompt_tokens,
        usage1.prompt_tokens + usage2.prompt_tokens
    );
    assert_eq!(
        usage3.completion_tokens,
        usage1.completion_tokens + usage2.completion_tokens
    );
    assert_eq!(
        usage3.total_tokens,
        usage1.total_tokens + usage2.total_tokens
    );
    assert_eq!(
        usage3.reasoning_tokens,
        usage1.reasoning_tokens.or(usage2.reasoning_tokens)
    );
}
