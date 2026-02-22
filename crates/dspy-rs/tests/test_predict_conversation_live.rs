use dspy_rs::{ChatAdapter, LM, Predict, Signature, configure};
use std::sync::LazyLock;
use tokio::sync::Mutex;

static SETTINGS_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

#[derive(Signature, Clone, Debug, PartialEq)]
/// Live multi-turn conversation signature.
struct LiveConversation {
    #[input]
    prompt: String,

    #[output]
    answer: String,
}

#[tokio::test]
#[ignore] // Requires real network access and provider API key(s)
async fn live_forward_with_history_two_turn_roundtrip() {
    let _lock = SETTINGS_LOCK.lock().await;

    let lm = LM::builder()
        .model("openai:gpt-4o-mini".to_string())
        .temperature(0.0)
        .max_tokens(256)
        .build()
        .await
        .expect("failed to build LM for live smoke test");
    configure(lm, ChatAdapter {});

    let predict = Predict::<LiveConversation>::new();

    // First turn: build and call
    let first_input = LiveConversationInput {
        prompt: "Reply with the word ONE.".to_string(),
    };
    let first = predict
        .forward(first_input, None)
        .await
        .expect("first turn forward failed");
    let chat = first.chat().clone();
    assert!(
        !first.answer.trim().is_empty(),
        "first turn answer should not be empty"
    );

    // Second turn: continue with typed follow-up and prior history
    let second_input = LiveConversationInput {
        prompt: "Now reply with the word TWO. Use the same answer field format.".to_string(),
    };

    let second = predict
        .forward(second_input, Some(chat))
        .await
        .expect("second turn forward failed");
    let chat2 = second.chat();

    assert!(
        second.answer.to_ascii_lowercase().contains("two"),
        "second turn answer should include 'two', got: {}",
        second.answer
    );
    assert!(chat2.len() >= 5, "chat should grow across turns");
}
