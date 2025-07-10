use openai_api_rs::v1::chat_completion::{Content, MessageRole};

use dsrs::premitives::dummy_lm::DummyLM;
use dsrs::premitives::lm::{LMConfig, LMProvider};

#[tokio::test]
#[cfg_attr(miri, ignore)] // Miri doesn't support async runtime
async fn test_dummy_lm() {
    let mut dummy_lm = DummyLM::new(
        Some(LMProvider::OpenAI),
        "test".to_string(),
        "test".to_string(),
        LMConfig::default(),
        None,
    );

    assert_eq!(dummy_lm.history.len(), 0);

    let output = dummy_lm
        .forward(
            "Hello, world!".to_string(),
            "Hello, world!".to_string(),
            "test".to_string(),
        )
        .await;

    assert_eq!(output, "Hello, world!");
    assert_eq!(dummy_lm.history.len(), 1);
    assert_eq!(dummy_lm.history[0].input.len(), 1);
    assert_eq!(dummy_lm.history[0].input[0].role, MessageRole::user);
    assert_eq!(
        dummy_lm.history[0].input[0].content,
        Content::Text("Hello, world!".to_string())
    );
    assert_eq!(dummy_lm.history[0].output, "Hello, world!");
    assert_eq!(dummy_lm.history[0].signature, "test");
    assert_eq!(dummy_lm.history[0].model, "test");

    let history = dummy_lm.inspect_history(1);
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].input.len(), 1);
    assert_eq!(history[0].input[0].role, MessageRole::user);
    assert_eq!(
        history[0].input[0].content,
        Content::Text("Hello, world!".to_string())
    );
    assert_eq!(history[0].output, "Hello, world!");
    assert_eq!(history[0].signature, "test");
    assert_eq!(history[0].model, "test");
    assert_eq!(history[0].provider, LMProvider::OpenAI);
}
