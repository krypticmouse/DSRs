use dspy_rs::{Cache, Chat, DummyLM, LM, LMConfig, LmUsage, Message};
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

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_lm_with_cache_enabled() {
    // Create LM with cache enabled
    let config = LMConfig {
        cache: true,
        ..Default::default()
    };

    let mut lm = LM::builder()
        .api_key("test_key".to_string().into())
        .config(config)
        .build();

    // Setup the LM client and cache
    lm.setup_client().await;

    // Verify cache handler is initialized
    assert!(lm.cache_handler.is_some());
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_lm_with_cache_disabled() {
    // Create LM with cache explicitly disabled
    let config = LMConfig {
        cache: false,
        ..Default::default()
    };

    let mut lm = LM::builder()
        .api_key("test_key".to_string().into())
        .config(config)
        .build();

    // Setup the LM client
    lm.setup_client().await;

    // Verify cache handler is NOT initialized when cache is disabled
    assert!(lm.cache_handler.is_none());
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_lm_cache_initialization_on_first_call() {
    // Create LM with cache enabled
    let config = LMConfig {
        cache: true,
        ..Default::default()
    };

    let mut lm = LM::builder()
        .api_key("test_key".to_string().into())
        .config(config)
        .build();

    // Initially, cache_handler should be None
    assert!(lm.cache_handler.is_none());

    // Setup happens on first call
    lm.setup_client().await;

    // After setup, cache_handler should be initialized
    assert!(lm.cache_handler.is_some());
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_lm_cache_direct_operations() {
    use dspy_rs::{Example, Prediction};
    use std::collections::HashMap;

    // Create LM with cache enabled
    let config = LMConfig {
        cache: true,
        ..Default::default()
    };

    let mut lm = LM::builder()
        .api_key("test_key".to_string().into())
        .config(config)
        .build();

    lm.setup_client().await;

    // Get cache handler
    let cache = lm
        .cache_handler
        .as_ref()
        .expect("Cache should be initialized");

    // Create test data
    let mut input_data = HashMap::new();
    input_data.insert(
        "question".to_string(),
        serde_json::json!("What is the capital of France?"),
    );
    let key = Example::new(input_data, vec!["question".to_string()], vec![]);

    // Initially cache should be empty
    let cached = cache.get(key.clone()).await.unwrap();
    assert!(cached.is_none());

    // Insert data
    let mut output_data = HashMap::new();
    output_data.insert("answer".to_string(), serde_json::json!("Paris"));
    output_data.insert("confidence".to_string(), serde_json::json!(0.95));
    let value = Prediction::new(output_data, LmUsage::default());

    cache.insert(key.clone(), value.clone()).unwrap();

    // Now cache should return the value
    let cached = cache.get(key).await.unwrap();
    assert!(cached.is_some());

    let cached_prediction = cached.unwrap();
    assert_eq!(
        cached_prediction.data.get("answer"),
        value.data.get("answer")
    );
    assert_eq!(
        cached_prediction.data.get("confidence"),
        value.data.get("confidence")
    );
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_lm_cache_with_different_models() {
    // Test that cache works with different model configurations
    let models = vec![
        "gpt-4o-mini",
        "openai/gpt-3.5-turbo",
        "anthropic/claude-3-haiku-20240307",
    ];

    for model in models {
        let config = LMConfig {
            cache: true,
            model: model.to_string(),
            ..Default::default()
        };

        let mut lm = LM::builder()
            .api_key("test_key".to_string().into())
            .config(config)
            .build();

        lm.setup_client().await;

        // Cache should be initialized regardless of model
        assert!(
            lm.cache_handler.is_some(),
            "Cache should be initialized for model: {}",
            model
        );
    }
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_cache_with_complex_inputs() {
    use dspy_rs::{Example, Prediction};
    use std::collections::HashMap;

    // Create LM with cache enabled
    let config = LMConfig {
        cache: true,
        ..Default::default()
    };

    let mut lm = LM::builder()
        .api_key("test_key".to_string().into())
        .config(config)
        .build();

    lm.setup_client().await;

    let cache = lm
        .cache_handler
        .as_ref()
        .expect("Cache should be initialized");

    // Create complex example with multiple fields
    let mut data = HashMap::new();
    data.insert("context".to_string(), serde_json::json!("The quick brown fox jumps over the lazy dog. This is a common pangram used in typography."));
    data.insert(
        "question".to_string(),
        serde_json::json!("What animal jumps over another animal?"),
    );
    data.insert("format".to_string(), serde_json::json!("detailed"));
    data.insert("temperature".to_string(), serde_json::json!(0.7));

    let key = Example::new(
        data.clone(),
        vec![
            "context".to_string(),
            "question".to_string(),
            "format".to_string(),
            "temperature".to_string(),
        ],
        vec![],
    );

    // Create prediction with multiple outputs
    let mut output = HashMap::new();
    output.insert(
        "answer".to_string(),
        serde_json::json!("A fox jumps over a dog"),
    );
    output.insert("confidence".to_string(), serde_json::json!(0.85));
    output.insert(
        "reasoning".to_string(),
        serde_json::json!("The text mentions 'The quick brown fox jumps over the lazy dog'"),
    );

    let value = Prediction::new(
        output.clone(),
        LmUsage {
            prompt_tokens: 50,
            completion_tokens: 30,
            total_tokens: 80,
            reasoning_tokens: Some(15),
        },
    );

    // Insert and retrieve
    cache.insert(key.clone(), value.clone()).unwrap();

    let cached = cache.get(key).await.unwrap().unwrap();
    assert_eq!(cached.data.len(), 3);
    assert_eq!(cached.data.get("answer"), output.get("answer"));
    assert_eq!(cached.data.get("confidence"), output.get("confidence"));
    assert_eq!(cached.data.get("reasoning"), output.get("reasoning"));
    // Note: lm_usage is reset to default when converting from cached Vec<(String, Value)>
    // This is expected behavior due to the From<Vec<(String, Value)>> implementation
    assert_eq!(cached.lm_usage.prompt_tokens, 0); // Default value
    assert_eq!(cached.lm_usage.completion_tokens, 0); // Default value
}
