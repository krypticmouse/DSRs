use dspy_rs::{Cache, Chat, DummyLM, Example, LM, LMConfig, LmUsage, Message, hashmap};
use rstest::*;

#[cfg_attr(miri, ignore)] // Miri doesn't support tokio's I/O driver
#[tokio::test]
async fn test_dummy_lm() {
    let dummy_lm = DummyLM::new().await;

    let chat = Chat::new(vec![
        Message::system("You are a helpful assistant."),
        Message::user("Hello, world!"),
    ]);

    let example = Example::new(
        hashmap! {
            "input".to_string() => "test".to_string().into(),
        },
        vec!["input".to_string()],
        vec![],
    );

    let output = dummy_lm
        .call(example.clone(), chat.clone(), "Hello, world!".to_string())
        .await
        .unwrap();
    assert_eq!(output.output.content(), "Hello, world!");

    // Verify the response structure
    assert_eq!(output.chat.len(), 3); // original 2 messages + assistant response
    assert_eq!(
        output.chat.messages[0].content(),
        "You are a helpful assistant.".to_string(),
    );
    assert_eq!(
        output.chat.messages[1].content(),
        "Hello, world!".to_string(),
    );
    assert_eq!(
        output.chat.messages[2].content(),
        "Hello, world!".to_string(),
    );

    // Check cache functionality if caching is enabled
    if dummy_lm.config.cache {
        let history = dummy_lm.inspect_history(1).await;
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].prompt, chat.to_json().to_string());
    }
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

    let lm = LM::builder()
        .api_key("test_key".to_string().into())
        .config(config)
        .build()
        .await;

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

    let lm = LM::builder()
        .api_key("test_key".to_string().into())
        .config(config)
        .build()
        .await;

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

    let lm = LM::builder()
        .api_key("test_key".to_string().into())
        .config(config)
        .build()
        .await;

    // After build, cache_handler should be initialized
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

    let lm = LM::builder()
        .api_key("test_key".to_string().into())
        .config(config)
        .build()
        .await;

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
    let cached = cache.lock().await.get(key.clone()).await.unwrap();
    assert!(cached.is_none());

    // Insert data
    let mut output_data = HashMap::new();
    output_data.insert("answer".to_string(), serde_json::json!("Paris"));
    output_data.insert("confidence".to_string(), serde_json::json!(0.95));
    let value = Prediction::new(output_data, LmUsage::default());

    // Create a channel to send the result
    let (tx, rx) = tokio::sync::mpsc::channel(1);
    use dspy_rs::CallResult;
    tx.send(CallResult {
        prompt: "test prompt".to_string(),
        prediction: value.clone(),
    })
    .await
    .unwrap();

    cache.lock().await.insert(key.clone(), rx).await.unwrap();

    // Now cache should return the value
    let cached = cache.lock().await.get(key).await.unwrap();
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

        let lm = LM::builder()
            .api_key("test_key".to_string().into())
            .config(config)
            .build()
            .await;

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

    let lm = LM::builder()
        .api_key("test_key".to_string().into())
        .config(config)
        .build()
        .await;

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
    let (tx, rx) = tokio::sync::mpsc::channel(1);
    use dspy_rs::CallResult;
    tx.send(CallResult {
        prompt: "complex test prompt".to_string(),
        prediction: value.clone(),
    })
    .await
    .unwrap();

    cache.lock().await.insert(key.clone(), rx).await.unwrap();

    let cached = cache.lock().await.get(key).await.unwrap().unwrap();
    assert_eq!(cached.data.len(), 3);
    assert_eq!(cached.data.get("answer"), output.get("answer"));
    assert_eq!(cached.data.get("confidence"), output.get("confidence"));
    assert_eq!(cached.data.get("reasoning"), output.get("reasoning"));
    // The cache stores and retrieves the full Prediction including usage stats
    assert_eq!(cached.lm_usage.prompt_tokens, 50); // Preserved from original
    assert_eq!(cached.lm_usage.completion_tokens, 30); // Preserved from original
    assert_eq!(cached.lm_usage.total_tokens, 80); // Preserved from original
    assert_eq!(cached.lm_usage.reasoning_tokens, Some(15)); // Preserved from original
}
