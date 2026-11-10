use dspy_rs::{Chat, LM, LMClient, LMConfig, LmUsage, Message, TestCompletionModel};
use rstest::*;

#[rstest]
fn lm_config_serializes_without_secrets() {
    let config = LMConfig {
        api_key: Some("sk-super-secret".to_string()),
        model: "openai:gpt-4o".to_string(),
        ..Default::default()
    };

    let json = serde_json::to_string(&config).unwrap();
    assert!(!json.contains("sk-super-secret"));
    assert!(!json.contains("api_key"));

    let restored: LMConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.api_key, None);
    assert_eq!(restored.model, "openai:gpt-4o");
    assert_eq!(restored.temperature, config.temperature);
    assert_eq!(restored.max_tokens, config.max_tokens);
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn lm_from_config_builds_live_client() {
    let config = LMConfig {
        model: "openai:gpt-4o-mini".to_string(),
        cache: true,
        ..Default::default()
    };

    let lm = temp_env::async_with_vars(
        [("OPENAI_API_KEY", Some("test"))],
        LM::from_config(config.clone()),
    )
    .await
    .unwrap();

    // The live LM carries its config verbatim and initialized live state.
    assert_eq!(lm.config, config);
    assert!(lm.cache_handler.is_some());
}

#[rstest]
fn test_lm_usage_add() {
    let usage1 = LmUsage {
        prompt_tokens: 10,
        completion_tokens: 20,
        total_tokens: 30,
    };
    let usage2 = LmUsage {
        prompt_tokens: 10,
        completion_tokens: 20,
        total_tokens: 30,
    };

    let usage3 = usage1 + usage2;

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
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_lm_with_cache_enabled() {
    let lm = temp_env::async_with_vars(
        [("OPENAI_API_KEY", Some("test"))],
        LM::builder()
            .model("openai:gpt-4o-mini".to_string())
            .cache(true)
            .build(),
    )
    .await
    .unwrap();

    // Verify cache handler is initialized
    assert!(lm.cache_handler.is_some());
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_lm_with_cache_disabled() {
    let lm = temp_env::async_with_vars(
        [("OPENAI_API_KEY", Some("test"))],
        LM::builder()
            .model("openai:gpt-4o-mini".to_string())
            .cache(false)
            .build(),
    )
    .await
    .unwrap();

    // Verify cache handler is NOT initialized when cache is disabled
    assert!(lm.cache_handler.is_none());
}

#[rstest]
#[case::default(None)]
#[case::disabled(Some(false))]
#[case::enabled(Some(true))]
#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn inspect_history_after_completion(#[case] cache: Option<bool>) {
    let mut config = LMConfig {
        base_url: Some("http://localhost:11434/v1".to_string()),
        model: "llama3.2".to_string(),
        ..Default::default()
    };
    if let Some(cache) = cache {
        config.cache = cache;
    }
    let client = TestCompletionModel::new([rig::completion::AssistantContent::text("Paris")]);
    let lm = LM::from_config(config)
        .await
        .unwrap()
        .with_client(LMClient::Test(client))
        .await
        .unwrap();

    assert_eq!(lm.cache_handler.is_some(), cache == Some(true));
    assert!(lm.inspect_history(1).await.is_empty());
    let chat = Chat::new(vec![Message::user("What is the capital of France?")]);
    let prompt = chat.to_json().to_string();
    let response = lm.call(chat, vec![]).await.unwrap();
    assert_eq!(response.output.content(), "Paris");

    assert!(lm.inspect_history(0).await.is_empty());
    let history = lm.inspect_history(10).await;
    if cache == Some(true) {
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].prompt, prompt);
        assert_eq!(history[0].raw_output.as_deref(), Some("Paris"));
    } else {
        assert!(history.is_empty());
    }
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn replacing_client_isolates_cached_responses_and_history() {
    use rig::completion::AssistantContent;

    let original_client = TestCompletionModel::new([AssistantContent::text("old backend")]);
    let original = LM::builder()
        .base_url("http://localhost:11434/v1".to_string())
        .model("llama3.2".to_string())
        .cache(true)
        .build()
        .await
        .unwrap()
        .with_client(LMClient::Test(original_client))
        .await
        .unwrap();
    let chat = Chat::new(vec![Message::user("hello")]);
    let first = original.call(chat.clone(), vec![]).await.unwrap();
    assert_eq!(first.output.content(), "old backend");

    let replacement_client = TestCompletionModel::new([AssistantContent::text("new backend")]);
    let replacement = original
        .clone()
        .with_client(LMClient::Test(replacement_client.clone()))
        .await
        .unwrap();
    assert!(replacement.inspect_history(1).await.is_empty());

    let fresh = replacement.call(chat.clone(), vec![]).await.unwrap();
    assert_eq!(fresh.output.content(), "new backend");
    assert!(replacement_client.last_request().is_some());

    // Each client had only one response: both repeat calls must hit their own cache.
    let original_hit = original.call(chat.clone(), vec![]).await.unwrap();
    let replacement_hit = replacement.call(chat, vec![]).await.unwrap();
    assert_eq!(original_hit.output.content(), "old backend");
    assert_eq!(replacement_hit.output.content(), "new backend");

    let original_history = original.inspect_history(10).await;
    let replacement_history = replacement.inspect_history(10).await;
    assert_eq!(original_history.len(), 1);
    assert_eq!(replacement_history.len(), 1);
    assert_eq!(
        original_history[0].raw_output.as_deref(),
        Some("old backend")
    );
    assert_eq!(
        replacement_history[0].raw_output.as_deref(),
        Some("new backend")
    );
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_lm_cache_initialization_on_first_call() {
    let lm = temp_env::async_with_vars(
        [("OPENAI_API_KEY", Some("test"))],
        LM::builder()
            .model("openai:gpt-4o-mini".to_string())
            .cache(true)
            .build(),
    )
    .await
    .unwrap();

    // After build, cache_handler should be initialized
    assert!(lm.cache_handler.is_some());
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_lm_cache_direct_operations() {
    use dspy_rs::utils::cache::{CacheEntry, CacheKey};

    let lm = temp_env::async_with_vars(
        [("OPENAI_API_KEY", Some("test"))],
        LM::builder()
            .model("openai:gpt-4o-mini".to_string())
            .cache(true)
            .build(),
    )
    .await
    .unwrap();

    // Get cache handler
    let cache = lm
        .cache_handler
        .as_ref()
        .expect("Cache should be initialized");

    let key: CacheKey = 0xDEAD_BEEF;

    // Initially cache should be empty
    let cached = cache.get_entry(key).await.unwrap();
    assert!(cached.is_none());

    // Insert an entry
    let entry = CacheEntry {
        prompt: "test prompt".to_string(),
        usage: LmUsage::default(),
        raw_output: Some("answer: Paris".to_string()),
    };
    cache.insert_entry(key, entry.clone());

    // Now cache should return the entry
    let cached = cache
        .get_entry(key)
        .await
        .unwrap()
        .expect("entry should be cached");
    assert_eq!(cached.prompt, entry.prompt);
    assert_eq!(cached.raw_output, entry.raw_output);

    // Unknown keys still miss
    let missing = cache.get_entry(key ^ 1).await.unwrap();
    assert!(missing.is_none());
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_lm_cache_with_different_models() {
    // Test that cache works with different model configurations
    let models = vec!["openai:gpt-3.5-turbo", "anthropic:claude-3-haiku-20240307"];

    for model in models {
        let lm = temp_env::async_with_vars(
            [
                ("OPENAI_API_KEY", Some("test")),
                ("ANTHROPIC_API_KEY", Some("test")),
            ],
            LM::builder().model(model.to_string()).cache(true).build(),
        )
        .await
        .unwrap();

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
async fn test_cache_preserves_usage_and_history() {
    use dspy_rs::utils::cache::{CacheEntry, CacheKey};

    let lm = temp_env::async_with_vars(
        [("OPENAI_API_KEY", Some("test"))],
        LM::builder()
            .model("openai:gpt-4o-mini".to_string())
            .cache(true)
            .build(),
    )
    .await
    .unwrap();

    let cache = lm
        .cache_handler
        .as_ref()
        .expect("Cache should be initialized");

    let key: CacheKey = 42;
    let entry = CacheEntry {
        prompt: "complex test prompt".to_string(),
        usage: LmUsage {
            prompt_tokens: 50,
            completion_tokens: 30,
            total_tokens: 80,
        },
        raw_output: Some("answer: A fox jumps over a dog".to_string()),
    };

    cache.insert_entry(key, entry.clone());

    // The cache stores and retrieves the full entry including usage stats.
    let cached = cache.get_entry(key).await.unwrap().unwrap();
    assert_eq!(cached.prompt, entry.prompt);
    assert_eq!(cached.raw_output, entry.raw_output);
    assert_eq!(cached.usage.prompt_tokens, 50);
    assert_eq!(cached.usage.completion_tokens, 30);
    assert_eq!(cached.usage.total_tokens, 80);

    // Insertions land in the sliding history window, newest first.
    let later = CacheEntry {
        prompt: "second prompt".to_string(),
        usage: LmUsage::default(),
        raw_output: None,
    };
    cache.insert_entry(43, later.clone());

    let history = lm.inspect_history(2).await;
    assert_eq!(history.len(), 2);
    assert_eq!(history[0].prompt, later.prompt);
    assert_eq!(history[1].prompt, entry.prompt);
    assert_eq!(lm.inspect_history(1).await[0].prompt, later.prompt);
}
