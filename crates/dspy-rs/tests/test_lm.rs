use dspy_rs::{LM, LMConfig, LmUsage};
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
    let cached = cache.lock().await.get_entry(key).await.unwrap();
    assert!(cached.is_none());

    // Insert an entry
    let entry = CacheEntry {
        prompt: "test prompt".to_string(),
        usage: LmUsage::default(),
        raw_output: Some("answer: Paris".to_string()),
    };
    cache.lock().await.insert_entry(key, entry.clone());

    // Now cache should return the entry
    let cached = cache
        .lock()
        .await
        .get_entry(key)
        .await
        .unwrap()
        .expect("entry should be cached");
    assert_eq!(cached.prompt, entry.prompt);
    assert_eq!(cached.raw_output, entry.raw_output);

    // Unknown keys still miss
    let missing = cache.lock().await.get_entry(key ^ 1).await.unwrap();
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

    cache.lock().await.insert_entry(key, entry.clone());

    // The cache stores and retrieves the full entry including usage stats.
    let cached = cache.lock().await.get_entry(key).await.unwrap().unwrap();
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
    cache.lock().await.insert_entry(43, later.clone());

    let history = cache.lock().await.get_history(2).await.unwrap();
    assert_eq!(history.len(), 2);
    assert_eq!(history[0].prompt, later.prompt);
    assert_eq!(history[1].prompt, entry.prompt);
}
