use dspy_rs::{ChatAdapter, LM, LMConfig, configure, get_lm};

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_settings() {
    configure(
        LM::new(LMConfig {
            model: "openai:gpt-4o-mini".to_string(),
            ..LMConfig::default()
        }),
        ChatAdapter {},
    );

    let lm = get_lm();
    assert_eq!(lm.config.model, "openai:gpt-4o-mini");

    configure(
        LM::new(LMConfig {
            model: "openai:gpt-4o".to_string(),
            ..LMConfig::default()
        }),
        ChatAdapter {},
    );

    let lm = get_lm();

    assert_eq!(lm.config.model, "openai:gpt-4o");
}
