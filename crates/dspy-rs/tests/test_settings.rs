use dspy_rs::{LM, configure, get_lm};

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_settings() {
    let lm1 = temp_env::async_with_vars(
        [("OPENAI_API_KEY", Some("test"))],
        LM::builder()
            .model("openai:gpt-4o-mini".to_string())
            .build(),
    )
    .await
    .unwrap();
    configure(lm1);

    let lm = get_lm();
    assert_eq!(lm.config.model, "openai:gpt-4o-mini");

    let lm2 = temp_env::async_with_vars(
        [("OPENAI_API_KEY", Some("test"))],
        LM::builder().model("openai:gpt-4o".to_string()).build(),
    )
    .await
    .unwrap();
    configure(lm2);

    let lm = get_lm();
    assert_eq!(lm.config.model, "openai:gpt-4o");
}
