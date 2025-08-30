use dspy_rs::{ChatAdapter, GLOBAL_SETTINGS, LM, LMConfig, configure};
use rstest::rstest;
use secrecy::SecretString;

#[rstest]
fn test_settings() {
    configure(
        LM::builder().api_key(SecretString::from("test")).build(),
        ChatAdapter {},
    );

    assert_eq!(
        GLOBAL_SETTINGS
            .read()
            .unwrap()
            .as_ref()
            .unwrap()
            .lm
            .config
            .model,
        "gpt-4o-mini"
    );
    assert_eq!(
        GLOBAL_SETTINGS
            .read()
            .unwrap()
            .as_ref()
            .unwrap()
            .lm
            .base_url,
        "https://api.openai.com/v1".to_string()
    );

    configure(
        LM::builder()
            .config(LMConfig {
                model: "gpt-4o".to_string(),
                ..LMConfig::default()
            })
            .api_key(SecretString::from("test"))
            .build(),
        ChatAdapter {},
    );

    assert_eq!(
        GLOBAL_SETTINGS
            .read()
            .unwrap()
            .as_ref()
            .unwrap()
            .lm
            .config
            .model,
        "gpt-4o"
    );
    assert_eq!(
        GLOBAL_SETTINGS
            .read()
            .unwrap()
            .as_ref()
            .unwrap()
            .lm
            .base_url,
        "https://api.openai.com/v1".to_string()
    );
}
