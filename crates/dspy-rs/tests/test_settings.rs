use dspy_rs::{ChatAdapter, LM, LMConfig, configure, get_lm};

use secrecy::SecretString;

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_settings() {
    configure(
        LM::builder()
            .api_key(SecretString::from("test"))
            .build()
            .await,
        ChatAdapter {},
    );

    let lm = get_lm();
    assert_eq!(lm.config.model, "gpt-4o-mini");
    assert_eq!(lm.base_url, "https://api.openai.com/v1".to_string());

    configure(
        LM::builder()
            .config(LMConfig {
                model: "gpt-4o".to_string(),
                ..LMConfig::default()
            })
            .api_key(SecretString::from("test"))
            .build()
            .await,
        ChatAdapter {},
    );

    let lm = get_lm();

    assert_eq!(lm.config.model, "gpt-4o");
    assert_eq!(lm.base_url, "https://api.openai.com/v1".to_string());
}
