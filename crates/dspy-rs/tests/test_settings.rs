use rstest::rstest;

use dspy_rs::adapter::chat::ChatAdapter;
use dspy_rs::providers::lm::LM;
use dspy_rs::utils::settings::{SETTINGS, configure_settings};

#[rstest]
fn test_settings() {
    configure_settings(Some(LM::default()), Some(ChatAdapter {}));

    assert_eq!(SETTINGS.lock().unwrap().lm.model, "openai/gpt-4o-mini");
    assert_eq!(
        SETTINGS.lock().unwrap().lm.base_url,
        Some("https://api.openai.com/v1".to_string())
    );

    configure_settings(
        Some(LM {
            model: "openai/gpt-4o".to_string(),
            ..LM::default()
        }),
        Some(ChatAdapter {}),
    );

    assert_eq!(SETTINGS.lock().unwrap().lm.model, "openai/gpt-4o");
    assert_eq!(
        SETTINGS.lock().unwrap().lm.base_url,
        Some("https://api.openai.com/v1".to_string())
    );
}
