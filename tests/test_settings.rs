use rstest::rstest;

use dspy_rs::adapter::chat_adapter::ChatAdapter;
use dspy_rs::clients::lm::LM;
use dspy_rs::utils::settings::{SETTINGS, configure_settings};

#[rstest]
fn test_settings() {
    configure_settings(Some(LM::default()), Some(ChatAdapter {}));

    assert_eq!(SETTINGS.lock().unwrap().lm.model, "openai/gpt-4o-mini");
    assert_eq!(
        SETTINGS.lock().unwrap().lm.base_url,
        "https://openrouter.ai/api/v1"
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
        "https://openrouter.ai/api/v1"
    );
}
