use crate::core::LMConfig;
use crate::core::lm::LM;
use crate::providers::{ConcreteProvider, OpenAIProvider};
use anyhow::Result;

fn get_base_url(provider: &str) -> String {
    match provider {
        "openai" => "https://api.openai.com/v1".to_string(),
        "anthropic" => "https://api.anthropic.com/v1".to_string(),
        "google" => "https://generativelanguage.googleapis.com/v1beta/openai".to_string(),
        "cohere" => "https://api.cohere.ai/compatibility/v1".to_string(),
        "groq" => "https://api.groq.com/openai/v1".to_string(),
        "openrouter" => "https://openrouter.ai/api/v1".to_string(),
        "qwen" => "https://dashscope-intl.aliyuncs.com/compatible-mode/v1".to_string(),
        "together" => "https://api.together.xyz/v1".to_string(),
        "xai" => "https://api.x.ai/v1".to_string(),
        _ => "https://openrouter.ai/api/v1".to_string(),
    }
}

pub fn openai_lm(api_key: String, model: String) -> Result<LM> {
    let (model_provider, model_name) = model.split_once('/').unwrap();
    let base_url = get_base_url(model_provider);
    let provider = OpenAIProvider::new(api_key, base_url);

    let config = LMConfig::builder().model(model_name.to_string()).build();

    let lm = LM::builder()
        .provider(ConcreteProvider::OpenAI(provider))
        .config(config)
        .build();
    Ok(lm)
}
