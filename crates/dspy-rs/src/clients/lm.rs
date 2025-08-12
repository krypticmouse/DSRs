use async_openai::{
    Client,
    config::OpenAIConfig,
    types::{
        ChatCompletionStreamOptions, CreateChatCompletionRequestArgs, CreateChatCompletionResponse,
        ReasoningEffort,
    },
};
use smart_default::SmartDefault;

use crate::{clients::chat::Chat, data::history::History};
use derive_builder::Builder;
use std::collections::HashMap;
use std::error::Error;

#[derive(Clone, Debug, SmartDefault)]
pub struct LMConfig {
    #[default(Some(0.7))]
    pub temperature: Option<f32>,
    #[default(Some(1.0))]
    pub top_p: Option<f32>,
    #[default(Some(512))]
    pub max_tokens: Option<u32>,
    #[default(Some(512))]
    pub max_completion_tokens: Option<u32>,
    #[default(Some(1))]
    pub n: Option<u8>,
    #[default(Some(0.0))]
    pub presence_penalty: Option<f32>,
    #[default(Some(0.0))]
    pub frequency_penalty: Option<f32>,
    #[default(Some(42))]
    pub seed: Option<i64>,
    pub stream: Option<bool>,
    pub stream_options: Option<ChatCompletionStreamOptions>,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub logit_bias: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Clone, Debug, SmartDefault, Builder)]
pub struct LM {
    pub api_key: String,
    #[default(Some("https://api.openai.com/v1".to_string()))]
    #[builder(default = Some("https://api.openai.com/v1".to_string()))]
    pub base_url: Option<String>,
    #[default("openai/gpt-4o-mini".to_string())]
    #[builder(default = "openai/gpt-4o-mini".to_string())]
    pub model: String,
    #[default(Vec::new())]
    #[builder(default = Vec::new())]
    pub history: Vec<History>,
    #[default(LMConfig::default())]
    #[builder(default = LMConfig::default())]
    pub config: LMConfig,
}

impl LM {
    pub async fn call(
        &mut self,
        chat: &Chat,
        signature: String,
    ) -> Result<CreateChatCompletionResponse, Box<dyn Error>> {
        let (provider, model) = self.model.split_once('/').unwrap();
        let base_url = if let Some(base_url) = self.base_url.clone() {
            base_url
        } else {
            Self::get_base_url(provider)
        };

        let config = OpenAIConfig::default()
            .with_api_base(base_url)
            .with_api_key(self.api_key.clone());

        let client = Client::with_config(config);

        let request = CreateChatCompletionRequestArgs::default()
            .model(model)
            .messages(chat.messages.clone())
            .temperature(self.config.temperature.unwrap_or_default())
            .top_p(self.config.top_p.unwrap_or_default())
            .n(self.config.n.unwrap_or_default())
            .max_completion_tokens(self.config.max_completion_tokens.unwrap_or_default())
            .max_tokens(self.config.max_tokens.unwrap_or_default())
            .presence_penalty(self.config.presence_penalty.unwrap_or_default())
            .frequency_penalty(self.config.frequency_penalty.unwrap_or_default())
            .seed(self.config.seed.unwrap_or_default())
            .stream(self.config.stream.unwrap_or(false))
            .stream_options(
                self.config
                    .stream_options
                    .unwrap_or(ChatCompletionStreamOptions {
                        include_usage: false,
                    }),
            )
            .reasoning_effort(
                self.config
                    .reasoning_effort
                    .clone()
                    .unwrap_or(ReasoningEffort::Low),
            )
            .logit_bias(self.config.logit_bias.clone().unwrap_or_default())
            .build()?;

        let response = client.chat().create(request).await?;

        self.history.push(History {
            input: chat.clone(),
            output: response.clone(),
            signature,
            model: self.model.clone(),
        });

        Ok(response)
    }

    pub fn inspect_history(&self, n: usize) -> Vec<&History> {
        self.history.iter().rev().take(n).collect()
    }

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

    pub fn builder() -> LMBuilder {
        LMBuilder::default()
    }
}
