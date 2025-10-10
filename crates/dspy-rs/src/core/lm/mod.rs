pub mod chat;
pub mod config;
pub mod usage;

pub use chat::*;
pub use config::*;
pub use usage::*;

use anyhow::Result;
use async_openai::types::CreateChatCompletionRequestArgs;
use async_openai::{Client, config::OpenAIConfig};

use crate::utils::cache::{Cache, ResponseCache};
use bon::Builder;
use secrecy::{ExposeSecretMut, SecretString};

#[derive(Clone, Debug)]
pub struct LMResponse {
    pub chat: Chat,
    pub config: LMConfig,
    pub output: Message,
    pub signature: String,
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

#[derive(Clone, Builder)]
pub struct LM {
    pub api_key: SecretString,
    #[builder(default = "https://api.openai.com/v1".to_string())]
    pub base_url: String,
    #[builder(default = LMConfig::default())]
    pub config: LMConfig,
    #[builder(default = Vec::new())]
    pub history: Vec<LMResponse>,
    client: Option<Client<OpenAIConfig>>,
    pub cache_handler: Option<ResponseCache>,
}

impl LM {
    pub async fn setup_client(&mut self) {
        let config = OpenAIConfig::new()
            .with_api_key(self.api_key.expose_secret_mut().to_string())
            .with_api_base(self.base_url.clone());

        self.client = Some(Client::with_config(config));

        if self.config.cache {
            self.cache_handler = Some(ResponseCache::new().await);
        }
    }

    pub async fn call(&mut self, messages: Chat, signature: &str) -> Result<(Message, LmUsage)> {
        if self.client.is_none() {
            if self.config.model.contains("/") {
                let model_str = self.config.model.clone();
                let (provider, model_id) = model_str.split_once("/").unwrap();
                self.config.model = model_id.to_string();
                self.base_url = get_base_url(provider);
            }
            self.setup_client().await;
        }

        let request_messages = messages.get_async_openai_messages();

        // Check if we're using a Gemini model
        let is_gemini = self.config.model.starts_with("gemini-");

        // Build the base request
        let mut builder = CreateChatCompletionRequestArgs::default();

        builder
            .model(self.config.model.clone())
            .messages(request_messages)
            .temperature(self.config.temperature)
            .top_p(self.config.top_p)
            .n(self.config.n)
            .max_tokens(self.config.max_tokens)
            .presence_penalty(self.config.presence_penalty);

        // Only add frequency_penalty, seed, and logit_bias for non-Gemini models
        if !is_gemini {
            builder
                .frequency_penalty(self.config.frequency_penalty)
                .seed(self.config.seed)
                .logit_bias(self.config.logit_bias.clone().unwrap_or_default());
        }

        let request = builder.build()?;

        let response = self.client.as_ref().unwrap().chat().create(request).await?;
        let first_choice = Message::from(response.choices.first().unwrap().message.clone());
        let usage = LmUsage::from(response.usage.unwrap());

        self.history.push(LMResponse {
            chat: messages.clone(),
            output: first_choice.clone(),
            config: self.config.clone(),
            signature: signature.to_string(),
        });

        Ok((first_choice, usage))
    }

    pub fn inspect_history(&self, n: usize) -> Vec<LMResponse> {
        self.history.iter().rev().take(n).cloned().collect()
    }
}

#[derive(Clone, Builder, Default)]
pub struct DummyLM {
    pub api_key: SecretString,
    #[builder(default = "https://api.openai.com/v1".to_string())]
    pub base_url: String,
    #[builder(default = LMConfig::default())]
    pub config: LMConfig,
    #[builder(default = Vec::new())]
    pub history: Vec<LMResponse>,
}

impl DummyLM {
    pub async fn call(
        &mut self,
        messages: Chat,
        signature: &str,
        prediction: String,
    ) -> Result<(Message, LmUsage)> {
        self.history.push(LMResponse {
            chat: messages.clone(),
            output: Message::Assistant {
                content: prediction.clone(),
            },
            config: self.config.clone(),
            signature: signature.to_string(),
        });

        Ok((
            Message::Assistant {
                content: prediction.clone(),
            },
            LmUsage::default(),
        ))
    }

    pub fn inspect_history(&self, n: usize) -> Vec<LMResponse> {
        self.history.iter().rev().take(n).cloned().collect()
    }
}
