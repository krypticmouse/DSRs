pub mod chat;
pub mod config;
pub mod usage;

pub use chat::*;
pub use config::*;
pub use usage::*;

use anyhow::Result;
use async_openai::types::CreateChatCompletionRequestArgs;
use async_openai::{Client, config::OpenAIConfig};

use crate::{Cache, CallResult, Example, Prediction, ResponseCache};
use bon::Builder;
use secrecy::{ExposeSecret, SecretString};
use std::{collections::HashMap, sync::Arc};
use tokio::sync::Mutex;

#[derive(Clone, Debug)]
pub struct LMResponse {
    pub output: Message,
    pub usage: LmUsage,
    pub chat: Chat,
}

fn get_base_url_by_provider(provider: &str) -> &str {
    match provider {
        "openai" => "https://api.openai.com/v1",
        "anthropic" => "https://api.anthropic.com/v1",
        "google" => "https://generativelanguage.googleapis.com/v1beta/openai",
        "cohere" => "https://api.cohere.ai/compatibility/v1",
        "groq" => "https://api.groq.com/openai/v1",
        "openrouter" => "https://openrouter.ai/api/v1",
        "qwen" => "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "together" => "https://api.together.xyz/v1",
        "xai" => "https://api.x.ai/v1",
        _ => "https://openrouter.ai/api/v1",
    }
}

#[derive(Builder)]
#[builder(finish_fn(vis = "", name = build_internal))]
pub struct LM {
    #[builder(getter)]
    pub api_key: SecretString,
    #[builder(default = "https://api.openai.com/v1".to_string(), getter)]
    pub base_url: String,
    #[builder(default = LMConfig::default(), getter)]
    pub config: LMConfig,
    client: Option<Client<OpenAIConfig>>,
    pub cache_handler: Option<Arc<Mutex<ResponseCache>>>,
}

impl Clone for LM {
    fn clone(&self) -> Self {
        Self {
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            config: self.config.clone(),
            client: self.client.clone(),
            cache_handler: self.cache_handler.clone(),
        }
    }
}

use l_m_builder::{IsSet, IsUnset, State};

impl<S: State> LMBuilder<S> {
    pub async fn build(self) -> LM
    where
        S::ApiKey: IsSet,
        S::Client: IsUnset,
        S::CacheHandler: IsUnset,
    {
        let mut lm = self.build_internal();

        if lm.config.model.contains("/") {
            let model_str = lm.config.model.clone();
            let (provider, model_id) = model_str.split_once("/").unwrap();
            lm.config.model = model_id.to_string();
            lm.base_url = get_base_url_by_provider(provider).to_string();
        }

        let openai_config = OpenAIConfig::new()
            .with_api_key(lm.api_key.expose_secret().to_string())
            .with_api_base(lm.base_url.clone());
        let client = Client::with_config(openai_config);
        lm.client = Some(client);

        if lm.config.cache {
            let cache_handler = Arc::new(Mutex::new(ResponseCache::new().await));
            lm.cache_handler = Some(cache_handler);
        }
        lm
    }
}

impl LM {
    pub async fn call(&self, messages: Chat) -> Result<LMResponse> {
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

        let mut full_chat = messages.clone();
        full_chat.push_message(first_choice.clone());

        Ok(LMResponse {
            output: first_choice,
            usage,
            chat: full_chat,
        })
    }

    pub async fn inspect_history(&self, n: usize) -> Vec<CallResult> {
        self.cache_handler
            .as_ref()
            .unwrap()
            .lock()
            .await
            .get_history(n)
            .await
            .unwrap()
    }
}

#[derive(Clone, Builder, Default)]
pub struct DummyLM {
    pub api_key: SecretString,
    #[builder(default = "https://api.openai.com/v1".to_string())]
    pub base_url: String,
    #[builder(default = LMConfig::default())]
    pub config: LMConfig,
    pub cache_handler: Option<Arc<Mutex<ResponseCache>>>,
}

impl DummyLM {
    pub async fn new() -> Self {
        let cache_handler = Arc::new(Mutex::new(ResponseCache::new().await));
        Self {
            api_key: "".into(),
            base_url: "https://api.openai.com/v1".to_string(),
            config: LMConfig::default(),
            cache_handler: Some(cache_handler),
        }
    }

    pub async fn call(
        &self,
        example: Example,
        messages: Chat,
        prediction: String,
    ) -> Result<LMResponse> {
        let mut full_chat = messages.clone();
        full_chat.push_message(Message::Assistant {
            content: prediction.clone(),
        });

        if self.config.cache
            && let Some(cache) = self.cache_handler.as_ref()
        {
            let (tx, rx) = tokio::sync::mpsc::channel(1);
            let cache_clone = cache.clone();
            let example_clone = example.clone();

            // Spawn the cache insert operation to avoid deadlock
            tokio::spawn(async move {
                let _ = cache_clone.lock().await.insert(example_clone, rx).await;
            });

            // Send the result to the cache
            tx.send(CallResult {
                prompt: messages.to_json().to_string(),
                prediction: Prediction::new(
                    HashMap::from([("prediction".to_string(), prediction.clone().into())]),
                    LmUsage::default(),
                ),
            })
            .await
            .map_err(|_| anyhow::anyhow!("Failed to send to cache"))?;
        }

        Ok(LMResponse {
            output: Message::Assistant {
                content: prediction.clone(),
            },
            usage: LmUsage::default(),
            chat: full_chat,
        })
    }

    pub async fn inspect_history(&self, n: usize) -> Vec<CallResult> {
        self.cache_handler
            .as_ref()
            .unwrap()
            .lock()
            .await
            .get_history(n)
            .await
            .unwrap()
    }
}
