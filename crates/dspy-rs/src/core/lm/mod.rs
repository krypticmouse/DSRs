pub mod chat;
pub mod config;
pub mod usage;

pub use chat::*;
pub use config::*;
pub use usage::*;

use anyhow::Result;
use rig::{
    client::builder::DynClientBuilder,
    completion::{AssistantContent, CompletionModelDyn},
};

use bon::Builder;
use rig::client::builder::ClientBuildError;
use std::{collections::HashMap, sync::Arc};
use tokio::sync::Mutex;

use crate::{Cache, CallResult, Example, Prediction, ResponseCache};

#[derive(Clone, Debug)]
pub struct LMResponse {
    pub output: Message,
    pub usage: LmUsage,
    pub chat: Chat,
}

pub struct LM {
    pub config: LMConfig,
    client: Arc<Box<dyn CompletionModelDyn>>,
    pub cache_handler: Option<Arc<Mutex<ResponseCache>>>,
}

impl Default for LM {
    fn default() -> Self {
        Self::new(LMConfig::default())
    }
}

impl Clone for LM {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            client: self.client.clone(),
            cache_handler: self.cache_handler.clone(),
        }
    }
}

impl LM {
    pub fn new(config: LMConfig) -> Self {
        let client_builder = DynClientBuilder::default();
        let (provider, model_id) = client_builder.parse(&config.model).unwrap();

        let client = client_builder
            .build(provider)
            .unwrap()
            .as_completion()
            .ok_or(ClientBuildError::UnsupportedFeature(
                provider.to_string(),
                "completion".to_owned(),
            ))
            .unwrap()
            .completion_model(model_id);

        Self {
            config,
            client: Arc::new(client),
            cache_handler: None,
        }
    }

    pub async fn call(&self, messages: Chat) -> Result<LMResponse> {
        let request_messages = messages.get_rig_messages();

        // Build and send the completion request
        let response = self
            .client
            .completion_request(request_messages.prompt)
            .preamble(request_messages.system)
            .messages(request_messages.conversation)
            .temperature(self.config.temperature as f64)
            .max_tokens(self.config.max_tokens as u64)
            .send()
            .await?;

        let first_choice = match response.choice.first() {
            AssistantContent::Text(text) => Message::assistant(text.text),
            AssistantContent::Reasoning(reasoning) => {
                Message::assistant(reasoning.reasoning.join("\n"))
            }
            AssistantContent::ToolCall(_tool_call) => {
                todo!()
            }
        };

        let usage = LmUsage::from(response.usage);

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
    pub api_key: String,
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
