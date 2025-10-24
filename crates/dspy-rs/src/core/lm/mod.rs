pub mod chat;
pub mod client_registry;
pub mod config;
pub mod usage;

pub use chat::*;
pub use client_registry::*;
pub use config::*;
pub use usage::*;

use anyhow::Result;
use rig::completion::AssistantContent;

use bon::Builder;
use std::{collections::HashMap, sync::Arc};
use tokio::sync::Mutex;

use crate::{Cache, CallResult, Example, Prediction, ResponseCache};

#[derive(Clone, Debug)]
pub struct LMResponse {
    /// Assistant message chosen by the provider.
    pub output: Message,
    /// Token usage reported by the provider for this call.
    pub usage: LmUsage,
    /// Chat history including the freshly appended assistant response.
    pub chat: Chat,
}

pub struct LM {
    pub config: LMConfig,
    client: Arc<LMClient>,
    pub cache_handler: Option<Arc<Mutex<ResponseCache>>>,
}

impl Default for LM {
    fn default() -> Self {
        // Use a blocking tokio runtime to call the async new function
        tokio::runtime::Runtime::new()
            .expect("Failed to create tokio runtime")
            .block_on(Self::new(LMConfig::default()))
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
    /// Creates a new LM with the given configuration.
    /// Uses enum dispatch for optimal runtime performance.
    ///
    /// This is an async function because it initializes the cache handler when
    /// `config.cache` is `true`. For synchronous contexts where cache initialization
    /// is not needed, use `new_sync` instead.
    pub async fn new(config: LMConfig) -> Self {
        let client = LMClient::from_model_string(&config.model)
            .expect("Failed to create client from model string");

        let cache_handler = if config.cache {
            Some(Arc::new(Mutex::new(ResponseCache::new().await)))
        } else {
            None
        };

        Self {
            config,
            client: Arc::new(client),
            cache_handler,
        }
    }

    /// Executes a chat completion against the configured provider.
    ///
    /// `messages` must already be formatted as OpenAI-compatible chat turns.
    /// The call returns an [`LMResponse`] containing the assistant output,
    /// token usage, and chat history including the new response.
    pub async fn call(&self, messages: Chat) -> Result<LMResponse> {
        use rig::OneOrMany;
        use rig::completion::CompletionRequest;

        let request_messages = messages.get_rig_messages();

        // Build the completion request manually
        let mut chat_history = request_messages.conversation;
        chat_history.push(request_messages.prompt);

        let request = CompletionRequest {
            preamble: Some(request_messages.system),
            chat_history: if chat_history.len() == 1 {
                OneOrMany::one(chat_history.into_iter().next().unwrap())
            } else {
                OneOrMany::many(chat_history).expect("chat_history should not be empty")
            },
            documents: Vec::new(),
            tools: Vec::new(),
            temperature: Some(self.config.temperature as f64),
            max_tokens: Some(self.config.max_tokens as u64),
            tool_choice: None,
            additional_params: None,
        };

        // Execute the completion using enum dispatch (zero-cost abstraction)
        let response = self.client.completion(request).await?;

        let first_choice = match response.choice.first() {
            AssistantContent::Text(text) => Message::assistant(&text.text),
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

    /// Returns the `n` most recent cached calls.
    ///
    /// Panics if caching is disabled for this `LM`.
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

/// In-memory LM used for deterministic tests and examples.
#[derive(Clone, Builder, Default)]
pub struct DummyLM {
    pub api_key: String,
    #[builder(default = "https://api.openai.com/v1".to_string())]
    pub base_url: String,
    /// Static configuration applied to stubbed responses.
    #[builder(default = LMConfig::default())]
    pub config: LMConfig,
    /// Cache backing storage shared with the real implementation.
    pub cache_handler: Option<Arc<Mutex<ResponseCache>>>,
}

impl DummyLM {
    /// Creates a new [`DummyLM`] with an enabled in-memory cache.
    pub async fn new() -> Self {
        let cache_handler = Arc::new(Mutex::new(ResponseCache::new().await));
        Self {
            api_key: "".into(),
            base_url: "https://api.openai.com/v1".to_string(),
            config: LMConfig::default(),
            cache_handler: Some(cache_handler),
        }
    }

    /// Mimics [`LM::call`] without hitting a remote provider.
    ///
    /// The provided `prediction` becomes the assistant output and is inserted
    /// into the shared cache when caching is enabled.
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

    /// Returns cached entries just like [`LM::inspect_history`].
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
