pub mod chat;
pub mod client_registry;
pub mod usage;

pub use chat::*;
pub use client_registry::*;
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

#[derive(Builder)]
#[builder(finish_fn(vis = "", name = __internal_build))]
pub struct LM {
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    #[builder(default = "openai:gpt-4o-mini".to_string())]
    pub model: String,
    #[builder(default = 0.7)]
    pub temperature: f32,
    #[builder(default = 512)]
    pub max_tokens: u32,
    #[builder(default = false)]
    pub cache: bool,
    pub cache_handler: Option<Arc<Mutex<ResponseCache>>>,
    #[builder(skip)]
    client: Option<Arc<LMClient>>,
}

impl Default for LM {
    fn default() -> Self {
        tokio::runtime::Handle::current().block_on(async { Self::builder().build().await.unwrap() })
    }
}

impl Clone for LM {
    fn clone(&self) -> Self {
        Self {
            base_url: self.base_url.clone(),
            api_key: self.api_key.clone(),
            model: self.model.clone(),
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            cache: self.cache,
            cache_handler: self.cache_handler.clone(),
            client: self.client.clone(),
        }
    }
}

impl LM {
    /// Finalizes construction of an [`LM`], initializing the HTTP client and
    /// optional response cache based on provided parameters.
    ///
    /// Supports 3 build cases:
    /// 1. OpenAI-compatible with auth: `base_url` + `api_key` provided
    ///    → Uses OpenAI client with custom base URL
    /// 2. Local OpenAI-compatible: `base_url` only (no `api_key`)
    ///    → Uses OpenAI client for vLLM/local servers (dummy key)
    /// 3. Provider via model string: no `base_url`, model in "provider:model" format
    ///    → Uses provider-specific client (openai, anthropic, gemini, etc.)
    async fn initialize_client(mut self) -> Result<Self> {
        // Determine which build case based on what's provided
        let client = match (&self.base_url, &self.api_key, &self.model) {
            // Case 1: OpenAI-compatible with authentication (base_url + api_key)
            // For custom OpenAI-compatible APIs that require API keys
            (Some(base_url), Some(api_key), _) => Arc::new(LMClient::from_openai_compatible(
                base_url,
                api_key,
                &self.model,
            )?),
            // Case 2: Local OpenAI-compatible server (base_url only, no api_key)
            // For vLLM, text-generation-inference, and other local OpenAI-compatible servers
            (Some(base_url), None, _) => Arc::new(LMClient::from_local(base_url, &self.model)?),
            // Case 3: Provider via model string (no base_url, model in "provider:model" format)
            // Uses provider-specific clients
            (None, api_key, model) if model.contains(':') => {
                Arc::new(LMClient::from_model_string(model, api_key.as_deref())?)
            }
            // Default case: assume OpenAI provider if no colon in model name
            (None, api_key, model) => {
                let model_str = if model.contains(':') {
                    model.to_string()
                } else {
                    format!("openai:{}", model)
                };
                Arc::new(LMClient::from_model_string(&model_str, api_key.as_deref())?)
            }
        };

        self.client = Some(client);

        // Initialize cache if enabled
        if self.cache && self.cache_handler.is_none() {
            self.cache_handler = Some(Arc::new(Mutex::new(ResponseCache::new().await)));
        }

        Ok(self)
    }
}

// Implement build() for all builder states since optional fields don't require setting
impl<S: l_m_builder::State> LMBuilder<S> {
    /// Builds the LM instance with proper client initialization
    ///
    /// Supports 3 build cases:
    /// 1. OpenAI-compatible with auth: `base_url` + `api_key` provided
    /// 2. Local OpenAI-compatible: `base_url` only (for vLLM, etc.)
    /// 3. Provider via model string: model in "provider:model" format
    pub async fn build(self) -> Result<LM> {
        let lm = self.__internal_build();
        lm.initialize_client().await
    }
}

impl LM {
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
            temperature: Some(self.temperature as f64),
            max_tokens: Some(self.max_tokens as u64),
            tool_choice: None,
            additional_params: None,
        };

        // Execute the completion using enum dispatch (zero-cost abstraction)
        let response = self
            .client
            .as_ref()
            .ok_or_else(|| {
                anyhow::anyhow!("LM client not initialized. Call build() on LMBuilder.")
            })?
            .completion(request)
            .await?;

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
    #[builder(default = 0.7)]
    pub temperature: f32,
    #[builder(default = 512)]
    pub max_tokens: u32,
    #[builder(default = true)]
    pub cache: bool,
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
            temperature: 0.7,
            max_tokens: 512,
            cache: true,
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

        if self.cache
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
