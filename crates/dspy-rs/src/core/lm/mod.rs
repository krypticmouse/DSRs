pub mod chat;
pub mod client_registry;
pub mod usage;

pub use chat::*;
pub use client_registry::*;
pub use usage::*;

use anyhow::Result;
use rig::{completion::AssistantContent, message::ToolCall, message::ToolChoice, tool::ToolDyn};

use bon::Builder;
use std::{collections::HashMap, sync::Arc};
use tokio::sync::Mutex;
use tracing::{Instrument, debug, trace, warn};

use crate::utils::cache::CacheEntry;
use crate::{Cache, Prediction, RawExample, ResponseCache};

#[derive(Clone, Debug)]
pub struct LMResponse {
    /// Assistant message chosen by the provider.
    pub output: Message,
    /// Token usage reported by the provider for this call.
    pub usage: LmUsage,
    /// Chat history including the freshly appended assistant response.
    pub chat: Chat,
    /// Tool calls made by the provider.
    pub tool_calls: Vec<ToolCall>,
    /// Tool executions made by the provider.
    pub tool_executions: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ToolLoopMode {
    Auto,
    CallerManaged,
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
    #[builder(default = 10)]
    pub max_tool_iterations: u32,
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
            max_tool_iterations: self.max_tool_iterations,
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
    #[tracing::instrument(
        name = "dsrs.lm.initialize_client",
        level = "debug",
        skip(self),
        fields(
            model = %self.model,
            base_url_present = self.base_url.is_some(),
            api_key_present = self.api_key.is_some(),
            cache_enabled = self.cache,
            max_tokens = self.max_tokens,
            temperature = self.temperature,
            max_tool_iterations = self.max_tool_iterations
        )
    )]
    async fn initialize_client(mut self) -> Result<Self> {
        // Determine which build case based on what's provided
        let client = match (&self.base_url, &self.api_key, &self.model) {
            // Case 1: OpenAI-compatible with authentication (base_url + api_key)
            // For custom OpenAI-compatible APIs that require API keys
            (Some(base_url), Some(api_key), _) => {
                debug!(build_case = 1, "using openai-compatible client with auth");
                Arc::new(LMClient::from_openai_compatible(
                    base_url,
                    api_key,
                    &self.model,
                )?)
            }
            // Case 2: Local OpenAI-compatible server (base_url only, no api_key)
            // For vLLM, text-generation-inference, and other local OpenAI-compatible servers
            (Some(base_url), None, _) => {
                debug!(build_case = 2, "using local openai-compatible client");
                Arc::new(LMClient::from_local(base_url, &self.model)?)
            }
            // Case 3: Provider via model string (no base_url, model in "provider:model" format)
            // Uses provider-specific clients
            (None, api_key, model) if model.contains(':') => {
                debug!(build_case = 3, "using provider:model client");
                Arc::new(LMClient::from_model_string(model, api_key.as_deref())?)
            }
            // Default case: assume OpenAI provider if no colon in model name
            (None, api_key, model) => {
                debug!(build_case = 4, "defaulting model to openai provider");
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
            debug!("initializing response cache");
            self.cache_handler = Some(Arc::new(Mutex::new(ResponseCache::new().await)));
        }

        debug!("lm client initialized");
        Ok(self)
    }

    pub async fn with_client(self, client: LMClient) -> Result<Self> {
        Ok(LM {
            client: Some(Arc::new(client)),
            ..self
        })
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
    #[tracing::instrument(name = "dsrs.lm.build", level = "debug", skip(self))]
    pub async fn build(self) -> Result<LM> {
        let lm = self.__internal_build();
        debug!(
            model = %lm.model,
            base_url_present = lm.base_url.is_some(),
            api_key_present = lm.api_key.is_some(),
            cache_enabled = lm.cache,
            "building lm"
        );
        lm.initialize_client().await
    }
}

struct ToolLoopResult {
    message: Message,
    chat_history: Vec<rig::message::Message>,
    tool_calls: Vec<ToolCall>,
    tool_executions: Vec<String>,
}

/// What the model actually wants to do, extracted from a potentially multi-block response.
/// Reasoning blocks are preserved in `full_content` for faithful history replay.
enum ChoiceAction {
    /// Terminal text response (possibly preceded by reasoning).
    Text(String),
    /// One or more tool calls to execute. Carries the full `OneOrMany` so
    /// reasoning blocks are preserved when we push the assistant turn into
    /// chat history. Supports parallel tool calling (Anthropic multi-tool-use,
    /// OpenAI parallel function calls).
    ToolCalls {
        calls: Vec<ToolCall>,
        full_content: Box<rig::OneOrMany<AssistantContent>>,
        assistant_text: Option<String>,
    },
}

/// Scan all content blocks in a response to find actionable items.
/// Anthropic returns `[Reasoning, ToolCall]` or `[Reasoning, Text]`;
/// OpenAI Responses API returns `[Reasoning, FunctionCall]`.
/// Multiple tool calls in one response are supported (parallel tool calling).
fn classify_choice(choice: rig::OneOrMany<AssistantContent>) -> ChoiceAction {
    let mut text: Option<String> = None;
    let mut tool_calls: Vec<ToolCall> = Vec::new();

    for item in choice.iter() {
        match item {
            AssistantContent::ToolCall(tc) => {
                tool_calls.push(tc.clone());
            }
            AssistantContent::Text(t) => {
                text = Some(t.text.clone());
            }
            AssistantContent::Reasoning(_) | AssistantContent::Image(_) => {}
        }
    }

    if !tool_calls.is_empty() {
        return ChoiceAction::ToolCalls {
            calls: tool_calls,
            full_content: Box::new(choice),
            assistant_text: text,
        };
    }

    if let Some(t) = text {
        return ChoiceAction::Text(t);
    }

    // Fallback: only reasoning blocks — extract display text
    let display = choice
        .iter()
        .filter_map(|item| match item {
            AssistantContent::Reasoning(r) => Some(r.display_text()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    ChoiceAction::Text(display)
}

/// Look up a tool by name in the tool list and execute it.
/// Returns `(result_string, was_found)`.
async fn find_and_execute_tool(
    tools: &mut [Arc<dyn ToolDyn>],
    tool_name: &str,
    args: &str,
) -> Result<(String, bool)> {
    for tool in tools.iter_mut() {
        let def = tool.definition("".to_string()).await;
        if def.name == tool_name {
            let result = tool.call(args.to_string()).await?;
            return Ok((result, true));
        }
    }
    Ok((format!("Tool '{}' not found", tool_name), false))
}

impl LM {
    fn chat_from_rig_history(system_prompt: &str, history: &[rig::message::Message]) -> Chat {
        let mut chat = Chat::new(Vec::new());
        if !system_prompt.is_empty() {
            chat.push_message(Message::system(system_prompt.to_string()));
        }
        for message in history {
            chat.push_message(Message::from(message.clone()));
        }
        chat
    }

    /// Execute all tool calls in a batch, returning results paired with their calls.
    async fn execute_tool_batch(
        tools: &mut [Arc<dyn ToolDyn>],
        calls: &[ToolCall],
        context: &str,
    ) -> Result<Vec<(ToolCall, String)>> {
        let mut results = Vec::with_capacity(calls.len());
        for tc in calls {
            let (result, found) =
                find_and_execute_tool(tools, &tc.function.name, &tc.function.arguments.to_string())
                    .await
                    .map_err(|err| {
                        anyhow::anyhow!(
                            "tool `{}` execution failed ({}): {:?}",
                            tc.function.name,
                            context,
                            err
                        )
                    })?;
            if !found {
                warn!(tool = %tc.function.name, context, "tool not found");
            }
            trace!(tool = %tc.function.name, result_len = result.len(), "tool executed");
            results.push((tc.clone(), result));
        }
        Ok(results)
    }

    /// Push tool results into chat history as a single User message.
    fn push_tool_results(
        chat_history: &mut Vec<rig::message::Message>,
        results: &[(ToolCall, String)],
    ) {
        use rig::OneOrMany;
        use rig::message::UserContent;

        let tool_result_contents: Vec<UserContent> = results
            .iter()
            .map(|(tc, result)| {
                if let Some(call_id) = &tc.call_id {
                    UserContent::tool_result_with_call_id(
                        tc.id.clone(),
                        call_id.clone(),
                        OneOrMany::one(result.clone().into()),
                    )
                } else {
                    UserContent::tool_result(tc.id.clone(), OneOrMany::one(result.clone().into()))
                }
            })
            .collect();

        chat_history.push(rig::message::Message::User {
            content: OneOrMany::many(tool_result_contents).expect("results should not be empty"),
        });
    }

    #[tracing::instrument(
        name = "dsrs.lm.tools.loop",
        level = "debug",
        skip(
            self,
            initial_calls,
            initial_assistant_content,
            tools,
            tool_definitions,
            chat_history,
            system_prompt,
            accumulated_usage
        ),
        fields(
            initial_tool_count = initial_calls.len(),
            max_iterations = self.max_tool_iterations as usize
        )
    )]
    #[allow(clippy::too_many_arguments)]
    async fn execute_tool_loop(
        &self,
        initial_calls: &[ToolCall],
        initial_assistant_content: rig::OneOrMany<AssistantContent>,
        mut tools: Vec<Arc<dyn ToolDyn>>,
        tool_definitions: Vec<rig::completion::ToolDefinition>,
        mut chat_history: Vec<rig::message::Message>,
        system_prompt: String,
        accumulated_usage: &mut LmUsage,
    ) -> Result<ToolLoopResult> {
        use rig::OneOrMany;
        use rig::completion::CompletionRequest;

        let max_iterations = self.max_tool_iterations as usize;
        let mut all_tool_calls = Vec::new();
        let mut all_tool_executions = Vec::new();

        // Execute the initial tool call batch
        debug!(count = initial_calls.len(), "executing initial tool calls");
        let results = Self::execute_tool_batch(&mut tools, initial_calls, "initial").await?;
        for (tc, result) in &results {
            all_tool_calls.push(tc.clone());
            all_tool_executions.push(result.clone());
        }

        // Add initial assistant turn to history, preserving ALL content blocks
        // (reasoning + tool calls) so providers like Anthropic get their thinking
        // blocks back with signatures intact.
        chat_history.push(rig::message::Message::Assistant {
            id: None,
            content: initial_assistant_content,
        });
        Self::push_tool_results(&mut chat_history, &results);

        // Now loop until we get a text response
        for iteration in 1..max_iterations {
            let request = CompletionRequest {
                model: None,
                preamble: Some(system_prompt.clone()),
                chat_history: if chat_history.len() == 1 {
                    OneOrMany::one(chat_history.clone().into_iter().next().unwrap())
                } else {
                    OneOrMany::many(chat_history.clone()).expect("chat_history should not be empty")
                },
                documents: Vec::new(),
                tools: tool_definitions.clone(),
                temperature: Some(self.temperature as f64),
                max_tokens: Some(self.max_tokens as u64),
                tool_choice: Some(ToolChoice::Auto),
                additional_params: None,
                output_schema: None,
            };

            let response = self
                .client
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("LM client not initialized"))?
                .completion(request)
                .await?;

            accumulated_usage.prompt_tokens += response.usage.input_tokens;
            accumulated_usage.completion_tokens += response.usage.output_tokens;
            accumulated_usage.total_tokens += response.usage.total_tokens;
            debug!(
                iteration,
                prompt_tokens = accumulated_usage.prompt_tokens,
                completion_tokens = accumulated_usage.completion_tokens,
                total_tokens = accumulated_usage.total_tokens,
                "tool loop usage updated"
            );

            // Scan ALL content blocks — don't just look at .first(), since
            // responses can be [Reasoning, ToolCall] or [Reasoning, Text].
            match classify_choice(response.choice) {
                ChoiceAction::Text(text) => {
                    debug!(iteration, "tool loop completed with text");
                    return Ok(ToolLoopResult {
                        message: Message::assistant(&text),
                        chat_history,
                        tool_calls: all_tool_calls,
                        tool_executions: all_tool_executions,
                    });
                }
                ChoiceAction::ToolCalls {
                    calls,
                    full_content,
                    ..
                } => {
                    let context = format!("iteration {}", iteration);
                    debug!(iteration, count = calls.len(), "executing tool calls");
                    let results = Self::execute_tool_batch(&mut tools, &calls, &context).await?;
                    for (tc, result) in &results {
                        all_tool_calls.push(tc.clone());
                        all_tool_executions.push(result.clone());
                    }

                    // Preserve full content (reasoning + tool calls) in history
                    chat_history.push(rig::message::Message::Assistant {
                        id: None,
                        content: *full_content,
                    });
                    Self::push_tool_results(&mut chat_history, &results);
                }
            }
        }

        warn!(max_iterations, "max tool iterations reached");
        Err(anyhow::anyhow!("Max tool iterations reached"))
    }

    pub async fn call(&self, messages: Chat, tools: Vec<Arc<dyn ToolDyn>>) -> Result<LMResponse> {
        self.call_with_tool_loop_mode(messages, tools, ToolLoopMode::Auto)
            .await
    }

    #[tracing::instrument(
        name = "dsrs.lm.call_with_tool_loop_mode",
        level = "debug",
        skip(self, messages, tools),
        fields(
            model = %self.model,
            message_count = messages.len(),
            tool_count = tools.len(),
            cache_enabled = self.cache,
            tool_loop_mode = ?tool_loop_mode
        )
    )]
    pub async fn call_with_tool_loop_mode(
        &self,
        messages: Chat,
        tools: Vec<Arc<dyn ToolDyn>>,
        tool_loop_mode: ToolLoopMode,
    ) -> Result<LMResponse> {
        use rig::OneOrMany;
        use rig::completion::CompletionRequest;
        let system_prompt = messages.system_prompt();
        let chat_history = messages.to_rig_chat_history();

        let mut tool_definitions = Vec::new();
        for tool in &tools {
            tool_definitions.push(tool.definition("".to_string()).await);
        }
        trace!(
            conversation_messages = chat_history.len(),
            tool_definitions = tool_definitions.len(),
            "prepared completion request inputs"
        );

        let request = CompletionRequest {
            model: None,
            preamble: Some(system_prompt.clone()),
            chat_history: if chat_history.len() == 1 {
                OneOrMany::one(chat_history.clone().into_iter().next().unwrap())
            } else {
                OneOrMany::many(chat_history.clone()).expect("chat_history should not be empty")
            },
            documents: Vec::new(),
            tools: tool_definitions.clone(),
            temperature: Some(self.temperature as f64),
            max_tokens: Some(self.max_tokens as u64),
            tool_choice: if !tool_definitions.is_empty() {
                Some(ToolChoice::Auto)
            } else {
                None
            },
            additional_params: None,
            output_schema: None,
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
        debug!(
            prompt_tokens = response.usage.input_tokens,
            completion_tokens = response.usage.output_tokens,
            total_tokens = response.usage.total_tokens,
            "lm completion received"
        );

        let mut accumulated_usage = LmUsage::from(response.usage);

        // Scan ALL content blocks in the response — don't just look at .first().
        // Responses can be [Reasoning, ToolCall] or [Reasoning, Text].
        let mut tool_loop_result = None;
        let mut returned_tool_calls = Vec::new();
        let mut assistant_content_for_history: Option<rig::OneOrMany<AssistantContent>> = None;
        let mut output_override: Option<Message> = None;
        let mut append_output_after_history = false;
        let classified = classify_choice(response.choice.clone());
        let first_choice = match classified {
            ChoiceAction::Text(text) => Message::assistant(&text),
            ChoiceAction::ToolCalls {
                calls,
                full_content,
                assistant_text: _,
            } if tool_loop_mode == ToolLoopMode::Auto && !tools.is_empty() => {
                debug!(count = calls.len(), "entering tool loop");
                let result = self
                    .execute_tool_loop(
                        &calls,
                        *full_content,
                        tools,
                        tool_definitions,
                        chat_history,
                        system_prompt.clone(),
                        &mut accumulated_usage,
                    )
                    .await?;
                let message = result.message.clone();
                tool_loop_result = Some(result);
                append_output_after_history = true;
                message
            }
            ChoiceAction::ToolCalls { calls, .. }
                if tool_loop_mode == ToolLoopMode::Auto && tools.is_empty() =>
            {
                let names: Vec<_> = calls.iter().map(|tc| tc.function.name.as_str()).collect();
                warn!(?names, "tools requested but no tools available");
                let msg = format!("Tool calls requested: {:?}, but no tools available", names);
                assistant_content_for_history = Some(rig::OneOrMany::many(
                    calls
                        .into_iter()
                        .map(AssistantContent::ToolCall)
                        .collect::<Vec<_>>(),
                )?);
                append_output_after_history = true;
                Message::assistant(&msg)
            }
            ChoiceAction::ToolCalls {
                calls,
                assistant_text,
                full_content,
            } => {
                returned_tool_calls = calls;
                let content = *full_content;
                assistant_content_for_history = Some(content.clone());
                output_override = Some(Message::from(rig::message::Message::Assistant {
                    id: None,
                    content,
                }));
                Message::assistant(assistant_text.unwrap_or_default())
            }
        };
        let output = output_override.unwrap_or_else(|| first_choice.clone());

        let mut full_chat = if let Some(result) = tool_loop_result.as_ref() {
            Self::chat_from_rig_history(&system_prompt, &result.chat_history)
        } else {
            let mut chat = messages.clone();
            if let Some(content) = assistant_content_for_history {
                // Convert grouped rig content into a single grouped Message.
                let rig_msg = rig::message::Message::Assistant { id: None, content };
                chat.push_message(Message::from(rig_msg));
            } else {
                // Text-only path: preserve a single assistant response turn.
                chat.push_message(first_choice.clone());
            }
            chat
        };
        if append_output_after_history {
            full_chat.push_message(first_choice.clone());
        }
        debug!(
            tool_calls = tool_loop_result
                .as_ref()
                .map(|result| result.tool_calls.len())
                .unwrap_or(0),
            tool_executions = tool_loop_result
                .as_ref()
                .map(|result| result.tool_executions.len())
                .unwrap_or(0),
            total_tokens = accumulated_usage.total_tokens,
            "lm call completed"
        );

        Ok(LMResponse {
            output,
            usage: accumulated_usage,
            chat: full_chat,
            tool_calls: tool_loop_result
                .as_ref()
                .map(|result| result.tool_calls.clone())
                .unwrap_or(returned_tool_calls),
            tool_executions: tool_loop_result
                .map(|result| result.tool_executions)
                .unwrap_or_default(),
        })
    }

    /// Returns the `n` most recent cached calls.
    ///
    /// Panics if caching is disabled for this `LM`.
    #[tracing::instrument(
        name = "dsrs.lm.inspect_history",
        level = "trace",
        skip(self),
        fields(n)
    )]
    pub async fn inspect_history(&self, n: usize) -> Vec<CacheEntry> {
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
    #[tracing::instrument(
        name = "dsrs.lm.dummy_call",
        level = "debug",
        skip(self, example, messages, prediction),
        fields(
            cache_enabled = self.cache,
            cache_handler_present = self.cache_handler.is_some(),
            message_count = messages.len()
        )
    )]
    pub async fn call(
        &self,
        example: RawExample,
        messages: Chat,
        prediction: String,
    ) -> Result<LMResponse> {
        let mut full_chat = messages.clone();
        full_chat.push_message(Message::assistant(prediction.clone()));

        if self.cache
            && let Some(cache) = self.cache_handler.as_ref()
        {
            let (tx, rx) = tokio::sync::mpsc::channel(1);
            let cache_clone = cache.clone();
            let example_clone = example.clone();

            // Spawn the cache insert operation to avoid deadlock
            tokio::spawn(
                async move {
                    let _ = cache_clone.lock().await.insert(example_clone, rx).await;
                }
                .instrument(tracing::Span::current()),
            );
            debug!("spawned async cache insert");

            // Send the result to the cache
            tx.send(CacheEntry {
                prompt: messages.to_json().to_string(),
                prediction: Prediction::new(
                    HashMap::from([("prediction".to_string(), prediction.clone().into())]),
                    LmUsage::default(),
                ),
            })
            .await
            .map_err(|_| anyhow::anyhow!("Failed to send to cache"))?;
            trace!("sent dummy response to cache task");
        }

        Ok(LMResponse {
            output: Message::assistant(prediction.clone()),
            usage: LmUsage::default(),
            chat: full_chat,
            tool_calls: Vec::new(),
            tool_executions: Vec::new(),
        })
    }

    /// Returns cached entries just like [`LM::inspect_history`].
    #[tracing::instrument(
        name = "dsrs.lm.dummy.inspect_history",
        level = "trace",
        skip(self),
        fields(n)
    )]
    pub async fn inspect_history(&self, n: usize) -> Vec<CacheEntry> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use rig::OneOrMany;
    use rig::completion::AssistantContent;
    use rig::completion::ToolDefinition;
    use rig::tool::Tool;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn make_tool_call(name: &str) -> AssistantContent {
        AssistantContent::tool_call(
            format!("id_{name}"),
            name.to_string(),
            serde_json::json!({"arg": "val"}),
        )
    }

    fn make_reasoning(text: &str) -> AssistantContent {
        AssistantContent::reasoning(text)
    }

    fn make_text(text: &str) -> AssistantContent {
        AssistantContent::text(text)
    }

    #[test]
    fn classify_text_only() {
        let choice = OneOrMany::one(make_text("hello"));
        match classify_choice(choice) {
            ChoiceAction::Text(t) => assert_eq!(t, "hello"),
            ChoiceAction::ToolCalls { .. } => panic!("expected Text, got ToolCalls"),
        }
    }

    #[test]
    fn classify_single_tool_call() {
        let choice = OneOrMany::one(make_tool_call("search"));
        match classify_choice(choice) {
            ChoiceAction::ToolCalls {
                calls,
                full_content,
                assistant_text,
            } => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].function.name, "search");
                assert_eq!(full_content.iter().count(), 1);
                assert!(assistant_text.is_none());
            }
            ChoiceAction::Text(_) => panic!("expected ToolCalls, got Text"),
        }
    }

    #[test]
    fn classify_reasoning_then_tool_call() {
        let choice = OneOrMany::many(vec![
            make_reasoning("thinking..."),
            make_tool_call("search"),
        ])
        .unwrap();

        match classify_choice(choice) {
            ChoiceAction::ToolCalls {
                calls,
                full_content,
                assistant_text,
            } => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].function.name, "search");
                // full_content preserves both blocks
                assert_eq!(full_content.iter().count(), 2);
                assert!(assistant_text.is_none());
            }
            ChoiceAction::Text(_) => panic!("expected ToolCalls, got Text"),
        }
    }

    #[test]
    fn classify_reasoning_then_text() {
        let choice = OneOrMany::many(vec![
            make_reasoning("let me think"),
            make_text("the answer is 42"),
        ])
        .unwrap();

        match classify_choice(choice) {
            ChoiceAction::Text(t) => assert_eq!(t, "the answer is 42"),
            ChoiceAction::ToolCalls { .. } => panic!("expected Text, got ToolCalls"),
        }
    }

    #[test]
    fn classify_reasoning_only_fallback() {
        let choice = OneOrMany::one(make_reasoning("just thinking"));
        match classify_choice(choice) {
            ChoiceAction::Text(t) => assert_eq!(t, "just thinking"),
            ChoiceAction::ToolCalls { .. } => panic!("expected Text, got ToolCalls"),
        }
    }

    #[test]
    fn classify_tool_call_wins_over_text() {
        let choice =
            OneOrMany::many(vec![make_text("some text"), make_tool_call("search")]).unwrap();

        match classify_choice(choice) {
            ChoiceAction::ToolCalls {
                calls,
                assistant_text,
                ..
            } => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].function.name, "search");
                assert_eq!(assistant_text.as_deref(), Some("some text"));
            }
            ChoiceAction::Text(_) => panic!("expected ToolCalls, got Text"),
        }
    }

    #[test]
    fn classify_multiple_tool_calls() {
        let choice = OneOrMany::many(vec![
            make_reasoning("planning"),
            make_tool_call("search"),
            make_tool_call("calculate"),
        ])
        .unwrap();

        match classify_choice(choice) {
            ChoiceAction::ToolCalls {
                calls,
                full_content,
                assistant_text,
            } => {
                assert_eq!(calls.len(), 2);
                assert_eq!(calls[0].function.name, "search");
                assert_eq!(calls[1].function.name, "calculate");
                assert_eq!(full_content.iter().count(), 3);
                assert!(assistant_text.is_none());
            }
            ChoiceAction::Text(_) => panic!("expected ToolCalls, got Text"),
        }
    }

    #[test]
    fn classify_image_only_fallback() {
        let choice = OneOrMany::one(AssistantContent::Image(
            rig::completion::message::Image::default(),
        ));
        match classify_choice(choice) {
            ChoiceAction::Text(t) => assert!(t.is_empty()),
            ChoiceAction::ToolCalls { .. } => panic!("expected Text, got ToolCalls"),
        }
    }

    #[derive(Clone)]
    struct CountingTool {
        calls: Arc<AtomicUsize>,
    }

    #[derive(Debug)]
    struct CountingToolError;

    impl std::fmt::Display for CountingToolError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "counting tool error")
        }
    }

    impl std::error::Error for CountingToolError {}

    impl Tool for CountingTool {
        const NAME: &'static str = "counter";
        type Error = CountingToolError;
        type Args = serde_json::Value;
        type Output = String;

        async fn definition(&self, _prompt: String) -> ToolDefinition {
            ToolDefinition {
                name: Self::NAME.to_string(),
                description: "counter tool".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "additionalProperties": true
                }),
            }
        }

        async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok("counted".to_string())
        }
    }

    fn test_lm_with_model(model: TestCompletionModel) -> LM {
        LM {
            base_url: None,
            api_key: None,
            model: "openai:gpt-4o-mini".to_string(),
            temperature: 0.0,
            max_tokens: 128,
            max_tool_iterations: 4,
            cache: false,
            cache_handler: None,
            client: Some(Arc::new(LMClient::Test(model))),
        }
    }

    #[tokio::test]
    async fn call_with_caller_managed_mode_returns_tool_calls_without_executing() {
        let model = TestCompletionModel::new([make_tool_call("counter")]);
        let lm = test_lm_with_model(model);

        let call_count = Arc::new(AtomicUsize::new(0));
        let tools: Vec<Arc<dyn ToolDyn>> = vec![Arc::new(CountingTool {
            calls: Arc::clone(&call_count),
        })];

        let chat = Chat::new(vec![Message::user("Use the counter tool")]);
        let response = lm
            .call_with_tool_loop_mode(chat, tools, ToolLoopMode::CallerManaged)
            .await
            .expect("caller-managed call should succeed");

        assert_eq!(response.tool_calls.len(), 1);
        assert!(response.tool_executions.is_empty());
        assert_eq!(call_count.load(Ordering::SeqCst), 0);
        assert!(response.output.has_tool_calls());
        assert!(response.output.content().contains("counter"));
        assert_eq!(response.chat.len(), 2);
        assert!(response.chat.messages[1].has_tool_calls());
    }

    #[tokio::test]
    async fn call_default_auto_mode_executes_tool_loop() {
        let model = TestCompletionModel::new([make_tool_call("counter"), make_text("done")]);
        let lm = test_lm_with_model(model);

        let call_count = Arc::new(AtomicUsize::new(0));
        let tools: Vec<Arc<dyn ToolDyn>> = vec![Arc::new(CountingTool {
            calls: Arc::clone(&call_count),
        })];

        let chat = Chat::new(vec![Message::user("Use the counter tool")]);
        let response = lm
            .call(chat, tools)
            .await
            .expect("auto call should succeed");

        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_executions.len(), 1);
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
        assert_eq!(response.output.content(), "done");
        assert_eq!(response.chat.len(), 4);
        assert!(response.chat.messages[1].has_tool_calls());
        assert!(response.chat.messages[2].has_tool_results());
        assert_eq!(response.chat.messages[3].role, Role::Assistant);
    }
}
