pub mod chat;
pub mod client_registry;
pub mod usage;

pub use chat::*;
pub use client_registry::*;
pub use usage::*;

use anyhow::Result;
use rig::{
    completion::{AssistantContent, CompletionError, CompletionRequest, CompletionResponse},
    message::ToolCall,
    message::ToolChoice,
    tool::ToolDyn,
};

use bon::Builder;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc, time::Duration};
use tokio::sync::Mutex;
use tracing::{debug, trace, warn};

use crate::trace::SpanEvent;
use crate::utils::cache::CacheEntry;
use crate::ResponseCache;

#[derive(Clone, Debug)]
pub struct LMResponse {
    /// Assistant message chosen by the provider.
    pub output: Message,
    /// Token usage reported by the provider for this call (aggregate).
    pub usage: LmUsage,
    /// Chat history including the freshly appended assistant response.
    pub chat: Chat,
    /// Tool calls made by the provider. Deprecated by `events`.
    pub tool_calls: Vec<ToolCall>,
    /// Tool executions made by the provider. Deprecated by `events`.
    pub tool_executions: Vec<String>,
    /// Ordered per-round-trip record: one `Exchange` per provider call with
    /// that round-trip's own usage, `ToolRun` entries interleaved in execution
    /// order. Cache-served responses synthesize a single `Exchange`.
    pub events: Vec<SpanEvent>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ToolLoopMode {
    Auto,
    CallerManaged,
}

/// Pre-fetched tool definitions plus name-indexed executors.
///
/// Fetching a rig `ToolDyn` definition costs a boxed-future allocation and a
/// fresh `ToolDefinition` build — doing that per tool on every LM call is pure
/// waste when the tool set is fixed. Build a `ToolSet` once (e.g.
/// [`Predict`](crate::Predict) caches one per instance) and reuse it across calls
/// via [`LM::call_with_toolset`].
#[derive(Clone, Default)]
pub struct ToolSet {
    definitions: Vec<rig::completion::ToolDefinition>,
    by_name: HashMap<String, Arc<dyn ToolDyn>>,
}

impl ToolSet {
    /// Fetches every tool's definition once and indexes the executors by
    /// definition name. Duplicate names keep the first tool.
    pub async fn build(tools: &[Arc<dyn ToolDyn>]) -> Self {
        let mut definitions = Vec::with_capacity(tools.len());
        let mut by_name: HashMap<String, Arc<dyn ToolDyn>> = HashMap::with_capacity(tools.len());
        for tool in tools {
            let definition = tool.definition(String::new()).await;
            by_name
                .entry(definition.name.clone())
                .or_insert_with(|| Arc::clone(tool));
            definitions.push(definition);
        }
        Self {
            definitions,
            by_name,
        }
    }

    /// Builds a `ToolSet` from pre-built definitions with **no executors** —
    /// for caller-managed loops where the caller executes tools itself (the IR
    /// interpreter's `AgentLoop` builds definitions from declared tool
    /// signatures and overlay-resolved descriptions).
    pub fn from_definitions(definitions: Vec<rig::completion::ToolDefinition>) -> Self {
        Self {
            definitions,
            by_name: HashMap::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.definitions.is_empty()
    }

    pub fn definitions(&self) -> &[rig::completion::ToolDefinition] {
        &self.definitions
    }
}

/// The data half of an LM: every generation parameter, no live state.
///
/// This is the serializable artifact — what a program file records, what an
/// optimizer can hash, diff, and mutate. The live half ([`LM`]) is built from
/// it with [`LM::from_config`]. `api_key` is deliberately `#[serde(skip)]`:
/// secrets never serialize, and on load the key is resolved from provider env
/// vars at client initialization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Builder)]
#[builder(finish_fn(vis = "", name = __internal_build))]
pub struct LMConfig {
    pub base_url: Option<String>,
    #[serde(skip)]
    pub api_key: Option<String>,
    #[builder(default = "openai:gpt-4o-mini".to_string())]
    pub model: String,
    #[builder(default = 0.7)]
    pub temperature: f32,
    #[builder(default = 512)]
    pub max_tokens: u32,
    #[builder(default = 10)]
    pub max_tool_iterations: u32,
    /// Additional attempts after a transient failure (429/5xx/network/timeout).
    /// `0` disables retries entirely.
    #[builder(default = 2)]
    pub max_retries: u32,
    /// Base delay for exponential backoff between retries. Attempt `n` waits
    /// `base * 2^n` plus up to 50% random jitter.
    #[builder(default = 250)]
    pub retry_base_delay_ms: u64,
    #[builder(default = false)]
    pub cache: bool,
}

impl Default for LMConfig {
    fn default() -> Self {
        LMConfig::builder().__internal_build()
    }
}

/// The live half: an [`LMConfig`] plus the initialized provider client and
/// response cache. Constructed via [`LM::builder()`] or [`LM::from_config`].
#[derive(Clone)]
pub struct LM {
    pub config: LMConfig,
    pub cache_handler: Option<Arc<Mutex<ResponseCache>>>,
    client: Option<Arc<LMClient>>,
}

impl Default for LM {
    fn default() -> Self {
        tokio::runtime::Handle::current().block_on(async { Self::builder().build().await.unwrap() })
    }
}

impl LM {
    /// Entry point for fluent construction: `LM::builder().model(...).build().await`.
    /// The builder collects an [`LMConfig`]; `build()` initializes the live client.
    pub fn builder() -> LMConfigBuilder {
        LMConfig::builder()
    }

    /// Builds the live [`LM`] from a config, initializing the provider client and
    /// optional response cache.
    ///
    /// Supports 3 build cases:
    /// 1. OpenAI-compatible with auth: `base_url` + `api_key` provided
    ///    → Uses OpenAI client with custom base URL
    /// 2. Local OpenAI-compatible: `base_url` only (no `api_key`)
    ///    → Uses OpenAI client for vLLM/local servers (dummy key)
    /// 3. Provider via model string: no `base_url`, model in "provider:model" format
    ///    → Uses provider-specific client (openai, anthropic, gemini, etc.)
    #[tracing::instrument(
        name = "dsrs.lm.from_config",
        level = "debug",
        skip(config),
        fields(
            model = %config.model,
            base_url_present = config.base_url.is_some(),
            api_key_present = config.api_key.is_some(),
            cache_enabled = config.cache,
            max_tokens = config.max_tokens,
            temperature = config.temperature,
            max_tool_iterations = config.max_tool_iterations
        )
    )]
    pub async fn from_config(config: LMConfig) -> Result<Self> {
        // Determine which build case based on what's provided
        let client = match (&config.base_url, &config.api_key, &config.model) {
            // Case 1: OpenAI-compatible with authentication (base_url + api_key)
            // For custom OpenAI-compatible APIs that require API keys
            (Some(base_url), Some(api_key), _) => {
                debug!(build_case = 1, "using openai-compatible client with auth");
                Arc::new(LMClient::from_openai_compatible(
                    base_url,
                    api_key,
                    &config.model,
                )?)
            }
            // Case 2: Local OpenAI-compatible server (base_url only, no api_key)
            // For vLLM, text-generation-inference, and other local OpenAI-compatible servers
            (Some(base_url), None, _) => {
                debug!(build_case = 2, "using local openai-compatible client");
                Arc::new(LMClient::from_local(base_url, &config.model)?)
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

        let cache_handler = if config.cache {
            debug!("initializing response cache");
            Some(Arc::new(Mutex::new(ResponseCache::new().await)))
        } else {
            None
        };

        debug!("lm client initialized");
        Ok(LM {
            config,
            cache_handler,
            client: Some(client),
        })
    }

    pub async fn with_client(self, client: LMClient) -> Result<Self> {
        Ok(LM {
            client: Some(Arc::new(client)),
            ..self
        })
    }
}

// Implement build() for all builder states since optional fields don't require setting
impl<S: l_m_config_builder::State> LMConfigBuilder<S> {
    /// Finishes the config and initializes the live client — see [`LM::from_config`].
    #[tracing::instrument(name = "dsrs.lm.build", level = "debug", skip(self))]
    pub async fn build(self) -> Result<LM> {
        let config = self.__internal_build();
        debug!(
            model = %config.model,
            base_url_present = config.base_url.is_some(),
            api_key_present = config.api_key.is_some(),
            cache_enabled = config.cache,
            "building lm"
        );
        LM::from_config(config).await
    }
}

struct ToolLoopResult {
    message: Message,
    chat_history: Vec<rig::message::Message>,
    tool_calls: Vec<ToolCall>,
    tool_executions: Vec<String>,
    events: Vec<SpanEvent>,
}

/// One executed tool call with the details the trace format records.
struct ToolRunRecord {
    call: ToolCall,
    result: String,
    duration_us: u64,
    /// Failure reported back to the model as text (e.g. tool not found).
    error: Option<String>,
}

impl ToolRunRecord {
    fn to_event(&self) -> SpanEvent {
        SpanEvent::ToolRun {
            id: self.call.id.clone(),
            name: self.call.function.name.clone(),
            args: self.call.function.arguments.clone(),
            result: self.result.clone(),
            duration_us: self.duration_us,
            error: self.error.clone(),
        }
    }
}

/// Converts grouped rig assistant content into a single [`Message`] for event
/// recording, preserving reasoning and tool-call blocks.
fn assistant_message_from_content(content: &rig::OneOrMany<AssistantContent>) -> Message {
    Message::from(rig::message::Message::Assistant {
        id: None,
        content: content.clone(),
    })
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

/// Whether a rig completion error is worth retrying.
///
/// HTTP-layer failures (connect, timeout) always are. Provider errors are string-typed
/// in rig, so transient markers (429/5xx/overload) are matched textually.
fn is_retryable_completion_error(err: &CompletionError) -> bool {
    match err {
        CompletionError::HttpError(_) => true,
        CompletionError::ProviderError(message) => {
            let message = message.to_ascii_lowercase();
            [
                "429",
                "rate limit",
                "rate_limit",
                "too many requests",
                "overloaded",
                "timeout",
                "timed out",
                "500",
                "502",
                "503",
                "529",
                "server error",
                "internal error",
                "unavailable",
            ]
            .iter()
            .any(|marker| message.contains(marker))
        }
        _ => false,
    }
}

impl LM {
    /// Builds the response-cache key: a streaming hash over the full message
    /// history plus every generation parameter that changes the completion.
    /// Demos and instructions live inside the messages, so they are covered
    /// automatically. Hashing streams through the `Debug` representation — no
    /// intermediate JSON tree or string is materialized. Uses the same stable
    /// hasher as the trace format's `request_hash`.
    fn cache_key_for(&self, messages: &Chat) -> u64 {
        use crate::utils::hash::{HashWriter, StableHasher};
        use std::hash::Hasher;

        let mut hasher = StableHasher::new();
        hasher.write(self.config.model.as_bytes());
        hasher.write(&self.config.temperature.to_bits().to_le_bytes());
        hasher.write(&self.config.max_tokens.to_le_bytes());
        use std::fmt::Write as _;
        let _ = write!(HashWriter(&mut hasher), "{messages:?}");
        hasher.finish()
    }

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

    /// Builds a rig completion request from borrowed parts.
    ///
    /// Kept as a separate builder (rather than constructing once) so the retry
    /// loop can rebuild the request per attempt — the success-first-try path
    /// then pays exactly one build, with no defensive clone.
    fn build_completion_request(
        &self,
        system_prompt: &str,
        chat_history: &[rig::message::Message],
        tool_definitions: &[rig::completion::ToolDefinition],
        tool_choice: Option<ToolChoice>,
    ) -> CompletionRequest {
        use rig::OneOrMany;
        CompletionRequest {
            model: None,
            preamble: Some(system_prompt.to_string()),
            chat_history: if chat_history.len() == 1 {
                OneOrMany::one(chat_history[0].clone())
            } else {
                OneOrMany::many(chat_history.to_vec()).expect("chat_history should not be empty")
            },
            documents: Vec::new(),
            tools: tool_definitions.to_vec(),
            temperature: Some(self.config.temperature as f64),
            max_tokens: Some(self.config.max_tokens as u64),
            tool_choice,
            additional_params: None,
            output_schema: None,
        }
    }

    /// Calls the provider, retrying transient failures with jittered exponential backoff.
    ///
    /// Takes a request *builder* so each attempt constructs its own request from
    /// borrowed parts — no whole-request clone on the common no-retry path.
    async fn completion_with_retry<F>(&self, build_request: F) -> Result<CompletionResponse<()>>
    where
        F: Fn() -> CompletionRequest,
    {
        let client = self
            .client
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("LM client not initialized. Call build() on LMBuilder."))?;

        let mut attempt = 0u32;
        loop {
            match client.completion(build_request()).await {
                Ok(response) => return Ok(response),
                Err(err) if attempt < self.config.max_retries && is_retryable_completion_error(&err) => {
                    let backoff = self
                        .config
                        .retry_base_delay_ms
                        .saturating_mul(1u64 << attempt.min(16));
                    // Scope the RNG so it drops before the await (thread_rng is !Send).
                    let jitter = {
                        use rand::Rng;
                        rand::thread_rng().gen_range(0..=backoff / 2)
                    };
                    let delay = Duration::from_millis(backoff.saturating_add(jitter));
                    warn!(
                        attempt = attempt + 1,
                        max_retries = self.config.max_retries,
                        delay_ms = delay.as_millis() as u64,
                        error = %err,
                        "retrying transient lm completion failure"
                    );
                    tokio::time::sleep(delay).await;
                    attempt += 1;
                }
                Err(err) => return Err(err.into()),
            }
        }
    }

    /// Execute all tool calls in a batch concurrently, returning results paired with
    /// their calls in request order.
    async fn execute_tool_batch(
        tools_by_name: &HashMap<String, Arc<dyn ToolDyn>>,
        calls: &[ToolCall],
        context: &str,
    ) -> Result<Vec<ToolRunRecord>> {
        let executions = calls.iter().map(|tc| {
            let tool = tools_by_name.get(&tc.function.name).cloned();
            async move {
                let started = std::time::Instant::now();
                let (result, error) = match tool {
                    Some(tool) => {
                        let result = tool
                            .call(tc.function.arguments.to_string())
                            .await
                            .map_err(|err| {
                                anyhow::anyhow!(
                                    "tool `{}` execution failed ({}): {:?}",
                                    tc.function.name,
                                    context,
                                    err
                                )
                            })?;
                        (result, None)
                    }
                    None => {
                        warn!(tool = %tc.function.name, context, "tool not found");
                        let message = format!("Tool '{}' not found", tc.function.name);
                        (message.clone(), Some(message))
                    }
                };
                trace!(tool = %tc.function.name, result_len = result.len(), "tool executed");
                Ok::<_, anyhow::Error>(ToolRunRecord {
                    call: tc.clone(),
                    result,
                    duration_us: started.elapsed().as_micros() as u64,
                    error,
                })
            }
        });

        futures::future::try_join_all(executions).await
    }

    /// Push tool results into chat history as a single User message.
    fn push_tool_results(chat_history: &mut Vec<rig::message::Message>, results: &[ToolRunRecord]) {
        use rig::OneOrMany;
        use rig::message::UserContent;

        let tool_result_contents: Vec<UserContent> = results
            .iter()
            .map(|record| {
                let tc = &record.call;
                if let Some(call_id) = &tc.call_id {
                    UserContent::tool_result_with_call_id(
                        tc.id.clone(),
                        call_id.clone(),
                        OneOrMany::one(record.result.clone().into()),
                    )
                } else {
                    UserContent::tool_result(
                        tc.id.clone(),
                        OneOrMany::one(record.result.clone().into()),
                    )
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
            chat_history,
            system_prompt,
            accumulated_usage
        ),
        fields(
            initial_tool_count = initial_calls.len(),
            max_iterations = self.config.max_tool_iterations as usize
        )
    )]
    #[allow(clippy::too_many_arguments)]
    async fn execute_tool_loop(
        &self,
        initial_calls: &[ToolCall],
        initial_assistant_content: rig::OneOrMany<AssistantContent>,
        initial_usage: LmUsage,
        tools: &ToolSet,
        mut chat_history: Vec<rig::message::Message>,
        system_prompt: String,
        accumulated_usage: &mut LmUsage,
    ) -> Result<ToolLoopResult> {
        let max_iterations = self.config.max_tool_iterations as usize;
        let mut all_tool_calls = Vec::new();
        let mut all_tool_executions = Vec::new();
        let mut events = vec![SpanEvent::Exchange {
            message: assistant_message_from_content(&initial_assistant_content),
            usage: initial_usage,
        }];

        // Execute the initial tool call batch
        debug!(count = initial_calls.len(), "executing initial tool calls");
        let results = Self::execute_tool_batch(&tools.by_name, initial_calls, "initial").await?;
        for record in &results {
            all_tool_calls.push(record.call.clone());
            all_tool_executions.push(record.result.clone());
            events.push(record.to_event());
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
            let response = self
                .completion_with_retry(|| {
                    self.build_completion_request(
                        &system_prompt,
                        &chat_history,
                        tools.definitions(),
                        Some(ToolChoice::Auto),
                    )
                })
                .await?;

            let round_usage = LmUsage::from(response.usage);
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
                    let message = Message::assistant(&text);
                    events.push(SpanEvent::Exchange {
                        message: message.clone(),
                        usage: round_usage,
                    });
                    return Ok(ToolLoopResult {
                        message,
                        chat_history,
                        tool_calls: all_tool_calls,
                        tool_executions: all_tool_executions,
                        events,
                    });
                }
                ChoiceAction::ToolCalls {
                    calls,
                    full_content,
                    ..
                } => {
                    events.push(SpanEvent::Exchange {
                        message: assistant_message_from_content(&full_content),
                        usage: round_usage,
                    });
                    let context = format!("iteration {}", iteration);
                    debug!(iteration, count = calls.len(), "executing tool calls");
                    let results = Self::execute_tool_batch(&tools.by_name, &calls, &context).await?;
                    for record in &results {
                        all_tool_calls.push(record.call.clone());
                        all_tool_executions.push(record.result.clone());
                        events.push(record.to_event());
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

    /// [`call`](LM::call) with an explicit tool-loop mode. Builds an ad-hoc
    /// [`ToolSet`] per call — reuse [`call_with_toolset`](LM::call_with_toolset)
    /// with a cached set when calling repeatedly with the same tools.
    pub async fn call_with_tool_loop_mode(
        &self,
        messages: Chat,
        tools: Vec<Arc<dyn ToolDyn>>,
        tool_loop_mode: ToolLoopMode,
    ) -> Result<LMResponse> {
        let toolset = if tools.is_empty() {
            ToolSet::default()
        } else {
            ToolSet::build(&tools).await
        };
        self.call_with_toolset(messages, &toolset, tool_loop_mode)
            .await
    }

    #[tracing::instrument(
        name = "dsrs.lm.call_with_toolset",
        level = "debug",
        skip(self, messages, tools),
        fields(
            model = %self.config.model,
            message_count = messages.len(),
            tool_count = tools.definitions.len(),
            cache_enabled = self.config.cache,
            tool_loop_mode = ?tool_loop_mode
        )
    )]
    pub async fn call_with_toolset(
        &self,
        messages: Chat,
        tools: &ToolSet,
        tool_loop_mode: ToolLoopMode,
    ) -> Result<LMResponse> {
        let system_prompt = messages.system_prompt();
        let chat_history = messages.to_rig_chat_history();

        // Response cache: only tool-free calls are cached — tool loops execute
        // side-effectful user code and must not be replayed from cache.
        let cache_key = if self.config.cache && self.cache_handler.is_some() && tools.is_empty() {
            Some(self.cache_key_for(&messages))
        } else {
            None
        };
        if let (Some(key), Some(cache)) = (cache_key, self.cache_handler.as_ref())
            && let Some(entry) = cache.lock().await.get_entry(key).await?
            && let Some(raw_output) = entry.raw_output
        {
            debug!("lm response served from cache");
            let output = Message::assistant(&raw_output);
            let mut chat = messages;
            chat.push_message(output.clone());
            return Ok(LMResponse {
                output: output.clone(),
                usage: entry.usage,
                chat,
                tool_calls: Vec::new(),
                tool_executions: Vec::new(),
                events: vec![SpanEvent::Exchange {
                    message: output,
                    usage: entry.usage,
                }],
            });
        }

        let tool_definitions = tools.definitions();
        trace!(
            conversation_messages = chat_history.len(),
            tool_definitions = tool_definitions.len(),
            "prepared completion request inputs"
        );

        let tool_choice = if !tool_definitions.is_empty() {
            Some(ToolChoice::Auto)
        } else {
            None
        };

        // Execute the completion using enum dispatch (zero-cost abstraction),
        // retrying transient failures with backoff.
        let response = self
            .completion_with_retry(|| {
                self.build_completion_request(
                    &system_prompt,
                    &chat_history,
                    tool_definitions,
                    tool_choice.clone(),
                )
            })
            .await?;
        debug!(
            prompt_tokens = response.usage.input_tokens,
            completion_tokens = response.usage.output_tokens,
            total_tokens = response.usage.total_tokens,
            "lm completion received"
        );

        let first_usage = LmUsage::from(response.usage);
        let mut accumulated_usage = first_usage;

        // Scan ALL content blocks in the response — don't just look at .first().
        // Responses can be [Reasoning, ToolCall] or [Reasoning, Text].
        let mut tool_loop_result = None;
        let mut returned_tool_calls = Vec::new();
        let mut assistant_content_for_history: Option<rig::OneOrMany<AssistantContent>> = None;
        let mut append_output_after_history = false;
        let mut events: Vec<SpanEvent> = Vec::new();
        let classified = classify_choice(response.choice);
        let first_choice = match classified {
            ChoiceAction::Text(text) => {
                let message = Message::assistant(&text);
                events.push(SpanEvent::Exchange {
                    message: message.clone(),
                    usage: first_usage,
                });
                message
            }
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
                        first_usage,
                        tools,
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
                let content = rig::OneOrMany::many(
                    calls
                        .into_iter()
                        .map(AssistantContent::ToolCall)
                        .collect::<Vec<_>>(),
                )?;
                events.push(SpanEvent::Exchange {
                    message: assistant_message_from_content(&content),
                    usage: first_usage,
                });
                assistant_content_for_history = Some(content);
                append_output_after_history = true;
                Message::assistant(&msg)
            }
            ChoiceAction::ToolCalls {
                calls,
                assistant_text,
                full_content,
            } => {
                events.push(SpanEvent::Exchange {
                    message: assistant_message_from_content(&full_content),
                    usage: first_usage,
                });
                returned_tool_calls = calls;
                assistant_content_for_history = Some(*full_content);
                Message::assistant(assistant_text.unwrap_or_default())
            }
        };

        // Populate the cache on the plain-text path only: `append_output_after_history`
        // covers the tool-loop and "no tools available" fallbacks, and
        // `returned_tool_calls` covers caller-managed tool responses. This runs
        // before `messages` is moved into the returned chat below.
        if let (Some(key), Some(cache)) = (cache_key, self.cache_handler.as_ref())
            && returned_tool_calls.is_empty()
            && !append_output_after_history
        {
            let entry = CacheEntry {
                prompt: messages.to_json().to_string(),
                usage: accumulated_usage,
                raw_output: Some(first_choice.content()),
            };
            cache.lock().await.insert_entry(key, entry);
            trace!("lm response cached");
        }

        let mut full_chat = if let Some(result) = tool_loop_result.as_ref() {
            Self::chat_from_rig_history(&system_prompt, &result.chat_history)
        } else {
            let mut chat = messages;
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
            output: first_choice,
            usage: accumulated_usage,
            chat: full_chat,
            tool_calls: tool_loop_result
                .as_ref()
                .map(|result| result.tool_calls.clone())
                .unwrap_or(returned_tool_calls),
            tool_executions: tool_loop_result
                .as_ref()
                .map(|result| result.tool_executions.clone())
                .unwrap_or_default(),
            events: tool_loop_result
                .map(|result| result.events)
                .unwrap_or(events),
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
    fn retry_classifier_matches_transient_errors_only() {
        assert!(is_retryable_completion_error(
            &CompletionError::ProviderError("429 Too Many Requests".to_string())
        ));
        assert!(is_retryable_completion_error(
            &CompletionError::ProviderError("Anthropic: Overloaded".to_string())
        ));
        assert!(is_retryable_completion_error(
            &CompletionError::ProviderError("HTTP 503 service unavailable".to_string())
        ));
        assert!(!is_retryable_completion_error(
            &CompletionError::ProviderError("invalid api key".to_string())
        ));
        assert!(!is_retryable_completion_error(
            &CompletionError::ResponseError("test response queue is empty".to_string())
        ));
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
            config: LMConfig {
                base_url: None,
                api_key: None,
                model: "openai:gpt-4o-mini".to_string(),
                temperature: 0.0,
                max_tokens: 128,
                max_tool_iterations: 4,
                max_retries: 0,
                retry_base_delay_ms: 1,
                cache: false,
            },
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
        assert_eq!(response.output.content(), "");
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

        // Ordered per-round-trip record: tool-call exchange, tool run, final text.
        assert_eq!(response.events.len(), 3);
        assert!(
            matches!(&response.events[0], SpanEvent::Exchange { message, .. } if message.has_tool_calls())
        );
        // rig's ToolDyn::call JSON-encodes the output, hence contains not equals.
        assert!(
            matches!(&response.events[1], SpanEvent::ToolRun { name, result, .. } if name == "counter" && result.contains("counted"))
        );
        assert!(
            matches!(&response.events[2], SpanEvent::Exchange { message, .. } if message.content() == "done")
        );
    }

    #[tokio::test]
    async fn plain_text_call_records_single_exchange_event() {
        let model = TestCompletionModel::new([make_text("hello")]);
        let lm = test_lm_with_model(model);

        let chat = Chat::new(vec![Message::user("hi")]);
        let response = lm.call(chat, vec![]).await.expect("call should succeed");

        assert!(
            matches!(response.events.as_slice(), [SpanEvent::Exchange { message, .. }] if message.content() == "hello")
        );
    }
}
