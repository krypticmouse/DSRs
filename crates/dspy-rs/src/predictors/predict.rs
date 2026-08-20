use anyhow::Result;
use rig::tool::ToolDyn;
use serde_json::{Map, Value};
use std::marker::PhantomData;
use std::ops::ControlFlow;
use std::sync::{Arc, OnceLock};
use tracing::{debug, trace};

use crate as dsrs;
use crate::core::lm::ToolSet;
use crate::core::{DynPredictor, Module, PredictAccessorFns, PredictState, Signature, StateUpdate};
use crate::{
    CallMetadata, Chat, ChatAdapter, FieldSchema, GLOBAL_SETTINGS, LmError, LmUsage, Message,
    PredictError, Predicted, Schema, SignatureSchema,
};

/// A typed input/output pair for few-shot prompting.
///
/// Demos are formatted as user/assistant exchanges in the prompt, showing the LM
/// what good responses look like. The types enforce that demos match the signature —
/// you can't accidentally pass a `QAOutput` demo to a `Predict<SummarizeSig>`.
///
/// ```
/// use dspy_rs::*;
/// use dspy_rs::doctest::*;
///
/// let demo = Demo::<QA>::new(
///     QAInput { question: "What is 2+2?".into() },
///     QAOutput { answer: "4".into() },
/// );
/// ```
#[derive(Clone, Debug, facet::Facet)]
#[facet(crate = facet)]
pub struct Demo<S: Signature> {
    pub input: S::Input,
    pub output: S::Output,
}

impl<S: Signature> Demo<S> {
    pub fn new(input: S::Input, output: S::Output) -> Self {
        Self { input, output }
    }
}

fn predict_dyn_visit<S>(
    value: *mut (),
    visitor: &mut dyn FnMut(&mut dyn DynPredictor) -> ControlFlow<()>,
) -> ControlFlow<()>
where
    S: Signature,
{
    // SAFETY: this function is only called through the shape-local
    // `dsrs::predict_accessor` payload attached to a shape with strict
    // `Predict` identity (`type_identifier` + `module_path`).
    let typed = unsafe { &mut *(value.cast::<Predict<S>>()) };
    visitor(typed)
}

type VisitPredictorMutFn =
    fn(*mut (), &mut dyn FnMut(&mut dyn DynPredictor) -> ControlFlow<()>) -> ControlFlow<()>;

trait PredictAccessorProvider {
    const VISIT_MUT: VisitPredictorMutFn;
}

impl<S> PredictAccessorProvider for S
where
    S: Signature,
{
    const VISIT_MUT: VisitPredictorMutFn = predict_dyn_visit::<S>;
}

/// The leaf module. The only thing in the system that actually calls the LM.
///
/// One `Predict` = one prompt template = one LM call. It takes a [`Signature`]'s fields
/// and instruction, formats them into a prompt (with any demos and tools), calls the
/// configured LM, and parses the response back into `S::Output`. Every other module —
/// [`ChainOfThought`](crate::ChainOfThought), custom pipelines — ultimately
/// delegates to one or more `Predict` leaves.
///
/// This is also the unit of optimization. When an optimizer tunes your program, it's
/// adjusting `Predict` leaves: their demos (few-shot examples) and instructions.
/// The optimizer's Facet walker discovers leaves automatically from struct fields —
/// no `#[parameter]` annotations or manual traversal needed.
///
/// # Optimizer discovery
///
/// `Predict<S>` encodes shape-local discovery payloads:
/// - strict shape identity (`type_identifier` + `module_path`) identifies the leaf
/// - `dsrs::predict_accessor` stores the typed mutable accessor visitor
///
/// The optimizer walker consumes these through `visit_named_predictors_mut`.
/// There is no runtime registration side effect in `new()` or `build()`.
///
/// ```no_run
/// # async fn example() -> Result<(), dspy_rs::PredictError> {
/// use dspy_rs::*;
/// use dspy_rs::doctest::*;
///
/// // Minimal
/// let predict = Predict::<QA>::new();
/// let result = predict.call(QAInput { question: "What is 2+2?".into() }).await?;
/// println!("{}", result.answer);
///
/// // With demos and custom instruction
/// let predict = Predict::<QA>::builder()
///     .demo(Demo::new(
///         QAInput { question: "What is 1+1?".into() },
///         QAOutput { answer: "2".into() },
///     ))
///     .instruction("Answer in one word.")
///     .build();
/// # Ok(())
/// # }
/// ```
#[derive(facet::Facet)]
#[facet(crate = facet, opaque)]
#[facet(dsrs::predict_accessor = &PredictAccessorFns {
    visit_mut: <S as PredictAccessorProvider>::VISIT_MUT,
})]
pub struct Predict<S: Signature> {
    #[facet(skip, opaque)]
    tools: Vec<Arc<dyn ToolDyn>>,
    #[facet(skip, opaque)]
    demos: Vec<Demo<S>>,
    instruction_override: Option<String>,
    #[facet(skip, opaque)]
    lm: Option<Arc<crate::core::LM>>,
    /// Formatted system + demo messages, built once per (instruction, demos)
    /// configuration. Reset by every mutator (`set_instruction`,
    /// `set_demos_from_examples`, `load_state`).
    #[facet(skip, opaque)]
    prompt_prefix: OnceLock<Vec<Message>>,
    /// Pre-fetched tool definitions + name-indexed executors. Tools are only
    /// settable at build time, so this never needs invalidation.
    #[facet(skip, opaque)]
    toolset: tokio::sync::OnceCell<Arc<ToolSet>>,
    /// Component name recorded on trace spans; set by
    /// [`PredictBuilder::named`], [`fx::predict`](crate::fx::predict), or the
    /// optimizer naming pass.
    #[facet(skip, opaque)]
    trace_name: Option<String>,
    #[facet(skip, opaque)]
    _marker: PhantomData<S>,
}

impl<S: Signature> Predict<S> {
    /// Creates a new `Predict` with no demos, no instruction override, and no tools.
    pub fn new() -> Self {
        Self {
            tools: Vec::new(),
            demos: Vec::new(),
            instruction_override: None,
            lm: None,
            prompt_prefix: OnceLock::new(),
            toolset: tokio::sync::OnceCell::new(),
            trace_name: None,
            _marker: PhantomData,
        }
    }

    /// Returns a builder for configuring demos, instruction, and tools.
    pub fn builder() -> PredictBuilder<S> {
        PredictBuilder::new()
    }

    /// The typed write path for optimizable state (instruction override + demos).
    ///
    /// This is the only place that assigns those fields and invalidates the
    /// cached prompt prefix; the builder and the type-erased
    /// `DynPredictor::apply_update` seam both funnel here. `None` leaves a
    /// field untouched.
    fn apply_state(
        &mut self,
        instruction: Option<Option<String>>,
        demos: Option<Vec<Demo<S>>>,
    ) {
        if let Some(instruction) = instruction {
            self.instruction_override = instruction;
        }
        if let Some(demos) = demos {
            self.demos = demos;
        }
        self.prompt_prefix = OnceLock::new();
    }

    /// The typed direct call: builds the prompt, calls the LM, and parses the response.
    ///
    /// The full pipeline:
    /// 1. Format system message from the signature's schema and instruction override
    /// 2. Format demo examples as user/assistant exchanges
    /// 3. Format the input as the final user message
    /// 4. Call the LM (with any tools attached)
    /// 5. Parse the response into `S::Output` via the `[[ ## field ## ]]` protocol
    /// 6. Record a trace span if inside a [`capture()`](crate::trace::capture) scope
    ///
    /// [`Module::forward`] delegates here; for multi-turn conversations, build the
    /// chat yourself and use [`call_and_parse`](Predict::call_and_parse).
    ///
    /// # Errors
    ///
    /// - [`PredictError::Lm`] if the LM call fails (network, rate limit, timeout)
    /// - [`PredictError::Parse`] if the response can't be parsed into the output fields
    #[tracing::instrument(
        name = "dsrs.predict.call",
        level = "debug",
        skip(self, input),
        fields(
            signature = std::any::type_name::<S>(),
            demo_count = self.demos.len(),
            tool_count = self.tools.len(),
            instruction_override = self.instruction_override.is_some(),
            capturing = crate::trace::is_capturing()
        )
    )]
    pub async fn call(&self, input: S::Input) -> Result<Predicted<S::Output>, PredictError>
    where
        S::Input: Schema,
        S::Output: Schema,
    {
        // Serialize the input for trace recording only when a scope is active.
        let capture_input = if crate::trace::is_capturing() {
            json_map_from_input::<S>(&input).ok()
        } else {
            None
        };
        let chat = self.build_chat(&input)?;
        // The chat is prefix + one live user message; everything before the
        // final message is the interned span prefix.
        let prefix_len = chat.len().saturating_sub(1);
        let (predicted, _) = self
            .call_and_parse_with_input(chat, capture_input, prefix_len)
            .await?;
        Ok(predicted)
    }

    /// Builds the first-turn chat from the signature, demos, and input.
    ///
    /// Returns a [`Chat`] ready to pass to [`call_and_parse`](Predict::call_and_parse).
    /// Useful when you need to inspect or modify the prompt before sending it to
    /// the LM.
    ///
    /// The system message and demo turns are formatted once per (instruction,
    /// demos) configuration and cached on the instance — only the live user
    /// message is formatted per call.
    #[allow(clippy::result_large_err)]
    pub fn build_chat(&self, input: &S::Input) -> Result<Chat, PredictError>
    where
        S::Input: Schema,
    {
        let prefix = self.prompt_prefix()?;
        let user = ChatAdapter.format_user_message_typed::<S>(input);

        let mut messages = Vec::with_capacity(prefix.len() + 1);
        messages.extend(prefix.iter().cloned());
        messages.push(Message::user(user));
        let chat = Chat::new(messages);
        trace!(message_count = chat.len(), "chat constructed");
        Ok(chat)
    }

    /// Returns the cached system + demo message prefix, building it on first use.
    #[allow(clippy::result_large_err)]
    fn prompt_prefix(&self) -> Result<&[Message], PredictError>
    where
        S::Input: Schema,
    {
        if self.prompt_prefix.get().is_none() {
            let built = self.build_prompt_prefix()?;
            // A concurrent forward may have won the race — that's fine, both
            // builds produce identical messages.
            let _ = self.prompt_prefix.set(built);
        }
        Ok(self
            .prompt_prefix
            .get()
            .expect("prompt prefix initialized above"))
    }

    #[allow(clippy::result_large_err)]
    fn build_prompt_prefix(&self) -> Result<Vec<Message>, PredictError>
    where
        S::Input: Schema,
    {
        let chat_adapter = ChatAdapter;
        let system = match chat_adapter
            .format_system_message_typed_with_instruction::<S>(self.instruction_override.as_deref())
        {
            Ok(system) => system,
            Err(err) => {
                return Err(PredictError::Lm {
                    source: LmError::Provider {
                        provider: "internal".to_string(),
                        message: err.to_string(),
                        source: None,
                    },
                });
            }
        };
        trace!(system_len = system.len(), "typed system prompt formatted");

        let mut messages = Vec::with_capacity(1 + self.demos.len() * 2);
        messages.push(Message::system(system));
        for demo in &self.demos {
            messages.push(Message::user(
                chat_adapter.format_user_message_typed::<S>(&demo.input),
            ));
            messages.push(Message::assistant(
                chat_adapter.format_assistant_message_typed::<S>(&demo.output),
            ));
        }
        Ok(messages)
    }

    /// The component name recorded on trace spans: the assigned `trace_name`
    /// (fx slot name / [`PredictBuilder::named`]) or, for unnamed predictors,
    /// the signature type name.
    fn component_name(&self) -> &str {
        match self.trace_name.as_deref() {
            Some(name) => name,
            None => {
                tracing::warn!(
                    signature = std::any::type_name::<S>(),
                    "dsrs.unnamed_component: span recorded under signature name; \
                     assign a name via PredictBuilder::named or fx::predict"
                );
                std::any::type_name::<S>()
            }
        }
    }

    /// Returns the cached [`ToolSet`], fetching tool definitions on first use.
    async fn cached_toolset(&self) -> Option<Arc<ToolSet>> {
        if self.tools.is_empty() {
            return None;
        }
        Some(Arc::clone(
            self.toolset
                .get_or_init(|| async { Arc::new(ToolSet::build(&self.tools).await) })
                .await,
        ))
    }

    /// The chat-level call: sends `chat` to the LM and parses the response,
    /// returning both the prediction and the updated conversation history.
    ///
    /// This is the one conversation seam. The caller owns the `Chat` between
    /// turns:
    /// 1. Build the first turn with [`build_chat`](Predict::build_chat) (or take
    ///    the `Chat` returned by a previous `call_and_parse`).
    /// 2. Append follow-up user messages to the returned `Chat`.
    /// 3. Call `call_and_parse` again with the updated `Chat`.
    ///
    /// Every turn parses with the same `[[ ## field ## ]]` protocol; the caller
    /// is responsible for including format instructions in follow-up messages if
    /// the model needs reminding of the output format.
    pub async fn call_and_parse(
        &self,
        chat: Chat,
    ) -> Result<(Predicted<S::Output>, Chat), PredictError>
    where
        S::Input: Schema,
        S::Output: Schema,
    {
        trace!(message_count = chat.len(), "chat-level call");
        self.call_and_parse_with_input(chat, None, 0).await
    }

    /// [`call_and_parse`](Predict::call_and_parse) with the typed input captured
    /// for trace recording. `capture_input` is only recorded when a capture
    /// scope is active; pass `None` when the input is unavailable (e.g.
    /// multi-turn continuations). `prefix_len` is the number of leading chat
    /// messages that are the cached system+demos prefix (0 for caller-owned chats).
    async fn call_and_parse_with_input(
        &self,
        chat: Chat,
        capture_input: Option<Map<String, Value>>,
        prefix_len: usize,
    ) -> Result<(Predicted<S::Output>, Chat), PredictError>
    where
        S::Input: Schema,
        S::Output: Schema,
    {
        let lm = match &self.lm {
            Some(lm) => Arc::clone(lm),
            None => {
                let guard = GLOBAL_SETTINGS.read().unwrap();
                let settings = guard.as_ref().unwrap();
                Arc::clone(&settings.lm)
            }
        };

        // Open the span before the LM call: a call that dies mid-flight still
        // leaves (component, seq, prompt, input) in the trace.
        let guard = crate::trace::begin_span(crate::trace::SpanRequest {
            component: self.component_name(),
            prefix: (prefix_len > 0).then(|| &chat.messages[..prefix_len]),
            suffix: &chat.messages[prefix_len..],
            input: capture_input,
            model: &lm.config,
            request_hash: None,
        });

        // Replay scope (RFC 0001 §4d/e): intercept above the LM. A served call
        // constructs no client, reaches no provider, and re-executes no tool.
        match crate::trace::replay::intercept(self.component_name(), &lm.config, &chat.messages) {
            Some(crate::trace::replay::ReplayDirective::Serve(span)) => {
                return self.serve_recorded_span(*span, chat, guard);
            }
            Some(crate::trace::replay::ReplayDirective::Refuse(err)) => {
                if let Some(guard) = guard {
                    guard.finish(crate::trace::SpanOutcome {
                        events: Vec::new(),
                        raw_output: None,
                        output: None,
                        usage: LmUsage::default(),
                        error: Some(crate::trace::SpanError {
                            kind: crate::trace::SpanErrorKind::Lm,
                            message: err.to_string(),
                        }),
                    });
                }
                return Err(PredictError::Replay { source: err });
            }
            // Live directive (post-divergence) or no replay scope: proceed.
            Some(crate::trace::replay::ReplayDirective::Live) | None => {}
        }

        let toolset = self.cached_toolset().await;
        let empty_toolset = ToolSet::default();
        let toolset_ref = toolset.as_deref().unwrap_or(&empty_toolset);
        let response = match lm
            .call_with_toolset(chat, toolset_ref, crate::ToolLoopMode::Auto)
            .await
        {
            Ok(response) => response,
            Err(err) => {
                if let Some(guard) = guard {
                    guard.finish(crate::trace::SpanOutcome {
                        events: Vec::new(),
                        raw_output: None,
                        output: None,
                        usage: LmUsage::default(),
                        error: Some(crate::trace::SpanError {
                            kind: crate::trace::SpanErrorKind::Lm,
                            message: err.to_string(),
                        }),
                    });
                }
                return Err(PredictError::Lm {
                    source: LmError::Provider {
                        provider: lm.config.model.clone(),
                        message: err.to_string(),
                        source: None,
                    },
                });
            }
        };
        debug!(
            prompt_tokens = response.usage.prompt_tokens,
            completion_tokens = response.usage.completion_tokens,
            total_tokens = response.usage.total_tokens,
            tool_calls = response.tool_calls.len(),
            "lm response received"
        );

        let crate::core::lm::LMResponse {
            output,
            usage,
            chat,
            tool_calls,
            tool_executions,
            events,
        } = response;

        let chat_adapter = ChatAdapter;
        let raw_response = output.content();
        let lm_usage = usage;

        let (typed_output, field_metas) = match chat_adapter.parse_response_typed::<S>(&output) {
            Ok(parsed) => parsed,
            Err(err) => {
                let failed_fields = err.fields();
                debug!(
                    failed_fields = failed_fields.len(),
                    fields = ?failed_fields,
                    raw_response_len = raw_response.len(),
                    "typed parse failed"
                );
                // Parse failures keep raw_output in the span — the model's
                // unparseable prose is prime reflection material.
                if let Some(guard) = guard {
                    guard.finish(crate::trace::SpanOutcome {
                        events,
                        raw_output: Some(raw_response.clone()),
                        output: None,
                        usage: lm_usage,
                        error: Some(crate::trace::SpanError {
                            kind: crate::trace::SpanErrorKind::Parse,
                            message: err.to_string(),
                        }),
                    });
                }
                return Err(PredictError::Parse {
                    source: err,
                    raw_response,
                    lm_usage,
                });
            }
        };

        let span_id = guard.as_ref().map(|guard| guard.id());
        if let Some(guard) = guard {
            guard.finish(crate::trace::SpanOutcome {
                events,
                raw_output: Some(raw_response.clone()),
                output: json_map_from_output::<S>(&typed_output).ok(),
                usage: lm_usage,
                error: None,
            });
        }

        let checks_total = field_metas
            .values()
            .map(|meta| meta.checks.len())
            .sum::<usize>();
        let checks_failed = field_metas
            .values()
            .flat_map(|meta| meta.checks.iter())
            .filter(|check| !check.passed)
            .count();
        let flagged_fields = field_metas
            .values()
            .filter(|meta| !meta.flags.is_empty())
            .count();
        debug!(
            output_fields = field_metas.len(),
            checks_total, checks_failed, flagged_fields, "typed parse completed"
        );

        let metadata = CallMetadata::new(
            raw_response,
            lm_usage,
            tool_calls,
            tool_executions,
            span_id,
            field_metas,
        );

        Ok((Predicted::new(typed_output, metadata), chat))
    }

    /// Serves one call from a recorded span (replay scope, RFC 0001 §4d):
    /// deserializes the recorded parsed output into `S::Output`, extends the
    /// chat with the recorded completion, and records the served span into any
    /// active capture scope. Zero provider calls, zero tool executions.
    #[allow(clippy::result_large_err)]
    fn serve_recorded_span(
        &self,
        span: crate::trace::Span,
        mut chat: Chat,
        guard: Option<crate::trace::SpanGuard>,
    ) -> Result<(Predicted<S::Output>, Chat), PredictError>
    where
        S::Output: Schema,
    {
        let output_map = span
            .output
            .clone()
            .expect("replay serves only spans with parsed output");
        let typed_output: S::Output = match serde_json::from_value(Value::Object(output_map)) {
            Ok(output) => output,
            Err(err) => {
                let source = crate::trace::ReplayError::OutputDecode {
                    component: self.component_name().to_string(),
                    seq: span.seq,
                    span: span.id,
                    message: err.to_string(),
                };
                if let Some(guard) = guard {
                    guard.finish(crate::trace::SpanOutcome {
                        events: Vec::new(),
                        raw_output: span.raw_output.clone(),
                        output: None,
                        usage: span.usage,
                        error: Some(crate::trace::SpanError {
                            kind: crate::trace::SpanErrorKind::Parse,
                            message: source.to_string(),
                        }),
                    });
                }
                return Err(PredictError::Replay { source });
            }
        };

        let raw_response = span.raw_output.clone().unwrap_or_default();
        debug!(
            component = self.component_name(),
            seq = span.seq,
            "predict call served from recorded trace"
        );

        // Rebuild the conversation the live call would have returned: the
        // recorded exchanges and tool results, in order. Tool effects are baked
        // into the recording — nothing re-executes.
        let mut completion = span.completion_messages();
        if completion.is_empty() && !raw_response.is_empty() {
            completion.push(Message::assistant(raw_response.clone()));
        }
        let tool_calls = completion
            .iter()
            .flat_map(|message| message.tool_calls().into_iter().cloned())
            .collect();
        let tool_executions = span
            .events
            .iter()
            .filter_map(|event| match event {
                crate::trace::SpanEvent::ToolRun { result, .. } => Some(result.clone()),
                _ => None,
            })
            .collect();
        for message in &completion {
            chat.push_message(message.clone());
        }

        let span_id = guard.as_ref().map(|guard| guard.id());
        if let Some(guard) = guard {
            guard.finish(crate::trace::SpanOutcome {
                events: span.events.clone(),
                raw_output: span.raw_output.clone(),
                output: span.output.clone(),
                usage: span.usage,
                error: None,
            });
        }

        // Served predictions carry no per-field parse metadata: the recording
        // stores the parsed output, not the parser's field-level bookkeeping.
        let metadata = CallMetadata::new(
            raw_response,
            span.usage,
            tool_calls,
            tool_executions,
            span_id,
            Default::default(),
        );
        Ok((Predicted::new(typed_output, metadata), chat))
    }
}

impl<S: Signature> Default for Predict<S> {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for [`Predict`] with demos, tools, and instruction override.
///
/// ```ignore
/// let predict = Predict::<QA>::builder()
///     .demo(demo1)
///     .demo(demo2)
///     .instruction("Answer in one word.")
///     .add_tool(my_tool)
///     .build();
/// ```
pub struct PredictBuilder<S: Signature> {
    tools: Vec<Arc<dyn ToolDyn>>,
    demos: Vec<Demo<S>>,
    instruction_override: Option<String>,
    lm: Option<Arc<crate::core::LM>>,
    trace_name: Option<String>,
    _marker: PhantomData<S>,
}

impl<S: Signature> PredictBuilder<S> {
    fn new() -> Self {
        Self {
            tools: Vec::new(),
            demos: Vec::new(),
            instruction_override: None,
            lm: None,
            trace_name: None,
            _marker: PhantomData,
        }
    }

    /// Assigns a human-readable name recorded on this predictor's trace spans.
    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.trace_name = Some(name.into());
        self
    }

    /// Adds a single demo (few-shot example) to the predictor.
    pub fn demo(mut self, demo: Demo<S>) -> Self {
        self.demos.push(demo);
        self
    }

    /// Adds multiple demos from an iterator.
    pub fn with_demos(mut self, demos: impl IntoIterator<Item = Demo<S>>) -> Self {
        self.demos.extend(demos);
        self
    }

    /// Adds a tool the LM can invoke during this call.
    pub fn add_tool(mut self, tool: impl ToolDyn + 'static) -> Self {
        self.tools.push(Arc::new(tool));
        self
    }

    /// Adds multiple tools from an iterator.
    pub fn with_tools(mut self, tools: impl IntoIterator<Item = Arc<dyn ToolDyn>>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Overrides the signature's default instruction for this predictor.
    pub fn instruction(mut self, instruction: impl Into<String>) -> Self {
        self.instruction_override = Some(instruction.into());
        self
    }

    /// Sets a per-instance LM for this predictor, bypassing the global.
    ///
    /// When set, this `Predict` will use the given LM instead of the one
    /// configured via [`configure()`](crate::configure). This enables
    /// concurrent calls with different models — each `Predict` leaf can
    /// target a different provider without contention on the global setting.
    ///
    /// ```ignore
    /// let predict = Predict::<QA>::builder()
    ///     .lm(LM::builder().model("anthropic:claude-sonnet-4-20250514").build().await?)
    ///     .build();
    /// ```
    pub fn lm(mut self, lm: crate::core::LM) -> Self {
        self.lm = Some(Arc::new(lm));
        self
    }

    /// Builds the [`Predict`], routing state through the same applicator the
    /// mutation seam uses.
    pub fn build(self) -> Predict<S> {
        let mut predict = Predict {
            tools: self.tools,
            demos: Vec::new(),
            instruction_override: None,
            lm: self.lm,
            prompt_prefix: OnceLock::new(),
            toolset: tokio::sync::OnceCell::new(),
            trace_name: self.trace_name,
            _marker: PhantomData,
        };
        predict.apply_state(Some(self.instruction_override), Some(self.demos));
        predict
    }
}

/// Picks the named schema fields out of a flat demo row.
fn pick_schema_fields(row: &Map<String, Value>, fields: &[FieldSchema]) -> Map<String, Value> {
    let mut map = Map::new();
    for field in fields {
        if let Some(value) = row.get(&field.rust_name) {
            map.insert(field.rust_name.clone(), value.clone());
        }
    }
    map
}

/// Parses a typed demo from a flat demo row (field name → value, input and
/// output fields merged into one object). The signature schema decides which
/// fields belong to the input and which to the output.
fn demo_from_json<S: Signature>(row: &Map<String, Value>) -> Result<Demo<S>>
where
    S::Input: Schema,
    S::Output: Schema,
{
    let schema = S::schema();
    let input = serde_json::from_value::<S::Input>(Value::Object(pick_schema_fields(
        row,
        schema.input_fields(),
    )))
    .map_err(|err| anyhow::anyhow!(err))?;
    let output = serde_json::from_value::<S::Output>(Value::Object(pick_schema_fields(
        row,
        schema.output_fields(),
    )))
    .map_err(|err| anyhow::anyhow!(err))?;
    Ok(Demo::new(input, output))
}

/// Serializes a typed demo into a flat demo row (input and output fields
/// merged into one object).
fn json_from_demo<S: Signature>(example: &Demo<S>) -> Result<Map<String, Value>>
where
    S::Input: Schema,
    S::Output: Schema,
{
    let mut row = json_map_from_input::<S>(&example.input)?;
    row.extend(json_map_from_output::<S>(&example.output)?);
    Ok(row)
}

fn json_map_from_input<S: Signature>(input: &S::Input) -> Result<Map<String, Value>>
where
    S::Input: Schema,
{
    match serde_json::to_value(input)? {
        Value::Object(map) => Ok(map),
        _ => Err(anyhow::anyhow!("expected object for signature input")),
    }
}

fn json_map_from_output<S: Signature>(output: &S::Output) -> Result<Map<String, Value>>
where
    S::Output: Schema,
{
    match serde_json::to_value(output)? {
        Value::Object(map) => Ok(map),
        _ => Err(anyhow::anyhow!("expected object for signature output")),
    }
}

impl<S> Module for Predict<S>
where
    S: Signature + Clone,
    S::Input: Schema,
    S::Output: Schema,
{
    type Input = S::Input;
    type Output = S::Output;

    #[tracing::instrument(
        name = "dsrs.module.forward",
        level = "debug",
        skip(self, input),
        fields(
            signature = std::any::type_name::<S>(),
            typed = true
        )
    )]
    async fn forward(&self, input: S::Input) -> Result<Predicted<S::Output>, PredictError> {
        Predict::call(self, input).await
    }
}

impl<S> DynPredictor for Predict<S>
where
    S: Signature,
    S::Input: Schema,
    S::Output: Schema,
{
    fn schema(&self) -> &SignatureSchema {
        S::schema()
    }

    fn instruction(&self) -> String {
        self.instruction_override
            .clone()
            .unwrap_or_else(|| S::instruction().to_string())
    }

    fn demos_as_json(&self) -> Vec<Map<String, Value>> {
        self.demos
            .iter()
            .map(|example| {
                json_from_demo::<S>(example)
                    .expect("typed Predict demo conversion should succeed")
            })
            .collect()
    }

    fn dump_state(&self) -> PredictState {
        PredictState {
            demos: self.demos_as_json(),
            instruction_override: self.instruction_override.clone(),
        }
    }

    fn apply_update(&mut self, update: StateUpdate) -> Result<()> {
        // Convert demos before touching any state so a schema mismatch leaves
        // the predictor unchanged.
        let demos = update
            .demos
            .map(|demos| {
                demos
                    .iter()
                    .map(demo_from_json::<S>)
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;
        self.apply_state(update.instruction, demos);
        Ok(())
    }

    fn set_trace_name(&mut self, name: &str) {
        self.trace_name = Some(name.to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[derive(crate::Signature, Clone, Debug)]
    struct PredictConversionSig {
        #[input]
        prompt: String,

        #[output]
        answer: String,
    }

    fn typed_row(prompt: &str, answer: &str) -> Demo<PredictConversionSig> {
        Demo::new(
            PredictConversionSigInput {
                prompt: prompt.to_string(),
            },
            PredictConversionSigOutput {
                answer: answer.to_string(),
            },
        )
    }

    #[test]
    fn typed_and_json_row_round_trip_preserves_fields() {
        let typed = typed_row("question", "response");
        let row = json_from_demo::<PredictConversionSig>(&typed)
            .expect("typed example should convert to a flat row");

        assert_eq!(row.get("prompt"), Some(&json!("question")));
        assert_eq!(row.get("answer"), Some(&json!("response")));

        let round_trip = demo_from_json::<PredictConversionSig>(&row)
            .expect("flat row should convert back to typed example");
        assert_eq!(round_trip.input.prompt, "question");
        assert_eq!(round_trip.output.answer, "response");
    }

    #[test]
    fn demo_from_json_splits_fields_by_schema() {
        let row = Map::from_iter([
            ("prompt".to_string(), json!("schema-input")),
            ("answer".to_string(), json!("schema-output")),
            ("extra".to_string(), json!("ignored")),
        ]);

        let typed = demo_from_json::<PredictConversionSig>(&row)
            .expect("schema keys should split the row into typed fields");
        assert_eq!(typed.input.prompt, "schema-input");
        assert_eq!(typed.output.answer, "schema-output");
    }

    #[test]
    fn dyn_predictor_apply_update_round_trips_json_demo_rows() {
        let typed = typed_row("demo-input", "demo-output");
        let row = json_from_demo::<PredictConversionSig>(&typed)
            .expect("typed demo should convert to a flat row");
        let mut predictor = Predict::<PredictConversionSig>::new();

        let update = StateUpdate {
            instruction: None,
            demos: Some(vec![row]),
        };
        DynPredictor::apply_update(&mut predictor, update)
            .expect("predictor should accept JSON demo rows");

        let demos = DynPredictor::demos_as_json(&predictor);
        assert_eq!(demos.len(), 1);
        assert_eq!(demos[0].get("prompt"), Some(&json!("demo-input")));
        assert_eq!(demos[0].get("answer"), Some(&json!("demo-output")));
    }
}
