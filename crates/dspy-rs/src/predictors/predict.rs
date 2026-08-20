use anyhow::Result;
use indexmap::IndexMap;
use rig::tool::ToolDyn;
use serde_json::{Map, Value};
use std::marker::PhantomData;
use std::sync::{Arc, OnceLock};
use tracing::{debug, trace};

use crate::core::lm::ToolSet;
use crate::core::{Module, PredictState, Signature};
use crate::ir::{
    self, Budget, Interpreter, Overlay, Program, RunError, RunOutput, RuntimeEnv, SignatureDef,
};
use crate::{
    CallMetadata, Chat, ChatAdapter, FieldSchema, GLOBAL_SETTINGS, LmError, LmUsage, Message,
    ParseError, PredictError, Predicted, Schema, SignatureSchema,
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
///
/// # Optimizer discovery
///
/// Modules declare their `Predict` leaves by name through the
/// [`Predictors`](crate::Predictors) trait (see the `predictors!` macro);
/// optimizers read each leaf through its [`PredictorInfo`](crate::PredictorInfo)
/// view and inject candidates ambiently per call
/// ([`fx::with_params`](crate::fx::with_params)) — never by mutating the leaf.
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
    /// `set_demos_from_examples`, `load_state`). Serves the conversation seam
    /// ([`build_chat`](Predict::build_chat)/[`call_and_parse`](Predict::call_and_parse));
    /// the typed [`call`](Predict::call) path renders through the interpreter.
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
    /// The cached 1-node IR program this predictor executes: a `predict` leaf
    /// named [`component_name`](Predict::component_name) over
    /// [`SignatureDef::of::<S>()`]. Reset by [`set_trace_name`] (the leaf name
    /// is part of the program).
    #[facet(skip, opaque)]
    program: OnceLock<Arc<Program>>,
    /// Instance state (instruction override + demos) minted as an [`Overlay`]
    /// against [`program`](Self::program). `None` inner = no overrides (the
    /// program defaults read through). Reset by every state mutator and by
    /// `set_trace_name`.
    #[facet(skip, opaque)]
    instance_overlay: OnceLock<Option<Arc<Overlay>>>,
    /// The loaded interpreter, keyed by the LM it was bound with; rebuilt when
    /// the resolved LM changes (per-instance LM is fixed at build, but the
    /// global [`configure`](crate::configure)d LM can change between calls).
    #[facet(skip, opaque)]
    engine: tokio::sync::Mutex<Option<(Arc<crate::core::LM>, Arc<Interpreter>)>>,
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
            program: OnceLock::new(),
            instance_overlay: OnceLock::new(),
            engine: tokio::sync::Mutex::new(None),
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
    /// cached prompt prefix; the builder and the [`PredictorInfo::load_state`]
    /// install seam both funnel here. `None` leaves a field untouched.
    ///
    /// [`PredictorInfo::load_state`]: crate::core::PredictorInfo::load_state
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
        self.instance_overlay = OnceLock::new();
    }

    /// The typed direct call: renders the prompt, calls the LM, and parses the
    /// response — all through the IR interpreter.
    ///
    /// `Predict<S>` is a thin typed handle over a 1-node IR [`Program`] (a
    /// `predict` leaf over [`SignatureDef::of::<S>()`]) executed by
    /// [`Interpreter::run_collecting`]:
    /// 1. Instance state (instruction override + demos) reads through an
    ///    [`Overlay`]; an ambient optimizer overlay
    ///    ([`ir::current_overlay`](crate::ir::current_overlay)) composes on
    ///    top (ambient entries win per slot).
    /// 2. The interpreter renders system/demos/input via the def-lane
    ///    [`ChatAdapter`] (byte-identical prompts), calls the bound LM, and
    ///    parses the `[[ ## field ## ]]` response.
    /// 3. A trace span is recorded under this predictor's component name when
    ///    inside a [`capture()`](crate::trace::capture) scope; replay scopes
    ///    intercept above the LM exactly as before.
    /// 4. The typed `Predicted<S::Output>` is reassembled from the run's
    ///    single [`LeafOutcome`](crate::ir::LeafOutcome).
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
        let program = self.program().await?;
        let overlay = self.effective_overlay(&program)?;
        let lm = self.resolve_lm();
        let interpreter = self.interpreter(&program, &lm).await?;
        let input_map = json_map_from_input::<S>(&input)
            .map_err(|err| internal_error(format!("failed to serialize input: {err}")))?;
        let run = interpreter
            .run_collecting(input_map, overlay, Budget::unlimited())
            .await
            .map_err(map_run_error::<S>)?;
        predicted_from_run::<S>(run)
    }

    /// Returns the cached 1-node program, building it on first use: a
    /// `predict` leaf, or an `agent` leaf when tools are attached (the IR
    /// says `Predict` carries no tools — a tooled predictor *is* an agent
    /// loop).
    async fn program(&self) -> Result<Arc<Program>, PredictError> {
        if let Some(program) = self.program.get() {
            return Ok(Arc::clone(program));
        }
        let built = if self.tools.is_empty() {
            self.build_predict_program()?
        } else {
            let toolset = self
                .cached_toolset()
                .await
                .expect("tools are non-empty, so the toolset exists");
            self.build_agent_program(toolset.definitions())?
        };
        // A concurrent call may have won the race — both builds are identical.
        let _ = self.program.set(Arc::new(built));
        Ok(Arc::clone(self.program.get().expect("program set above")))
    }

    /// Builds the 1-node `agent` program for a tooled predictor: same leaf
    /// name and signature as the `predict` form, plus a host-tool declaration
    /// per attached tool. Stop behavior is the IR's [`StopSpec`](crate::ir::StopSpec)
    /// default — `until_parse` with `max_turns = 8` — the closest IR
    /// expression of the old LM-layer auto tool loop.
    #[allow(clippy::result_large_err)]
    fn build_agent_program(
        &self,
        definitions: &[rig::completion::ToolDefinition],
    ) -> Result<Program, PredictError> {
        let leaf = self.component_name();
        let def = SignatureDef::of::<S>();
        let mut b = ir::ProgramBuilder::new(leaf);
        let model = b.model("default", crate::ir::module_build::unbound_model_config("default"));
        let sid = b.sig_of::<S>();
        b.add_types(&<S::Input as crate::typesys::Schema>::output_schema().types);

        let mut tool_ids = Vec::with_capacity(definitions.len());
        let mut seen = std::collections::HashSet::new();
        for definition in definitions {
            // Duplicate names keep the first tool, mirroring `ToolSet::build`.
            if !seen.insert(definition.name.as_str()) {
                continue;
            }
            let (tool_sig, tool_types) =
                tool_signature_from_definition(&definition.name, &definition.parameters);
            b.add_types(&tool_types);
            let tool_sid = b.sig(tool_sig);
            tool_ids.push(b.host_tool(&definition.name, &definition.description, tool_sid, &[]));
        }

        let mut ns = ir::agent(leaf, sid).model(model).tools(tool_ids);
        for field in def.inputs.iter() {
            ns = ns.bind(&field.name, ir::input(&field.name));
        }
        let mut root = ir::seq([ns]);
        for field in def.outputs.iter() {
            root = root.out(&field.name, ir::out(leaf, &field.name));
        }
        b.main(sid, root)
            .map_err(|err| internal_error(format!("failed to build agent program: {err}")))
    }

    /// Builds the 1-node `predict` program: leaf name = the trace-name
    /// convention (so span identity and capture/replay keying are unchanged),
    /// signature = `SignatureDef::of::<S>()`, model = the `default` ref bound
    /// through [`RuntimeEnv`] at load.
    #[allow(clippy::result_large_err)]
    fn build_predict_program(&self) -> Result<Program, PredictError> {
        let leaf = self.component_name();
        let def = SignatureDef::of::<S>();
        let mut b = ir::ProgramBuilder::new(leaf);
        let model = b.model("default", crate::ir::module_build::unbound_model_config("default"));
        let sid = b.sig_of::<S>();
        // `sig_of` merges output-reachable class/enum defs; input-reachable
        // ones are needed too (the interpreter type-checks run inputs).
        b.add_types(&<S::Input as crate::typesys::Schema>::output_schema().types);
        let mut ns = ir::predict(leaf, sid).model(model);
        for field in def.inputs.iter() {
            ns = ns.bind(&field.name, ir::input(&field.name));
        }
        let mut root = ir::seq([ns]);
        for field in def.outputs.iter() {
            root = root.out(&field.name, ir::out(leaf, &field.name));
        }
        b.main(sid, root)
            .map_err(|err| internal_error(format!("failed to build predict program: {err}")))
    }

    /// The instance overlay: instruction override + demos as slot values
    /// against this predictor's program. `None` when neither is set.
    #[allow(clippy::result_large_err)]
    fn instance_overlay(&self, program: &Arc<Program>) -> Result<Option<Arc<Overlay>>, PredictError>
    where
        S::Input: Schema,
        S::Output: Schema,
    {
        if let Some(cached) = self.instance_overlay.get() {
            return Ok(cached.clone());
        }
        let overlay = if self.instruction_override.is_none() && self.demos.is_empty() {
            None
        } else {
            let state = PredictState {
                demos: crate::core::PredictorInfo::demos_as_json(self),
                instruction_override: self.instruction_override.clone(),
            };
            let minted =
                crate::ir::bridge::states_to_overlay(program, [(self.component_name(), &state)])
                    .map_err(|err| {
                        internal_error(format!("failed to mint instance overlay: {err}"))
                    })?;
            Some(Arc::new(minted))
        };
        let _ = self.instance_overlay.set(overlay);
        Ok(self
            .instance_overlay
            .get()
            .expect("instance overlay set above")
            .clone())
    }

    /// The overlay a run reads through: instance state composed with the
    /// ambient candidate scopes, later layers winning per slot:
    ///
    /// 1. instance state (instruction override + demos);
    /// 2. an ambient optimizer overlay
    ///    ([`ir::current_overlay`](crate::ir::current_overlay)) — ignored when
    ///    minted against a *different* program (one scope can span several
    ///    modules; only the matching one accepts it);
    /// 3. the ambient [`fx::Params`](crate::fx::Params) entry matching this
    ///    predictor's component name ([`fx::with_params`](crate::fx::with_params)
    ///    — the optimizer's candidate-injection scope). Explicit clears
    ///    resolve to the program slot defaults, so a candidate can reset a
    ///    slot past instance state.
    ///
    /// Ambient entries winning over instance entries preserves exactly the
    /// precedence the old apply/restore mutation seam had.
    #[allow(clippy::result_large_err)]
    fn effective_overlay(
        &self,
        program: &Arc<Program>,
    ) -> Result<Option<Arc<Overlay>>, PredictError>
    where
        S::Input: Schema,
        S::Output: Schema,
    {
        let instance = self.instance_overlay(program)?;
        let ambient = crate::ir::bridge::current_overlay()
            .filter(|overlay| overlay.base == program.meta.program_hash);
        let params_values = match crate::fx::ambient_entry(self.component_name()) {
            Some(entry) => crate::ir::bridge::entry_slot_values(program, self.component_name(), &entry)
                .map_err(|err| {
                    internal_error(format!("failed to bind ambient params: {err}"))
                })?,
            None => Vec::new(),
        };

        let mut merged: Option<Overlay> = match (instance, ambient) {
            (None, None) => None,
            (Some(instance), None) => Some((*instance).clone()),
            (None, Some(ambient)) => Some((*ambient).clone()),
            (Some(instance), Some(ambient)) => {
                let mut merged = (*instance).clone();
                for (id, value) in ambient.entries() {
                    merged.set(program, id, value.clone()).map_err(|err| {
                        internal_error(format!("failed to compose ambient overlay: {err}"))
                    })?;
                }
                Some(merged)
            }
        };

        if !params_values.is_empty() {
            let mut overlay = merged.take().unwrap_or_else(|| Overlay::new(program));
            for (id, value) in params_values {
                overlay.set(program, id, value).map_err(|err| {
                    internal_error(format!("failed to compose ambient params: {err}"))
                })?;
            }
            merged = Some(overlay);
        }

        Ok(merged.map(Arc::new))
    }

    /// Resolves the LM this call uses: instance LM > global
    /// [`configure()`](crate::configure) settings. Panics exactly like the
    /// pre-IR path when no global LM is configured and no instance LM is set.
    fn resolve_lm(&self) -> Arc<crate::core::LM> {
        match &self.lm {
            Some(lm) => Arc::clone(lm),
            None => {
                let guard = GLOBAL_SETTINGS.read().unwrap();
                let settings = guard.as_ref().unwrap();
                Arc::clone(&settings.lm)
            }
        }
    }

    /// Returns the loaded interpreter for `lm`, reloading when the resolved
    /// LM changed since the last call (the program is bound to its model at
    /// load, not per call).
    async fn interpreter(
        &self,
        program: &Arc<Program>,
        lm: &Arc<crate::core::LM>,
    ) -> Result<Arc<Interpreter>, PredictError> {
        let mut slot = self.engine.lock().await;
        if let Some((bound, interpreter)) = slot.as_ref()
            && Arc::ptr_eq(bound, lm)
        {
            return Ok(Arc::clone(interpreter));
        }
        let env = self.runtime_env(lm);
        let interpreter = Interpreter::load((**program).clone(), env)
            .await
            .map_err(|err| internal_error(format!("failed to load predict program: {err}")))?;
        let interpreter = Arc::new(interpreter);
        *slot = Some((Arc::clone(lm), Arc::clone(&interpreter)));
        Ok(interpreter)
    }

    /// The runtime environment a load binds against: the resolved model under
    /// the `default` ref, plus every attached tool bound as a host tool.
    fn runtime_env(&self, lm: &Arc<crate::core::LM>) -> RuntimeEnv {
        let mut env = RuntimeEnv::new().bind_model("default", Arc::clone(lm));
        for tool in &self.tools {
            env = env.bind_host_tool(&tool.name(), Arc::clone(tool));
        }
        env
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
    ///
    /// Part of the conversation seam
    /// (see [`call_and_parse`](Predict::call_and_parse)).
    // TODO(dsrs-phase4-conversation): fold the caller-owned-chat seam into the
    // interpreter once it grows a conversation-in/conversation-out surface.
    #[allow(clippy::result_large_err)]
    pub fn build_chat(&self, input: &S::Input) -> Result<Chat, PredictError>
    where
        S::Input: Schema,
    {
        let prefix = self.prompt_prefix()?;
        let input_map = json_map_from_input::<S>(input)
            .map_err(|err| internal_error(format!("failed to serialize input: {err}")))?;
        let user = ChatAdapter.format_input_def(SignatureDef::of::<S>(), &input_map);

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
        let def = SignatureDef::of::<S>();
        let types = SignatureDef::types_of::<S>();
        let system =
            ChatAdapter.build_system_def(def, types, self.instruction_override.as_deref());
        trace!(system_len = system.len(), "typed system prompt formatted");

        let mut messages = Vec::with_capacity(1 + self.demos.len() * 2);
        messages.push(Message::system(system));
        for demo in &self.demos {
            let input = json_map_from_input::<S>(&demo.input)
                .map_err(|err| internal_error(format!("failed to serialize demo input: {err}")))?;
            let output = json_map_from_output::<S>(&demo.output).map_err(|err| {
                internal_error(format!("failed to serialize demo output: {err}"))
            })?;
            messages.push(Message::user(ChatAdapter.format_input_def(def, &input)));
            messages.push(Message::assistant(ChatAdapter.format_output_def(def, &output)));
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
    ///
    /// This seam is a compat shim on the LM-layer call path: the interpreter
    /// has no conversation-in/conversation-out surface yet, so caller-owned
    /// chats (and the caller-managed tool pattern built on them) stay here.
    // TODO(dsrs-phase4-conversation): fold this seam into the interpreter
    // once it can accept and return a caller-owned conversation.
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

        // TODO(dsrs-phase4-caller-managed): this LM-layer tool loop only
        // serves the conversation seam; typed `call`s with tools run through
        // the interpreter's AgentLoop. Fold this into the interpreter when it
        // can express caller-managed conversations.
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

        let raw_response = output.content();
        let lm_usage = usage;

        let def = SignatureDef::of::<S>();
        let types = SignatureDef::types_of::<S>();
        let (output_map, def_metas) = match ChatAdapter.parse_output_def(def, types, &output) {
            Ok(parsed) => parsed,
            Err(err) => {
                let err = translate_parse_error::<S>(err);
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

        let typed_output: S::Output = match typed_output_from_map::<S>(&output_map, &raw_response)
        {
            Ok(typed) => typed,
            Err(err) => {
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
        let field_metas = translate_field_meta::<S>(&def_metas);

        let span_id = guard.as_ref().map(|guard| guard.id());
        if let Some(guard) = guard {
            guard.finish(crate::trace::SpanOutcome {
                events,
                raw_output: Some(raw_response.clone()),
                output: Some(output_map),
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
    /// install seam ([`PredictorInfo::load_state`](crate::core::PredictorInfo::load_state))
    /// uses.
    pub fn build(self) -> Predict<S> {
        let mut predict = Predict {
            tools: self.tools,
            demos: Vec::new(),
            instruction_override: None,
            lm: self.lm,
            prompt_prefix: OnceLock::new(),
            toolset: tokio::sync::OnceCell::new(),
            trace_name: self.trace_name,
            program: OnceLock::new(),
            instance_overlay: OnceLock::new(),
            engine: tokio::sync::Mutex::new(None),
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

// ---------------------------------------------------------------------------
// Tool schema projection (JSON Schema → SignatureDef)
// ---------------------------------------------------------------------------

/// Best-effort projection of a rig tool's JSON-Schema `parameters` object
/// into a tool [`SignatureDef`] (input side) plus the class/enum definitions
/// it references. The interpreter regenerates the model-facing schema from
/// this signature via [`ir::input_schema_of`](crate::ir::input_schema_of);
/// JSON-Schema features outside [`FieldType`](crate::typesys::FieldType)
/// (open `additionalProperties`, `oneOf`, per-property formats, …) degrade to
/// their closest `FieldType` equivalent.
fn tool_signature_from_definition(
    tool_name: &str,
    parameters: &Value,
) -> (SignatureDef, crate::typesys::TypeTable) {
    let mut types = crate::typesys::TypeTable::default();
    let empty = Map::new();
    let properties = parameters
        .get("properties")
        .and_then(Value::as_object)
        .unwrap_or(&empty);
    let required: Vec<&str> = parameters
        .get("required")
        .and_then(Value::as_array)
        .map(|entries| entries.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();

    let mut inputs = Vec::with_capacity(properties.len());
    for (name, schema) in properties {
        let token = format!("{tool_name}_{name}");
        let mut ty = field_type_from_json_schema(&token, schema, &mut types);
        if !required.contains(&name.as_str()) {
            ty = crate::typesys::FieldType::optional(ty);
        }
        let mut field = ir::FieldDef::new(name, ty);
        if let Some(docs) = schema.get("description").and_then(Value::as_str) {
            field = field.with_docs(docs);
        }
        inputs.push(field);
    }

    let def = SignatureDef {
        name: format!("{tool_name}_tool").into(),
        instruction: "".into(),
        inputs: inputs.into_boxed_slice(),
        // The output side is unused for host tools (results come back as raw
        // strings); a single string field keeps the signature well-formed.
        outputs: Box::new([ir::FieldDef::new(
            "result",
            crate::typesys::FieldType::String,
        )]),
    };
    (def, types)
}

fn field_type_from_json_schema(
    token: &str,
    schema: &Value,
    types: &mut crate::typesys::TypeTable,
) -> crate::typesys::FieldType {
    use crate::typesys::FieldType;

    if let Some(any_of) = schema.get("anyOf").and_then(Value::as_array) {
        let items = any_of
            .iter()
            .enumerate()
            .map(|(i, sub)| field_type_from_json_schema(&format!("{token}_{i}"), sub, types))
            .collect();
        return FieldType::Union(items);
    }
    if let Some(values) = schema.get("enum").and_then(Value::as_array) {
        let names: Vec<String> = values
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect();
        if !names.is_empty() {
            types.enums.insert(
                token.to_string(),
                crate::typesys::EnumDef {
                    internal_name: token.to_string(),
                    rendered_name: token.to_string(),
                    docs: None,
                    values: names
                        .into_iter()
                        .map(|name| crate::typesys::EnumValueDef {
                            rendered_name: name.clone(),
                            name,
                            docs: None,
                        })
                        .collect(),
                },
            );
            return FieldType::Enum(token.to_string());
        }
    }
    match schema.get("type").and_then(Value::as_str) {
        Some("string") => FieldType::String,
        Some("integer") => FieldType::Int,
        Some("number") => FieldType::Float,
        Some("boolean") => FieldType::Bool,
        Some("array") => FieldType::List(Box::new(
            schema
                .get("items")
                .map(|items| field_type_from_json_schema(&format!("{token}_items"), items, types))
                .unwrap_or(FieldType::String),
        )),
        Some("object") => {
            if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
                let required: Vec<&str> = schema
                    .get("required")
                    .and_then(Value::as_array)
                    .map(|entries| entries.iter().filter_map(Value::as_str).collect())
                    .unwrap_or_default();
                let fields = properties
                    .iter()
                    .map(|(name, sub)| {
                        let mut ty =
                            field_type_from_json_schema(&format!("{token}_{name}"), sub, types);
                        if !required.contains(&name.as_str()) {
                            ty = crate::typesys::FieldType::optional(ty);
                        }
                        crate::typesys::FieldDef {
                            name: name.clone(),
                            rendered_name: name.clone(),
                            field_type: ty,
                            docs: sub
                                .get("description")
                                .and_then(Value::as_str)
                                .map(str::to_string),
                            constraints: Vec::new(),
                        }
                    })
                    .collect();
                types.classes.insert(
                    token.to_string(),
                    crate::typesys::ClassDef {
                        internal_name: token.to_string(),
                        rendered_name: token.to_string(),
                        docs: None,
                        fields,
                        constraints: Vec::new(),
                    },
                );
                FieldType::Class(token.to_string())
            } else {
                let inner = match schema.get("additionalProperties") {
                    Some(additional) if additional.is_object() => field_type_from_json_schema(
                        &format!("{token}_value"),
                        additional,
                        types,
                    ),
                    _ => FieldType::String,
                };
                FieldType::Map(Box::new(FieldType::String), Box::new(inner))
            }
        }
        _ => FieldType::String,
    }
}

// ---------------------------------------------------------------------------
// Interpreter boundary: RunError → PredictError, RunOutput → Predicted
// ---------------------------------------------------------------------------

/// Internal (non-provider, non-parse) failure surfaced through the historical
/// `PredictError::Lm { provider: "internal" }` shape.
fn internal_error(message: String) -> PredictError {
    PredictError::Lm {
        source: LmError::Provider {
            provider: "internal".to_string(),
            message,
            source: None,
        },
    }
}

/// Deserializes the canonical output map into the typed output struct —
/// failure is the historical `ParseError::ExtractionFailed` shape.
fn typed_output_from_map<S: Signature>(
    output_map: &Map<String, Value>,
    raw_response: &str,
) -> std::result::Result<S::Output, ParseError> {
    serde_json::from_value(Value::Object(output_map.clone())).map_err(|err| {
        ParseError::ExtractionFailed {
            field: "<all>".to_string(),
            raw_response: raw_response.to_string(),
            reason: err.to_string(),
        }
    })
}

/// Re-keys def-lane [`FieldMeta`](crate::FieldMeta) entries (canonical
/// `FieldDef::name`s) to the static lane's `rust_name` keying, preserving
/// schema field order — the user-visible `CallMetadata` contract.
fn translate_field_meta<S: Signature>(
    leaf_meta: &IndexMap<String, crate::FieldMeta>,
) -> IndexMap<String, crate::FieldMeta> {
    let mut field_meta = IndexMap::new();
    for field in S::schema().output_fields() {
        let leaf_name = field.path().iter().last().unwrap_or(field.lm_name);
        if let Some(meta) = leaf_meta.get(leaf_name) {
            field_meta.insert(field.rust_name.clone(), meta.clone());
        }
    }
    field_meta
}

/// The canonical (leaf) name of an output field → the static lane's
/// `rust_name` (dotted flatten path). Identity for non-flattened signatures.
fn output_rust_name<S: Signature>(canonical: &str) -> String {
    for field in S::schema().output_fields() {
        let leaf = field.path().iter().last().unwrap_or(field.lm_name);
        if leaf == canonical {
            return field.rust_name.clone();
        }
    }
    canonical.to_string()
}

/// Re-keys a def-lane [`ParseError`] (canonical `FieldDef::name`s) to the
/// static lane's `rust_name` keying, preserving the user-visible error shape
/// for flattened signatures.
fn translate_parse_error<S: Signature>(err: ParseError) -> ParseError {
    match err {
        ParseError::MissingField {
            field,
            raw_response,
        } => ParseError::MissingField {
            field: output_rust_name::<S>(&field),
            raw_response,
        },
        ParseError::ExtractionFailed {
            field,
            raw_response,
            reason,
        } => ParseError::ExtractionFailed {
            field: output_rust_name::<S>(&field),
            raw_response,
            reason,
        },
        ParseError::CoercionFailed {
            field,
            expected_type,
            raw_text,
            source,
        } => ParseError::CoercionFailed {
            field: output_rust_name::<S>(&field),
            expected_type,
            raw_text,
            source,
        },
        ParseError::AssertFailed {
            field,
            label,
            expression,
            value,
        } => ParseError::AssertFailed {
            field: output_rust_name::<S>(&field),
            label,
            expression,
            value,
        },
        ParseError::Multiple { errors, partial } => ParseError::Multiple {
            errors: errors.into_iter().map(translate_parse_error::<S>).collect(),
            partial,
        },
    }
}

/// Maps an interpreter [`RunError`] onto the historical [`PredictError`]
/// variants: `Lm`/`Parse`/`Replay` map structurally; everything else (input
/// validation, internal invariants) surfaces as an internal LM error.
fn map_run_error<S: Signature>(err: RunError) -> PredictError {
    match err {
        RunError::Lm { source, .. } => PredictError::Lm { source },
        RunError::Parse {
            raw,
            source,
            usage,
            ..
        } => PredictError::Parse {
            source: match source {
                Some(parse) => translate_parse_error::<S>(*parse),
                None => ParseError::ExtractionFailed {
                    field: "<all>".to_string(),
                    raw_response: raw.clone(),
                    reason: "response did not match the output signature".to_string(),
                },
            },
            raw_response: raw,
            lm_usage: usage,
        },
        RunError::Replay { source, .. } => PredictError::Replay { source },
        other => internal_error(other.to_string()),
    }
}

/// Reassembles the typed [`Predicted`] from a 1-node run: typed deserialize of
/// the output map + [`CallMetadata`] from the single [`LeafOutcome`](crate::ir::LeafOutcome),
/// with `FieldMeta` re-keyed from canonical names to the static lane's
/// `rust_name` keying (schema field order preserved).
#[allow(clippy::result_large_err)]
fn predicted_from_run<S: Signature>(run: RunOutput) -> Result<Predicted<S::Output>, PredictError>
where
    S::Output: Schema,
{
    let leaf = run.leaves.into_iter().next();
    let (raw_response, lm_usage, span_id, leaf_meta, tool_calls, tool_executions) = match leaf {
        Some(leaf) => (
            leaf.raw_response,
            leaf.usage,
            leaf.span_id,
            leaf.field_meta,
            leaf.tool_calls,
            leaf.tool_executions,
        ),
        None => (
            String::new(),
            LmUsage::default(),
            None,
            IndexMap::new(),
            Vec::new(),
            Vec::new(),
        ),
    };

    let typed: S::Output =
        typed_output_from_map::<S>(&run.output, &raw_response).map_err(|err| {
            PredictError::Parse {
                source: err,
                raw_response: raw_response.clone(),
                lm_usage,
            }
        })?;
    let field_meta = translate_field_meta::<S>(&leaf_meta);

    let checks_total = field_meta
        .values()
        .map(|meta| meta.checks.len())
        .sum::<usize>();
    let checks_failed = field_meta
        .values()
        .flat_map(|meta| meta.checks.iter())
        .filter(|check| !check.passed)
        .count();
    debug!(
        output_fields = field_meta.len(),
        checks_total, checks_failed, "typed parse completed"
    );

    let metadata = CallMetadata::new(
        raw_response,
        lm_usage,
        tool_calls,
        tool_executions,
        span_id,
        field_meta,
    );
    Ok(Predicted::new(typed, metadata))
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

impl<S: Signature> Predict<S> {
    /// Assigns the component name recorded on this predictor's trace spans.
    ///
    /// The leaf name is part of the 1-node program (and its hash), so the
    /// cached program, the overlay minted against it, and the loaded
    /// interpreter are all invalidated when the name changes.
    pub(crate) fn assign_trace_name(&mut self, name: &str) {
        if self.trace_name.as_deref() == Some(name) {
            return;
        }
        self.trace_name = Some(name.to_string());
        self.program = OnceLock::new();
        self.instance_overlay = OnceLock::new();
        *self.engine.get_mut() = None;
    }
}

impl<S> crate::core::PredictorInfo for Predict<S>
where
    S: Signature,
    S::Input: Schema,
    S::Output: Schema,
{
    fn schema(&self) -> &'static SignatureSchema {
        S::schema()
    }

    fn instruction(&self) -> String {
        self.instruction_override
            .clone()
            .unwrap_or_else(|| S::instruction().to_string())
    }

    fn default_instruction(&self) -> String {
        S::instruction().to_string()
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
            demos: crate::core::PredictorInfo::demos_as_json(self),
            instruction_override: self.instruction_override.clone(),
        }
    }

    fn load_state(&mut self, state: PredictState) -> Result<()> {
        // Convert demos before touching any state so a schema mismatch leaves
        // the predictor unchanged.
        let demos = state
            .demos
            .iter()
            .map(demo_from_json::<S>)
            .collect::<Result<Vec<_>>>()?;
        self.apply_state(Some(state.instruction_override), Some(demos));
        Ok(())
    }

    fn set_trace_name(&mut self, name: &str) {
        Predict::assign_trace_name(self, name);
    }
}

impl<S> crate::core::Predictors for Predict<S>
where
    S: Signature,
    S::Input: Schema,
    S::Output: Schema,
{
    /// A bare `Predict` used as a module is itself the one leaf. Its name is
    /// the assigned trace name when present, else `"self"`.
    fn predictors(&self) -> Vec<(String, &dyn crate::core::PredictorInfo)> {
        let name = self.trace_name.clone().unwrap_or_else(|| "self".to_string());
        vec![(name, self as &dyn crate::core::PredictorInfo)]
    }

    fn predictors_mut(&mut self) -> Vec<(String, &mut dyn crate::core::PredictorInfo)> {
        let name = self.trace_name.clone().unwrap_or_else(|| "self".to_string());
        vec![(name, self as &mut dyn crate::core::PredictorInfo)]
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
    fn predictor_info_load_state_round_trips_json_demo_rows() {
        use crate::core::PredictorInfo;

        let typed = typed_row("demo-input", "demo-output");
        let row = json_from_demo::<PredictConversionSig>(&typed)
            .expect("typed demo should convert to a flat row");
        let mut predictor = Predict::<PredictConversionSig>::new();

        PredictorInfo::load_state(
            &mut predictor,
            PredictState {
                demos: vec![row],
                instruction_override: None,
            },
        )
        .expect("predictor should accept JSON demo rows");

        let demos = PredictorInfo::demos_as_json(&predictor);
        assert_eq!(demos.len(), 1);
        assert_eq!(demos[0].get("prompt"), Some(&json!("demo-input")));
        assert_eq!(demos[0].get("answer"), Some(&json!("demo-output")));
    }
}
