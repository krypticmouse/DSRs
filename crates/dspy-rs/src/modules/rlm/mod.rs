use std::collections::BTreeSet;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use indexmap::IndexMap;
use pyo3::types::{
    PyAnyMethods, PyBool, PyDict, PyDictMethods, PyFloat, PyInt, PyList, PyListMethods, PyModule,
    PySet, PyString, PyStringMethods, PyTuple, PyTypeMethods,
};
use pyo3::{Bound, Py, Python};
use rig::message::ToolCall;
use tracing::{debug, info, info_span};

use crate::{
    BamlType, BamlValue, CallMetadata, Chat, ChatAdapter, Facet, FieldMeta, LmUsage, Module,
    Predict, PredictError, Predicted, Signature,
};

mod exec;
mod previews;
mod prompt;
mod py_bridge;
pub mod runtime;
mod submit;
mod tools;
use previews::render_previews;
use prompt::{render_action_instruction, render_extract_instruction};
pub use runtime::{
    DynRuntime, LlmTools, PyO3Runtime, RlmInputFields, RlmRuntime, StubRuntime, SubmitError,
    SubmitHandler, SubmitResultDyn, SubmitSlot, clear_submit_slot, take_submit_result,
};
pub use tools::LlmQuery;

const DEFAULT_MAX_ITERATIONS: usize = 20;
const DEFAULT_MAX_LLM_CALLS: usize = 50;
const DEFAULT_MAX_OUTPUT_CHARS: usize = 100_000;
const DEFAULT_ENABLE_EXTRACTION_FALLBACK: bool = true;
const MAX_RECOVERABLE_PARSE_SNIPPET_CHARS: usize = 80;

const REPL_HISTORY_INPUT_RENDER_TEMPLATE: &str = r#"{% if this.entries|length == 0 %}(no executed REPL turns captured){% else %}{% for entry in this.entries %}=== Turn {{ entry.turn }} ===
Code:
{{ entry.code }}

Output:
{% if entry.output %}{{ entry.output }}{% else %}<empty>{% endif %}{% if not loop.last %}

{% endif %}{% endfor %}{% endif %}"#;

#[derive(Signature, Clone, Debug)]
struct RlmActionSig {
    #[input]
    perception: String,

    #[output]
    code: String,
}

#[derive(Clone, Debug)]
#[BamlType]
struct REPLHistory {
    entries: Vec<REPLEntry>,
}

#[derive(Clone, Debug)]
#[BamlType]
struct REPLEntry {
    turn: u32,
    code: String,
    output: String,
}

#[derive(Clone, Debug)]
#[BamlType]
struct RlmExtractInput {
    variables_info: String,
    repl_history: REPLHistory,
}

struct RlmExtractSig<S: Signature>(PhantomData<S>);

impl<S> Signature for RlmExtractSig<S>
where
    S: Signature,
    S::Input: BamlType + for<'a> Facet<'a> + Clone + Send + Sync + RlmInputFields,
    S::Output: BamlType + for<'a> Facet<'a> + Clone + Send + Sync,
{
    type Input = RlmExtractInput;
    type Output = S::Output;

    fn instruction() -> &'static str {
        S::instruction()
    }

    fn input_shape() -> &'static facet::Shape {
        facet::shape_of::<RlmExtractInput>()
    }

    fn output_shape() -> &'static facet::Shape {
        facet::shape_of::<S::Output>()
    }

    fn input_field_metadata() -> &'static [crate::FieldMetadataSpec] {
        const INPUT_META: [crate::FieldMetadataSpec; 2] = [
            crate::FieldMetadataSpec {
                rust_name: "variables_info",
                alias: None,
                constraints: &[],
                input_render: crate::InputRenderSpec::Default,
            },
            crate::FieldMetadataSpec {
                rust_name: "repl_history",
                alias: None,
                constraints: &[],
                input_render: crate::InputRenderSpec::Jinja(REPL_HISTORY_INPUT_RENDER_TEMPLATE),
            },
        ];
        &INPUT_META
    }

    fn output_field_metadata() -> &'static [crate::FieldMetadataSpec] {
        S::output_field_metadata()
    }
}

#[derive(Debug, Clone, facet::Facet)]
#[facet(crate = facet)]
pub struct RlmConfig {
    pub max_iterations: usize,
    pub max_llm_calls: usize,
    pub max_output_chars: usize,
    pub enable_extraction_fallback: bool,
}

impl Default for RlmConfig {
    fn default() -> Self {
        Self {
            max_iterations: DEFAULT_MAX_ITERATIONS,
            max_llm_calls: DEFAULT_MAX_LLM_CALLS,
            max_output_chars: DEFAULT_MAX_OUTPUT_CHARS,
            enable_extraction_fallback: DEFAULT_ENABLE_EXTRACTION_FALLBACK,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct MetadataAcc {
    lm_usage: LmUsage,
    tool_calls: Vec<ToolCall>,
    tool_executions: Vec<String>,
    raw_responses: Vec<String>,
    field_meta: IndexMap<String, FieldMeta>,
}

impl MetadataAcc {
    fn absorb_call_metadata(&mut self, metadata: CallMetadata) {
        self.lm_usage = std::mem::take(&mut self.lm_usage) + metadata.lm_usage;
        self.tool_calls.extend(metadata.tool_calls);
        self.tool_executions.extend(metadata.tool_executions);
        self.raw_responses.push(metadata.raw_response);
        self.field_meta.extend(metadata.field_meta);
    }

    fn absorb_parse_metadata(&mut self, raw_response: String, lm_usage: LmUsage) {
        self.lm_usage = std::mem::take(&mut self.lm_usage) + lm_usage;
        self.raw_responses.push(raw_response);
    }

    fn into_call_metadata(self) -> CallMetadata {
        let raw_response = if self.raw_responses.is_empty() {
            String::new()
        } else {
            self.raw_responses.join("\n\n")
        };

        CallMetadata::new(
            raw_response,
            self.lm_usage,
            self.tool_calls,
            self.tool_executions,
            None,
            self.field_meta,
        )
    }
}

enum ActionTurn {
    Parsed(Predicted<RlmActionSigOutput>),
    RecoverableParse {
        raw_response: String,
        lm_usage: LmUsage,
        chat: Chat,
        reason: String,
    },
}

enum ExecOutcome {
    Continue {
        output: String,
    },
    SubmitAccepted {
        value: BamlValue,
        field_meta: IndexMap<String, FieldMeta>,
    },
    SubmitValidationError {
        message: String,
        errors: Vec<String>,
        raw_output: String,
    },
    SubmitAssertionFailed {
        label: String,
        expression: String,
        raw_output: String,
    },
    PythonException {
        message: String,
    },
    RecoverableParse {
        message: String,
    },
}

enum TurnDecision {
    Continue,
    Finalization,
    Fallback,
}

#[derive(Debug, Clone, Default)]
struct PerceptionFeedback {
    stdout: Option<String>,
    stderr: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum RlmError {
    #[error("configuration error: {message}")]
    Configuration { message: String },

    #[error("action predict failed")]
    ActionPredict {
        #[source]
        source: PredictError,
    },

    #[error("python execution failed: {message}")]
    PythonExec { message: String },

    #[error("extraction fallback failed")]
    ExtractFallback {
        #[source]
        source: PredictError,
    },

    #[error("max iterations reached ({max})")]
    MaxIterationsReached { max: usize },

    #[error("internal invariant violated: {message}")]
    Invariant { message: String },
}

impl From<RlmError> for PredictError {
    fn from(value: RlmError) -> Self {
        match value {
            RlmError::ActionPredict { source } => source,
            RlmError::ExtractFallback { source } => source,
            other => PredictError::Module {
                module: "Rlm",
                source: Box::new(other),
            },
        }
    }
}

#[derive(facet::Facet)]
#[facet(crate = facet)]
pub struct Rlm<S>
where
    S: Signature,
    S::Input: BamlType + for<'a> Facet<'a> + Clone + Send + Sync + RlmInputFields,
    S::Output: BamlType + for<'a> Facet<'a> + Clone + Send + Sync,
{
    extract: Predict<RlmExtractSig<S>>,

    #[facet(skip)]
    config: RlmConfig,
    #[facet(skip)]
    instruction_override: Option<String>,
    #[facet(skip, opaque)]
    sub_lm: Option<Arc<crate::LM>>,
    #[facet(skip, opaque)]
    runtime: Arc<dyn RlmRuntime<S>>,
}

impl<S> Default for Rlm<S>
where
    S: Signature,
    S::Input: BamlType + for<'a> Facet<'a> + Clone + Send + Sync + RlmInputFields,
    S::Output: BamlType + for<'a> Facet<'a> + Clone + Send + Sync,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<S> Rlm<S>
where
    S: Signature,
    S::Input: BamlType + for<'a> Facet<'a> + Clone + Send + Sync + RlmInputFields,
    S::Output: BamlType + for<'a> Facet<'a> + Clone + Send + Sync,
{
    pub fn new() -> Self {
        Self::builder().build()
    }

    pub fn builder() -> RlmBuilder<S> {
        RlmBuilder::new()
    }

    pub async fn call(&self, input: S::Input) -> Result<Predicted<S::Output>, PredictError> {
        self.forward(input).await
    }

    pub async fn forward(&self, input: S::Input) -> Result<Predicted<S::Output>, PredictError> {
        self.run_loop(&input).await.map_err(Into::into)
    }

    async fn run_loop(&self, input: &S::Input) -> Result<Predicted<S::Output>, RlmError> {
        if self.config.max_iterations == 0 {
            return Err(RlmError::Configuration {
                message: "max_iterations must be >= 1".to_string(),
            });
        }
        info!(
            max_iterations = self.config.max_iterations,
            max_llm_calls = self.config.max_llm_calls,
            max_output_chars = self.config.max_output_chars,
            extraction_fallback = self.config.enable_extraction_fallback,
            "rlm run started"
        );

        let submit_slot: SubmitSlot = Arc::new(Mutex::new(None));
        let submit_handler = SubmitHandler::new::<S>(Arc::clone(&submit_slot));
        let sub_lm = self.sub_lm.clone().or_else(|| {
            let guard = crate::GLOBAL_SETTINGS.read().ok()?;
            guard.as_ref().map(|settings| Arc::clone(&settings.lm))
        });
        if self.runtime.requires_sub_lm_tools() && sub_lm.is_none() {
            return Err(RlmError::Configuration {
                message: "Rlm runtime requires a configured sub-LM (global configure() or builder.sub_lm(...))"
                    .to_string(),
            });
        }
        let llm_tools = if self.runtime.requires_sub_lm_tools() {
            Some(LlmTools::with_budget(
                sub_lm.expect("sub_lm present when required by runtime"),
                self.config.max_llm_calls,
                tokio::runtime::Handle::try_current().map_err(|err| RlmError::Configuration {
                    message: format!("Rlm requires an active Tokio runtime handle: {err}"),
                })?,
            ))
        } else {
            None
        };
        let input_fields = input.rlm_field_names().len();
        let setup = {
            let _inject_span = info_span!(
                "rlm.inject",
                input_fields,
                sub_lm_tools = llm_tools.is_some()
            )
            .entered();
            Python::attach(|py| {
                self.runtime.setup_interpreter_globals(
                    py,
                    input,
                    &submit_handler,
                    llm_tools.as_ref(),
                )
            })
        }
        .map_err(|err| RlmError::Configuration {
            message: err.to_string(),
        })?;
        debug!(
            input_fields,
            injected_objects = setup.methods_by_var.len(),
            "interpreter globals injected"
        );
        let globals = setup.globals;

        let preview_span = info_span!(
            "rlm.preview",
            input_fields,
            preview_len = tracing::field::Empty
        );
        let previews = {
            let _preview_span = preview_span.enter();
            render_previews::<S>(input, &setup.methods_by_var)
        };
        let preview_len = previews.chars().count();
        preview_span.record("preview_len", preview_len);
        info!(preview_len, "rlm preview generated");

        let action_instruction = render_action_instruction::<S>(
            &self.config,
            self.instruction_override.as_deref(),
            &previews,
        );
        // TODO(dsrs-rlm): This local Predict is a runtime-workaround so instruction
        // composition can include runtime-collected method metadata and rendered
        // input schemas. Structural fix options:
        // 1) public post-build instruction override on Predict, or
        // 2) build-time instruction composition using compile-time method metadata.
        let generate_action = Predict::<RlmActionSig>::builder()
            .instruction(action_instruction)
            .adapter(ChatAdapter::passthrough())
            .build();
        let task_hint = task_hint_from_input::<S>(input).unwrap_or_else(|| {
            if let Some(instruction) = self.instruction_override.as_deref() {
                instruction.trim().to_string()
            } else {
                S::instruction().trim().to_string()
            }
        });

        let mut history: Option<Chat> = None;
        let mut feedback: Option<PerceptionFeedback> = None;
        let mut turn_index = 1usize;
        let mut acc = MetadataAcc::default();
        let mut repl_history = REPLHistory {
            entries: Vec::new(),
        };

        loop {
            let is_first_turn = turn_index == 1;
            let _turn_span = info_span!(
                "rlm.turn",
                iteration = turn_index,
                first_turn = is_first_turn
            )
            .entered();
            match self.decide_turn_policy(turn_index, self.config.max_iterations) {
                TurnDecision::Fallback => {
                    if self.config.enable_extraction_fallback {
                        return self
                            .run_extraction_fallback(&previews, repl_history, &mut acc)
                            .await;
                    }
                    return Err(RlmError::MaxIterationsReached {
                        max: self.config.max_iterations,
                    });
                }
                TurnDecision::Continue | TurnDecision::Finalization => {}
            }

            let budget_remaining = self
                .config
                .max_iterations
                .saturating_sub(turn_index)
                .saturating_add(1);
            let sub_lm_remaining = self.runtime.sub_lm_budget_remaining(llm_tools.as_ref());
            let perception = Python::attach(|py| {
                build_perception_message::<S>(
                    py,
                    &globals,
                    input,
                    &task_hint,
                    feedback.as_ref(),
                    budget_remaining,
                    sub_lm_remaining,
                    is_first_turn,
                )
            })
            .map_err(|message| RlmError::Configuration { message })?;
            let action_input = RlmActionSigInput::new(perception);

            info!(
                iteration = turn_index,
                first_turn = is_first_turn,
                budget_remaining,
                "running action predict call"
            );
            let turn_history = history.take();
            match self
                .run_action_turn(&generate_action, action_input, turn_history)
                .await?
            {
                ActionTurn::RecoverableParse {
                    raw_response,
                    lm_usage,
                    chat,
                    reason,
                } => {
                    debug!(
                        iteration = turn_index,
                        response_kind = "error",
                        error_kind = "recoverable_parse",
                        "predict response received"
                    );
                    acc.absorb_parse_metadata(raw_response, lm_usage);
                    history = Some(chat);
                    feedback = Some(PerceptionFeedback {
                        stdout: None,
                        stderr: Some(reason),
                    });
                    turn_index += 1;
                }
                ActionTurn::Parsed(predicted) => {
                    let (action_output, action_metadata, action_chat) = predicted.into_parts();
                    acc.absorb_call_metadata(action_metadata);
                    history = Some(action_chat);

                    let code = action_output.code;
                    clear_submit_slot(&submit_slot);

                    let exec_result = Python::attach(|py| {
                        self.runtime.execute_repl_code(
                            py,
                            &globals,
                            &code,
                            self.config.max_output_chars,
                        )
                    });
                    let submit_result = take_submit_result(&submit_slot);
                    let outcome = classify_exec_outcome(exec_result, submit_result);

                    match outcome {
                        ExecOutcome::SubmitAccepted { value, field_meta } => {
                            info!(
                                iteration = turn_index,
                                response_kind = "submit",
                                "predict response received"
                            );
                            let typed_output =
                                S::Output::try_from_baml_value(value).map_err(|err| {
                                    RlmError::Invariant {
                                        message: format!(
                                            "SUBMIT produced invalid output value: {err}"
                                        ),
                                    }
                                })?;
                            acc.field_meta.extend(field_meta);

                            let final_chat = history.unwrap_or_else(|| Chat::new(vec![]));
                            return Ok(Predicted::new(
                                typed_output,
                                acc.into_call_metadata(),
                                final_chat,
                            ));
                        }
                        other => {
                            debug!(
                                iteration = turn_index,
                                response_kind = predict_response_kind_from_outcome(&other),
                                outcome = exec_outcome_kind(&other),
                                "predict response received"
                            );
                            feedback = Some(perception_feedback_from_outcome(&other));
                            repl_history.entries.push(REPLEntry {
                                turn: turn_index.min(u32::MAX as usize) as u32,
                                code,
                                output: outcome_to_raw_output(&other),
                            });
                            turn_index += 1;
                        }
                    }
                }
            }
        }
    }

    async fn run_action_turn(
        &self,
        generate_action: &Predict<RlmActionSig>,
        action_input: RlmActionSigInput,
        history: Option<Chat>,
    ) -> Result<ActionTurn, RlmError> {
        match generate_action.forward(action_input, history).await {
            Ok(predicted) => Ok(ActionTurn::Parsed(predicted)),
            Err(error) => match error {
                PredictError::Parse {
                    source,
                    raw_response,
                    lm_usage,
                    chat,
                } if raw_response.trim().is_empty() => {
                    let reason = format_empty_response_recovery_reason(&raw_response, &source);
                    Ok(ActionTurn::RecoverableParse {
                        raw_response,
                        lm_usage,
                        chat,
                        reason,
                    })
                }
                other => Err(RlmError::ActionPredict { source: other }),
            },
        }
    }

    fn decide_turn_policy(&self, turn_index: usize, max_iterations: usize) -> TurnDecision {
        if turn_index < max_iterations {
            TurnDecision::Continue
        } else if turn_index == max_iterations {
            TurnDecision::Finalization
        } else {
            TurnDecision::Fallback
        }
    }

    async fn run_extraction_fallback(
        &self,
        previews: &str,
        repl_history: REPLHistory,
        acc: &mut MetadataAcc,
    ) -> Result<Predicted<S::Output>, RlmError> {
        let extract_input = RlmExtractInput {
            variables_info: previews.to_string(),
            repl_history,
        };
        let predicted = self
            .extract
            .forward(extract_input, None)
            .await
            .map_err(|source| RlmError::ExtractFallback { source })?;
        let (output, metadata, chat) = predicted.into_parts();
        acc.absorb_call_metadata(metadata);
        let metadata = std::mem::take(acc).into_call_metadata();
        Ok(Predicted::new(output, metadata, chat))
    }
}

impl<S> Module for Rlm<S>
where
    S: Signature,
    S::Input: BamlType + for<'a> Facet<'a> + Clone + Send + Sync + RlmInputFields,
    S::Output: BamlType + for<'a> Facet<'a> + Clone + Send + Sync,
{
    type Input = S::Input;
    type Output = S::Output;

    async fn forward(&self, input: S::Input) -> Result<Predicted<S::Output>, PredictError> {
        Rlm::forward(self, input).await
    }
}

pub struct RlmBuilder<S>
where
    S: Signature,
    S::Input: BamlType + for<'a> Facet<'a> + Clone + Send + Sync + RlmInputFields,
    S::Output: BamlType + for<'a> Facet<'a> + Clone + Send + Sync,
{
    config: RlmConfig,
    instruction_override: Option<String>,
    sub_lm: Option<Arc<crate::LM>>,
    runtime: Option<Arc<dyn RlmRuntime<S>>>,
    _marker: PhantomData<S>,
}

impl<S> RlmBuilder<S>
where
    S: Signature,
    S::Input: BamlType + for<'a> Facet<'a> + Clone + Send + Sync + RlmInputFields,
    S::Output: BamlType + for<'a> Facet<'a> + Clone + Send + Sync,
{
    fn new() -> Self {
        Self {
            config: RlmConfig::default(),
            instruction_override: None,
            sub_lm: None,
            runtime: None,
            _marker: PhantomData,
        }
    }

    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.config.max_iterations = max_iterations;
        self
    }

    pub fn max_llm_calls(mut self, max_llm_calls: usize) -> Self {
        self.config.max_llm_calls = max_llm_calls;
        self
    }

    pub fn max_output_chars(mut self, max_output_chars: usize) -> Self {
        self.config.max_output_chars = max_output_chars;
        self
    }

    pub fn enable_extraction_fallback(mut self, enable_extraction_fallback: bool) -> Self {
        self.config.enable_extraction_fallback = enable_extraction_fallback;
        self
    }

    pub fn instruction(mut self, instruction: impl Into<String>) -> Self {
        self.instruction_override = Some(instruction.into());
        self
    }

    pub fn sub_lm(mut self, sub_lm: Arc<crate::LM>) -> Self {
        self.sub_lm = Some(sub_lm);
        self
    }

    pub fn runtime(mut self, runtime: Arc<dyn RlmRuntime<S>>) -> Self {
        self.runtime = Some(runtime);
        self
    }

    pub fn build(self) -> Rlm<S> {
        let extract_instruction =
            render_extract_instruction::<S>(self.instruction_override.as_deref());
        let extract = Predict::<RlmExtractSig<S>>::builder()
            .instruction(extract_instruction)
            .adapter(ChatAdapter::new())
            .build();

        let runtime = self
            .runtime
            .unwrap_or_else(|| default_runtime::<S>(self.config.max_llm_calls));

        Rlm {
            extract,
            config: self.config,
            instruction_override: self.instruction_override,
            sub_lm: self.sub_lm,
            runtime,
        }
    }
}

fn default_runtime<S: Signature>(max_llm_calls: usize) -> Arc<dyn RlmRuntime<S>>
where
    S::Input: BamlType + for<'a> Facet<'a> + Clone + Send + Sync + RlmInputFields,
    S::Output: BamlType + for<'a> Facet<'a> + Clone + Send + Sync,
{
    if let Ok(runtime_override) = std::env::var("DSPY_RS_RLM_RUNTIME") {
        match runtime_override.trim().to_ascii_lowercase().as_str() {
            "stub" => return Arc::new(StubRuntime::new(max_llm_calls)),
            "pyo3" => return Arc::new(PyO3Runtime),
            _ => {}
        }
    }

    #[cfg(test)]
    {
        Arc::new(StubRuntime::new(max_llm_calls))
    }
    #[cfg(not(test))]
    {
        let _ = max_llm_calls;
        Arc::new(PyO3Runtime)
    }
}

fn task_hint_from_input<S>(input: &S::Input) -> Option<String>
where
    S: Signature,
    S::Input: BamlType,
{
    let value = input.to_baml_value();
    let question = match &value {
        BamlValue::Class(_, fields) | BamlValue::Map(fields) => fields.get("question"),
        _ => None,
    }?;
    if let BamlValue::String(text) = question {
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }
    None
}

fn build_perception_message<S>(
    py: Python<'_>,
    globals: &Py<PyDict>,
    input: &S::Input,
    task_hint: &str,
    feedback: Option<&PerceptionFeedback>,
    budget_remaining: usize,
    sub_lm_remaining: usize,
    first_turn: bool,
) -> Result<String, String>
where
    S: Signature,
    S::Input: BamlType + RlmInputFields,
{
    let mut lines = Vec::new();
    let turns_label = if budget_remaining == 1 {
        "1 turn".to_string()
    } else {
        format!("{budget_remaining} turns")
    };
    lines.push(format!(
        "[env] {turns_label} | {sub_lm_remaining} sub-LLM calls"
    ));

    if first_turn {
        lines.push(format!("[query] {}", truncate_chars(task_hint, 180)));
    }

    if let Some(feedback) = feedback {
        if let Some(stdout) = feedback.stdout.as_deref()
            && !stdout.trim().is_empty()
        {
            lines.push(String::new());
            lines.push("[stdout]".to_string());
            lines.push(stdout.to_string());
        }
        if let Some(stderr) = feedback.stderr.as_deref()
            && !stderr.trim().is_empty()
        {
            lines.push(String::new());
            lines.push("[stderr]".to_string());
            lines.push(stderr.to_string());
        }
    }

    if budget_remaining == 1 {
        lines.push(String::new());
        lines.push("⚠ LAST TURN — you MUST call SUBMIT() now with your best answer.".to_string());
    }

    let namespace = collect_namespace_snapshot(py, globals, input.rlm_field_names())?;
    lines.push(String::new());
    if first_turn {
        lines.push("--- namespace ---".to_string());
    } else {
        lines.push(format!("--- namespace ({} names) ---", namespace.len()));
    }
    for (name, repr_value) in namespace {
        lines.push(format!("{name} = {repr_value}"));
    }
    lines.push(String::new());
    lines.push(">>>".to_string());

    Ok(lines.join("\n"))
}

fn collect_namespace_snapshot(
    py: Python<'_>,
    globals: &Py<PyDict>,
    injected_roots: &[&str],
) -> Result<Vec<(String, String)>, String> {
    let dict = globals.bind(py);
    let roots = injected_roots
        .iter()
        .map(|name| (*name).to_string())
        .collect::<BTreeSet<_>>();

    let mut out = Vec::new();
    for root in injected_roots {
        if let Some(value) = dict
            .get_item(*root)
            .map_err(|err| format!("failed to fetch root `{root}` from globals: {err}"))?
        {
            out.push(((*root).to_string(), safe_namespace_repr(&value, true)?));
        }
    }

    let mut extras = Vec::new();
    for (name, value) in dict.iter() {
        let Ok(name) = name.extract::<String>() else {
            continue;
        };
        if roots.contains(name.as_str()) {
            continue;
        }
        if !include_in_namespace(name.as_str(), &value, &roots) {
            continue;
        }
        extras.push((name, safe_namespace_repr(&value, false)?));
    }
    extras.sort_by(|a, b| a.0.cmp(&b.0));
    out.extend(extras);

    Ok(out)
}

fn include_in_namespace(
    name: &str,
    value: &Bound<'_, pyo3::PyAny>,
    roots: &BTreeSet<String>,
) -> bool {
    if roots.contains(name) {
        return true;
    }
    if name.starts_with('_') {
        return false;
    }
    if name.chars().count() <= 1 {
        return false;
    }
    if value.is_instance_of::<PyModule>() {
        return false;
    }
    if value.is_callable() {
        return false;
    }
    true
}

fn safe_namespace_repr(value: &Bound<'_, pyo3::PyAny>, is_root: bool) -> Result<String, String> {
    if is_root {
        if value.is_instance_of::<PyList>() {
            let len = value.len().unwrap_or_default();
            if len > 5 {
                if let Ok(list) = value.cast::<PyList>() {
                    let mut preview = Vec::new();
                    for item in list.iter().take(2) {
                        let rendered = sanitize_python_surface(&repr_value(&item)?);
                        preview.push(truncate_chars(&rendered, 100));
                    }
                    if !preview.is_empty() {
                        return Ok(format!("[{}, ... ({} total)]", preview.join(", "), len));
                    }
                }
                return Ok(format!("list({len} items)"));
            }
        }
        return Ok(truncate_chars(&repr_value(value)?, 200));
    }

    if value.is_instance_of::<PyString>() {
        let text = value
            .extract::<String>()
            .map_err(|err| format!("string extract failed: {err}"))?;
        return Ok(format!("{:?}", truncate_chars(&text, 50)));
    }
    if value.is_instance_of::<PyBool>()
        || value.is_instance_of::<PyInt>()
        || value.is_instance_of::<PyFloat>()
    {
        return repr_value(value);
    }

    if value.is_instance_of::<PyList>() {
        let len = value.len().unwrap_or_default();
        if len <= 5 {
            return Ok(truncate_chars(
                &sanitize_python_surface(&repr_value(value)?),
                120,
            ));
        }
        return Ok(format!("<list of {len} items>"));
    }
    if value.is_instance_of::<PyTuple>() {
        let len = value.len().unwrap_or_default();
        if len <= 5 {
            return Ok(truncate_chars(
                &sanitize_python_surface(&repr_value(value)?),
                120,
            ));
        }
        return Ok(format!("<tuple of {len} items>"));
    }
    if value.is_instance_of::<PySet>() {
        let len = value.len().unwrap_or_default();
        if len <= 5 {
            return Ok(truncate_chars(
                &sanitize_python_surface(&repr_value(value)?),
                120,
            ));
        }
        return Ok(format!("<set of {len} items>"));
    }
    if value.is_instance_of::<PyDict>() {
        let len = value.len().unwrap_or_default();
        if len <= 5 {
            return Ok(truncate_chars(
                &sanitize_python_surface(&repr_value(value)?),
                120,
            ));
        }
        return Ok(format!("<dict of {len} items>"));
    }

    let class_name = value
        .get_type()
        .name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|_| "Object".to_string());

    if let Ok(len) = value.len() {
        return Ok(format!("<{class_name}: {len} items>"));
    }

    Ok(format!("<{class_name}>"))
}

fn repr_value(value: &Bound<'_, pyo3::PyAny>) -> Result<String, String> {
    let repr = value
        .repr()
        .map_err(|err| format!("repr() failed: {err}"))?;
    Ok(repr.to_string_lossy().to_string())
}

fn sanitize_python_surface(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut token = String::new();

    let flush = |out: &mut String, token: &mut String| {
        if token.is_empty() {
            return;
        }
        if let Some(last) = token.rsplit("::").next() {
            out.push_str(last);
        } else {
            out.push_str(token);
        }
        token.clear();
    };

    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == ':' {
            token.push(ch);
        } else {
            flush(&mut out, &mut token);
            out.push(ch);
        }
    }
    flush(&mut out, &mut token);
    out
}

fn perception_feedback_from_outcome(outcome: &ExecOutcome) -> PerceptionFeedback {
    match outcome {
        ExecOutcome::Continue { output } => PerceptionFeedback {
            stdout: (!output.trim().is_empty()).then(|| output.clone()),
            stderr: None,
        },
        ExecOutcome::SubmitAccepted { .. } => PerceptionFeedback::default(),
        ExecOutcome::SubmitValidationError { .. }
        | ExecOutcome::SubmitAssertionFailed { .. }
        | ExecOutcome::PythonException { .. }
        | ExecOutcome::RecoverableParse { .. } => PerceptionFeedback {
            stdout: None,
            stderr: Some(outcome_to_raw_output(outcome)),
        },
    }
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let mut out = String::new();
    for ch in text.chars().take(max_chars) {
        out.push(ch);
    }
    out.push_str("...");
    out
}

#[cfg(test)]
fn recoverable_outcome_from_parse_error(error: &PredictError) -> Option<(String, Chat)> {
    match error {
        PredictError::Parse {
            raw_response,
            chat,
            source,
            ..
        } if raw_response.trim().is_empty() => Some((
            format_empty_response_recovery_reason(raw_response, source),
            chat.clone(),
        )),
        _ => None,
    }
}

fn format_empty_response_recovery_reason(
    raw_response: &str,
    source: &impl std::fmt::Display,
) -> String {
    let total_chars = raw_response.chars().count();
    let mut snippet = raw_response
        .chars()
        .take(MAX_RECOVERABLE_PARSE_SNIPPET_CHARS)
        .collect::<String>();
    if total_chars > MAX_RECOVERABLE_PARSE_SNIPPET_CHARS {
        snippet.push_str("...");
    }

    format!(
        "Empty response from model ({source}). Write executable Python code. Raw response: len={total_chars}, snippet={snippet:?}."
    )
}

fn classify_exec_outcome(
    exec_result: Result<String, String>,
    submit_result: Option<SubmitResultDyn>,
) -> ExecOutcome {
    if let Some(submit_result) = submit_result {
        let raw_exec_output = match exec_result {
            Ok(output) => output,
            Err(message) => message,
        };
        return match submit_result {
            Ok((value, field_meta)) => ExecOutcome::SubmitAccepted { value, field_meta },
            Err(SubmitError::ValidationError { message, errors }) => {
                ExecOutcome::SubmitValidationError {
                    message,
                    errors,
                    raw_output: raw_exec_output,
                }
            }
            Err(SubmitError::AssertionFailed { label, expression }) => {
                ExecOutcome::SubmitAssertionFailed {
                    label,
                    expression,
                    raw_output: raw_exec_output,
                }
            }
        };
    }

    match exec_result {
        Ok(output) => ExecOutcome::Continue { output },
        Err(message) => ExecOutcome::PythonException { message },
    }
}

fn predict_response_kind_from_outcome(outcome: &ExecOutcome) -> &'static str {
    match outcome {
        ExecOutcome::SubmitAccepted { .. } => "submit",
        ExecOutcome::Continue { .. } => "code",
        ExecOutcome::SubmitValidationError { .. }
        | ExecOutcome::SubmitAssertionFailed { .. }
        | ExecOutcome::PythonException { .. }
        | ExecOutcome::RecoverableParse { .. } => "error",
    }
}

fn exec_outcome_kind(outcome: &ExecOutcome) -> &'static str {
    match outcome {
        ExecOutcome::Continue { .. } => "continue",
        ExecOutcome::SubmitAccepted { .. } => "submit_accepted",
        ExecOutcome::SubmitValidationError { .. } => "submit_validation_error",
        ExecOutcome::SubmitAssertionFailed { .. } => "submit_assertion_failed",
        ExecOutcome::PythonException { .. } => "python_exception",
        ExecOutcome::RecoverableParse { .. } => "recoverable_parse",
    }
}

fn outcome_to_raw_output(outcome: &ExecOutcome) -> String {
    match outcome {
        ExecOutcome::Continue { output, .. } => output.clone(),
        ExecOutcome::SubmitAccepted { .. } => String::new(),
        ExecOutcome::SubmitValidationError {
            message,
            errors,
            raw_output,
        } => {
            if !raw_output.is_empty() {
                return raw_output.clone();
            }
            if errors.is_empty() {
                message.clone()
            } else {
                format!("{message}\n{}", errors.join("\n"))
            }
        }
        ExecOutcome::SubmitAssertionFailed {
            label,
            expression,
            raw_output,
        } => {
            if !raw_output.is_empty() {
                return raw_output.clone();
            }
            format!("Submit assertion failed: `{label}` ({expression})")
        }
        ExecOutcome::PythonException { message } => message.clone(),
        ExecOutcome::RecoverableParse { message } => message.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ParseError, Signature};
    use pyo3::Python;
    use pyo3::types::{PyDict, PyDictMethods, PyModule};
    use std::sync::Arc;
    use temp_env::with_var;

    #[derive(Signature, Clone, Debug)]
    struct RuntimePolicySig {
        #[input]
        prompt: String,
        #[output]
        answer: String,
    }

    #[test]
    fn default_runtime_in_tests_uses_stub_policy() {
        let runtime = default_runtime::<RuntimePolicySig>(3);
        assert!(
            !runtime.requires_sub_lm_tools(),
            "test default runtime should be StubRuntime without required sub-LM tools"
        );
    }

    #[test]
    fn default_runtime_override_to_pyo3_is_explicit() {
        with_var("DSPY_RS_RLM_RUNTIME", Some("pyo3"), || {
            let runtime = default_runtime::<RuntimePolicySig>(3);
            assert!(
                runtime.requires_sub_lm_tools(),
                "explicit pyo3 override should require sub-LM tools"
            );
        });
    }

    #[test]
    fn default_runtime_override_to_stub_is_explicit() {
        with_var("DSPY_RS_RLM_RUNTIME", Some("stub"), || {
            let runtime = default_runtime::<RuntimePolicySig>(3);
            assert!(
                !runtime.requires_sub_lm_tools(),
                "explicit stub override should not require sub-LM tools"
            );
        });
    }

    #[test]
    fn perception_message_uses_env_task_namespace_and_prompt_markers() {
        Python::attach(|py| {
            let globals = PyDict::new(py);
            globals
                .set_item("prompt", "Where did signal drop?")
                .expect("set prompt");
            globals
                .set_item("result_count", 7)
                .expect("set result_count");
            globals.set_item("_tmp", 99).expect("set tmp");

            let input = RuntimePolicySigInput {
                prompt: "Where did signal drop?".to_string(),
            };
            let message = build_perception_message::<RuntimePolicySig>(
                py,
                &globals.unbind(),
                &input,
                "Inspect trajectories",
                None,
                3,
                11,
                true,
            )
            .expect("message");

            assert!(message.contains("[env] 3 turns | 11 sub-LLM calls"));
            assert!(message.contains("[query] Inspect trajectories"));
            assert!(message.contains("--- namespace ---"));
            assert!(message.contains("prompt ="));
            assert!(message.contains("result_count = 7"));
            assert!(!message.contains("_tmp ="));
            assert!(message.ends_with(">>>"));
        });
    }

    #[test]
    fn perception_message_turn_two_includes_stdout_and_last_turn_warning() {
        Python::attach(|py| {
            let globals = PyDict::new(py);
            globals.set_item("prompt", "x").expect("set prompt");
            let input = RuntimePolicySigInput {
                prompt: "x".to_string(),
            };
            let feedback = PerceptionFeedback {
                stdout: Some("computed summary".to_string()),
                stderr: None,
            };

            let message = build_perception_message::<RuntimePolicySig>(
                py,
                &globals.unbind(),
                &input,
                "Inspect trajectories",
                Some(&feedback),
                1,
                3,
                false,
            )
            .expect("message");

            assert!(message.contains("[env] 1 turn | 3 sub-LLM calls"));
            assert!(message.contains("[stdout]"));
            assert!(message.contains("computed summary"));
            assert!(
                message.contains("⚠ LAST TURN — you MUST call SUBMIT() now with your best answer.")
            );
            assert!(!message.contains("[query]"));
        });
    }

    #[test]
    fn namespace_filtering_excludes_noise_and_keeps_roots() {
        Python::attach(|py| {
            let globals = PyDict::new(py);
            globals
                .set_item("prompt", "Where did signal drop?")
                .expect("set prompt root");
            globals.set_item("i", 1).expect("set single char");
            globals
                .set_item("_scratch", "temp")
                .expect("set private name");

            let json_mod = PyModule::import(py, "json").expect("import json");
            globals
                .set_item("json", json_mod)
                .expect("set module variable");

            let builtins = PyModule::import(py, "builtins").expect("import builtins");
            let len_fn = builtins.getattr("len").expect("load len");
            globals
                .set_item("callable_fn", len_fn)
                .expect("set callable variable");

            globals
                .set_item("kept_value", 42)
                .expect("set regular value");

            let input = RuntimePolicySigInput {
                prompt: "Where did signal drop?".to_string(),
            };
            let message = build_perception_message::<RuntimePolicySig>(
                py,
                &globals.unbind(),
                &input,
                "Inspect trajectories",
                None,
                3,
                9,
                true,
            )
            .expect("message");

            assert!(message.contains("prompt ="));
            assert!(message.contains("kept_value = 42"));
            assert!(!message.contains("\ni = "));
            assert!(!message.contains("_scratch = "));
            assert!(!message.contains("json = "));
            assert!(!message.contains("callable_fn = "));
        });
    }

    #[test]
    fn sanitize_python_surface_strips_module_paths() {
        let rendered = sanitize_python_surface(
            "Sessions(items=[tanha::types::Session(id='abc')], kind=tanha::types::Kind::Fast)",
        );
        assert!(!rendered.contains("tanha::types::"));
        assert!(rendered.contains("Session(id='abc')"));
        assert!(rendered.contains("kind=Fast"));
    }

    #[test]
    fn root_namespace_repr_uses_object_repr_without_custom_heuristics() {
        Python::attach(|py| {
            let globals = PyDict::new(py);
            py.run(
                pyo3::ffi::c_str!(
                    "class Sessions:\n  def __repr__(self):\n    return 'Sessions(CUSTOM_REPR)'\nsessions = Sessions()\n"
                ),
                Some(&globals),
                Some(&globals),
            )
            .expect("python setup");
            let sessions = globals
                .get_item("sessions")
                .expect("sessions lookup should succeed")
                .expect("sessions should exist");
            let rendered = safe_namespace_repr(&sessions, true).expect("repr");
            assert_eq!(rendered, "Sessions(CUSTOM_REPR)");
        });
    }

    #[test]
    fn extract_signature_uses_custom_repl_history_render_template() {
        let fields = RlmExtractSig::<RuntimePolicySig>::input_field_metadata();
        assert_eq!(fields.len(), 2);
        match fields[1].input_render {
            crate::InputRenderSpec::Jinja(template) => {
                assert!(template.contains("=== Turn {{ entry.turn }} ==="));
                assert!(template.contains("Code:"));
                assert!(template.contains("Output:"));
            }
            other => panic!("expected jinja render template, got: {other:?}"),
        }
    }

    #[test]
    fn turn_policy_reserves_last_turn_for_finalization_then_fallback() {
        let module = Rlm::<RuntimePolicySig>::builder().build();

        assert!(matches!(
            module.decide_turn_policy(1, 3),
            TurnDecision::Continue
        ));
        assert!(matches!(
            module.decide_turn_policy(2, 3),
            TurnDecision::Continue
        ));
        assert!(matches!(
            module.decide_turn_policy(3, 3),
            TurnDecision::Finalization
        ));
        assert!(matches!(
            module.decide_turn_policy(4, 3),
            TurnDecision::Fallback
        ));
    }

    #[test]
    fn perception_feedback_maps_stdout_and_stderr_honestly() {
        let continue_feedback = perception_feedback_from_outcome(&ExecOutcome::Continue {
            output: "ok".to_string(),
        });
        assert_eq!(continue_feedback.stdout.as_deref(), Some("ok"));
        assert!(continue_feedback.stderr.is_none());

        let error_feedback = perception_feedback_from_outcome(&ExecOutcome::PythonException {
            message: "Traceback...".to_string(),
        });
        assert_eq!(error_feedback.stderr.as_deref(), Some("Traceback..."));
        assert!(error_feedback.stdout.is_none());
    }

    #[test]
    fn classify_exec_outcome_covers_all_variants_and_feedback_projection() {
        let continue_outcome = classify_exec_outcome(Ok("x\n".into()), None);
        assert!(matches!(
            continue_outcome,
            ExecOutcome::Continue { ref output } if output == "x\n"
        ));
        assert_eq!(outcome_to_raw_output(&continue_outcome), "x\n");

        let submit_ok = classify_exec_outcome(
            Ok(String::new()),
            Some(Ok((BamlValue::String("ok".to_string()), IndexMap::new()))),
        );
        assert!(matches!(submit_ok, ExecOutcome::SubmitAccepted { .. }));
        assert!(outcome_to_raw_output(&submit_ok).is_empty());

        let submit_validation = classify_exec_outcome(
            Err("Traceback...\nSubmitError".to_string()),
            Some(Err(SubmitError::ValidationError {
                message: "validation failed".to_string(),
                errors: vec!["field `answer` expected string".to_string()],
            })),
        );
        assert!(matches!(
            submit_validation,
            ExecOutcome::SubmitValidationError { .. }
        ));
        assert_eq!(
            outcome_to_raw_output(&submit_validation),
            "Traceback...\nSubmitError"
        );

        let submit_assert = classify_exec_outcome(
            Err("SubmitError: Assertion failed".to_string()),
            Some(Err(SubmitError::AssertionFailed {
                label: "non_empty".to_string(),
                expression: "this.len() > 0".to_string(),
            })),
        );
        assert!(matches!(
            submit_assert,
            ExecOutcome::SubmitAssertionFailed { .. }
        ));
        assert_eq!(
            outcome_to_raw_output(&submit_assert),
            "SubmitError: Assertion failed"
        );

        let python_exception = classify_exec_outcome(Err("Traceback...".into()), None);
        assert!(matches!(
            python_exception,
            ExecOutcome::PythonException { ref message } if message == "Traceback..."
        ));
        assert_eq!(outcome_to_raw_output(&python_exception), "Traceback...");

        let recoverable = ExecOutcome::RecoverableParse {
            message: "Your response was empty.".to_string(),
        };
        assert_eq!(
            outcome_to_raw_output(&recoverable),
            "Your response was empty."
        );
    }

    #[test]
    fn recoverable_parse_error_detection_only_triggers_on_empty_response() {
        let empty_err = PredictError::Parse {
            source: ParseError::ExtractionFailed {
                field: "code".to_string(),
                raw_response: String::new(),
                reason: "empty passthrough response".to_string(),
            },
            raw_response: "   \n\t".to_string(),
            lm_usage: LmUsage::default(),
            chat: Chat::new(vec![]),
        };
        let recovered = recoverable_outcome_from_parse_error(&empty_err)
            .expect("empty response should be recoverable");
        assert!(recovered.0.contains("Empty response from model"));
        assert!(recovered.0.contains("Raw response: len="));
        assert!(recovered.0.contains("\\n\\t"));

        let non_empty_err = PredictError::Parse {
            source: ParseError::ExtractionFailed {
                field: "code".to_string(),
                raw_response: "no code".to_string(),
                reason: "failed extraction".to_string(),
            },
            raw_response: "I refuse".to_string(),
            lm_usage: LmUsage::default(),
            chat: Chat::new(vec![]),
        };
        assert!(
            recoverable_outcome_from_parse_error(&non_empty_err).is_none(),
            "non-empty parse failures should remain terminal"
        );
    }

    #[tokio::test]
    async fn pyo3_runtime_requires_sub_lm_when_not_configured() {
        let module = Rlm::<RuntimePolicySig>::builder()
            .runtime(Arc::new(PyO3Runtime))
            .build();

        let err = module
            .call(RuntimePolicySigInput {
                prompt: "ping".to_string(),
            })
            .await
            .expect_err("missing sub-LM should fail before first action turn");
        match err {
            PredictError::Module { source, .. } => {
                assert!(
                    source.to_string().contains("configured sub-LM"),
                    "expected sub-LM config error, got: {source}"
                );
            }
            other => panic!("expected module error, got: {other}"),
        }
    }
}
