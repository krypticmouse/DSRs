use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use indexmap::IndexMap;
use pyo3::types::PyDict;
use pyo3::{Py, Python};
use rig::message::ToolCall;

use crate::{
    BamlType, BamlValue, CallMetadata, Chat, ChatAdapter, Facet, FieldMeta, LmUsage, Module,
    Predict, PredictError, Predicted, Signature, SignatureSchema,
};

mod exec;
mod py_bridge;
pub mod runtime;
mod submit;
mod tools;
pub use runtime::{
    DynRuntime, LlmTools, PyO3Runtime, RlmRuntime, StubRuntime, SubmitError, SubmitHandler,
    SubmitResultDyn, SubmitSlot, clear_submit_slot, take_submit_result,
};
pub use tools::LlmQuery;

const DEFAULT_MAX_ITERATIONS: usize = 20;
const DEFAULT_MAX_LLM_CALLS: usize = 50;
const DEFAULT_MAX_OUTPUT_CHARS: usize = 100_000;
const DEFAULT_ENABLE_EXTRACTION_FALLBACK: bool = true;

const ACTION_INSTRUCTION: &str = "You are operating inside a persistent Python REPL.\n\
Write executable Python code that advances the task.\n\
Use SUBMIT(field=value, ...) once you can return the final typed answer.\n\
Do not add prose or markdown fences unless needed by the task.";

const EXTRACT_INSTRUCTION: &str = "Extract the final typed answer from the REPL history.\n\
Use the expected output schema exactly.";

#[derive(Signature, Clone, Debug)]
struct RlmActionSig {
    #[input]
    variables_info: Option<String>,

    #[input]
    execution_feedback: Option<String>,

    #[input]
    budget_remaining: u32,

    #[output]
    code: String,
}

#[derive(Clone, Debug)]
#[BamlType]
pub struct REPLHistory {
    pub entries: Vec<REPLEntry>,
}

#[derive(Clone, Debug)]
#[BamlType]
pub struct REPLEntry {
    pub turn: u32,
    pub code: String,
    pub output: String,
}

#[derive(Clone, Debug)]
#[BamlType]
pub struct RlmExtractInput {
    pub variables_info: String,
    pub repl_history: REPLHistory,
}

pub struct RlmExtractSig<S: Signature>(PhantomData<S>);

impl<S> Signature for RlmExtractSig<S>
where
    S: Signature,
    S::Input: BamlType + for<'a> Facet<'a> + Clone + Send + Sync,
    S::Output: BamlType + for<'a> Facet<'a> + Clone + Send + Sync,
{
    type Input = RlmExtractInput;
    type Output = S::Output;

    fn instruction() -> &'static str {
        EXTRACT_INSTRUCTION
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
                input_render: crate::InputRenderSpec::Default,
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
pub struct MetadataAcc {
    pub lm_usage: LmUsage,
    pub tool_calls: Vec<ToolCall>,
    pub tool_executions: Vec<String>,
    pub raw_responses: Vec<String>,
    pub field_meta: IndexMap<String, FieldMeta>,
}

impl MetadataAcc {
    fn absorb_call_metadata(&mut self, metadata: CallMetadata) {
        self.lm_usage = self.lm_usage.clone() + metadata.lm_usage;
        self.tool_calls.extend(metadata.tool_calls);
        self.tool_executions.extend(metadata.tool_executions);
        self.raw_responses.push(metadata.raw_response);
        self.field_meta.extend(metadata.field_meta);
    }

    fn absorb_parse_metadata(&mut self, raw_response: String, lm_usage: LmUsage) {
        self.lm_usage = self.lm_usage.clone() + lm_usage;
        self.raw_responses.push(raw_response);
    }

    fn to_call_metadata(&self) -> CallMetadata {
        let raw_response = if self.raw_responses.is_empty() {
            String::new()
        } else {
            self.raw_responses.join("\n\n")
        };

        CallMetadata::new(
            raw_response,
            self.lm_usage.clone(),
            self.tool_calls.clone(),
            self.tool_executions.clone(),
            None,
            self.field_meta.clone(),
        )
    }
}

pub enum ActionTurn {
    Parsed(Predicted<RlmActionSigOutput>),
    RecoverableParse {
        raw_response: String,
        lm_usage: LmUsage,
        chat: Chat,
        reason: String,
    },
}

pub enum ExecOutcome {
    Continue {
        code: String,
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
    S::Input: BamlType + for<'a> Facet<'a> + Clone + Send + Sync,
    S::Output: BamlType + for<'a> Facet<'a> + Clone + Send + Sync,
{
    generate_action: Predict<RlmActionSig>,
    extract: Predict<RlmExtractSig<S>>,

    #[facet(skip)]
    config: RlmConfig,
    #[facet(skip, opaque)]
    sub_lm: Option<Arc<crate::LM>>,
    #[facet(skip, opaque)]
    runtime: Arc<dyn RlmRuntime<S>>,
}

impl<S> Default for Rlm<S>
where
    S: Signature,
    S::Input: BamlType + for<'a> Facet<'a> + Clone + Send + Sync,
    S::Output: BamlType + for<'a> Facet<'a> + Clone + Send + Sync,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<S> Rlm<S>
where
    S: Signature,
    S::Input: BamlType + for<'a> Facet<'a> + Clone + Send + Sync,
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

        let submit_slot: SubmitSlot = Arc::new(Mutex::new(None));
        let submit_handler = SubmitHandler::new::<S>(Arc::clone(&submit_slot));
        let sub_lm = self
            .sub_lm
            .clone()
            .or_else(|| {
                let guard = crate::GLOBAL_SETTINGS.read().ok()?;
                guard.as_ref().map(|settings| Arc::clone(&settings.lm))
            })
            .ok_or_else(|| RlmError::Configuration {
                message: "Rlm requires a configured LM (global configure() or builder.sub_lm(...))"
                    .to_string(),
            })?;
        let llm_tools = LlmTools::with_budget(
            sub_lm,
            self.config.max_llm_calls,
            tokio::runtime::Handle::try_current().map_err(|err| RlmError::Configuration {
                message: format!("Rlm requires an active Tokio runtime handle: {err}"),
            })?,
        );
        let globals: Py<PyDict> = Python::attach(|py| {
            self.runtime
                .setup_interpreter_globals(py, input, &submit_handler, &llm_tools)
        })
        .map_err(|err| RlmError::Configuration {
            message: err.to_string(),
        })?;

        let previews = render_previews::<S>(input);
        let mut history: Option<Chat> = None;
        let mut feedback: Option<String> = None;
        let mut turn_index = 1usize;
        let mut acc = MetadataAcc::default();
        let mut repl_history = REPLHistory {
            entries: Vec::new(),
        };

        loop {
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

            let mut execution_feedback = feedback.clone();
            if matches!(
                self.decide_turn_policy(turn_index, self.config.max_iterations),
                TurnDecision::Finalization
            ) {
                let directive = self.finalization_directive();
                execution_feedback = Some(match execution_feedback {
                    Some(existing) if !existing.is_empty() => format!("{existing}\n\n{directive}"),
                    _ => directive,
                });
            }

            let budget_remaining = self
                .config
                .max_iterations
                .saturating_sub(turn_index)
                .saturating_add(1);
            let action_input = self.build_action_input(
                turn_index,
                Some(previews.as_str()),
                execution_feedback.as_deref(),
                budget_remaining,
            );

            match self
                .run_action_turn(action_input, history.clone(), &mut acc)
                .await?
            {
                ActionTurn::RecoverableParse {
                    raw_response,
                    lm_usage,
                    chat,
                    reason,
                } => {
                    acc.absorb_parse_metadata(raw_response, lm_usage);
                    history = Some(chat);
                    let sub_lm_remaining = self.runtime.sub_lm_budget_remaining(&llm_tools);
                    let parsed_feedback = format_feedback(
                        turn_index,
                        self.config.max_iterations.saturating_sub(turn_index),
                        sub_lm_remaining,
                        self.config.max_llm_calls,
                        &ExecOutcome::RecoverableParse { message: reason },
                    );
                    feedback = Some(parsed_feedback);
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
                    let outcome = classify_exec_outcome(code.clone(), exec_result, submit_result);

                    match outcome {
                        ExecOutcome::SubmitAccepted { value, field_meta } => {
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
                                acc.to_call_metadata(),
                                final_chat,
                            ));
                        }
                        other => {
                            let sub_lm_remaining = self.runtime.sub_lm_budget_remaining(&llm_tools);
                            let rendered_feedback = format_feedback(
                                turn_index,
                                self.config.max_iterations.saturating_sub(turn_index),
                                sub_lm_remaining,
                                self.config.max_llm_calls,
                                &other,
                            );
                            feedback = Some(rendered_feedback);
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

    fn build_action_input(
        &self,
        turn_index: usize,
        previews: Option<&str>,
        execution_feedback: Option<&str>,
        budget_remaining: usize,
    ) -> RlmActionSigInput {
        let (variables_info, execution_feedback) = if turn_index == 1 {
            (previews.map(ToOwned::to_owned), None)
        } else {
            (None, execution_feedback.map(ToOwned::to_owned))
        };

        RlmActionSigInput::new(
            variables_info,
            execution_feedback,
            budget_remaining.min(u32::MAX as usize) as u32,
        )
    }

    async fn run_action_turn(
        &self,
        action_input: RlmActionSigInput,
        history: Option<Chat>,
        _acc: &mut MetadataAcc,
    ) -> Result<ActionTurn, RlmError> {
        match self.generate_action.forward(action_input, history).await {
            Ok(predicted) => Ok(ActionTurn::Parsed(predicted)),
            Err(error) => match error {
                PredictError::Parse {
                    source,
                    raw_response,
                    lm_usage,
                    chat,
                } if raw_response.trim().is_empty() => Ok(ActionTurn::RecoverableParse {
                    raw_response,
                    lm_usage,
                    chat,
                    reason: format!(
                        "Empty response from model ({source}). Write executable Python code."
                    ),
                }),
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
        Ok(Predicted::new(output, acc.to_call_metadata(), chat))
    }

    fn finalization_directive(&self) -> String {
        let output_fields = S::schema()
            .output_fields()
            .iter()
            .map(|field| format!("{}=...", field.lm_name))
            .collect::<Vec<_>>()
            .join(", ");
        format!("This is your final turn. Call SUBMIT({output_fields}) now with your best answer.")
    }
}

impl<S> Module for Rlm<S>
where
    S: Signature,
    S::Input: BamlType + for<'a> Facet<'a> + Clone + Send + Sync,
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
    S::Input: BamlType + for<'a> Facet<'a> + Clone + Send + Sync,
    S::Output: BamlType + for<'a> Facet<'a> + Clone + Send + Sync,
{
    config: RlmConfig,
    sub_lm: Option<Arc<crate::LM>>,
    runtime: Option<Arc<dyn RlmRuntime<S>>>,
    _marker: PhantomData<S>,
}

impl<S> RlmBuilder<S>
where
    S: Signature,
    S::Input: BamlType + for<'a> Facet<'a> + Clone + Send + Sync,
    S::Output: BamlType + for<'a> Facet<'a> + Clone + Send + Sync,
{
    fn new() -> Self {
        Self {
            config: RlmConfig::default(),
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

    pub fn sub_lm(mut self, sub_lm: Arc<crate::LM>) -> Self {
        self.sub_lm = Some(sub_lm);
        self
    }

    pub fn runtime(mut self, runtime: Arc<dyn RlmRuntime<S>>) -> Self {
        self.runtime = Some(runtime);
        self
    }

    pub fn build(self) -> Rlm<S> {
        let generate_action = Predict::<RlmActionSig>::builder()
            .instruction(ACTION_INSTRUCTION)
            .adapter(ChatAdapter::passthrough())
            .build();
        let extract = Predict::<RlmExtractSig<S>>::builder()
            .instruction(EXTRACT_INSTRUCTION)
            .build();

        let runtime = self
            .runtime
            .unwrap_or_else(|| Arc::new(StubRuntime::new(self.config.max_llm_calls)));

        Rlm {
            generate_action,
            extract,
            config: self.config,
            sub_lm: self.sub_lm,
            runtime,
        }
    }
}

pub fn format_feedback(
    turn_index: usize,
    budget_remaining: usize,
    sub_lm_remaining: usize,
    max_llm_calls: usize,
    outcome: &ExecOutcome,
) -> String {
    let header = format!(
        "[Turn {turn_index} | {budget_remaining} turns remaining, {sub_lm_remaining}/{max_llm_calls} sub-model calls remaining]"
    );
    let body = outcome_to_raw_output(outcome);
    if body.is_empty() {
        header
    } else {
        format!("{header}\n{body}")
    }
}

pub fn recoverable_outcome_from_parse_error(error: &PredictError) -> Option<(String, Chat)> {
    match error {
        PredictError::Parse {
            raw_response,
            chat,
            source,
            ..
        } if raw_response.trim().is_empty() => Some((
            format!("Empty response from model ({source}). Write executable Python code."),
            chat.clone(),
        )),
        _ => None,
    }
}

pub fn render_previews<S: Signature>(input: &S::Input) -> String
where
    S::Input: BamlType + for<'a> Facet<'a>,
{
    let schema = SignatureSchema::of::<S>();
    let value = input.to_baml_value();

    let mut lines = vec!["## Variables".to_string(), String::new()];
    for field in schema.input_fields() {
        let rendered_type = field.type_ir.diagnostic_repr().to_string();
        lines.push(format!("{}: {}", field.lm_name, rendered_type));
        if let Some(field_value) = schema.navigate_field(field.path(), &value) {
            lines.push(format!("  {}", render_value_preview(field_value, 0)));
        } else {
            lines.push("  <missing>".to_string());
        }
    }

    lines.push(String::new());
    lines.push("## Expected Output".to_string());
    for field in schema.output_fields() {
        lines.push(format!(
            "{}: {}",
            field.lm_name,
            field.type_ir.diagnostic_repr()
        ));
    }

    lines.join("\n")
}

fn classify_exec_outcome(
    code: String,
    exec_result: Result<String, String>,
    submit_result: Option<SubmitResultDyn>,
) -> ExecOutcome {
    let raw_exec_output = match &exec_result {
        Ok(output) => output.clone(),
        Err(message) => message.clone(),
    };

    if let Some(submit_result) = submit_result {
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
        Ok(output) => ExecOutcome::Continue { code, output },
        Err(message) => ExecOutcome::PythonException { message },
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

fn render_value_preview(value: &BamlValue, depth: usize) -> String {
    const MAX_DEPTH: usize = 2;
    if depth >= MAX_DEPTH {
        return summarize_value_shape(value);
    }

    match value {
        BamlValue::String(s) => {
            let len = s.chars().count();
            if len <= 200 {
                format!("String ({len} chars): {:?}", s)
            } else {
                let head: String = s.chars().take(100).collect();
                let tail: String = s
                    .chars()
                    .rev()
                    .take(100)
                    .collect::<String>()
                    .chars()
                    .rev()
                    .collect();
                format!(
                    "String ({len} chars): {:?} ... ({} chars omitted) ... {:?}",
                    head,
                    len.saturating_sub(200),
                    tail
                )
            }
        }
        BamlValue::Int(n) => format!("Int: {n}"),
        BamlValue::Float(f) => format!("Float: {f}"),
        BamlValue::Bool(b) => format!("Bool: {b}"),
        BamlValue::Null => "Null".to_string(),
        BamlValue::Enum(name, variant) => format!("Enum {name}::{variant}"),
        BamlValue::Media(_) => "Media (preview omitted)".to_string(),
        BamlValue::List(items) => {
            if items.is_empty() {
                return "List (0 items)".to_string();
            }
            let mut sample_indices = vec![0usize];
            let mid = items.len() / 2;
            if mid != 0 && mid != items.len() - 1 {
                sample_indices.push(mid);
            }
            if items.len() > 1 {
                sample_indices.push(items.len() - 1);
            }
            sample_indices.sort_unstable();
            sample_indices.dedup();

            let samples = sample_indices
                .into_iter()
                .map(|idx| {
                    format!(
                        "sample[{idx}] = {}",
                        render_value_preview(&items[idx], depth + 1)
                    )
                })
                .collect::<Vec<_>>()
                .join("; ");
            format!("List ({} items): {samples}", items.len())
        }
        BamlValue::Map(map) | BamlValue::Class(_, map) => {
            let mut fields = map
                .iter()
                .take(4)
                .map(|(k, v)| format!("{k}: {}", summarize_value_shape(v)))
                .collect::<Vec<_>>();
            if map.len() > 4 {
                fields.push(format!("... ({} more)", map.len() - 4));
            }
            format!("Object {{{}}}", fields.join(", "))
        }
    }
}

fn summarize_value_shape(value: &BamlValue) -> String {
    match value {
        BamlValue::String(s) => format!("String({} chars)", s.chars().count()),
        BamlValue::Int(_) => "Int".to_string(),
        BamlValue::Float(_) => "Float".to_string(),
        BamlValue::Bool(_) => "Bool".to_string(),
        BamlValue::Null => "Null".to_string(),
        BamlValue::Enum(name, variant) => format!("Enum {name}::{variant}"),
        BamlValue::Media(_) => "Media".to_string(),
        BamlValue::List(items) => format!("List({} items)", items.len()),
        BamlValue::Map(map) => format!("Map({} keys)", map.len()),
        BamlValue::Class(name, map) => format!("Class {name}({} fields)", map.len()),
    }
}
