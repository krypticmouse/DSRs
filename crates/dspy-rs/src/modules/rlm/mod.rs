use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use indexmap::IndexMap;
use pyo3::types::PyDict;
use pyo3::{Py, Python};
use rig::message::ToolCall;

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
    DynRuntime, LlmTools, PyO3Runtime, RlmRuntime, StubRuntime, SubmitError, SubmitHandler,
    SubmitResultDyn, SubmitSlot, clear_submit_slot, take_submit_result,
};
pub use tools::LlmQuery;

const DEFAULT_MAX_ITERATIONS: usize = 20;
const DEFAULT_MAX_LLM_CALLS: usize = 50;
const DEFAULT_MAX_OUTPUT_CHARS: usize = 100_000;
const DEFAULT_ENABLE_EXTRACTION_FALLBACK: bool = true;
const MAX_RECOVERABLE_PARSE_SNIPPET_CHARS: usize = 80;

const EXTRACT_INSTRUCTION: &str = "Extract the final typed answer from the REPL history.\n\
Use the expected output schema exactly.";
const REPL_HISTORY_INPUT_RENDER_TEMPLATE: &str = r#"{% if this.entries|length == 0 %}(no executed REPL turns captured){% else %}{% for entry in this.entries %}=== Turn {{ entry.turn }} ===
Code:
{{ entry.code }}

Output:
{% if entry.output %}{{ entry.output }}{% else %}<empty>{% endif %}{% if not loop.last %}

{% endif %}{% endfor %}{% endif %}"#;

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
        let globals: Py<PyDict> = Python::attach(|py| {
            self.runtime
                .setup_interpreter_globals(py, input, &submit_handler, llm_tools.as_ref())
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

            let budget_remaining = self
                .config
                .max_iterations
                .saturating_sub(turn_index)
                .saturating_add(1);
            let action_input = self.build_action_input(
                turn_index,
                Some(previews.as_str()),
                feedback.as_deref(),
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
                    let sub_lm_remaining = self.runtime.sub_lm_budget_remaining(llm_tools.as_ref());
                    let next_turn_index = turn_index.saturating_add(1);
                    let finalization_directive = (next_turn_index == self.config.max_iterations)
                        .then(|| self.finalization_directive());
                    let parsed_feedback = format_feedback(
                        next_turn_index,
                        self.config.max_iterations.saturating_sub(turn_index),
                        sub_lm_remaining,
                        self.config.max_llm_calls,
                        &ExecOutcome::RecoverableParse { message: reason },
                        finalization_directive.as_deref(),
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
                            let sub_lm_remaining =
                                self.runtime.sub_lm_budget_remaining(llm_tools.as_ref());
                            let next_turn_index = turn_index.saturating_add(1);
                            let finalization_directive = (next_turn_index
                                == self.config.max_iterations)
                                .then(|| self.finalization_directive());
                            let rendered_feedback = format_feedback(
                                next_turn_index,
                                self.config.max_iterations.saturating_sub(turn_index),
                                sub_lm_remaining,
                                self.config.max_llm_calls,
                                &other,
                                finalization_directive.as_deref(),
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
    instruction_override: Option<String>,
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
        let action_instruction =
            render_action_instruction::<S>(&self.config, self.instruction_override.as_deref());
        let extract_instruction =
            render_extract_instruction::<S>(self.instruction_override.as_deref());
        let generate_action = Predict::<RlmActionSig>::builder()
            .instruction(action_instruction)
            .adapter(ChatAdapter::passthrough())
            .build();
        let extract = Predict::<RlmExtractSig<S>>::builder()
            .instruction(extract_instruction)
            .adapter(ChatAdapter::new())
            .build();

        let runtime = self
            .runtime
            .unwrap_or_else(|| default_runtime::<S>(self.config.max_llm_calls));

        Rlm {
            generate_action,
            extract,
            config: self.config,
            sub_lm: self.sub_lm,
            runtime,
        }
    }
}

fn default_runtime<S: Signature>(max_llm_calls: usize) -> Arc<dyn RlmRuntime<S>>
where
    S::Input: BamlType + for<'a> Facet<'a> + Clone + Send + Sync,
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

pub fn format_feedback(
    turn_index: usize,
    budget_remaining: usize,
    sub_lm_remaining: usize,
    max_llm_calls: usize,
    outcome: &ExecOutcome,
    finalization_directive: Option<&str>,
) -> String {
    let header = format!(
        "[Turn {turn_index} | {budget_remaining} turns, {sub_lm_remaining}/{max_llm_calls} sub-model calls remaining]"
    );
    let body = outcome_to_raw_output(outcome);
    let mut rendered = if body.is_empty() {
        header
    } else {
        format!("{header}\n\n{body}")
    };
    if let Some(directive) = finalization_directive {
        rendered.push_str("\n\n");
        rendered.push_str(directive);
    }
    rendered
}

pub fn recoverable_outcome_from_parse_error(error: &PredictError) -> Option<(String, Chat)> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ParseError, Signature};
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
    fn action_input_is_asymmetric_between_first_and_later_turns() {
        let module = Rlm::<RuntimePolicySig>::builder().build();

        let turn1 = module.build_action_input(1, Some("preview block"), Some("feedback"), 20);
        assert_eq!(turn1.variables_info.as_deref(), Some("preview block"));
        assert!(turn1.execution_feedback.is_none());
        assert_eq!(turn1.budget_remaining, 20);

        let turn2 = module.build_action_input(2, Some("preview block"), Some("feedback"), 19);
        assert!(turn2.variables_info.is_none());
        assert_eq!(turn2.execution_feedback.as_deref(), Some("feedback"));
        assert_eq!(turn2.budget_remaining, 19);
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
    fn feedback_uses_next_turn_framing_and_supports_finalization_directive() {
        let feedback = format_feedback(
            2,
            19,
            50,
            50,
            &ExecOutcome::Continue {
                code: "print('ok')".to_string(),
                output: "ok".to_string(),
            },
            Some("This is your final turn. Call SUBMIT(answer=...) now with your best answer."),
        );

        assert!(feedback.contains("[Turn 2 | 19 turns, 50/50 sub-model calls remaining]"));
        assert!(feedback.contains("\n\nok"));
        assert!(feedback.contains("This is your final turn. Call SUBMIT(answer=...) now"));
    }

    #[test]
    fn classify_exec_outcome_covers_all_variants_and_feedback_projection() {
        let continue_outcome =
            classify_exec_outcome("print('x')".to_string(), Ok("x\n".into()), None);
        assert!(matches!(
            continue_outcome,
            ExecOutcome::Continue { ref code, ref output } if code == "print('x')" && output == "x\n"
        ));
        assert_eq!(outcome_to_raw_output(&continue_outcome), "x\n");

        let submit_ok = classify_exec_outcome(
            "SUBMIT(answer='ok')".to_string(),
            Ok(String::new()),
            Some(Ok((BamlValue::String("ok".to_string()), IndexMap::new()))),
        );
        assert!(matches!(submit_ok, ExecOutcome::SubmitAccepted { .. }));
        assert!(outcome_to_raw_output(&submit_ok).is_empty());

        let submit_validation = classify_exec_outcome(
            "SUBMIT(answer=123)".to_string(),
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
            "SUBMIT(answer='')".to_string(),
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

        let python_exception = classify_exec_outcome(
            "raise ValueError('boom')".to_string(),
            Err("Traceback...".into()),
            None,
        );
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
