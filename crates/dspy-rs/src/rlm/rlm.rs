#![cfg(feature = "rlm")]

use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use pyo3::types::{PyAnyMethods, PyDict, PyDictMethods};
use pyo3::{Py, PyResult, Python};
use regex::Regex;
use tokio::runtime::Handle;
use std::sync::LazyLock;

use crate::baml_bridge::{BamlValueConvert, ToBamlValue};
use crate::core::settings::GLOBAL_SETTINGS;
use crate::rlm_core::RlmInputFields;
use crate::{Predict, Signature, LM};

use super::adapter::RlmAdapter;
use super::history::REPLHistory;
use super::submit::{SubmitError, SubmitHandler, SubmitResultDyn};
use super::signatures::{RlmActionSig, RlmActionSigInput, RlmExtractInput, RlmExtractSig};
use super::{execute_repl_code, LlmTools, RlmConfig, RlmError, RlmResult};

/// Recursive Language Model module.
///
/// Uses a DSRs Signature to run RLM with typed inputs/outputs.
/// Follows DSRs patterns: uses global LM settings by default.
pub struct Rlm<S: Signature> {
    config: RlmConfig,
    lm_override: Option<Arc<LM>>,
    generate_action: Predict<RlmActionSig>,
    extract: Predict<RlmExtractSig<S>>,
    _marker: PhantomData<S>,
}

struct ExtractionFallbackContext<'a> {
    variables_info: &'a str,
    history: &'a REPLHistory,
    iterations: usize,
    llm_calls: usize,
}

impl<S: Signature> Rlm<S> {
    /// Create with default config, using global LM settings.
    pub fn new() -> Self {
        let config = RlmConfig::default();
        let (generate_action, extract) = Self::build_predictors(&config, None);
        Self {
            config,
            lm_override: None,
            generate_action,
            extract,
            _marker: PhantomData,
        }
    }

    /// Create with custom config.
    pub fn with_config(config: RlmConfig) -> Self {
        let (generate_action, extract) = Self::build_predictors(&config, None);
        Self {
            config,
            lm_override: None,
            generate_action,
            extract,
            _marker: PhantomData,
        }
    }

    /// Create with explicit LM (overrides global settings).
    pub fn with_lm(lm: Arc<LM>) -> Self {
        let config = RlmConfig::default();
        let (generate_action, extract) = Self::build_predictors(&config, Some(Arc::clone(&lm)));
        Self {
            config,
            lm_override: Some(lm),
            generate_action,
            extract,
            _marker: PhantomData,
        }
    }

    fn build_predictors(
        config: &RlmConfig,
        lm_override: Option<Arc<LM>>,
    ) -> (Predict<RlmActionSig>, Predict<RlmExtractSig<S>>) {
        let adapter = RlmAdapter::new(config.clone());
        let action_instruction = adapter.build_action_instruction::<S>();
        let extract_instruction = adapter.build_extract_instruction::<S>();

        let mut generate_action = Predict::<RlmActionSig>::builder()
            .instruction(action_instruction);
        let mut extract = Predict::<RlmExtractSig<S>>::builder()
            .instruction(extract_instruction);

        if let Some(lm) = lm_override {
            generate_action = generate_action.with_lm(Arc::clone(&lm));
            extract = extract.with_lm(lm);
        }

        let generate_action = generate_action.build();
        let extract = extract.build();
        (generate_action, extract)
    }

    fn get_lm(&self) -> Result<Arc<LM>, RlmError> {
        if let Some(lm) = &self.lm_override {
            return Ok(Arc::clone(lm));
        }
        let guard = GLOBAL_SETTINGS.read().map_err(|_| RlmError::ConfigurationError {
            message: "Settings lock poisoned".to_string(),
        })?;
        let settings = guard.as_ref().ok_or_else(|| RlmError::ConfigurationError {
            message: "DSRs not configured. Call dspy_rs::configure() first.".to_string(),
        })?;
        Ok(Arc::clone(&settings.lm))
    }

    pub async fn call(&self, input: S::Input) -> Result<RlmResult<S>, RlmError>
    where
        S::Input: RlmInputFields,
        S::Output: Clone + ToBamlValue,
    {
        let adapter = RlmAdapter::new(self.config.clone());
        let variable_descriptions = adapter.variable_previews::<S>(&input);
        let (submit_handler, submit_rx) = SubmitHandler::new::<S>();
        let schema = submit_handler.schema();
        let variables_info = adapter.build_variables_info::<S>(&variable_descriptions, &schema);
        let runtime = Handle::try_current().map_err(|err| RlmError::RuntimeUnavailable {
            message: err.to_string(),
        })?;
        let lm = self.get_lm()?;
        let tools = LlmTools::new(Arc::clone(&lm), self.config.max_llm_calls, runtime);
        let globals = setup_globals::<S>(&input, &tools, &submit_handler)?;
        let mut history = REPLHistory::with_max_output_chars(self.config.max_history_output_chars);
        let mut iterations = 0usize;
        let mut main_calls = 0usize;

        while iterations < self.config.max_iterations {
            iterations += 1;
            let action_input = RlmActionSigInput {
                variables_info: variables_info.clone(),
                repl_history: history.clone(),
                iteration: format!("{}/{}", iterations, self.config.max_iterations),
            };

            let action = self
                .generate_action
                .call(action_input)
                .await
                .map_err(|err| RlmError::PredictError {
                    stage: "action",
                    source: err,
                })?;
            main_calls += 1;

            let (_, action_output) = action.output.into_parts();
            let code = strip_code_fences(&action_output.code);

            let reasoning = action_output.reasoning;
            let mut output = if code.trim().is_empty() {
                "No code provided.".to_string()
            } else {
                match execute_repl_code(&globals, &code, self.config.max_output_chars) {
                    Ok(result) => result,
                    Err(err) => format!("[Error] {err}"),
                }
            };

            if let Some(result) = take_submit_result(&submit_rx) {
                match result {
                    Ok((baml_value, metas)) => {
                        let typed_output =
                            <S::Output as BamlValueConvert>::try_from_baml_value(
                                baml_value.clone(),
                                Vec::new(),
                            )
                            .map_err(|err| RlmError::ConversionError {
                                source: err,
                                value: baml_value,
                            })?;
                        let llm_calls = main_calls + tools.call_count();
                        let trajectory =
                            history.append_with_reasoning(reasoning, code.clone(), output);
                        return Ok(RlmResult::new(
                            input,
                            typed_output,
                            metas,
                            iterations,
                            llm_calls,
                            false,
                            trajectory,
                        ));
                    }
                    Err(SubmitError::AssertionFailed { label, expression }) => {
                        if self.config.strict_assertions {
                            return Err(RlmError::AssertionFailed { label, expression });
                        }
                        output = format!(
                            "[Error] Assertion '{}' failed: {}",
                            label, expression
                        );
                    }
                    Err(SubmitError::ValidationError { message, errors }) => {
                        output = format_submit_validation(&message, &errors, &schema);
                    }
                }
            }

            history = history.append_with_reasoning(reasoning, code, output);
        }

        if self.config.enable_extraction_fallback {
            let context = ExtractionFallbackContext {
                variables_info: &variables_info,
                history: &history,
                iterations,
                llm_calls: main_calls + tools.call_count(),
            };
            return self.extraction_fallback(input, context).await;
        }

        Err(RlmError::MaxIterationsReached {
            max: self.config.max_iterations,
        })
    }

    async fn extraction_fallback(
        &self,
        input: S::Input,
        context: ExtractionFallbackContext<'_>,
    ) -> Result<RlmResult<S>, RlmError>
    where
        S::Output: Clone + ToBamlValue,
    {
        let extract_input = RlmExtractInput {
            variables_info: context.variables_info.to_string(),
            repl_history: context.history.clone(),
        };

        let extract_result = self.extract.call(extract_input).await.map_err(|err| {
            RlmError::PredictError {
                stage: "extract",
                source: err,
            }
        })?;
        let field_metas = extract_result.field_metas().clone();
        let (_, typed_output) = extract_result.output.into_parts();

        Ok(RlmResult::new(
            input,
            typed_output,
            field_metas,
            context.iterations,
            context.llm_calls + 1,
            true,
            context.history.clone(),
        ))
    }
}

impl<S: Signature> Default for Rlm<S> {
    fn default() -> Self {
        Self::new()
    }
}

static CODE_FENCE_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)^```(?:repl|python|py)?\s*\r?\n(?P<code>.*)\n```\s*$")
        .expect("valid code fence regex")
});

fn strip_code_fences(code: &str) -> String {
    let trimmed = code.trim();
    if let Some(caps) = CODE_FENCE_PATTERN.captures(trimmed) {
        if let Some(capture) = caps.name("code") {
            return capture.as_str().to_string();
        }
    }
    trimmed.to_string()
}

impl<S: Signature> Rlm<S> {
    /// Start building an Rlm with fluent configuration.
    pub fn builder() -> RlmBuilder<S> {
        RlmBuilder::new()
    }
}

/// Builder for fluent Rlm configuration.
pub struct RlmBuilder<S: Signature> {
    config: RlmConfig,
    lm_override: Option<Arc<LM>>,
    _marker: PhantomData<S>,
}

impl<S: Signature> RlmBuilder<S> {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            config: RlmConfig::default(),
            lm_override: None,
            _marker: PhantomData,
        }
    }

    /// Set maximum iterations before giving up.
    pub fn max_iterations(mut self, max: usize) -> Self {
        self.config.max_iterations = max;
        self
    }

    /// Set maximum LLM calls allowed within Python code.
    pub fn max_llm_calls(mut self, max: usize) -> Self {
        self.config.max_llm_calls = max;
        self
    }

    /// Enable or disable extraction fallback on max iterations.
    pub fn enable_extraction_fallback(mut self, enable: bool) -> Self {
        self.config.enable_extraction_fallback = enable;
        self
    }

    /// Enable or disable strict assertion mode.
    pub fn strict_assertions(mut self, strict: bool) -> Self {
        self.config.strict_assertions = strict;
        self
    }

    /// Set maximum output characters for Python execution.
    pub fn max_output_chars(mut self, max: usize) -> Self {
        self.config.max_output_chars = max;
        self
    }

    /// Use an explicit LM instead of global settings.
    pub fn with_lm(mut self, lm: Arc<LM>) -> Self {
        self.lm_override = Some(lm);
        self
    }

    /// Build the Rlm instance.
    pub fn build(self) -> Rlm<S> {
        let (generate_action, extract) =
            Rlm::build_predictors(&self.config, self.lm_override.as_ref().map(Arc::clone));
        Rlm {
            config: self.config,
            lm_override: self.lm_override,
            generate_action,
            extract,
            _marker: PhantomData,
        }
    }
}

impl<S: Signature> Default for RlmBuilder<S> {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(clippy::result_large_err)]
fn setup_globals<S: Signature>(
    input: &S::Input,
    tools: &LlmTools,
    submit_handler: &SubmitHandler,
) -> Result<Py<PyDict>, RlmError>
where
    S::Input: RlmInputFields,
{
    Python::attach(|py| -> PyResult<Py<PyDict>> {
        let globals = PyDict::new(py);
        input.inject_into_python(py, &globals)?;
        let tools_py = Py::new(py, tools.clone())?;
        let tools_bound = tools_py.bind(py);
        let llm_query = tools_bound.getattr("llm_query")?;
        let llm_query_batched = tools_bound.getattr("llm_query_batched")?;
        globals.set_item("tools", tools_py.clone_ref(py))?;
        globals.set_item("llm_query", llm_query)?;
        globals.set_item("llm_query_batched", llm_query_batched)?;
        globals.set_item("SUBMIT", Py::new(py, submit_handler.clone())?)?;
        Ok(globals.unbind())
    })
    .map_err(RlmError::from)
}

fn take_submit_result(rx: &Arc<Mutex<Option<SubmitResultDyn>>>) -> Option<SubmitResultDyn> {
    rx.lock().ok().and_then(|mut guard| guard.take())
}

fn format_submit_validation(message: &str, errors: &[String], schema: &str) -> String {
    let joined = errors.join("\n");
    if schema.trim().is_empty() {
        if message.is_empty() {
            return joined;
        }
        if joined.is_empty() {
            return format!("[Error] {message}");
        }
        return format!("[Error] {message}\n{joined}");
    }

    let base = if message.is_empty() {
        joined
    } else if joined.is_empty() {
        format!("[Error] {message}")
    } else {
        format!("[Error] {message}\n{joined}")
    };

    if base.is_empty() {
        format!("Expected schema:\n{schema}")
    } else {
        format!("{base}\n\nExpected schema:\n{schema}")
    }
}
