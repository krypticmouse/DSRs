#![cfg(feature = "rlm")]

use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use pyo3::types::{PyDict, PyDictMethods};
use pyo3::{Py, PyResult, Python};
use rig::agent::Agent;
use rig::completion::Prompt;
use rig::providers::openai::CompletionModel;
use tokio::runtime::Handle;

use crate::baml_bridge::BamlValueConvert;
use crate::rlm_core::RlmInputFields;
use crate::{ChatAdapter, LmError, Message, Signature};

use super::history::{render_history, ReplHistoryEntry};
use super::submit::{SubmitError, SubmitHandler, SubmitResultDyn};
use super::{execute_repl_code, Command, LlmTools, RlmConfig, RlmError, RlmResult};

/// Typed Recursive Language Model.
///
/// Uses a DSRs Signature to run RLM with typed inputs/outputs.
#[allow(dead_code)]
pub struct TypedRlm<S: Signature> {
    agent: Agent<CompletionModel>,
    config: RlmConfig,
    _marker: PhantomData<S>,
}

impl<S: Signature> TypedRlm<S> {
    /// Create a new TypedRlm with the given agent and config.
    pub fn new(agent: Agent<CompletionModel>, config: RlmConfig) -> Self {
        Self {
            agent,
            config,
            _marker: PhantomData,
        }
    }

    /// Create with default config.
    pub fn with_agent(agent: Agent<CompletionModel>) -> Self {
        Self::new(agent, RlmConfig::default())
    }

    pub async fn call(&self, input: S::Input) -> Result<RlmResult<S>, RlmError>
    where
        S::Input: RlmInputFields,
    {
        let preamble = input.rlm_variable_descriptions();
        let (submit_handler, submit_rx) = SubmitHandler::new::<S>();
        let schema = submit_handler.schema();
        let runtime = Handle::try_current().map_err(|err| RlmError::RuntimeUnavailable {
            message: err.to_string(),
        })?;
        let tools = LlmTools::new(self.agent.clone(), self.config.max_llm_calls, runtime);
        let globals = setup_globals::<S>(&input, &tools, &submit_handler)?;
        let mut history = Vec::new();
        let mut iterations = 0usize;
        let mut main_calls = 0usize;

        while iterations < self.config.max_iterations {
            iterations += 1;
            let prompt = build_prompt(
                &preamble,
                &schema,
                &history,
                self.config.max_history_output_chars,
            );
            let response = self
                .agent
                .prompt(&prompt)
                .await
                .map_err(|err| RlmError::Lm {
                    source: LmError::Provider {
                        provider: "rig".to_string(),
                        message: err.to_string(),
                        source: None,
                    },
                })?;
            main_calls += 1;

            let Some(command) = Command::parse(&response) else {
                let output = "No executable command found. Wrap Python in ```repl``` or ```python``` fences, or call SUBMIT(...).".to_string();
                history.push(ReplHistoryEntry::new(response, output));
                continue;
            };

            let mut output = if command.code().trim().is_empty() {
                "No code provided.".to_string()
            } else {
                match execute_repl_code(&globals, command.code(), self.config.max_output_chars) {
                    Ok(result) => result,
                    Err(err) => format!("Python error: {err}"),
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
                            .map_err(|err| RlmError::Conversion {
                                source: err,
                                value: baml_value,
                            })?;
                        let llm_calls = main_calls + tools.call_count();
                        return Ok(RlmResult::new(
                            input,
                            typed_output,
                            metas,
                            iterations,
                            llm_calls,
                            false,
                        ));
                    }
                    Err(SubmitError::AssertionFailed { label, expression }) => {
                        if self.config.strict_assertions {
                            return Err(RlmError::SubmitAssertion { label, expression });
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

            history.push(ReplHistoryEntry::new(command.code().to_string(), output));
        }

        if self.config.enable_extraction_fallback {
            return self
                .extraction_fallback(
                    input,
                    &preamble,
                    &schema,
                    &history,
                    iterations,
                    main_calls + tools.call_count(),
                )
                .await;
        }

        Err(RlmError::MaxIterations {
            max: self.config.max_iterations,
        })
    }

    async fn extraction_fallback(
        &self,
        input: S::Input,
        preamble: &str,
        schema: &str,
        history: &[ReplHistoryEntry],
        iterations: usize,
        llm_calls: usize,
    ) -> Result<RlmResult<S>, RlmError> {
        let prompt = build_extraction_prompt::<S>(
            preamble,
            schema,
            history,
            self.config.max_history_output_chars,
        );
        let response = self
            .agent
            .prompt(&prompt)
            .await
            .map_err(|err| RlmError::Lm {
                source: LmError::Provider {
                    provider: "rig".to_string(),
                    message: err.to_string(),
                    source: None,
                },
            })?;

        let raw_response = response.clone();
        let message = Message::assistant(response);
        let chat_adapter = ChatAdapter::default();
        let (typed_output, field_metas) = chat_adapter
            .parse_response_typed::<S>(&message)
            .map_err(|err| RlmError::ExtractionParse {
                source: err,
                raw_response: raw_response.clone(),
            })?;

        Ok(RlmResult::new(
            input,
            typed_output,
            field_metas,
            iterations,
            llm_calls + 1,
            true,
        ))
    }
}

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
        globals.set_item("tools", Py::new(py, tools.clone())?)?;
        globals.set_item("SUBMIT", Py::new(py, submit_handler.clone())?)?;
        Ok(globals.unbind())
    })
    .map_err(|err| RlmError::PythonSetup {
        message: err.to_string(),
    })
}

fn take_submit_result(rx: &Arc<Mutex<Option<SubmitResultDyn>>>) -> Option<SubmitResultDyn> {
    rx.lock().ok().and_then(|mut guard| guard.take())
}

fn build_prompt(
    preamble: &str,
    schema: &str,
    history: &[ReplHistoryEntry],
    max_history_output_chars: usize,
) -> String {
    let mut prompt = String::new();
    prompt.push_str("You are running inside a Python REPL for a typed signature.\n");
    prompt.push_str("Use the provided variables, and call SUBMIT(...) when you have the final output.\n");
    prompt.push_str("You can call tools.llm_query or tools.llm_query_batched if needed.\n");

    if !preamble.trim().is_empty() {
        prompt.push_str("\nInputs:\n");
        prompt.push_str(preamble);
    }

    if !schema.trim().is_empty() {
        prompt.push_str("\n\nOutput schema:\n");
        prompt.push_str(schema);
    }

    if !history.is_empty() {
        prompt.push_str("\n\nHistory:\n");
        prompt.push_str(&render_history(history, max_history_output_chars));
    }

    prompt.push_str("\n\nReturn the next step as a ```repl``` or ```python``` code block. If you are done, call SUBMIT(field=value, ...).\n");
    prompt
}

fn build_extraction_prompt<S: Signature>(
    preamble: &str,
    schema: &str,
    history: &[ReplHistoryEntry],
    max_history_output_chars: usize,
) -> String {
    let mut prompt = String::new();
    prompt.push_str("You are performing fallback extraction for a typed signature.\n");
    prompt.push_str("Use the inputs, schema, and REPL history to extract the final output.\n");

    if !preamble.trim().is_empty() {
        prompt.push_str("\nInputs:\n");
        prompt.push_str(preamble);
    }

    if !schema.trim().is_empty() {
        prompt.push_str("\n\nOutput schema:\n");
        prompt.push_str(schema);
    }

    if !history.is_empty() {
        prompt.push_str("\n\nREPL history:\n");
        prompt.push_str(&render_history(history, max_history_output_chars));
    }

    prompt.push_str("\n\n");
    prompt.push_str(&format_output_instructions::<S>());
    prompt.push_str("\nRespond with only the structured output and no extra commentary.\n");
    prompt
}

fn format_output_instructions<S: Signature>() -> String {
    let mut fields = S::output_fields().iter();
    let Some(first) = fields.next() else {
        return "Respond with the marker for `[[ ## completed ## ]]`.".to_string();
    };

    let mut message = format!(
        "Respond with the output fields, starting with `[[ ## {} ## ]]`",
        first.name
    );
    for field in fields {
        message.push_str(&format!(", then `[[ ## {} ## ]]`", field.name));
    }
    message.push_str(", and then ending with the marker for `[[ ## completed ## ]]`.");
    message
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
