use std::sync::{Arc, Mutex};

use indexmap::IndexMap;
use pyo3::types::PyDict;
use pyo3::{Py, PyResult, Python};

use crate::{BamlValue, FieldMeta, Signature};

pub type SubmitResultDyn = Result<(BamlValue, IndexMap<String, FieldMeta>), SubmitError>;
pub type SubmitSlot = Arc<Mutex<Option<SubmitResultDyn>>>;

#[derive(Debug, Clone, thiserror::Error)]
pub enum SubmitError {
    #[error("validation failed: {message}")]
    ValidationError {
        message: String,
        errors: Vec<String>,
    },

    #[error("assertion `{label}` failed: {expression}")]
    AssertionFailed { label: String, expression: String },
}

pub fn clear_submit_slot(slot: &SubmitSlot) {
    let mut guard = slot.lock().expect("submit slot mutex poisoned");
    *guard = None;
}

pub fn take_submit_result(slot: &SubmitSlot) -> Option<SubmitResultDyn> {
    let mut guard = slot.lock().expect("submit slot mutex poisoned");
    guard.take()
}

#[derive(Debug, Default, Clone)]
pub struct SubmitHandler;

#[derive(Debug, Default, Clone)]
pub struct LlmTools {
    pub max_llm_calls: usize,
}

impl LlmTools {
    pub fn remaining_calls(&self) -> usize {
        self.max_llm_calls
    }
}

/// Runtime abstraction for REPL-backed RLM execution.
///
/// V1 ships with a stub implementation in this crate. Another module can provide
/// a concrete PyO3-backed implementation by implementing this trait and wiring it
/// through `RlmBuilder::runtime(...)`.
pub trait RlmRuntime<S: Signature>: Send + Sync {
    fn setup_interpreter_globals(
        &self,
        py: Python<'_>,
        input: &S::Input,
        submit_handler: &SubmitHandler,
        llm_tools: &LlmTools,
    ) -> PyResult<Py<PyDict>>;

    fn execute_repl_code(
        &self,
        py: Python<'_>,
        globals: &Py<PyDict>,
        code: &str,
        max_output_chars: usize,
    ) -> Result<String, String>;

    fn sub_lm_budget_remaining(&self, llm_tools: &LlmTools) -> usize;
}

#[derive(Debug)]
pub struct StubRuntime {
    sub_lm_remaining: Mutex<usize>,
}

impl StubRuntime {
    pub fn new(max_llm_calls: usize) -> Self {
        Self {
            sub_lm_remaining: Mutex::new(max_llm_calls),
        }
    }
}

impl<S: Signature> RlmRuntime<S> for StubRuntime {
    fn setup_interpreter_globals(
        &self,
        py: Python<'_>,
        _input: &S::Input,
        _submit_handler: &SubmitHandler,
        _llm_tools: &LlmTools,
    ) -> PyResult<Py<PyDict>> {
        Ok(PyDict::new(py).unbind())
    }

    fn execute_repl_code(
        &self,
        _py: Python<'_>,
        _globals: &Py<PyDict>,
        _code: &str,
        _max_output_chars: usize,
    ) -> Result<String, String> {
        Ok(String::new())
    }

    fn sub_lm_budget_remaining(&self, _llm_tools: &LlmTools) -> usize {
        *self
            .sub_lm_remaining
            .lock()
            .expect("stub runtime budget mutex poisoned")
    }
}

pub type DynRuntime<S> = Arc<dyn RlmRuntime<S>>;
