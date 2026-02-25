use std::sync::Arc;

use super::exec;
use super::py_bridge;
use super::submit;
use super::tools;
use crate::Signature;
use pyo3::types::PyDict;
use pyo3::{Py, PyResult, Python};

pub type SubmitResultDyn = submit::SubmitResultDyn;
pub type SubmitSlot = submit::SubmitSlot;
pub type SubmitError = submit::SubmitError;
pub type SubmitHandler = submit::SubmitHandler;
pub type LlmTools = tools::LlmTools;

pub use submit::{clear_submit_slot, take_submit_result};

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

#[derive(Default, Debug, Clone)]
pub struct StubRuntime;

impl StubRuntime {
    pub fn new(_max_llm_calls: usize) -> Self {
        Self
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
        0
    }
}

#[derive(Default, Debug, Clone)]
pub struct PyO3Runtime;

impl<S: Signature> RlmRuntime<S> for PyO3Runtime {
    fn setup_interpreter_globals(
        &self,
        py: Python<'_>,
        input: &S::Input,
        submit_handler: &SubmitHandler,
        llm_tools: &LlmTools,
    ) -> PyResult<Py<PyDict>> {
        py_bridge::setup_interpreter_globals::<S>(py, input, submit_handler, llm_tools)
    }

    fn execute_repl_code(
        &self,
        py: Python<'_>,
        globals: &Py<PyDict>,
        code: &str,
        max_output_chars: usize,
    ) -> Result<String, String> {
        exec::execute_repl_code(py, globals, code, max_output_chars)
    }

    fn sub_lm_budget_remaining(&self, llm_tools: &LlmTools) -> usize {
        llm_tools.remaining_calls()
    }
}

pub type DynRuntime<S> = Arc<dyn RlmRuntime<S>>;
