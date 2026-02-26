use std::collections::BTreeMap;
use std::sync::Arc;

use super::exec;
use super::py_bridge;
use super::submit;
use super::tools;
use crate::Signature;
use pyo3::types::{PyAny, PyDict, PyDictMethods};
use pyo3::{Bound, Py, PyResult, Python};

pub type SubmitResultDyn = submit::SubmitResultDyn;
pub type SubmitSlot = submit::SubmitSlot;
pub type SubmitError = submit::SubmitError;
pub type SubmitHandler = submit::SubmitHandler;
pub type LlmTools = tools::LlmTools;

pub use submit::{clear_submit_slot, take_submit_result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MethodSource {
    Generated,
    Custom,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct MethodSignature {
    pub name: String,
    pub signature: String,
    pub doc: String,
    pub source: MethodSource,
    pub is_dunder: bool,
}

#[derive(Debug, Clone)]
pub struct InterpreterSetup {
    pub globals: Py<PyDict>,
    pub methods_by_var: BTreeMap<String, Vec<MethodSignature>>,
}

pub trait RlmInputFields {
    fn rlm_field_names(&self) -> &'static [&'static str];

    fn rlm_py_fields(&self, py: Python<'_>) -> PyResult<Vec<(String, Py<PyAny>)>>;

    fn inject_into_python<'py>(
        &self,
        py: Python<'py>,
        globals: &Bound<'py, PyDict>,
    ) -> PyResult<()> {
        for (name, obj) in self.rlm_py_fields(py)? {
            globals.set_item(name, obj)?;
        }
        Ok(())
    }
}

/// Runtime abstraction for REPL-backed RLM execution.
///
/// V1 ships with a stub implementation in this crate. Another module can provide
/// a concrete PyO3-backed implementation by implementing this trait and wiring it
/// through `RlmBuilder::runtime(...)`.
pub trait RlmRuntime<S: Signature>: Send + Sync {
    /// Whether this runtime needs sub-LM tools (`llm_query*`) to be installed.
    /// Stub runtimes can return `false` so tests can run without sub-LM wiring.
    fn requires_sub_lm_tools(&self) -> bool {
        true
    }

    fn setup_interpreter_globals(
        &self,
        py: Python<'_>,
        input: &S::Input,
        submit_handler: &SubmitHandler,
        llm_tools: Option<&LlmTools>,
    ) -> PyResult<InterpreterSetup>
    where
        S::Input: RlmInputFields;

    fn execute_repl_code(
        &self,
        py: Python<'_>,
        globals: &Py<PyDict>,
        code: &str,
        max_output_chars: usize,
    ) -> Result<String, String>;

    fn sub_lm_budget_remaining(&self, llm_tools: Option<&LlmTools>) -> usize;
}

#[derive(Default, Debug, Clone)]
pub struct StubRuntime;

impl StubRuntime {
    pub fn new(_max_llm_calls: usize) -> Self {
        Self
    }
}

impl<S: Signature> RlmRuntime<S> for StubRuntime {
    fn requires_sub_lm_tools(&self) -> bool {
        false
    }

    fn setup_interpreter_globals(
        &self,
        py: Python<'_>,
        _input: &S::Input,
        _submit_handler: &SubmitHandler,
        _llm_tools: Option<&LlmTools>,
    ) -> PyResult<InterpreterSetup>
    where
        S::Input: RlmInputFields,
    {
        Ok(InterpreterSetup {
            globals: PyDict::new(py).unbind(),
            methods_by_var: BTreeMap::new(),
        })
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

    fn sub_lm_budget_remaining(&self, _llm_tools: Option<&LlmTools>) -> usize {
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
        llm_tools: Option<&LlmTools>,
    ) -> PyResult<InterpreterSetup>
    where
        S::Input: RlmInputFields,
    {
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

    fn sub_lm_budget_remaining(&self, llm_tools: Option<&LlmTools>) -> usize {
        llm_tools.map(LlmTools::remaining_calls).unwrap_or(0)
    }
}

pub type DynRuntime<S> = Arc<dyn RlmRuntime<S>>;
