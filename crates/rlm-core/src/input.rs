use crate::variable::RlmVariable;
use pyo3::types::{PyDict, PyDictMethods};
use pyo3::{Bound, Py, PyAny, PyResult, Python};

/// Exposes signature inputs as Python variables + prompt descriptions.
pub trait RlmInputFields {
    /// Returns input fields as Python objects with their variable names.
    fn rlm_py_fields(&self, py: Python<'_>) -> Vec<(String, Py<PyAny>)>;

    /// Returns prompt-ready variable descriptions for input fields.
    fn rlm_variables(&self) -> Vec<RlmVariable>;

    /// Returns formatted variable descriptions for prompt inclusion.
    fn rlm_variable_descriptions(&self) -> String {
        self.rlm_variables()
            .iter()
            .map(|variable| variable.format())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Injects input fields into a Python globals dict (REPL context).
    fn inject_into_python<'py>(
        &self,
        py: Python<'py>,
        globals: &Bound<'py, PyDict>,
    ) -> PyResult<()> {
        for (name, obj) in self.rlm_py_fields(py) {
            globals.set_item(name, obj)?;
        }
        Ok(())
    }
}
