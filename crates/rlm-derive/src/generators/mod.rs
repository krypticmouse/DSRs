//! Code generators for RlmType derive macro.
//!
//! This module contains the code generation logic for PyO3 integration:
//! - `pyclass`: Generates `#[pymethods]` impl blocks with field getters and `__baml__()`
//! - `repr`: Generates `__repr__` implementation from template strings

mod pyclass;
pub mod repr;

pub use pyclass::generate_pymethods;
