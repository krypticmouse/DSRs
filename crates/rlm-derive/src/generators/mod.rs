//! Code generators for RlmType derive macro.
//!
//! This module contains the code generation logic for PyO3 integration:
//! - `pyclass`: Generates `#[pymethods]` impl blocks with field getters and `__baml__()`
//! - `repr`: Generates `__repr__` implementation from template strings (Task 1.4)

mod pyclass;
pub mod repr;

pub use pyclass::{generate_pymethods, generate_pymethods_with_repr};
pub use repr::generate_repr;
