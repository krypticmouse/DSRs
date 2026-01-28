//! Code generators for RlmType derive macro.
//!
//! This module contains the code generation logic for PyO3 integration:
//! - `pyclass`: Generates `#[pymethods]` impl blocks with field getters and `__baml__()`
//! - `repr`: Generates `__repr__` implementation from template strings (Task 1.4)
//! - `iter`: Generates `__iter__`, `__len__`, `__getitem__` (Task 1.5)
//! - `properties`: Generates filter/flatten properties (Task 1.6)
//! - `describe`: Generates RlmDescribe impls (Task 1.8)

mod describe;
mod iter;
mod properties;
mod pyclass;
pub mod repr;

pub use describe::generate_describe;
pub use iter::generate_iter_support;
pub use properties::generate_properties;
pub use pyclass::generate_pymethods_with_repr;
pub use repr::generate_repr;
