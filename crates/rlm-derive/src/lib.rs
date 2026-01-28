//! Derive macros for RLM-compatible types.
//!
//! This crate provides two main proc-macros:
//!
//! - `#[rlm_type]` - Attribute macro that adds `#[pyclass]` and injects
//!   `#[derive(BamlType, RlmType)]`
//! - `#[derive(RlmType)]` - Derive macro that generates `#[pymethods]` impls
//!   with getters, `__repr__`, `__len__`, `__iter__`, `__getitem__`, and `__baml__`

use proc_macro::TokenStream;
use syn::parse_macro_input;

mod attrs;
mod generators;
mod rlm_attr;
mod rlm_type;

/// Attribute macro for ergonomic RLM type definitions.
///
/// Expands to:
/// - `#[pyo3::pyclass]` on the struct (with optional name override via `pyclass_name`)
/// - `#[derive(baml_bridge::BamlType, RlmType)]` (merged with existing derives)
/// - preserves existing attributes
///
/// # Example
///
/// ```ignore
/// #[rlm_type]
/// #[derive(Clone, Debug)]
/// pub struct Trajectory {
///     pub session_id: String,
///     pub steps: Vec<Step>,
/// }
/// ```
///
/// Expands to:
///
/// ```ignore
/// #[pyo3::pyclass]
/// #[derive(Clone, Debug, baml_bridge::BamlType, RlmType)]
/// pub struct Trajectory {
///     pub session_id: String,
///     pub steps: Vec<Step>,
/// }
/// ```
#[proc_macro_attribute]
pub fn rlm_type(attr: TokenStream, item: TokenStream) -> TokenStream {
    rlm_attr::expand(attr, item)
}

/// Derive macro for RLM-compatible types.
///
/// Generates `#[pymethods]` impl block with:
/// - Field getters
/// - `__repr__` (customizable via `#[rlm(repr = "...")]`)
/// - `__len__` and `__iter__` (if `#[rlm(iter = "field")]` specified)
/// - `__getitem__` (if `#[rlm(index = "field")]` specified)
/// - `__baml__()` for SUBMIT normalization
/// - `__rlm_schema__` class attribute for REPL discovery
///
/// # Container Attributes
///
/// - `#[rlm(repr = "...")]` - Custom `__repr__` template
/// - `#[rlm(iter = "field")]` - Field to iterate over
/// - `#[rlm(index = "field")]` - Field to index into
/// - `#[rlm(pyclass_name = "...")]` - Python class name override
/// - `#[rlm(property(name = "...", desc = "..."))]` - Document computed properties
///
/// # Field Attributes
///
/// - `#[rlm(desc = "...")]` - Human-readable description
/// - `#[rlm(skip_python)]` - Skip getter generation
/// - `#[rlm(filter_property = "...", filter_value = "...")]` - Generate filter property
/// - `#[rlm(flatten_property = "...")]` - Flatten nested collection
#[proc_macro_derive(RlmType, attributes(rlm))]
pub fn derive_rlm_type(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::DeriveInput);

    match rlm_type::derive(input) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}
