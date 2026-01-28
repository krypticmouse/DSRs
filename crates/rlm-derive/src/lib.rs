//! Derive macros for RLM-compatible types.
//!
//! This crate provides two main proc-macros:
//!
//! - `#[rlm_type]` - Attribute macro that adds `#[pyclass]` and injects
//!   `#[derive(BamlType, RlmType)]`
//! - `#[derive(RlmType)]` - Derive macro that generates `#[pymethods]` impls
//!   with getters, `__repr__`, `__len__`, `__iter__`, `__getitem__`, and `__baml__`

use darling::FromDeriveInput;
use proc_macro::TokenStream;
use syn::parse_macro_input;

mod attrs;
mod generators;
mod rlm_attr;

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

    match expand_derive_rlm_type(&input) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

fn expand_derive_rlm_type(input: &syn::DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    // Parse attributes using darling
    let attrs = attrs::RlmTypeAttrs::from_derive_input(input)
        .map_err(|e| syn::Error::new_spanned(input, e.to_string()))?;

    // Validate the parsed attributes
    let validation_errors = attrs.validate();
    if !validation_errors.is_empty() {
        return Err(syn::Error::new_spanned(
            input,
            validation_errors.join("; "),
        ));
    }

    // Generate __repr__ method (can fail if template is invalid)
    let repr_method = generators::generate_repr(&attrs)
        .map_err(|e| syn::Error::new_spanned(input, e.to_string()))?;

    let iter_support = generators::generate_iter_support(&attrs)?;
    let property_methods = generators::generate_properties(&attrs)?;

    // Generate the #[pymethods] impl block
    // TODO (Task 1.7): Add __rlm_schema__ generation
    let mut extra_methods = Vec::new();
    extra_methods.extend(iter_support.methods);
    extra_methods.extend(property_methods);

    let pymethods =
        generators::generate_pymethods_with_repr(&attrs, repr_method, &extra_methods);

    let mut output = proc_macro2::TokenStream::new();
    output.extend(iter_support.extra_items);
    output.extend(pymethods);

    Ok(output)
}
