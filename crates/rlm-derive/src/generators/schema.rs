//! Schema metadata generation for `__rlm_schema__`.
//!
//! Produces a method that returns a dict mapping field/property name to
//! `(type, desc)` tuples for REPL discovery.

use std::collections::HashSet;

use proc_macro2::TokenStream;
use quote::quote;

use crate::attrs::RlmTypeAttrs;

/// Generate the `__rlm_schema__()` method.
///
/// Returns a dict mapping field name -> (type, desc) for REPL discovery.
/// Computed properties declared via `#[rlm(property(...))]` are also included.
pub fn generate_schema_method(attrs: &RlmTypeAttrs) -> TokenStream {
    let field_names: HashSet<String> = attrs.fields().map(|field| field.name()).collect();

    let field_entries: Vec<TokenStream> = attrs
        .fields()
        .filter(|field| field.should_include_in_schema())
        .map(|field| {
            let name = field.name();
            let desc = field.desc.clone().unwrap_or_default();
            let field_ty = &field.ty;
            quote! {
                ::pyo3::types::PyDictMethods::set_item(
                    &schema,
                    #name,
                    (stringify!(#field_ty), #desc),
                )?;
            }
        })
        .collect();

    let property_entries: Vec<TokenStream> = attrs
        .property
        .iter()
        .filter(|prop| !field_names.contains(&prop.name))
        .map(|prop| {
            let name = &prop.name;
            let desc = prop.desc.clone().unwrap_or_default();
            quote! {
                ::pyo3::types::PyDictMethods::set_item(
                    &schema,
                    #name,
                    ("property", #desc),
                )?;
            }
        })
        .collect();

    quote! {
        /// Machine-readable field schema for REPL discovery.
        fn __rlm_schema__(&self, py: ::pyo3::Python<'_>) -> ::pyo3::PyResult<::pyo3::PyObject> {
            let schema = ::pyo3::types::PyDict::new(py);
            #(#field_entries)*
            #(#property_entries)*
            Ok(schema.into_any().unbind())
        }
    }
}
