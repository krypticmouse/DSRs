//! PyO3 code generation for `#[derive(RlmType)]`.
//!
//! Generates `#[pymethods]` impl blocks with:
//! - Field getters (handling Copy vs Clone vs &str)
//! - `__baml__()` method for SUBMIT normalization

use proc_macro2::TokenStream;
use quote::quote;
use syn::Type;

use crate::attrs::{RlmFieldAttrs, RlmTypeAttrs};
use crate::generators::schema;

/// Generate the `#[pymethods]` impl block for a struct.
///
/// This generates:
/// - Field getters for all fields not marked `skip_python`
/// - `__baml__()` method that converts to a JSON-like Python object
#[allow(dead_code)]
pub fn generate_pymethods(attrs: &RlmTypeAttrs) -> TokenStream {
    let struct_name = &attrs.ident;
    let (impl_generics, ty_generics, where_clause) = attrs.generics.split_for_impl();

    let getters: Vec<TokenStream> = attrs
        .fields()
        .filter(|f| f.should_generate_getter())
        .map(generate_getter)
        .collect();

    let schema_method = schema::generate_schema_method(attrs);
    let baml_method = generate_baml_method();

    quote! {
        #[::dspy_rs::pyo3::pymethods]
        impl #impl_generics #struct_name #ty_generics #where_clause {
            #(#getters)*

            #schema_method

            #baml_method
        }
    }
}

/// Generate the `#[pymethods]` impl block with a custom repr method.
///
/// This generates:
/// - Field getters for all fields not marked `skip_python`
/// - `__repr__()` method (from the provided TokenStream)
/// - `__baml__()` method that converts to a JSON-like Python object
#[allow(dead_code)]
pub fn generate_pymethods_with_repr(
    attrs: &RlmTypeAttrs,
    repr_method: TokenStream,
    extra_methods: &[TokenStream],
) -> TokenStream {
    let struct_name = &attrs.ident;
    let (impl_generics, ty_generics, where_clause) = attrs.generics.split_for_impl();

    let getters: Vec<TokenStream> = attrs
        .fields()
        .filter(|f| f.should_generate_getter())
        .map(generate_getter)
        .collect();

    let schema_method = schema::generate_schema_method(attrs);
    let baml_method = generate_baml_method();

    quote! {
        #[::dspy_rs::pyo3::pymethods]
        impl #impl_generics #struct_name #ty_generics #where_clause {
            #(#getters)*

            #repr_method

            #(#extra_methods)*

            #schema_method

            #baml_method
        }
    }
}

/// Generate a getter for a single field.
///
/// Return strategy:
/// - Copy types (primitives): return by value
/// - `String`: return as `&str`
/// - Other types: clone
fn generate_getter(field: &RlmFieldAttrs) -> TokenStream {
    let field_name = field.ident.as_ref().expect("named field required");
    let field_ty = &field.ty;

    // Determine getter return type and body based on the field type
    let (return_ty, getter_body) = getter_strategy(field_ty, field_name);

    // Generate docstring from description if available
    let doc_attr = field.desc.as_ref().map(|desc| {
        quote! { #[doc = #desc] }
    });

    quote! {
        #doc_attr
        #[getter]
        fn #field_name(&self) -> #return_ty {
            #getter_body
        }
    }
}

/// Determine the return type and body for a field getter.
///
/// Strategy:
/// - `String` -> `&str`, return `self.field.as_str()`
/// - Copy types (i32, f64, bool, etc.) -> return by value
/// - Other types -> clone
fn getter_strategy(ty: &Type, field_name: &syn::Ident) -> (TokenStream, TokenStream) {
    if is_string_type(ty) {
        // String fields return &str
        (quote! { &str }, quote! { self.#field_name.as_str() })
    } else if is_copy_type(ty) {
        // Copy types return by value
        (quote! { #ty }, quote! { self.#field_name })
    } else {
        // Everything else gets cloned
        (quote! { #ty }, quote! { self.#field_name.clone() })
    }
}

/// Check if a type is `String`.
fn is_string_type(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty && let Some(segment) = type_path.path.segments.last() {
        return segment.ident == "String";
    }
    false
}

/// Check if a type is a Copy primitive.
///
/// This is a heuristic - we check for common primitive types.
/// For complex types, we default to cloning.
fn is_copy_type(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty && let Some(segment) = type_path.path.segments.last() {
        let ident = segment.ident.to_string();
        return matches!(
            ident.as_str(),
            "bool"
                | "i8"
                | "i16"
                | "i32"
                | "i64"
                | "i128"
                | "isize"
                | "u8"
                | "u16"
                | "u32"
                | "u64"
                | "u128"
                | "usize"
                | "f32"
                | "f64"
                | "char"
        );
    }
    false
}

/// Generate the `__baml__()` method.
///
/// This method converts the struct to a JSON-like Python object for SUBMIT normalization.
/// Uses `ToBamlValue` to convert to `BamlValue`, then `baml_value_to_py` for Python conversion.
fn generate_baml_method() -> TokenStream {
    quote! {
        /// Convert this value to a JSON-like Python object for SUBMIT normalization.
        ///
        /// Returns a Python dict/list/primitive that can be used with BAML's SUBMIT.
        fn __baml__(&self, py: ::dspy_rs::pyo3::Python<'_>) -> ::dspy_rs::pyo3::PyResult<::dspy_rs::pyo3::PyObject> {
            let value = ::dspy_rs::baml_bridge::ToBamlValue::to_baml_value(self);
            Ok(::dspy_rs::baml_bridge::py::baml_value_to_py(py, &value))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_string_type() {
        let string_ty: Type = syn::parse_quote!(String);
        assert!(is_string_type(&string_ty));

        let i32_ty: Type = syn::parse_quote!(i32);
        assert!(!is_string_type(&i32_ty));

        let vec_ty: Type = syn::parse_quote!(Vec<String>);
        assert!(!is_string_type(&vec_ty));
    }

    #[test]
    fn test_is_copy_type() {
        let i32_ty: Type = syn::parse_quote!(i32);
        assert!(is_copy_type(&i32_ty));

        let bool_ty: Type = syn::parse_quote!(bool);
        assert!(is_copy_type(&bool_ty));

        let f64_ty: Type = syn::parse_quote!(f64);
        assert!(is_copy_type(&f64_ty));

        let string_ty: Type = syn::parse_quote!(String);
        assert!(!is_copy_type(&string_ty));

        let vec_ty: Type = syn::parse_quote!(Vec<i32>);
        assert!(!is_copy_type(&vec_ty));
    }
}
