//! Shared entry point for the `#[derive(RlmType)]` macro.

use darling::FromDeriveInput;
use proc_macro2::TokenStream;
use syn::DeriveInput;

use crate::{attrs, generators};

pub(crate) fn derive(input: DeriveInput) -> syn::Result<TokenStream> {
    ensure_pyclass(&input)?;

    let attrs = attrs::RlmTypeAttrs::from_derive_input(&input)
        .map_err(|e| syn::Error::new_spanned(&input, e.to_string()))?;

    let validation_errors = attrs.validate();
    if !validation_errors.is_empty() {
        return Err(syn::Error::new_spanned(
            &input,
            validation_errors.join("; "),
        ));
    }

    let repr_method = generators::generate_repr(&attrs)
        .map_err(|e| syn::Error::new_spanned(&input, e.to_string()))?;

    let iter_support = generators::generate_iter_support(&attrs)?;
    let describe_impl = generators::generate_describe(&attrs)?;
    let property_methods = generators::generate_properties(&attrs)?;

    let mut extra_methods = Vec::new();
    extra_methods.extend(iter_support.methods);
    extra_methods.extend(property_methods);

    let pymethods = generators::generate_pymethods_with_repr(&attrs, repr_method, &extra_methods);

    let mut output = TokenStream::new();
    output.extend(iter_support.extra_items);
    output.extend(pymethods);
    output.extend(describe_impl);

    Ok(output)
}

fn ensure_pyclass(input: &DeriveInput) -> syn::Result<()> {
    if has_pyclass_attr(&input.attrs) {
        Ok(())
    } else {
        Err(syn::Error::new_spanned(
            &input.ident,
            "RlmType requires a #[pyclass] attribute; consider using #[rlm_type] to add it",
        ))
    }
}

fn has_pyclass_attr(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|attr| {
        let path = attr.path();
        path.is_ident("pyclass")
            || path
                .segments
                .last()
                .map(|segment| segment.ident == "pyclass")
                .unwrap_or(false)
    })
}
