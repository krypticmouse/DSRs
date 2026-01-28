//! Implementation of the `#[rlm_type]` attribute macro.
//!
//! This macro rewrites a struct to add `#[pyclass]` and inject the necessary derives.

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Attribute, ItemStruct};

/// Parse `#[rlm(pyclass_name = "...")]` from attributes.
fn extract_pyclass_name(attrs: &[Attribute]) -> Option<String> {
    for attr in attrs {
        if !attr.path().is_ident("rlm") {
            continue;
        }
        let mut found: Option<String> = None;
        let _ = attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("pyclass_name") {
                let lit: syn::LitStr = meta.value()?.parse()?;
                found = Some(lit.value());
            }
            Ok(())
        });
        if found.is_some() {
            return found;
        }
    }
    None
}

/// Expand the `#[rlm_type]` attribute macro.
///
/// This:
/// 1. Adds `#[::dspy_rs::pyo3::pyclass]` (with optional name override)
/// 2. Merges `BamlType` and `RlmType` into existing `#[derive(...)]` or creates new one
/// 3. Preserves all other attributes
pub fn expand(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemStruct);
    let mut attrs = input.attrs.clone();

    // Add #[pyclass] with optional name override
    let pyclass_name = extract_pyclass_name(&attrs);
    if let Some(name) = pyclass_name {
        let lit = syn::LitStr::new(&name, proc_macro2::Span::call_site());
        attrs.push(syn::parse_quote!(#[::dspy_rs::pyo3::pyclass(name = #lit)]));
    } else {
        attrs.push(syn::parse_quote!(#[::dspy_rs::pyo3::pyclass]));
    }

    // Inject BamlType + RlmType derives (merge with existing derive list)
    let mut merged = false;
    for attr in attrs.iter_mut() {
        if attr.path().is_ident("derive") {
            let mut paths: syn::punctuated::Punctuated<syn::Path, syn::token::Comma> = attr
                .parse_args_with(syn::punctuated::Punctuated::parse_terminated)
                .unwrap_or_default();

            let to_add: [syn::Path; 2] = [
                syn::parse_quote!(::dspy_rs::baml_bridge::BamlType),
                syn::parse_quote!(RlmType),
            ];

            for p in to_add {
                // Check if already present (by last segment)
                let p_str = p
                    .segments
                    .last()
                    .map(|s| s.ident.to_string())
                    .unwrap_or_default();
                let already_present = paths.iter().any(|existing| {
                    existing
                        .segments
                        .last()
                        .map(|s| s.ident.to_string())
                        .unwrap_or_default()
                        == p_str
                });
                if !already_present {
                    paths.push(p);
                }
            }

            *attr = syn::parse_quote!(#[derive(#paths)]);
            merged = true;
            break;
        }
    }

    if !merged {
        attrs.push(syn::parse_quote!(#[derive(::dspy_rs::baml_bridge::BamlType, RlmType)]));
    }

    let ident = &input.ident;
    let generics = &input.generics;
    let vis = &input.vis;
    let fields = &input.fields;

    let expanded = quote! {
        #(#attrs)*
        #vis struct #ident #generics #fields
    };
    expanded.into()
}
