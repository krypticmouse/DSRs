use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
use syn::{Data, DeriveInput, Fields, Meta, parse_macro_input};

use crate::runtime_path::{ensure_facet_resolvable, resolve_dspy_rs_path};

pub(crate) fn expand(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr_args = parse_macro_input!(
        attr with syn::punctuated::Punctuated::<Meta, syn::Token![,]>::parse_terminated
    );
    let mut skip_repr = false;
    for meta in attr_args {
        match meta {
            Meta::Path(path) if path.is_ident("skip_repr") => {
                skip_repr = true;
            }
            other => {
                return syn::Error::new_spanned(
                    quote!(#other),
                    "unsupported #[rlm_type(...)] key; supported keys in V1: skip_repr",
                )
                .to_compile_error()
                .into();
            }
        }
    }

    let mut input = parse_macro_input!(item as DeriveInput);
    if let Err(err) = validate_input(&input) {
        return err.to_compile_error().into();
    }
    if let Err(err) = ensure_facet_resolvable() {
        return err.to_compile_error().into();
    }

    let runtime = match resolve_dspy_rs_path() {
        Ok(path) => path,
        Err(err) => return err.to_compile_error().into(),
    };
    let pyo3_crate = pyo3_crate_lit(&runtime);

    if has_baml_type_attr(&input.attrs) {
        return syn::Error::new_spanned(
            &input.ident,
            "#[rlm_type] subsumes #[BamlType]; remove #[BamlType] and keep only #[rlm_type]",
        )
        .to_compile_error()
        .into();
    }

    input
        .attrs
        .push(syn::parse_quote!(#[#runtime::__macro_support::pyo3::pyclass(crate = #pyo3_crate)]));
    input.attrs.push(syn::parse_quote!(#[#runtime::BamlType]));
    merge_derive(&mut input.attrs, &[syn::parse_quote!(#runtime::RlmType)]);
    if skip_repr {
        input.attrs.push(syn::parse_quote!(#[rlm(skip_repr)]));
    }

    TokenStream::from(quote! { #input })
}

fn validate_input(input: &DeriveInput) -> syn::Result<()> {
    if !input.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &input.ident,
            "rlm_type currently supports named structs only (no generics or lifetimes)",
        ));
    }

    match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(_) => Ok(()),
            _ => Err(syn::Error::new_spanned(
                &input.ident,
                "rlm_type currently supports named structs only",
            )),
        },
        _ => Err(syn::Error::new_spanned(
            &input.ident,
            "rlm_type currently supports named structs only",
        )),
    }
}

fn has_baml_type_attr(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|attr| {
        let path = attr.path();
        path.is_ident("BamlType")
            || path
                .segments
                .last()
                .map(|s| s.ident == "BamlType")
                .unwrap_or(false)
    })
}

fn merge_derive(attrs: &mut Vec<syn::Attribute>, required: &[syn::Path]) {
    for attr in attrs.iter_mut() {
        if !attr.path().is_ident("derive") {
            continue;
        }

        let mut derives: syn::punctuated::Punctuated<syn::Path, syn::Token![,]> = attr
            .parse_args_with(syn::punctuated::Punctuated::parse_terminated)
            .unwrap_or_default();

        for required_path in required {
            if !derives
                .iter()
                .any(|existing| paths_eq(existing, required_path))
            {
                derives.push(required_path.clone());
            }
        }

        *attr = syn::parse_quote!(#[derive(#derives)]);
        return;
    }

    attrs.push(syn::parse_quote!(#[derive(#(#required),*)]));
}

fn paths_eq(left: &syn::Path, right: &syn::Path) -> bool {
    if left
        .segments
        .iter()
        .map(|s| &s.ident)
        .eq(right.segments.iter().map(|s| &s.ident))
    {
        return true;
    }

    left.segments.last().map(|s| &s.ident) == right.segments.last().map(|s| &s.ident)
}

fn pyo3_crate_lit(runtime: &syn::Path) -> syn::LitStr {
    let runtime_str = runtime
        .segments
        .iter()
        .map(|segment| segment.ident.to_string())
        .collect::<Vec<_>>()
        .join("::");
    syn::LitStr::new(
        &format!("{runtime_str}::__macro_support::pyo3"),
        Span::call_site(),
    )
}
