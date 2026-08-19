//! `#[derive(Example)]` — marks a plain struct as a trainset row.
//!
//! The row stays yours: any fields, any types, no signature named anywhere.
//! The derive generates generic `ToInput`/`ToOutput` impls that project the
//! row into *any* target type by field name through serde: extra row fields
//! (gold labels, metric-only metadata) are ignored; a missing or mismatched
//! field is a runtime error on first use, naming both types.
//!
//! ```ignore
//! #[derive(Example, Clone, Debug, serde::Serialize)]
//! struct HotpotRow {
//!     question: String,              // → QAInput.question, matched by name
//!     answer: String,                // → QAOutput.answer, when seeding demos
//!     supporting_facts: Vec<String>, // metric-only: ignored by projection
//! }
//! ```
//!
//! The row must be `Serialize` (derive it alongside `Example`); target types
//! come from the signatures the row meets at the call site, so one row type
//! works with every module whose input it can fill.

use quote::quote;
use syn::{Data, DeriveInput, Error, Fields, Path, Result, parse_quote};

pub(crate) fn expand_example(
    input: &DeriveInput,
    runtime: &Path,
) -> Result<proc_macro2::TokenStream> {
    match &input.data {
        Data::Struct(data) => {
            if !matches!(data.fields, Fields::Named(_)) {
                return Err(Error::new_spanned(
                    &input.ident,
                    "#[derive(Example)] requires named struct fields (projection matches by field name)",
                ));
            }
        }
        _ => {
            return Err(Error::new_spanned(
                &input.ident,
                "#[derive(Example)] only supports structs",
            ));
        }
    }

    let name = &input.ident;
    let serde = quote! { #runtime::__macro_support::serde };
    let anyhow = quote! { #runtime::__macro_support::anyhow };

    let (_, ty_generics, _) = input.generics.split_for_impl();

    // The impls add one generic param (the projection target) on top of the
    // row's own generics, plus the serde bounds the projection needs.
    let mut generics = input.generics.clone();
    generics
        .params
        .push(parse_quote!(__DsrsTarget: #serde::de::DeserializeOwned));
    let where_clause = generics.make_where_clause();
    where_clause
        .predicates
        .push(parse_quote!(Self: #serde::Serialize));
    let (impl_generics, _, where_clause) = generics.split_for_impl();

    Ok(quote! {
        impl #impl_generics #runtime::ToInput<__DsrsTarget> for #name #ty_generics #where_clause {
            fn to_input(&self) -> #anyhow::Result<__DsrsTarget> {
                #runtime::core::example::project(self)
            }
        }

        impl #impl_generics #runtime::ToOutput<__DsrsTarget> for #name #ty_generics #where_clause {
            fn to_output(&self) -> #anyhow::Result<__DsrsTarget> {
                #runtime::core::example::project(self)
            }
        }
    })
}
