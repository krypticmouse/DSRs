//! `#[derive(Example)]` — connects a plain trainset-row struct to signatures.
//!
//! The row struct stays yours: any fields, any types. The derive generates
//! `ToInput`/`ToOutput` impls toward the signatures named in `#[example(...)]`,
//! matching row fields to signature Input/Output fields *by name* at compile
//! time — a typo or type mismatch is a compile error, not a runtime one.
//!
//! ```ignore
//! #[derive(Example)]
//! #[example(QA)]
//! struct HotpotRow {
//!     #[input]
//!     question: String,
//!     #[output]
//!     answer: String,
//!     supporting_facts: Vec<String>, // metric-only: not part of input or output
//! }
//! ```
//!
//! Partition rules:
//! - `#[input]` marks the fields projected into `Signature::Input`. At least one is required.
//! - With no explicit `#[output]` marks, every non-input field is an output.
//! - Marking any field `#[output]` switches to explicit mode: only marked fields
//!   are outputs, and unmarked non-input fields become metric-only metadata.
//! - `#[meta]` excludes a field from the default output partition without
//!   switching to explicit mode.
//! - An empty output set is fine: no `ToOutput` impl is generated. Metrics read
//!   the row directly, so gold fields that don't line up with the signature's
//!   Output struct (e.g. reasoning-bearing signatures) can simply stay `#[meta]`.

use quote::{format_ident, quote};
use syn::punctuated::Punctuated;
use syn::{Data, DeriveInput, Error, Fields, Ident, Path, Result, Token};

pub(crate) fn expand_example(input: &DeriveInput, runtime: &Path) -> Result<proc_macro2::TokenStream> {
    let signatures = collect_signatures(input)?;
    if signatures.is_empty() {
        return Err(Error::new_spanned(
            &input.ident,
            "#[derive(Example)] needs at least one target signature: #[example(MySignature)]",
        ));
    }

    let fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(named) => &named.named,
            _ => {
                return Err(Error::new_spanned(
                    &input.ident,
                    "#[derive(Example)] requires named struct fields",
                ));
            }
        },
        _ => {
            return Err(Error::new_spanned(
                &input.ident,
                "#[derive(Example)] only supports structs",
            ));
        }
    };

    let mut input_fields: Vec<&Ident> = Vec::new();
    let mut marked_outputs: Vec<&Ident> = Vec::new();
    let mut unmarked: Vec<&Ident> = Vec::new();

    for field in fields {
        let ident = field.ident.as_ref().expect("named field");
        let is_input = field.attrs.iter().any(|attr| attr.path().is_ident("input"));
        let is_output = field.attrs.iter().any(|attr| attr.path().is_ident("output"));
        let is_meta = field.attrs.iter().any(|attr| attr.path().is_ident("meta"));
        if is_meta && (is_input || is_output) {
            return Err(Error::new_spanned(
                ident,
                "#[meta] cannot be combined with #[input] or #[output]",
            ));
        }
        match (is_input, is_output) {
            (true, true) => {
                return Err(Error::new_spanned(
                    ident,
                    "a field cannot be both #[input] and #[output]",
                ));
            }
            (true, false) => input_fields.push(ident),
            (false, true) => marked_outputs.push(ident),
            (false, false) if is_meta => {}
            (false, false) => unmarked.push(ident),
        }
    }

    if input_fields.is_empty() {
        return Err(Error::new_spanned(
            &input.ident,
            "#[derive(Example)] needs at least one #[input] field",
        ));
    }

    // Explicit #[output] marks switch off the everything-else-is-output default.
    let output_fields: Vec<&Ident> = if marked_outputs.is_empty() {
        unmarked.clone()
    } else {
        marked_outputs
    };

    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let mut impls = proc_macro2::TokenStream::new();
    for (idx, sig) in signatures.iter().enumerate() {
        let input_alias = format_ident!("__DsrsExampleInput{idx}");
        let output_alias = format_ident!("__DsrsExampleOutput{idx}");
        let input_inits = input_fields.iter().map(|field| {
            quote! { #field: ::core::clone::Clone::clone(&self.#field) }
        });
        let output_inits = output_fields.iter().map(|field| {
            quote! { #field: ::core::clone::Clone::clone(&self.#field) }
        });

        impls.extend(quote! {
            #[allow(non_camel_case_types)]
            impl #impl_generics #runtime::ToInput<<#sig as #runtime::Signature>::Input>
                for #name #ty_generics #where_clause
            {
                fn to_input(&self) -> <#sig as #runtime::Signature>::Input {
                    type #input_alias = <#sig as #runtime::Signature>::Input;
                    #input_alias { #(#input_inits,)* }
                }
            }
        });

        if !output_fields.is_empty() {
            impls.extend(quote! {
                #[allow(non_camel_case_types)]
                impl #impl_generics #runtime::ToOutput<<#sig as #runtime::Signature>::Output>
                    for #name #ty_generics #where_clause
                {
                    fn to_output(&self) -> <#sig as #runtime::Signature>::Output {
                        type #output_alias = <#sig as #runtime::Signature>::Output;
                        #output_alias { #(#output_inits,)* }
                    }
                }
            });
        }
    }

    Ok(impls)
}

/// Collects target signature paths from every `#[example(...)]` attribute
/// (comma-separated and/or repeated attributes both work).
fn collect_signatures(input: &DeriveInput) -> Result<Vec<Path>> {
    let mut signatures = Vec::new();
    for attr in &input.attrs {
        if !attr.path().is_ident("example") {
            continue;
        }
        let paths =
            attr.parse_args_with(Punctuated::<Path, Token![,]>::parse_terminated)?;
        signatures.extend(paths);
    }
    Ok(signatures)
}
