//! Shared helpers for the RFC 0003 step macros (`#[predict]`, `#[cot]`,
//! `#[agent]`, `#[tool]`, `#[module]`).

use proc_macro2::TokenStream;
use quote::quote;
use syn::punctuated::Punctuated;
use syn::{Attribute, Expr, ExprLit, Lit, Meta, Token};

/// Joins a fn's `///` doc lines into one string (the tool description /
/// instruction source).
pub(crate) fn doc_string(attrs: &[Attribute]) -> String {
    let mut lines = Vec::new();
    for attr in attrs {
        if !attr.path().is_ident("doc") {
            continue;
        }
        if let Meta::NameValue(nv) = &attr.meta
            && let Expr::Lit(ExprLit {
                lit: Lit::Str(s), ..
            }) = &nv.value
        {
            lines.push(s.value().trim().to_string());
        }
    }
    lines.join("\n").trim().to_string()
}

/// Parses an attr list that may only contain `model = "@name"`. Returns the
/// ref with the leading `@` stripped.
pub(crate) fn parse_model_only_attr(attr: TokenStream) -> syn::Result<Option<String>> {
    if attr.is_empty() {
        return Ok(None);
    }
    let metas: Punctuated<Meta, Token![,]> =
        syn::parse::Parser::parse2(Punctuated::parse_terminated, attr)?;
    let mut model = None;
    for meta in metas {
        match &meta {
            Meta::NameValue(nv) if nv.path.is_ident("model") => {
                model = Some(model_ref_value(&nv.value)?);
            }
            other => {
                return Err(syn::Error::new_spanned(
                    other,
                    "#[predict]/#[cot] accept only `model = \"@name\"`",
                ));
            }
        }
    }
    Ok(model)
}

/// Extracts a `"@name"` model ref literal, stripping the `@`.
pub(crate) fn model_ref_value(value: &Expr) -> syn::Result<String> {
    let Expr::Lit(ExprLit {
        lit: Lit::Str(s), ..
    }) = value
    else {
        return Err(syn::Error::new_spanned(
            value,
            "model refs are string literals, e.g. `model = \"@fast\"`",
        ));
    };
    let raw = s.value();
    let name = raw.strip_prefix('@').unwrap_or(&raw).to_string();
    if name.is_empty() {
        return Err(syn::Error::new_spanned(s, "empty model ref"));
    }
    Ok(name)
}

/// `Option<&str>` → `Some("…")`/`None` tokens.
pub(crate) fn option_str_tokens(value: Option<&str>) -> TokenStream {
    match value {
        Some(s) => quote! { ::core::option::Option::Some(#s) },
        None => quote! { ::core::option::Option::None },
    }
}

/// FNV-1a 64 — the macro-side stable content hash for extracted holes.
/// Deterministic across builds and platforms; independent of the runtime's
/// hasher (the value is an opaque fingerprint, compared only to itself).
pub(crate) fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Maps a *simple* declared Rust type to `FieldType` constructor tokens.
/// `None` = not in the mappable subset (String, bool, iN/uN, fN, Vec<…>,
/// Option<…>).
pub(crate) fn map_simple_type(ty: &syn::Type, runtime: &syn::Path) -> Option<TokenStream> {
    let syn::Type::Path(tp) = ty else { return None };
    if tp.qself.is_some() {
        return None;
    }
    let segment = tp.path.segments.last()?;
    let ident = segment.ident.to_string();
    match ident.as_str() {
        "String" => Some(quote! { #runtime::ir::FieldType::String }),
        "bool" => Some(quote! { #runtime::ir::FieldType::Bool }),
        "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" => {
            Some(quote! { #runtime::ir::FieldType::Int })
        }
        "f32" | "f64" => Some(quote! { #runtime::ir::FieldType::Float }),
        "Vec" | "Option" => {
            let syn::PathArguments::AngleBracketed(args) = &segment.arguments else {
                return None;
            };
            let inner = args.args.iter().find_map(|arg| match arg {
                syn::GenericArgument::Type(t) => Some(t),
                _ => None,
            })?;
            let inner = map_simple_type(inner, runtime)?;
            if ident == "Vec" {
                Some(quote! { #runtime::ir::FieldType::List(::std::boxed::Box::new(#inner)) })
            } else {
                Some(quote! { #runtime::ir::FieldType::Optional(::std::boxed::Box::new(#inner)) })
            }
        }
        _ => None,
    }
}
