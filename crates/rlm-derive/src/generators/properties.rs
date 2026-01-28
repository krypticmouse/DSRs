//! Computed property generation for `#[derive(RlmType)]`.
//!
//! Generates:
//! - Filter properties (e.g., user_steps) based on a child field value
//! - Flatten properties (e.g., all_tool_calls) across nested collections

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{GenericArgument, PathArguments, Type};

use crate::attrs::{RlmFieldAttrs, RlmTypeAttrs};

pub fn generate_properties(attrs: &RlmTypeAttrs) -> syn::Result<Vec<TokenStream>> {
    let fields: Vec<&RlmFieldAttrs> = attrs.fields().collect();
    let mut methods = Vec::new();

    for field in &fields {
        if let Some(ref filter_prop) = field.filter_property {
            methods.push(generate_filter_property(field, filter_prop)?);
        }

        if let Some(ref flatten_prop) = field.flatten_property {
            methods.push(generate_flatten_property(attrs, &fields, field, flatten_prop)?);
        }
    }

    Ok(methods)
}

fn generate_filter_property(
    field: &RlmFieldAttrs,
    property_name: &str,
) -> syn::Result<TokenStream> {
    let field_ident = field
        .ident
        .as_ref()
        .expect("filter_property requires a named field");
    let property_ident = format_ident!("{}", property_name);

    let filter_value = field.filter_value.as_ref().ok_or_else(|| {
        syn::Error::new_spanned(
            &field.ty,
            format!("filter_property `{}` requires filter_value", property_name),
        )
    })?;
    let filter_field = field.filter_field.as_deref().unwrap_or("source");
    let filter_field_ident = format_ident!("{}", filter_field);

    let item_type = extract_vec_inner_type(&field.ty).ok_or_else(|| {
        syn::Error::new_spanned(
            &field.ty,
            format!(
                "filter_property `{}` only works on Vec<T> fields",
                property_name
            ),
        )
    })?;
    let filter_value_lit = syn::LitStr::new(filter_value, proc_macro2::Span::call_site());

    Ok(quote! {
        #[getter]
        fn #property_ident(&self) -> Vec<#item_type> {
            self.#field_ident
                .iter()
                .filter(|item| item.#filter_field_ident == #filter_value_lit)
                .cloned()
                .collect()
        }
    })
}

fn generate_flatten_property(
    attrs: &RlmTypeAttrs,
    fields: &[&RlmFieldAttrs],
    field: &RlmFieldAttrs,
    property_name: &str,
) -> syn::Result<TokenStream> {
    let property_ident = format_ident!("{}", property_name);

    let parent_ident = resolve_parent_collection(attrs, fields, field).ok_or_else(|| {
        syn::Error::new_spanned(
            &field.ty,
            "flatten_property requires flatten_parent when more than one Vec<T> exists",
        )
    })?;

    let nested_field_ident = field
        .ident
        .as_ref()
        .expect("flatten_property requires a named field");

    let inner_type = extract_innermost_vec_type(&field.ty).ok_or_else(|| {
        syn::Error::new_spanned(
            &field.ty,
            format!(
                "flatten_property `{}` requires Option<Vec<T>> or Vec<T>",
                property_name
            ),
        )
    })?;

    if is_option_type(&field.ty) {
        Ok(quote! {
            #[getter]
            fn #property_ident(&self) -> Vec<#inner_type> {
                self.#parent_ident
                    .iter()
                    .filter_map(|item| item.#nested_field_ident.as_ref())
                    .flatten()
                    .cloned()
                    .collect()
            }
        })
    } else {
        Ok(quote! {
            #[getter]
            fn #property_ident(&self) -> Vec<#inner_type> {
                self.#parent_ident
                    .iter()
                    .flat_map(|item| &item.#nested_field_ident)
                    .cloned()
                    .collect()
            }
        })
    }
}

fn resolve_parent_collection<'a>(
    _attrs: &'a RlmTypeAttrs,
    fields: &'a [&'a RlmFieldAttrs],
    child: &RlmFieldAttrs,
) -> Option<&'a syn::Ident> {
    if let Some(parent) = &child.flatten_parent {
        return fields
            .iter()
            .find(|f| f.ident.as_ref().map(|i| i == parent).unwrap_or(false))
            .and_then(|f| f.ident.as_ref());
    }

    let vec_fields: Vec<_> = fields.iter().filter(|f| is_vec_type(&f.ty)).collect();
    if vec_fields.len() == 1 {
        return vec_fields[0].ident.as_ref();
    }

    None
}

fn is_vec_type(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty && let Some(segment) = type_path.path.segments.last() {
        return segment.ident == "Vec";
    }
    false
}

fn extract_vec_inner_type(ty: &Type) -> Option<Type> {
    if let Type::Path(type_path) = ty {
        let segment = type_path.path.segments.last()?;
        if segment.ident != "Vec" {
            return None;
        }
        if let PathArguments::AngleBracketed(args) = &segment.arguments {
            if let Some(GenericArgument::Type(inner)) = args.args.first() {
                return Some(inner.clone());
            }
        }
    }
    None
}

fn extract_innermost_vec_type(ty: &Type) -> Option<Type> {
    if let Type::Path(type_path) = ty {
        let segment = type_path.path.segments.last()?;
        if segment.ident == "Option" {
            if let PathArguments::AngleBracketed(args) = &segment.arguments {
                if let Some(GenericArgument::Type(inner)) = args.args.first() {
                    return extract_innermost_vec_type(inner);
                }
            }
            return None;
        }
        if segment.ident == "Vec" {
            if let PathArguments::AngleBracketed(args) = &segment.arguments {
                if let Some(GenericArgument::Type(inner)) = args.args.first() {
                    return Some(inner.clone());
                }
            }
        }
    }
    None
}

fn is_option_type(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty && let Some(segment) = type_path.path.segments.last() {
        return segment.ident == "Option";
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_vec_inner_type() {
        let ty: Type = syn::parse_quote!(Vec<String>);
        let inner = extract_vec_inner_type(&ty).expect("expected Vec inner type");
        assert_eq!(quote!(#inner).to_string(), "String");
    }

    #[test]
    fn test_extract_innermost_vec_type_option() {
        let ty: Type = syn::parse_quote!(Option<Vec<u8>>);
        let inner = extract_innermost_vec_type(&ty).expect("expected inner type");
        assert_eq!(quote!(#inner).to_string(), "u8");
    }

    #[test]
    fn test_extract_innermost_vec_type_vec() {
        let ty: Type = syn::parse_quote!(Vec<ToolCall>);
        let inner = extract_innermost_vec_type(&ty).expect("expected inner type");
        assert_eq!(quote!(#inner).to_string(), "ToolCall");
    }

    #[test]
    fn test_extract_innermost_vec_type_none() {
        let ty: Type = syn::parse_quote!(Option<ToolCall>);
        assert!(extract_innermost_vec_type(&ty).is_none());
    }
}
