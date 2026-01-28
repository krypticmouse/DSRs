//! RlmDescribe trait generation for `#[derive(RlmType)]`.

use proc_macro2::TokenStream;
use quote::quote;
use syn::{GenericArgument, PathArguments, Type};

use crate::attrs::{RlmFieldAttrs, RlmPropertyAttrs, RlmTypeAttrs};

pub fn generate_describe(attrs: &RlmTypeAttrs) -> syn::Result<TokenStream> {
    let struct_name = &attrs.ident;
    let (impl_generics, ty_generics, where_clause) = attrs.generics.split_for_impl();
    let type_name = attrs.python_class_name();
    let type_name_lit = syn::LitStr::new(&type_name, proc_macro2::Span::call_site());

    let field_descs: Vec<TokenStream> = attrs
        .fields()
        .filter(|field| field.should_include_in_schema())
        .map(generate_field_desc)
        .collect();

    let property_descs = generate_property_descs(attrs)?;

    let is_iterable = attrs.iter_field.is_some();
    let is_indexable = attrs.index_field.is_some();

    Ok(quote! {
        impl #impl_generics ::rlm_core::describe::RlmDescribe for #struct_name #ty_generics #where_clause {
            fn type_name() -> &'static str {
                #type_name_lit
            }

            fn fields() -> Vec<::rlm_core::describe::RlmFieldDesc> {
                vec![
                    #(#field_descs),*
                ]
            }

            fn properties() -> Vec<::rlm_core::describe::RlmPropertyDesc> {
                vec![
                    #(#property_descs),*
                ]
            }

            fn is_iterable() -> bool {
                #is_iterable
            }

            fn is_indexable() -> bool {
                #is_indexable
            }

            fn describe_value(&self) -> String {
                let field_names: Vec<&str> = Self::fields().iter().map(|f| f.name).collect();
                if field_names.is_empty() {
                    format!("{} {{}}", Self::type_name())
                } else {
                    format!("{} {{ {} }}", Self::type_name(), field_names.join(", "))
                }
            }
        }
    })
}

fn generate_field_desc(field: &RlmFieldAttrs) -> TokenStream {
    let field_ident = field
        .ident
        .as_ref()
        .expect("RlmDescribe fields require named fields");
    let field_name = field_ident.to_string();
    let field_name_lit = syn::LitStr::new(&field_name, proc_macro2::Span::call_site());
    let field_ty = &field.ty;

    let mut expr = quote! {
        ::rlm_core::describe::RlmFieldDesc::new(#field_name_lit, stringify!(#field_ty))
    };

    if let Some(desc) = &field.desc {
        let desc_lit = syn::LitStr::new(desc, proc_macro2::Span::call_site());
        expr = quote! { #expr.with_desc(#desc_lit) };
    }

    if is_optional_type(field_ty) {
        expr = quote! { #expr.optional() };
    }

    if is_collection_type(field_ty) {
        expr = quote! { #expr.collection() };
    }

    expr
}

fn generate_property_descs(attrs: &RlmTypeAttrs) -> syn::Result<Vec<TokenStream>> {
    let mut descs = Vec::new();

    for prop in &attrs.property {
        descs.push(property_desc_from_attr(prop));
    }

    for field in attrs.fields() {
        if let Some(filter_name) = &field.filter_property {
            descs.push(generate_filter_property_desc(field, filter_name)?);
        }
        if let Some(flatten_name) = &field.flatten_property {
            descs.push(generate_flatten_property_desc(field, flatten_name)?);
        }
    }

    Ok(descs)
}

fn property_desc_from_attr(prop: &RlmPropertyAttrs) -> TokenStream {
    let name_lit = syn::LitStr::new(&prop.name, proc_macro2::Span::call_site());
    let mut expr = quote! {
        ::rlm_core::describe::RlmPropertyDesc::new(#name_lit, "Unknown")
    };

    if let Some(desc) = &prop.desc {
        let desc_lit = syn::LitStr::new(desc, proc_macro2::Span::call_site());
        expr = quote! { #expr.with_desc(#desc_lit) };
    }

    expr
}

fn generate_filter_property_desc(
    field: &RlmFieldAttrs,
    property_name: &str,
) -> syn::Result<TokenStream> {
    let name_lit = syn::LitStr::new(property_name, proc_macro2::Span::call_site());
    let inner_ty = extract_vec_inner_type(&field.ty).ok_or_else(|| {
        syn::Error::new_spanned(
            &field.ty,
            format!(
                "filter_property `{}` only works on Vec<T> fields",
                property_name
            ),
        )
    })?;

    Ok(quote! {
        ::rlm_core::describe::RlmPropertyDesc::new(#name_lit, stringify!(Vec<#inner_ty>))
    })
}

fn generate_flatten_property_desc(
    field: &RlmFieldAttrs,
    property_name: &str,
) -> syn::Result<TokenStream> {
    let name_lit = syn::LitStr::new(property_name, proc_macro2::Span::call_site());
    let inner_ty = extract_innermost_vec_type(&field.ty).ok_or_else(|| {
        syn::Error::new_spanned(
            &field.ty,
            format!(
                "flatten_property `{}` requires Option<Vec<T>> or Vec<T>",
                property_name
            ),
        )
    })?;

    Ok(quote! {
        ::rlm_core::describe::RlmPropertyDesc::new(#name_lit, stringify!(Vec<#inner_ty>))
    })
}

fn is_optional_type(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty && let Some(segment) = type_path.path.segments.last() {
        return segment.ident == "Option";
    }
    false
}

fn is_collection_type(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty && let Some(segment) = type_path.path.segments.last() {
        if segment.ident == "Vec" || segment.ident == "HashMap" || segment.ident == "BTreeMap" {
            return true;
        }
        if segment.ident == "Option"
            && let PathArguments::AngleBracketed(args) = &segment.arguments
            && let Some(GenericArgument::Type(inner)) = args.args.first()
        {
            return is_collection_type(inner);
        }
    }
    false
}

fn extract_vec_inner_type(ty: &Type) -> Option<Type> {
    if let Type::Path(type_path) = ty {
        let segment = type_path.path.segments.last()?;
        if segment.ident != "Vec" {
            return None;
        }
        if let PathArguments::AngleBracketed(args) = &segment.arguments
            && let Some(GenericArgument::Type(inner)) = args.args.first()
        {
            return Some(inner.clone());
        }
    }
    None
}

fn extract_innermost_vec_type(ty: &Type) -> Option<Type> {
    if let Type::Path(type_path) = ty {
        let segment = type_path.path.segments.last()?;
        if segment.ident == "Option"
            && let PathArguments::AngleBracketed(args) = &segment.arguments
            && let Some(GenericArgument::Type(inner)) = args.args.first()
        {
            return extract_innermost_vec_type(inner);
        }
        if segment.ident == "Option" {
            return None;
        }
        if segment.ident == "Vec"
            && let PathArguments::AngleBracketed(args) = &segment.arguments
            && let Some(GenericArgument::Type(inner)) = args.args.first()
        {
            return Some(inner.clone());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use darling::FromDeriveInput;

    #[test]
    fn test_is_optional_type() {
        let ty: Type = syn::parse_quote!(Option<String>);
        assert!(is_optional_type(&ty));
        let ty: Type = syn::parse_quote!(Vec<String>);
        assert!(!is_optional_type(&ty));
    }

    #[test]
    fn test_is_collection_type_vec() {
        let ty: Type = syn::parse_quote!(Vec<u8>);
        assert!(is_collection_type(&ty));
    }

    #[test]
    fn test_is_collection_type_option_vec() {
        let ty: Type = syn::parse_quote!(Option<Vec<u8>>);
        assert!(is_collection_type(&ty));
    }

    #[test]
    fn test_extract_vec_inner_type() {
        let ty: Type = syn::parse_quote!(Vec<Foo>);
        let inner = extract_vec_inner_type(&ty).expect("expected inner type");
        assert_eq!(quote!(#inner).to_string(), "Foo");
    }

    #[test]
    fn test_extract_innermost_vec_type_option() {
        let ty: Type = syn::parse_quote!(Option<Vec<Bar>>);
        let inner = extract_innermost_vec_type(&ty).expect("expected inner type");
        assert_eq!(quote!(#inner).to_string(), "Bar");
    }

    #[test]
    fn test_generate_describe_rejects_filter_on_non_vec() {
        let input: syn::DeriveInput = syn::parse_quote! {
            pub struct Example {
                #[rlm(filter_property = "user_steps", filter_value = "user")]
                pub steps: String,
            }
        };

        let attrs = RlmTypeAttrs::from_derive_input(&input).expect("attrs parse");
        let result = generate_describe(&attrs);
        assert!(result.is_err(), "expected error for non-Vec filter field");
    }

    #[test]
    fn test_generate_describe_rejects_flatten_on_non_vec() {
        let input: syn::DeriveInput = syn::parse_quote! {
            pub struct Example {
                #[rlm(flatten_property = "all_steps")]
                pub steps: Option<String>,
            }
        };

        let attrs = RlmTypeAttrs::from_derive_input(&input).expect("attrs parse");
        let result = generate_describe(&attrs);
        assert!(result.is_err(), "expected error for non-Vec flatten field");
    }
}
