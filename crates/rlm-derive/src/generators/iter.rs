//! Iterator-related PyO3 code generation for `#[derive(RlmType)]`.
//!
//! Generates:
//! - `__len__` based on the configured iter field
//! - `__iter__` returning a private iterator pyclass
//! - `__getitem__` with negative indexing support

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{spanned::Spanned, GenericArgument, PathArguments, Type};

use crate::attrs::{RlmFieldAttrs, RlmTypeAttrs};

pub struct IterSupport {
    pub extra_items: TokenStream,
    pub methods: Vec<TokenStream>,
}

pub fn generate_iter_support(attrs: &RlmTypeAttrs) -> syn::Result<IterSupport> {
    let mut extra_items = TokenStream::new();
    let mut methods = Vec::new();

    if let Some(iter_field) = attrs.iter_field_attrs() {
        let field_ident = iter_field
            .ident
            .as_ref()
            .expect("iter field must be named");
        let item_ty = extract_vec_inner_type(&iter_field.ty).ok_or_else(|| {
            syn::Error::new_spanned(
                &iter_field.ty,
                format!(
                    "iter field `{}` must be a Vec<T> or slice type",
                    iter_field.name()
                ),
            )
        })?;

        let iter_struct = format_ident!("__{}Iter", attrs.ident);

        methods.push(generate_len_method(field_ident));
        methods.push(generate_iter_method(field_ident, &iter_struct));

        extra_items = generate_iterator_struct(&iter_struct, &item_ty);
    }

    if let Some(index_field) = attrs.index_field_attrs() {
        let field_ident = index_field
            .ident
            .as_ref()
            .expect("index field must be named");
        let item_ty = extract_vec_inner_type(&index_field.ty).ok_or_else(|| {
            syn::Error::new_spanned(
                &index_field.ty,
                format!(
                    "index field `{}` must be a Vec<T> or slice type",
                    index_field.name()
                ),
            )
        })?;

        methods.push(generate_getitem_method(field_ident, &item_ty));
    }

    Ok(IterSupport {
        extra_items,
        methods,
    })
}

fn generate_len_method(field_ident: &syn::Ident) -> TokenStream {
    quote! {
        fn __len__(&self) -> usize {
            self.#field_ident.len()
        }
    }
}

fn generate_iter_method(field_ident: &syn::Ident, iter_struct: &syn::Ident) -> TokenStream {
    quote! {
        fn __iter__(&self) -> #iter_struct {
            #iter_struct {
                items: self.#field_ident.clone(),
                index: 0,
            }
        }
    }
}

fn generate_getitem_method(field_ident: &syn::Ident, item_ty: &Type) -> TokenStream {
    quote! {
        fn __getitem__(&self, index: isize) -> ::pyo3::PyResult<#item_ty> {
            let len = self.#field_ident.len() as isize;
            let index = if index < 0 { len + index } else { index };
            if index < 0 || index >= len {
                return Err(::pyo3::exceptions::PyIndexError::new_err("index out of range"));
            }
            Ok(self.#field_ident[index as usize].clone())
        }
    }
}

fn generate_iterator_struct(iter_struct: &syn::Ident, item_ty: &Type) -> TokenStream {
    quote! {
        #[pyo3::pyclass]
        struct #iter_struct {
            items: Vec<#item_ty>,
            index: usize,
        }

        #[pyo3::pymethods]
        impl #iter_struct {
            fn __iter__(slf: ::pyo3::PyRef<Self>) -> ::pyo3::PyRef<Self> {
                slf
            }

            fn __next__(mut slf: ::pyo3::PyRefMut<Self>) -> Option<#item_ty> {
                if slf.index >= slf.items.len() {
                    return None;
                }
                let item = slf.items[slf.index].clone();
                slf.index += 1;
                Some(item)
            }
        }
    }
}

fn extract_vec_inner_type(ty: &Type) -> Option<Type> {
    match ty {
        Type::Path(type_path) => {
            let segment = type_path.path.segments.last()?;
            if segment.ident != "Vec" {
                return None;
            }
            match &segment.arguments {
                PathArguments::AngleBracketed(args) => {
                    if args.args.len() != 1 {
                        return None;
                    }
                    match args.args.first()? {
                        GenericArgument::Type(inner) => Some(inner.clone()),
                        _ => None,
                    }
                }
                _ => None,
            }
        }
        Type::Reference(reference) => match reference.elem.as_ref() {
            Type::Slice(slice) => Some((*slice.elem).clone()),
            _ => None,
        },
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_vec_inner_type_vec() {
        let ty: Type = syn::parse_quote!(Vec<String>);
        let inner = extract_vec_inner_type(&ty).expect("expected Vec inner type");
        assert_eq!(quote!(#inner).to_string(), "String");
    }

    #[test]
    fn test_extract_vec_inner_type_slice() {
        let ty: Type = syn::parse_quote!(&[u8]);
        let inner = extract_vec_inner_type(&ty).expect("expected slice inner type");
        assert_eq!(quote!(#inner).to_string(), "u8");
    }

    #[test]
    fn test_extract_vec_inner_type_none() {
        let ty: Type = syn::parse_quote!(HashMap<String, String>);
        assert!(extract_vec_inner_type(&ty).is_none());
    }
}
