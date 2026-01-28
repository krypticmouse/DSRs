//! Iterator-related PyO3 code generation for `#[derive(RlmType)]`.
//!
//! Generates:
//! - `__len__` based on the configured iter field
//! - `__iter__` returning a private iterator pyclass
//! - `__getitem__` with negative indexing support

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{GenericArgument, PathArguments, Type};

use crate::attrs::RlmTypeAttrs;

pub struct IterSupport {
    pub extra_items: TokenStream,
    pub methods: Vec<TokenStream>,
}

#[derive(Debug, Copy, Clone)]
enum SequenceKind {
    Vec,
    SliceRef,
}

pub fn generate_iter_support(attrs: &RlmTypeAttrs) -> syn::Result<IterSupport> {
    let mut extra_items = TokenStream::new();
    let mut methods = Vec::new();

    if let Some(iter_field) = attrs.iter_field_attrs() {
        let field_ident = iter_field
            .ident
            .as_ref()
            .expect("iter field must be named");
        let (item_ty, kind) = extract_sequence_inner_type(&iter_field.ty).ok_or_else(|| {
            syn::Error::new_spanned(
                &iter_field.ty,
                format!(
                    "iter field `{}` must be a Vec<T> or &[T]",
                    iter_field.name()
                ),
            )
        })?;

        let iter_struct = format_ident!("__{}Iter", attrs.ident);

        methods.push(generate_len_method(field_ident));
        methods.push(generate_iter_method(field_ident, &iter_struct, kind));

        extra_items = generate_iterator_struct(&iter_struct, &item_ty);
    }

    if let Some(index_field) = attrs.index_field_attrs() {
        let field_ident = index_field
            .ident
            .as_ref()
            .expect("index field must be named");
        let (item_ty, _) = extract_sequence_inner_type(&index_field.ty).ok_or_else(|| {
            syn::Error::new_spanned(
                &index_field.ty,
                format!(
                    "index field `{}` must be a Vec<T> or &[T]",
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

fn generate_iter_method(
    field_ident: &syn::Ident,
    iter_struct: &syn::Ident,
    kind: SequenceKind,
) -> TokenStream {
    let items_expr = match kind {
        SequenceKind::Vec => quote! { self.#field_ident.clone() },
        SequenceKind::SliceRef => quote! { self.#field_ident.to_vec() },
    };

    quote! {
        fn __iter__(&self) -> #iter_struct {
            #iter_struct {
                items: #items_expr,
                index: 0,
            }
        }
    }
}

fn generate_getitem_method(field_ident: &syn::Ident, item_ty: &Type) -> TokenStream {
    quote! {
        fn __getitem__(&self, index: isize) -> ::dspy_rs::pyo3::PyResult<#item_ty> {
            let len = self.#field_ident.len() as isize;
            let index = if index < 0 { len + index } else { index };
            if index < 0 || index >= len {
                return Err(::dspy_rs::pyo3::exceptions::PyIndexError::new_err("index out of range"));
            }
            Ok(self.#field_ident[index as usize].clone())
        }
    }
}

fn generate_iterator_struct(iter_struct: &syn::Ident, item_ty: &Type) -> TokenStream {
    quote! {
        #[::dspy_rs::pyo3::pyclass]
        struct #iter_struct {
            items: Vec<#item_ty>,
            index: usize,
        }

        #[::dspy_rs::pyo3::pymethods]
        impl #iter_struct {
            fn __iter__(slf: ::dspy_rs::pyo3::PyRef<'_, Self>) -> ::dspy_rs::pyo3::PyRef<'_, Self> {
                slf
            }

            fn __next__(mut slf: ::dspy_rs::pyo3::PyRefMut<'_, Self>) -> Option<#item_ty> {
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

fn extract_sequence_inner_type(ty: &Type) -> Option<(Type, SequenceKind)> {
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
                        GenericArgument::Type(inner) => {
                            Some((inner.clone(), SequenceKind::Vec))
                        }
                        _ => None,
                    }
                }
                _ => None,
            }
        }
        Type::Reference(reference) => match reference.elem.as_ref() {
            Type::Slice(slice) => Some(((*slice.elem).clone(), SequenceKind::SliceRef)),
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
        let (inner, kind) =
            extract_sequence_inner_type(&ty).expect("expected Vec inner type");
        assert_eq!(quote!(#inner).to_string(), "String");
        assert!(matches!(kind, SequenceKind::Vec));
    }

    #[test]
    fn test_extract_vec_inner_type_slice() {
        let ty: Type = syn::parse_quote!(&[u8]);
        let (inner, kind) =
            extract_sequence_inner_type(&ty).expect("expected slice inner type");
        assert_eq!(quote!(#inner).to_string(), "u8");
        assert!(matches!(kind, SequenceKind::SliceRef));
    }

    #[test]
    fn test_extract_vec_inner_type_none() {
        let ty: Type = syn::parse_quote!(HashMap<String, String>);
        assert!(extract_sequence_inner_type(&ty).is_none());
    }
}
