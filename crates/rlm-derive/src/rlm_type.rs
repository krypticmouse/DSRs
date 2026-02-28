use proc_macro2::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Expr, ExprLit, Fields, Lit, Meta, Type};

use crate::runtime_path::resolve_dspy_rs_path;

pub(crate) fn derive(input: DeriveInput) -> syn::Result<TokenStream> {
    validate_struct_surface(&input)?;
    ensure_pyclass(&input)?;

    let runtime = resolve_dspy_rs_path()?;
    let pyo3_crate = pyo3_crate_lit(&runtime);
    let options = parse_container_options(&input.attrs)?;
    let fields = parse_fields(&input)?;
    validate_iter_index_fields(&input.ident, &fields, &options)?;

    let struct_name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let helper_impl = generate_helper_impl(
        struct_name,
        &impl_generics,
        &ty_generics,
        where_clause,
        &runtime,
    );
    let getter_methods = fields
        .iter()
        .filter(|f| !f.skip_python)
        .map(generate_getter_method)
        .collect::<Vec<_>>();
    let repr_method = (!options.skip_repr).then(|| generate_repr_method(&runtime));
    let baml_method = generate_baml_method(&runtime);
    let mut extra_methods = Vec::new();
    if let Some(iter_field) = options.iter_field {
        extra_methods.push(generate_len_method(&runtime, &iter_field));
        extra_methods.push(generate_iter_method(&runtime, &iter_field));
    }
    if let Some(index_field) = options.index_field {
        extra_methods.push(generate_getitem_method(&runtime, &index_field));
    }

    Ok(quote! {
        #helper_impl

        #[#runtime::__macro_support::pyo3::pymethods(crate = #pyo3_crate)]
        impl #impl_generics #struct_name #ty_generics #where_clause {
            #(#getter_methods)*
            #repr_method
            #(#extra_methods)*
            #baml_method
        }
    })
}

#[derive(Clone)]
struct FieldSpec {
    ident: syn::Ident,
    ty: Type,
    doc: String,
    skip_python: bool,
}

#[derive(Default)]
struct ContainerOptions {
    iter_field: Option<String>,
    index_field: Option<String>,
    skip_repr: bool,
}

fn validate_struct_surface(input: &DeriveInput) -> syn::Result<()> {
    if !input.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &input.ident,
            "RlmType currently supports named structs only (no generics or lifetimes)",
        ));
    }

    match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(_) => Ok(()),
            _ => Err(syn::Error::new_spanned(
                &input.ident,
                "RlmType currently supports named structs only",
            )),
        },
        _ => Err(syn::Error::new_spanned(
            &input.ident,
            "RlmType currently supports named structs only",
        )),
    }
}

fn ensure_pyclass(input: &DeriveInput) -> syn::Result<()> {
    if has_pyclass_attr(&input.attrs) {
        Ok(())
    } else {
        Err(syn::Error::new_spanned(
            &input.ident,
            "RlmType requires a #[pyclass] attribute; use #[rlm_type] instead of deriving RlmType directly",
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

fn parse_container_options(attrs: &[syn::Attribute]) -> syn::Result<ContainerOptions> {
    let mut out = ContainerOptions::default();

    for attr in attrs {
        if !attr.path().is_ident("rlm") {
            continue;
        }

        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("iter") {
                let lit: syn::LitStr = meta.value()?.parse()?;
                out.iter_field = Some(lit.value());
                return Ok(());
            }
            if meta.path.is_ident("index") {
                let lit: syn::LitStr = meta.value()?.parse()?;
                out.index_field = Some(lit.value());
                return Ok(());
            }
            if meta.path.is_ident("skip_repr") {
                out.skip_repr = true;
                return Ok(());
            }
            Err(meta.error(
                "unsupported #[rlm(...)] key for structs; supported keys in V1 are `iter`, `index`, and `skip_repr`",
            ))
        })?;
    }

    Ok(out)
}

fn parse_fields(input: &DeriveInput) -> syn::Result<Vec<FieldSpec>> {
    let Data::Struct(data) = &input.data else {
        return Err(syn::Error::new_spanned(
            &input.ident,
            "RlmType currently supports named structs only",
        ));
    };
    let Fields::Named(named) = &data.fields else {
        return Err(syn::Error::new_spanned(
            &input.ident,
            "RlmType currently supports named structs only",
        ));
    };

    let mut out = Vec::with_capacity(named.named.len());
    for field in &named.named {
        let ident = field
            .ident
            .clone()
            .ok_or_else(|| syn::Error::new_spanned(field, "RlmType fields must be named"))?;
        let mut doc = doc_from_attrs(&field.attrs);
        let mut skip_python = false;

        for attr in &field.attrs {
            if !attr.path().is_ident("rlm") {
                continue;
            }
            attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("skip_python") {
                    skip_python = true;
                    return Ok(());
                }
                if meta.path.is_ident("desc") {
                    let lit: syn::LitStr = meta.value()?.parse()?;
                    doc = lit.value();
                    return Ok(());
                }
                Err(meta.error(
                    "unsupported #[rlm(...)] key for fields; supported keys in V1 are `skip_python` and `desc`",
                ))
            })?;
        }

        out.push(FieldSpec {
            ident,
            ty: field.ty.clone(),
            doc,
            skip_python,
        });
    }

    Ok(out)
}

fn validate_iter_index_fields(
    struct_name: &syn::Ident,
    fields: &[FieldSpec],
    options: &ContainerOptions,
) -> syn::Result<()> {
    let has_field = |name: &str| fields.iter().any(|f| f.ident == name);

    if let Some(iter) = &options.iter_field
        && !has_field(iter)
    {
        return Err(syn::Error::new_spanned(
            struct_name,
            format!("iter field `{iter}` not found on `{struct_name}`"),
        ));
    }
    if let Some(index) = &options.index_field
        && !has_field(index)
    {
        return Err(syn::Error::new_spanned(
            struct_name,
            format!("index field `{index}` not found on `{struct_name}`"),
        ));
    }
    Ok(())
}

fn doc_from_attrs(attrs: &[syn::Attribute]) -> String {
    let mut docs = Vec::new();
    for attr in attrs {
        if !attr.path().is_ident("doc") {
            continue;
        }
        if let Meta::NameValue(meta) = &attr.meta
            && let Expr::Lit(ExprLit {
                lit: Lit::Str(lit), ..
            }) = &meta.value
        {
            docs.push(lit.value().trim().to_string());
        }
    }
    docs.join("\n")
}

fn generate_getter_method(field: &FieldSpec) -> TokenStream {
    let name = &field.ident;
    let ty = &field.ty;
    let doc = if field.doc.trim().is_empty() {
        format!("Get `{name}`.")
    } else {
        field.doc.trim().to_string()
    };

    let (ret_ty, body) = getter_strategy(ty, name);
    quote! {
        #[doc = #doc]
        #[getter]
        fn #name(&self) -> #ret_ty {
            #body
        }
    }
}

fn getter_strategy(ty: &Type, field_name: &syn::Ident) -> (TokenStream, TokenStream) {
    if is_string_type(ty) {
        (quote! { &str }, quote! { self.#field_name.as_str() })
    } else if is_copy_primitive(ty) {
        (quote! { #ty }, quote! { self.#field_name })
    } else {
        (quote! { #ty }, quote! { self.#field_name.clone() })
    }
}

fn is_string_type(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty
        && let Some(segment) = type_path.path.segments.last()
    {
        return segment.ident == "String";
    }
    false
}

fn is_copy_primitive(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty
        && let Some(segment) = type_path.path.segments.last()
    {
        return matches!(
            segment.ident.to_string().as_str(),
            "bool"
                | "i8"
                | "i16"
                | "i32"
                | "i64"
                | "i128"
                | "isize"
                | "u8"
                | "u16"
                | "u32"
                | "u64"
                | "u128"
                | "usize"
                | "f32"
                | "f64"
                | "char"
        );
    }
    false
}

fn generate_helper_impl(
    struct_name: &syn::Ident,
    impl_generics: &syn::ImplGenerics<'_>,
    ty_generics: &syn::TypeGenerics<'_>,
    where_clause: Option<&syn::WhereClause>,
    runtime: &syn::Path,
) -> TokenStream {
    quote! {
        impl #impl_generics #struct_name #ty_generics #where_clause {
            fn __rlm_truncate_chars(input: &str, max_chars: usize) -> String {
                let mut out = String::new();
                let mut iter = input.chars();
                for _ in 0..max_chars {
                    if let Some(ch) = iter.next() {
                        out.push(ch);
                    } else {
                        return out;
                    }
                }
                if iter.next().is_some() {
                    out.push_str("...");
                }
                out
            }

            fn __rlm_render_repr_value(value: &#runtime::BamlValue, depth: usize) -> String {
                const MAX_ITEMS: usize = 3;
                const MAX_STRING_CHARS: usize = 200;
                match value {
                    #runtime::BamlValue::String(s) => {
                        format!("\"{}\"", Self::__rlm_truncate_chars(s, MAX_STRING_CHARS))
                    }
                    #runtime::BamlValue::Int(v) => v.to_string(),
                    #runtime::BamlValue::Float(v) => v.to_string(),
                    #runtime::BamlValue::Bool(v) => v.to_string(),
                    #runtime::BamlValue::Null => "None".to_string(),
                    #runtime::BamlValue::Enum(name, variant) => format!("{name}.{variant}"),
                    #runtime::BamlValue::Media(_) => "<media>".to_string(),
                    #runtime::BamlValue::List(items) => {
                        let mut rendered = items
                            .iter()
                            .take(MAX_ITEMS)
                            .map(|item| Self::__rlm_render_repr_value(item, depth + 1))
                            .collect::<Vec<_>>();
                        if items.len() > MAX_ITEMS {
                            rendered.push(format!("... ({} total)", items.len()));
                        }
                        format!("[{}]", rendered.join(", "))
                    }
                    #runtime::BamlValue::Map(fields) => {
                        let mut rendered = fields
                            .iter()
                            .take(MAX_ITEMS)
                            .map(|(k, v)| format!("{k}={}", Self::__rlm_render_repr_value(v, depth + 1)))
                            .collect::<Vec<_>>();
                        if fields.len() > MAX_ITEMS {
                            rendered.push(format!("... ({} total)", fields.len()));
                        }
                        if depth > 0 {
                            format!("Map({})", rendered.join(", "))
                        } else {
                            format!("{{{}}}", rendered.join(", "))
                        }
                    }
                    #runtime::BamlValue::Class(name, fields) => {
                        let mut rendered = fields
                            .iter()
                            .take(MAX_ITEMS)
                            .map(|(k, v)| format!("{k}={}", Self::__rlm_render_repr_value(v, depth + 1)))
                            .collect::<Vec<_>>();
                        if fields.len() > MAX_ITEMS {
                            rendered.push(format!("... ({} total)", fields.len()));
                        }
                        if depth > 0 {
                            format!("{name}({})", rendered.join(", "))
                        } else {
                            format!("{}({})", stringify!(#struct_name), rendered.join(", "))
                        }
                    }
                }
            }

            fn __rlm_baml_to_py(
                py: #runtime::__macro_support::pyo3::Python<'_>,
                value: &#runtime::BamlValue,
            ) -> #runtime::__macro_support::pyo3::PyResult<#runtime::__macro_support::pyo3::Py<#runtime::__macro_support::pyo3::PyAny>> {
                use #runtime::__macro_support::pyo3::IntoPyObjectExt;
                use #runtime::__macro_support::pyo3::types::{PyAnyMethods, PyDictMethods, PyListMethods};

                match value {
                    #runtime::BamlValue::String(v) => v.clone().into_py_any(py),
                    #runtime::BamlValue::Int(v) => v.into_py_any(py),
                    #runtime::BamlValue::Float(v) => v.into_py_any(py),
                    #runtime::BamlValue::Bool(v) => v.into_py_any(py),
                    #runtime::BamlValue::Null => Ok(py.None()),
                    #runtime::BamlValue::Enum(_, variant) => variant.clone().into_py_any(py),
                    #runtime::BamlValue::List(items) => {
                        let list = #runtime::__macro_support::pyo3::types::PyList::empty(py);
                        for item in items {
                            list.append(Self::__rlm_baml_to_py(py, item)?)?;
                        }
                        Ok(list.into_any().unbind())
                    }
                    #runtime::BamlValue::Map(map) | #runtime::BamlValue::Class(_, map) => {
                        let dict = #runtime::__macro_support::pyo3::types::PyDict::new(py);
                        for (key, item) in map.iter() {
                            dict.set_item(key, Self::__rlm_baml_to_py(py, item)?)?;
                        }
                        Ok(dict.into_any().unbind())
                    }
                    #runtime::BamlValue::Media(_) => Err(#runtime::__macro_support::pyo3::exceptions::PyTypeError::new_err(
                        "Media values are not supported in rlm_type::__baml__",
                    )),
                }
            }

            fn __rlm_list_field<'a>(
                value: &'a #runtime::BamlValue,
                field_name: &str,
            ) -> #runtime::__macro_support::pyo3::PyResult<&'a Vec<#runtime::BamlValue>> {
                let fields = match value {
                    #runtime::BamlValue::Class(_, fields) | #runtime::BamlValue::Map(fields) => fields,
                    _ => {
                        return Err(#runtime::__macro_support::pyo3::exceptions::PyTypeError::new_err(
                            "expected object-like value for list access",
                        ))
                    }
                };
                let Some(field_value) = fields.get(field_name) else {
                    return Err(#runtime::__macro_support::pyo3::exceptions::PyKeyError::new_err(format!(
                        "missing field `{field_name}`"
                    )));
                };
                match field_value {
                    #runtime::BamlValue::List(items) => Ok(items),
                    _ => Err(#runtime::__macro_support::pyo3::exceptions::PyTypeError::new_err(format!(
                        "field `{field_name}` is not a list"
                    ))),
                }
            }
        }
    }
}

fn generate_repr_method(runtime: &syn::Path) -> TokenStream {
    quote! {
        #[doc = "Compact model-safe representation of this object."]
        fn __repr__(&self) -> String {
            const MAX_TOTAL_CHARS: usize = 500;
            let value = <Self as #runtime::BamlType>::to_baml_value(self);
            let raw = Self::__rlm_render_repr_value(&value, 0);
            Self::__rlm_truncate_chars(&raw, MAX_TOTAL_CHARS)
        }
    }
}

fn generate_baml_method(runtime: &syn::Path) -> TokenStream {
    quote! {
        #[doc = "Convert this object to a dict/list/scalar representation for serialization or delegation."]
        #[pyo3(text_signature = "() -> dict")]
        fn __baml__(
            &self,
            py: #runtime::__macro_support::pyo3::Python<'_>,
        ) -> #runtime::__macro_support::pyo3::PyResult<#runtime::__macro_support::pyo3::Py<#runtime::__macro_support::pyo3::PyAny>> {
            let value = <Self as #runtime::BamlType>::to_baml_value(self);
            Self::__rlm_baml_to_py(py, &value)
        }
    }
}

fn generate_len_method(runtime: &syn::Path, field_name: &str) -> TokenStream {
    quote! {
        #[doc = "Return the number of elements in the configured collection field."]
        fn __len__(&self) -> #runtime::__macro_support::pyo3::PyResult<usize> {
            let value = <Self as #runtime::BamlType>::to_baml_value(self);
            Ok(Self::__rlm_list_field(&value, #field_name)?.len())
        }
    }
}

fn generate_iter_method(runtime: &syn::Path, field_name: &str) -> TokenStream {
    quote! {
        #[doc = "Iterate over elements in the configured collection field."]
        fn __iter__(
            &self,
            py: #runtime::__macro_support::pyo3::Python<'_>,
        ) -> #runtime::__macro_support::pyo3::PyResult<#runtime::__macro_support::pyo3::Py<#runtime::__macro_support::pyo3::PyAny>> {
            use #runtime::__macro_support::pyo3::types::{PyAnyMethods, PyListMethods};

            let value = <Self as #runtime::BamlType>::to_baml_value(self);
            let items = Self::__rlm_list_field(&value, #field_name)?;
            let list = #runtime::__macro_support::pyo3::types::PyList::empty(py);
            for item in items {
                list.append(Self::__rlm_baml_to_py(py, item)?)?;
            }
            let iter = list.into_any().call_method0("__iter__")?;
            Ok(iter.unbind())
        }
    }
}

fn generate_getitem_method(runtime: &syn::Path, field_name: &str) -> TokenStream {
    quote! {
        #[doc = "Index into the configured collection field. Supports negative indexing."]
        fn __getitem__(
            &self,
            py: #runtime::__macro_support::pyo3::Python<'_>,
            index: isize,
        ) -> #runtime::__macro_support::pyo3::PyResult<#runtime::__macro_support::pyo3::Py<#runtime::__macro_support::pyo3::PyAny>> {
            let value = <Self as #runtime::BamlType>::to_baml_value(self);
            let items = Self::__rlm_list_field(&value, #field_name)?;

            let len = items.len() as isize;
            let normalized = if index < 0 { len + index } else { index };
            if normalized < 0 || normalized >= len {
                return Err(#runtime::__macro_support::pyo3::exceptions::PyIndexError::new_err("index out of range"));
            }

            Self::__rlm_baml_to_py(py, &items[normalized as usize])
        }
    }
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
        proc_macro2::Span::call_site(),
    )
}
