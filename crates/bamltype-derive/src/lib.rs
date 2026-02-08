use convert_case::{Case, Casing};
use proc_macro::TokenStream;
use proc_macro_crate::{FoundCrate, crate_name};
use proc_macro2::Span;
use quote::{format_ident, quote};
use syn::spanned::Spanned;
use syn::{
    Attribute, Data, DeriveInput, Expr, ExprLit, Field, Fields, Lit, Meta, Path, Type,
    parse_macro_input,
};

#[derive(Default)]
struct ContainerCompatAttrs {
    rename: Option<String>,
    rename_all: Option<RenameRule>,
    tag: Option<String>,
    description: Option<String>,
    internal_name: Option<String>,
    constraints: Vec<ConstraintCompatAttr>,
    as_union: bool,
    as_enum: bool,
}

#[derive(Default)]
struct FieldCompatAttrs {
    rename: Option<String>,
    preserve_original_name: bool,
    skip: bool,
    default: bool,
    with_adapter: Option<Path>,
    description: Option<String>,
    int_repr: Option<String>,
    map_key_repr: Option<String>,
    constraints: Vec<ConstraintCompatAttr>,
}

#[derive(Default)]
struct VariantCompatAttrs {
    rename: Option<String>,
    description: Option<String>,
}

#[derive(Clone)]
struct ConstraintCompatAttr {
    kind: ConstraintKind,
    label: String,
    expr: String,
}

#[derive(Clone, Copy)]
enum ConstraintKind {
    Check,
    Assert,
}

#[derive(Clone, Copy)]
enum RenameRule {
    Camel,
    Snake,
    Pascal,
    Kebab,
    ScreamingSnake,
    Lower,
    Upper,
    ScreamingKebab,
}

impl RenameRule {
    fn facet_value(self) -> Option<&'static str> {
        match self {
            RenameRule::Camel => Some("camelCase"),
            RenameRule::Snake => Some("snake_case"),
            RenameRule::Pascal => Some("PascalCase"),
            RenameRule::Kebab => Some("kebab-case"),
            RenameRule::ScreamingSnake => Some("SCREAMING_SNAKE_CASE"),
            RenameRule::ScreamingKebab => Some("SCREAMING-KEBAB-CASE"),
            RenameRule::Lower | RenameRule::Upper => None,
        }
    }

    fn apply(self, name: &str) -> String {
        let case = match self {
            RenameRule::Camel => Case::Camel,
            RenameRule::Snake => Case::Snake,
            RenameRule::Pascal => Case::Pascal,
            RenameRule::Kebab => Case::Kebab,
            RenameRule::ScreamingSnake => Case::UpperSnake,
            RenameRule::Lower => Case::Lower,
            RenameRule::Upper => Case::Upper,
            RenameRule::ScreamingKebab => Case::UpperKebab,
        };
        name.to_case(case)
    }
}

/// Attribute macro that makes a struct or enum usable with BAML.
///
/// This macro normalizes `#[baml(...)]` (and relevant `#[serde(...)]`) attributes
/// onto the facet model, then derives `facet::Facet` and implements `BamlSchema`.
#[allow(non_snake_case)]
#[proc_macro_attribute]
pub fn BamlType(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut input = parse_macro_input!(item as DeriveInput);
    let name = input.ident.clone();

    let runtime_crate = match resolve_runtime_crate() {
        Ok(path) => path,
        Err(err) => return TokenStream::from(err.into_compile_error()),
    };

    let runtime_ns_ident = format_ident!("__bamltype_runtime_{}", name);
    let runtime_ns_path: Path = syn::parse_quote!(#runtime_ns_ident);

    if let Err(err) = validate_input(&input) {
        return TokenStream::from(err.into_compile_error());
    }

    if let Err(err) = normalize_attrs(&mut input, &runtime_crate, &runtime_ns_path) {
        return TokenStream::from(err.into_compile_error());
    }

    let is_enum = matches!(input.data, Data::Enum(_));
    let has_repr = input.attrs.iter().any(|attr| attr.path().is_ident("repr"));

    // Enums need an explicit repr for facet — add #[repr(u8)] if missing.
    if is_enum && !has_repr {
        input.attrs.insert(0, syn::parse_quote!(#[repr(u8)]));
    }

    // Put derive first so helper attrs (`#[facet(...)]`) are recognized without
    // tripping legacy helper ordering lints in downstream crates.
    let mut reordered_attrs = Vec::with_capacity(input.attrs.len() + 2);
    reordered_attrs.push(syn::parse_quote!(#[derive(#runtime_crate::facet::Facet)]));
    reordered_attrs.push(syn::parse_quote!(#[facet(crate = #runtime_crate::facet)]));
    reordered_attrs.extend(std::mem::take(&mut input.attrs));
    input.attrs = reordered_attrs;

    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let runtime_ns_import = quote! {
        #[allow(unused_imports)]
        use #runtime_crate as #runtime_ns_ident;
    };

    let expanded = quote! {
        #runtime_ns_import
        #input

        impl #impl_generics #runtime_crate::BamlSchema for #name #ty_generics #where_clause {
            fn baml_schema() -> &'static #runtime_crate::SchemaBundle {
                static SCHEMA: ::std::sync::OnceLock<#runtime_crate::SchemaBundle> =
                    ::std::sync::OnceLock::new();
                SCHEMA.get_or_init(|| {
                    #runtime_crate::SchemaBundle::from_shape(
                        <Self as #runtime_crate::facet::Facet<'_>>::SHAPE
                    )
                })
            }
        }
    };

    TokenStream::from(expanded)
}

fn resolve_runtime_crate() -> syn::Result<Path> {
    if let Some(path) = find_crate_path("bamltype") {
        return Ok(path);
    }

    if let Some(dspy_path) = find_crate_path("dspy-rs") {
        return Ok(syn::parse_quote!(#dspy_path::bamltype));
    }

    Err(syn::Error::new(
        Span::call_site(),
        "could not resolve bamltype runtime crate; expected dependency on `bamltype` or `dspy-rs`",
    ))
}

fn find_crate_path(package_name: &str) -> Option<Path> {
    match crate_name(package_name).ok()? {
        FoundCrate::Itself => Some(syn::parse_quote!(crate)),
        FoundCrate::Name(name) => {
            let ident = syn::Ident::new(&name.replace('-', "_"), Span::call_site());
            Some(syn::parse_quote!(::#ident))
        }
    }
}

fn normalize_attrs(
    input: &mut DeriveInput,
    runtime_crate: &Path,
    runtime_ns: &Path,
) -> syn::Result<()> {
    let keep_serde_attrs = has_serde_derive(&input.attrs)?;
    let manual_rename_all =
        normalize_container_attrs(&mut input.attrs, keep_serde_attrs, runtime_ns)?;

    match &mut input.data {
        Data::Struct(data) => normalize_fields(
            &mut data.fields,
            keep_serde_attrs,
            manual_rename_all,
            runtime_crate,
            runtime_ns,
        )?,
        Data::Enum(data) => {
            for variant in &mut data.variants {
                let variant_name = variant.ident.to_string();
                normalize_variant_attrs(
                    &mut variant.attrs,
                    &variant_name,
                    keep_serde_attrs,
                    manual_rename_all,
                )?;
                normalize_fields_with_name_strategy(
                    &mut variant.fields,
                    |field, index| field_name_for_alias(field, index, &variant_name),
                    keep_serde_attrs,
                    manual_rename_all,
                    runtime_crate,
                    runtime_ns,
                )?;
            }
        }
        Data::Union(union) => {
            return Err(syn::Error::new(
                union.union_token.span(),
                "BamlType does not support `union` items; hint: use a struct or enum instead",
            ));
        }
    }

    Ok(())
}

fn normalize_fields(
    fields: &mut Fields,
    keep_serde_attrs: bool,
    rename_all: Option<RenameRule>,
    runtime_crate: &Path,
    runtime_ns: &Path,
) -> syn::Result<()> {
    normalize_fields_with_name_strategy(
        fields,
        |field, index| field_name_for_alias(field, index, "field"),
        keep_serde_attrs,
        rename_all,
        runtime_crate,
        runtime_ns,
    )
}

fn normalize_fields_with_name_strategy<F>(
    fields: &mut Fields,
    mut name_of: F,
    keep_serde_attrs: bool,
    rename_all: Option<RenameRule>,
    runtime_crate: &Path,
    runtime_ns: &Path,
) -> syn::Result<()>
where
    F: FnMut(&Field, usize) -> String,
{
    match fields {
        Fields::Named(named) => {
            for (index, field) in named.named.iter_mut().enumerate() {
                let original_name = name_of(field, index);
                normalize_field_attrs(
                    &mut field.attrs,
                    &field.ty,
                    &original_name,
                    keep_serde_attrs,
                    rename_all,
                    runtime_crate,
                    runtime_ns,
                )?;
            }
        }
        Fields::Unit => {}
        Fields::Unnamed(_) => {
            return Err(syn::Error::new(
                Span::call_site(),
                "tuple fields are not supported for BamlType; use named fields",
            ));
        }
    }

    Ok(())
}

fn field_name_for_alias(field: &Field, index: usize, fallback_prefix: &str) -> String {
    field
        .ident
        .as_ref()
        .map(std::string::ToString::to_string)
        .unwrap_or_else(|| format!("{fallback_prefix}_{index}"))
}

fn validate_input(input: &DeriveInput) -> syn::Result<()> {
    let mut container = ContainerCompatAttrs::default();
    for attr in &input.attrs {
        if attr.path().is_ident("baml") {
            parse_baml_container_meta(attr, &mut container)?;
        }
        if attr.path().is_ident("serde") {
            parse_serde_container_meta(attr, &mut container)?;
        }
    }

    match &input.data {
        Data::Struct(data) => validate_struct(input, data),
        Data::Enum(data) => validate_enum(input, data, &container),
        Data::Union(union) => Err(syn::Error::new(
            union.union_token.span(),
            "BamlType does not support `union` items; hint: use a struct or enum instead",
        )),
    }
}

fn validate_struct(input: &DeriveInput, data: &syn::DataStruct) -> syn::Result<()> {
    match &data.fields {
        Fields::Named(fields) => {
            for field in &fields.named {
                validate_field(field)?;
            }
            Ok(())
        }
        Fields::Unit => Err(syn::Error::new_spanned(
            input,
            "Unit structs are not supported for BAML outputs; hint: use a named-field struct or enum",
        )),
        Fields::Unnamed(_) => Err(syn::Error::new_spanned(
            input,
            "Tuple structs are not supported for BAML outputs; hint: use a named-field struct",
        )),
    }
}

fn validate_enum(
    input: &DeriveInput,
    data: &syn::DataEnum,
    container: &ContainerCompatAttrs,
) -> syn::Result<()> {
    let mut has_data_variant = false;

    for variant in &data.variants {
        for attr in &variant.attrs {
            if attr.path().is_ident("baml") {
                let mut out = VariantCompatAttrs::default();
                parse_baml_variant_meta(attr, &mut out)?;
            }
            if attr.path().is_ident("serde") {
                let mut out = VariantCompatAttrs::default();
                parse_serde_variant_meta(attr, &mut out)?;
            }
        }

        match &variant.fields {
            Fields::Unit => {}
            Fields::Unnamed(_) => {
                return Err(syn::Error::new_spanned(
                    variant,
                    "Tuple enum variants are not supported; hint: use a unit or struct-like variant",
                ));
            }
            Fields::Named(fields) => {
                if !fields.named.is_empty() {
                    has_data_variant = true;
                }
                for field in &fields.named {
                    validate_field(field)?;
                }
            }
        }
    }

    if container.as_enum && has_data_variant {
        return Err(syn::Error::new_spanned(
            input,
            "as_enum is only valid for unit enums; hint: remove #[baml(as_enum)] or convert variants to unit",
        ));
    }

    Ok(())
}

fn validate_field(field: &Field) -> syn::Result<()> {
    let mut field_attrs = FieldCompatAttrs::default();
    for attr in &field.attrs {
        if attr.path().is_ident("baml") {
            parse_baml_field_meta(attr, &mut field_attrs)?;
        }
        if attr.path().is_ident("serde") {
            parse_serde_field_meta(attr, &mut field_attrs)?;
        }
    }

    if field_attrs.with_adapter.is_some() {
        return Ok(());
    }

    if let Some(ty) = find_type_match(&field.ty, &|ty| matches!(ty, Type::BareFn(_))) {
        return Err(syn::Error::new_spanned(
            ty,
            "function types are not supported in BAML outputs; hint: remove the field or use #[baml(with = \"...\")] to adapt it",
        ));
    }

    if let Some(ty) = find_type_match(&field.ty, &|ty| matches!(ty, Type::Tuple(_))) {
        return Err(syn::Error::new_spanned(
            ty,
            "tuple types are not supported in BAML outputs; hint: use a struct with named fields or a list",
        ));
    }

    if let Some(ty) = find_type_match(&field.ty, &|ty| matches!(ty, Type::TraitObject(_))) {
        return Err(syn::Error::new_spanned(
            ty,
            "trait objects are not supported in BAML outputs; hint: use a concrete type or a custom adapter",
        ));
    }

    if let Some(ty) = find_type_match(&field.ty, &is_serde_json_value) {
        return Err(syn::Error::new_spanned(
            ty,
            "serde_json::Value is not supported without a #[baml(with = \"...\")] adapter; hint: use a concrete type or provide a custom adapter",
        ));
    }

    if field_attrs.map_key_repr.is_some() && map_types_for_repr(&field.ty).is_none() {
        let span_ty = map_key_repr_error_span_type(&field.ty);
        return Err(syn::Error::new_spanned(
            span_ty,
            "map_key_repr only applies to map fields (HashMap/BTreeMap), optionally wrapped in Option/Vec/Box/Arc/Rc; hint: remove the attribute or change the field type",
        ));
    }

    if field_attrs.map_key_repr.is_none()
        && let Some((key, _)) = map_types_for_repr(&field.ty)
        && !is_string_type(key)
    {
        return Err(syn::Error::new_spanned(
            &field.ty,
            "map keys must be String for object maps; hint: use HashMap<String, V> or add #[baml(map_key_repr = \"string\"|\"pairs\")], or use a custom adapter",
        ));
    }

    if field_attrs.int_repr.is_none()
        && let Some(ty) = find_type_match(&field.ty, &is_large_int_type)
    {
        return Err(syn::Error::new_spanned(
            ty,
            "unsupported integer width for BAML outputs; hint: use #[baml(int_repr = \"string\"|\"i64\")] or a smaller integer type",
        ));
    }

    Ok(())
}

fn find_type_match<'a, F>(ty: &'a Type, predicate: &F) -> Option<&'a Type>
where
    F: Fn(&Type) -> bool,
{
    if predicate(ty) {
        return Some(ty);
    }

    match ty {
        Type::Array(array) => find_type_match(&array.elem, predicate),
        Type::Group(group) => find_type_match(&group.elem, predicate),
        Type::Paren(paren) => find_type_match(&paren.elem, predicate),
        Type::Ptr(ptr) => find_type_match(&ptr.elem, predicate),
        Type::Reference(reference) => find_type_match(&reference.elem, predicate),
        Type::Slice(slice) => find_type_match(&slice.elem, predicate),
        Type::Tuple(tuple) => {
            for elem in &tuple.elems {
                if let Some(found) = find_type_match(elem, predicate) {
                    return Some(found);
                }
            }
            None
        }
        Type::Path(path) => {
            for segment in &path.path.segments {
                if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                    for arg in &args.args {
                        if let syn::GenericArgument::Type(inner) = arg
                            && let Some(found) = find_type_match(inner, predicate)
                        {
                            return Some(found);
                        }
                    }
                }
            }
            None
        }
        _ => None,
    }
}

fn type_ident(ty: &Type) -> Option<&syn::Ident> {
    match ty {
        Type::Path(path) if path.qself.is_none() => path.path.segments.last().map(|s| &s.ident),
        _ => None,
    }
}

fn unwrap_repr_wrapper(ty: &Type) -> Option<&Type> {
    extract_single_arg(ty, "Option")
        .or_else(|| extract_single_arg(ty, "Vec"))
        .or_else(|| extract_single_arg(ty, "Box"))
        .or_else(|| extract_single_arg(ty, "Arc"))
        .or_else(|| extract_single_arg(ty, "Rc"))
}

fn extract_single_arg<'a>(ty: &'a Type, ident: &str) -> Option<&'a Type> {
    if let Type::Path(path) = ty
        && let Some(segment) = path.path.segments.last()
        && segment.ident == ident
        && let syn::PathArguments::AngleBracketed(args) = &segment.arguments
        && let Some(syn::GenericArgument::Type(inner)) = args.args.first()
    {
        return Some(inner);
    }
    None
}

fn map_types(ty: &Type) -> Option<(&Type, &Type)> {
    if let Type::Path(path) = ty
        && let Some(segment) = path.path.segments.last()
        && (segment.ident == "HashMap" || segment.ident == "BTreeMap")
        && let syn::PathArguments::AngleBracketed(args) = &segment.arguments
    {
        let mut iter = args.args.iter();
        let key = match iter.next() {
            Some(syn::GenericArgument::Type(t)) => t,
            _ => return None,
        };
        let value = match iter.next() {
            Some(syn::GenericArgument::Type(t)) => t,
            _ => return None,
        };
        return Some((key, value));
    }
    None
}

fn map_types_for_repr(ty: &Type) -> Option<(&Type, &Type)> {
    let mut current = ty;
    loop {
        if let Some((key, value)) = map_types(current) {
            return Some((key, value));
        }

        let inner = unwrap_repr_wrapper(current)?;
        current = inner;
    }
}

fn map_key_repr_error_span_type(ty: &Type) -> &Type {
    let mut current = ty;
    while let Some(inner) = unwrap_repr_wrapper(current) {
        current = inner;
    }
    current
}

fn is_string_type(ty: &Type) -> bool {
    type_ident(ty)
        .map(|ident| ident == "String")
        .unwrap_or(false)
}

fn is_large_int_type(ty: &Type) -> bool {
    match type_ident(ty).map(|ident| ident.to_string()) {
        Some(name) => matches!(name.as_str(), "u64" | "usize" | "i128" | "u128"),
        None => false,
    }
}

fn is_serde_json_value(ty: &Type) -> bool {
    if let Type::Path(path) = ty
        && let Some(segment) = path.path.segments.last()
        && segment.ident == "Value"
    {
        return path
            .path
            .segments
            .iter()
            .any(|seg| seg.ident == "serde_json");
    }
    false
}

fn has_serde_derive(attrs: &[Attribute]) -> syn::Result<bool> {
    for attr in attrs {
        if !attr.path().is_ident("derive") {
            continue;
        }

        let derives = attr.parse_args_with(
            syn::punctuated::Punctuated::<Path, syn::Token![,]>::parse_terminated,
        )?;

        for derive in derives {
            if derive.is_ident("Serialize") || derive.is_ident("Deserialize") {
                return Ok(true);
            }
        }
    }

    Ok(false)
}

fn normalize_container_attrs(
    attrs: &mut Vec<Attribute>,
    keep_serde_attrs: bool,
    runtime_ns: &Path,
) -> syn::Result<Option<RenameRule>> {
    let mut out = Vec::with_capacity(attrs.len());
    let mut compat = ContainerCompatAttrs::default();

    for attr in std::mem::take(attrs) {
        if attr.path().is_ident("baml") {
            parse_baml_container_meta(&attr, &mut compat)?;
            continue;
        }

        if attr.path().is_ident("serde") {
            parse_serde_container_meta(&attr, &mut compat)?;
            if keep_serde_attrs {
                out.push(attr);
            }
            continue;
        }

        out.push(attr);
    }

    if let Some(rename) = compat.rename {
        let lit = syn::LitStr::new(&rename, Span::call_site());
        out.push(syn::parse_quote!(#[facet(rename = #lit)]));
    }
    let mut manual_rename_all = None;
    if let Some(rename_all) = compat.rename_all {
        if let Some(facet_value) = rename_all.facet_value() {
            let lit = syn::LitStr::new(facet_value, Span::call_site());
            out.push(syn::parse_quote!(#[facet(rename_all = #lit)]));
        } else {
            manual_rename_all = Some(rename_all);
        }
    }
    if compat.as_union {
        out.push(syn::parse_quote!(#[facet(untagged)]));
    }
    if let Some(tag) = compat.tag {
        let lit = syn::LitStr::new(&tag, Span::call_site());
        out.push(syn::parse_quote!(#[facet(tag = #lit)]));
    }
    if let Some(internal_name) = compat.internal_name {
        let lit = syn::LitStr::new(&internal_name, Span::call_site());
        out.push(syn::parse_quote!(#[facet(#runtime_ns::internal_name = #lit)]));
    }
    push_constraint_attrs(&mut out, &compat.constraints, runtime_ns);
    if let Some(description) = compat.description {
        replace_doc_attrs(&mut out, &description);
    }

    *attrs = out;
    Ok(manual_rename_all)
}

fn normalize_field_attrs(
    attrs: &mut Vec<Attribute>,
    field_ty: &Type,
    original_name: &str,
    keep_serde_attrs: bool,
    rename_all: Option<RenameRule>,
    runtime_crate: &Path,
    runtime_ns: &Path,
) -> syn::Result<()> {
    let mut out = Vec::with_capacity(attrs.len());
    let mut compat = FieldCompatAttrs::default();

    for attr in std::mem::take(attrs) {
        if attr.path().is_ident("baml") {
            parse_baml_field_meta(&attr, &mut compat)?;
            continue;
        }

        if attr.path().is_ident("serde") {
            parse_serde_field_meta(&attr, &mut compat)?;
            if keep_serde_attrs {
                out.push(attr);
            }
            continue;
        }

        out.push(attr);
    }

    apply_rename_all(&mut compat.rename, rename_all, original_name);

    if let Some(rename) = compat.rename {
        let lit = syn::LitStr::new(&rename, Span::call_site());
        out.push(syn::parse_quote!(#[facet(rename = #lit)]));

        // Old BAML alias behavior accepted both the alias and the original field name.
        if compat.preserve_original_name && rename != original_name {
            let original = syn::LitStr::new(original_name, Span::call_site());
            out.push(syn::parse_quote!(#[facet(alias = #original)]));
        }
    }

    if compat.skip {
        out.push(syn::parse_quote!(#[facet(skip)]));
    }
    // Keep legacy bridge compatibility: skipped fields deserialize from Default.
    if compat.default || compat.skip {
        out.push(syn::parse_quote!(#[facet(default)]));
    }
    if let Some(adapter) = compat.with_adapter {
        let with_expr = quote! {
            &#runtime_crate::facet_ext::WithAdapterFns {
                type_ir: || <#adapter as #runtime_crate::adapters::FieldCodec<#field_ty>>::type_ir(),
                register: |reg| <#adapter as #runtime_crate::adapters::FieldCodec<#field_ty>>::register(reg),
                apply: |partial, value, path| {
                    let converted = <#adapter as #runtime_crate::adapters::FieldCodec<#field_ty>>::try_from_baml(value, path)?;
                    partial.set(converted).map_err(|err| {
                        #runtime_crate::BamlConvertError::new(
                            ::std::vec::Vec::new(),
                            "compatible type",
                            err.to_string(),
                            err.to_string(),
                        )
                    })
                },
            }
        };
        out.push(syn::parse_quote!(#[facet(#runtime_ns::with = #with_expr)]));
    }
    if let Some(int_repr) = compat.int_repr {
        let lit = syn::LitStr::new(&int_repr, Span::call_site());
        out.push(syn::parse_quote!(#[facet(#runtime_ns::int_repr = #lit)]));
    }
    if let Some(map_key_repr) = compat.map_key_repr {
        let lit = syn::LitStr::new(&map_key_repr, Span::call_site());
        out.push(syn::parse_quote!(#[facet(#runtime_ns::map_key_repr = #lit)]));
    }
    push_constraint_attrs(&mut out, &compat.constraints, runtime_ns);
    if let Some(description) = compat.description {
        replace_doc_attrs(&mut out, &description);
    }

    *attrs = out;
    Ok(())
}

fn normalize_variant_attrs(
    attrs: &mut Vec<Attribute>,
    variant_name: &str,
    keep_serde_attrs: bool,
    rename_all: Option<RenameRule>,
) -> syn::Result<()> {
    let mut out = Vec::with_capacity(attrs.len());
    let mut compat = VariantCompatAttrs::default();

    for attr in std::mem::take(attrs) {
        if attr.path().is_ident("baml") {
            parse_baml_variant_meta(&attr, &mut compat)?;
            continue;
        }

        if attr.path().is_ident("serde") {
            parse_serde_variant_meta(&attr, &mut compat)?;
            if keep_serde_attrs {
                out.push(attr);
            }
            continue;
        }

        out.push(attr);
    }

    apply_rename_all(&mut compat.rename, rename_all, variant_name);

    if let Some(rename) = compat.rename {
        let lit = syn::LitStr::new(&rename, Span::call_site());
        out.push(syn::parse_quote!(#[facet(rename = #lit)]));
    }
    if let Some(description) = compat.description {
        replace_doc_attrs(&mut out, &description);
    }

    *attrs = out;
    Ok(())
}

fn apply_rename_all(
    rename: &mut Option<String>,
    rename_all: Option<RenameRule>,
    original_name: &str,
) {
    if rename.is_none()
        && let Some(rule) = rename_all
    {
        *rename = Some(rule.apply(original_name));
    }
}

fn push_constraint_attrs(
    out: &mut Vec<Attribute>,
    constraints: &[ConstraintCompatAttr],
    runtime_ns: &Path,
) {
    for constraint in constraints {
        let label = syn::LitStr::new(&constraint.label, Span::call_site());
        let expr = syn::LitStr::new(&constraint.expr, Span::call_site());
        match constraint.kind {
            ConstraintKind::Check => {
                out.push(
                    syn::parse_quote!(#[facet(#runtime_ns::check(label = #label, expr = #expr))]),
                );
            }
            ConstraintKind::Assert => {
                out.push(
                    syn::parse_quote!(#[facet(#runtime_ns::assert(label = #label, expr = #expr))]),
                );
            }
        }
    }
}

const UNSUPPORTED_BAML_ATTR_HINT: &str =
    "unsupported #[baml(...)] attribute; hint: check the supported keys in the bridge docs";

fn parse_baml_container_meta(attr: &Attribute, out: &mut ContainerCompatAttrs) -> syn::Result<()> {
    for meta in parse_meta_list(attr)? {
        match meta {
            Meta::NameValue(meta) if meta.path.is_ident("name") => {
                out.rename = Some(parse_string_expr(&meta.value, meta.path.span())?);
            }
            Meta::NameValue(meta) if meta.path.is_ident("rename_all") => {
                out.rename_all = Some(parse_rename_rule(&meta.value, meta.span())?);
            }
            Meta::NameValue(meta) if meta.path.is_ident("tag") => {
                out.tag = Some(parse_string_expr(&meta.value, meta.span())?);
            }
            Meta::NameValue(meta) if meta.path.is_ident("description") => {
                out.description = parse_optional_string(&meta.value, meta.span())?;
            }

            // Accepted for source compatibility (runtime behavior parity is handled
            // in bamltype runtime code paths).
            Meta::NameValue(meta) if meta.path.is_ident("internal_name") => {
                out.internal_name = Some(parse_string_expr(&meta.value, meta.span())?);
            }
            Meta::List(meta) if meta.path.is_ident("check") => {
                out.constraints
                    .push(parse_constraint_meta(&meta, ConstraintKind::Check)?);
            }
            Meta::List(meta) if meta.path.is_ident("assert") => {
                out.constraints
                    .push(parse_constraint_meta(&meta, ConstraintKind::Assert)?);
            }
            Meta::Path(path) if path.is_ident("as_union") => {
                out.as_union = true;
            }
            Meta::Path(path) if path.is_ident("as_enum") => {
                out.as_enum = true;
            }

            _ => {
                return Err(syn::Error::new_spanned(meta, UNSUPPORTED_BAML_ATTR_HINT));
            }
        }
    }

    Ok(())
}

fn parse_baml_field_meta(attr: &Attribute, out: &mut FieldCompatAttrs) -> syn::Result<()> {
    for meta in parse_meta_list(attr)? {
        match meta {
            Meta::NameValue(meta) if meta.path.is_ident("alias") => {
                out.rename = Some(parse_string_expr(&meta.value, meta.span())?);
                out.preserve_original_name = true;
            }
            Meta::NameValue(meta) if meta.path.is_ident("description") => {
                out.description = parse_optional_string(&meta.value, meta.span())?;
            }
            Meta::Path(path) if path.is_ident("skip") => {
                out.skip = true;
            }
            Meta::Path(path) if path.is_ident("default") => {
                out.default = true;
            }

            Meta::NameValue(meta) if meta.path.is_ident("with") => {
                let path_str = parse_string_expr(&meta.value, meta.span())?;
                out.with_adapter = Some(syn::parse_str::<Path>(&path_str)?);
            }
            Meta::NameValue(meta) if meta.path.is_ident("int_repr") => {
                out.int_repr = Some(parse_int_repr(&meta.value, meta.span())?);
            }
            Meta::NameValue(meta) if meta.path.is_ident("map_key_repr") => {
                out.map_key_repr = Some(parse_map_key_repr(&meta.value, meta.span())?);
            }
            Meta::List(meta) if meta.path.is_ident("check") => {
                out.constraints
                    .push(parse_constraint_meta(&meta, ConstraintKind::Check)?);
            }
            Meta::List(meta) if meta.path.is_ident("assert") => {
                out.constraints
                    .push(parse_constraint_meta(&meta, ConstraintKind::Assert)?);
            }

            _ => {
                return Err(syn::Error::new_spanned(meta, UNSUPPORTED_BAML_ATTR_HINT));
            }
        }
    }

    Ok(())
}

fn parse_baml_variant_meta(attr: &Attribute, out: &mut VariantCompatAttrs) -> syn::Result<()> {
    for meta in parse_meta_list(attr)? {
        match meta {
            Meta::NameValue(meta) if meta.path.is_ident("alias") => {
                out.rename = Some(parse_string_expr(&meta.value, meta.span())?);
            }
            Meta::NameValue(meta) if meta.path.is_ident("description") => {
                out.description = parse_optional_string(&meta.value, meta.span())?;
            }
            _ => {
                return Err(syn::Error::new_spanned(meta, UNSUPPORTED_BAML_ATTR_HINT));
            }
        }
    }
    Ok(())
}

fn parse_serde_container_meta(attr: &Attribute, out: &mut ContainerCompatAttrs) -> syn::Result<()> {
    for meta in parse_meta_list(attr)? {
        match meta {
            Meta::NameValue(meta) if meta.path.is_ident("rename") => {
                if out.rename.is_none() {
                    out.rename = Some(parse_string_expr(&meta.value, meta.span())?);
                }
            }
            Meta::NameValue(meta) if meta.path.is_ident("rename_all") => {
                if out.rename_all.is_none() {
                    out.rename_all = Some(parse_rename_rule(&meta.value, meta.span())?);
                }
            }
            Meta::NameValue(meta) if meta.path.is_ident("tag") => {
                if out.tag.is_none() {
                    out.tag = Some(parse_string_expr(&meta.value, meta.span())?);
                }
            }
            Meta::Path(path) if path.is_ident("untagged") => {
                return Err(syn::Error::new_spanned(
                    path,
                    "serde(untagged) is not supported; hint: use #[baml(tag = \"...\")] for data enums",
                ));
            }
            Meta::Path(path) if path.is_ident("flatten") => {
                return Err(syn::Error::new_spanned(
                    path,
                    "serde(flatten) is not supported; hint: model fields explicitly",
                ));
            }
            _ => {}
        }
    }
    Ok(())
}

fn parse_serde_field_meta(attr: &Attribute, out: &mut FieldCompatAttrs) -> syn::Result<()> {
    for meta in parse_meta_list(attr)? {
        match meta {
            Meta::NameValue(meta) if meta.path.is_ident("rename") => {
                if out.rename.is_none() {
                    out.rename = Some(parse_string_expr(&meta.value, meta.span())?);
                    out.preserve_original_name = true;
                }
            }
            Meta::Path(path) if path.is_ident("skip") => {
                out.skip = true;
            }
            Meta::Path(path) if path.is_ident("default") => {
                out.default = true;
            }
            Meta::NameValue(meta) if meta.path.is_ident("default") => {
                return Err(syn::Error::new_spanned(
                    meta,
                    "serde(default = \"path\") is not supported; hint: use #[baml(default)] or Default::default",
                ));
            }
            Meta::Path(path) if path.is_ident("flatten") => {
                return Err(syn::Error::new_spanned(
                    path,
                    "serde(flatten) is not supported; hint: model fields explicitly",
                ));
            }
            _ => {}
        }
    }
    Ok(())
}

fn parse_serde_variant_meta(attr: &Attribute, out: &mut VariantCompatAttrs) -> syn::Result<()> {
    for meta in parse_meta_list(attr)? {
        match meta {
            Meta::NameValue(meta) if meta.path.is_ident("rename") => {
                if out.rename.is_none() {
                    out.rename = Some(parse_string_expr(&meta.value, meta.span())?);
                }
            }
            Meta::Path(path) if path.is_ident("skip") => {
                return Err(syn::Error::new_spanned(
                    path,
                    "serde(skip) is not supported on enum variants; hint: remove the variant or use a separate enum",
                ));
            }
            _ => {}
        }
    }
    Ok(())
}

fn parse_meta_list(attr: &Attribute) -> syn::Result<Vec<Meta>> {
    let metas = attr
        .parse_args_with(syn::punctuated::Punctuated::<Meta, syn::Token![,]>::parse_terminated)?;
    Ok(metas.into_iter().collect())
}

fn parse_string_expr(expr: &Expr, span: Span) -> syn::Result<String> {
    match expr {
        Expr::Lit(ExprLit {
            lit: Lit::Str(value),
            ..
        }) => Ok(value.value()),
        _ => Err(syn::Error::new(
            span,
            "expected string literal; hint: wrap the value in quotes",
        )),
    }
}

fn parse_optional_string(expr: &Expr, span: Span) -> syn::Result<Option<String>> {
    if let Expr::Path(path) = expr
        && path.path.is_ident("None")
    {
        return Ok(None);
    }
    Ok(Some(parse_string_expr(expr, span)?))
}

fn parse_rename_rule(expr: &Expr, span: Span) -> syn::Result<RenameRule> {
    let value = parse_string_expr(expr, span)?;
    match value.as_str() {
        "camelCase" => Ok(RenameRule::Camel),
        "snake_case" => Ok(RenameRule::Snake),
        "PascalCase" => Ok(RenameRule::Pascal),
        "kebab-case" => Ok(RenameRule::Kebab),
        "SCREAMING_SNAKE_CASE" => Ok(RenameRule::ScreamingSnake),
        "lowercase" => Ok(RenameRule::Lower),
        "UPPERCASE" => Ok(RenameRule::Upper),
        "SCREAMING-KEBAB-CASE" => Ok(RenameRule::ScreamingKebab),
        _ => Err(syn::Error::new(span, "unsupported rename_all value")),
    }
}

fn parse_int_repr(expr: &Expr, span: Span) -> syn::Result<String> {
    let value = parse_string_expr(expr, span)?;
    match value.as_str() {
        "string" | "i64" => Ok(value),
        _ => Err(syn::Error::new(
            span,
            "int_repr must be \"string\" or \"i64\"",
        )),
    }
}

fn parse_map_key_repr(expr: &Expr, span: Span) -> syn::Result<String> {
    let value = parse_string_expr(expr, span)?;
    match value.as_str() {
        "string" | "pairs" => Ok(value),
        _ => Err(syn::Error::new(
            span,
            "map_key_repr must be \"string\" or \"pairs\"",
        )),
    }
}

fn parse_constraint_meta(
    meta: &syn::MetaList,
    kind: ConstraintKind,
) -> syn::Result<ConstraintCompatAttr> {
    let nested = meta
        .parse_args_with(syn::punctuated::Punctuated::<Meta, syn::Token![,]>::parse_terminated)?;
    let mut label = None;
    let mut expr = None;

    for item in nested {
        match item {
            Meta::NameValue(m) if m.path.is_ident("label") => {
                label = Some(parse_string_expr(&m.value, m.span())?);
            }
            Meta::NameValue(m) if m.path.is_ident("expr") => {
                expr = Some(parse_string_expr(&m.value, m.span())?);
            }
            other => {
                return Err(syn::Error::new_spanned(
                    other,
                    "unsupported constraint attribute; hint: use #[baml(check(...))] or #[baml(assert(...))]",
                ));
            }
        }
    }

    let Some(label) = label else {
        return Err(syn::Error::new(
            meta.span(),
            "constraint missing label; hint: use label = \"...\"",
        ));
    };
    let Some(expr) = expr else {
        return Err(syn::Error::new(
            meta.span(),
            "constraint missing expr; hint: use expr = \"...\"",
        ));
    };

    Ok(ConstraintCompatAttr { kind, label, expr })
}

fn replace_doc_attrs(attrs: &mut Vec<Attribute>, description: &str) {
    attrs.retain(|attr| !attr.path().is_ident("doc"));
    for line in description.lines() {
        let lit = syn::LitStr::new(line, Span::call_site());
        attrs.push(syn::parse_quote!(#[doc = #lit]));
    }
}
