use proc_macro::TokenStream;
use quote::{format_ident, quote};
use std::collections::{HashMap, HashSet};
use syn::{
    Attribute, Data, DeriveInput, Expr, ExprLit, Fields, Ident, Lit, LitStr, Meta, MetaNameValue,
    Token, Visibility,
    parse::{Parse, ParseStream},
    parse_macro_input,
    spanned::Spanned,
    visit::Visit,
};

mod runtime_path;

use runtime_path::resolve_dspy_rs_path;

#[proc_macro_derive(
    Signature,
    attributes(input, output, check, assert, alias, format, render, flatten)
)]
pub fn derive_signature(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let runtime = match resolve_dspy_rs_path() {
        Ok(path) => path,
        Err(err) => return err.to_compile_error().into(),
    };

    match expand_signature(&input, &runtime) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

#[proc_macro_derive(Augmentation, attributes(output, augment, alias))]
pub fn derive_augmentation(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let runtime = match resolve_dspy_rs_path() {
        Ok(path) => path,
        Err(err) => return err.to_compile_error().into(),
    };

    match expand_augmentation(&input, &runtime) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

fn expand_signature(
    input: &DeriveInput,
    runtime: &syn::Path,
) -> syn::Result<proc_macro2::TokenStream> {
    let data = match &input.data {
        Data::Struct(data) => data,
        _ => {
            return Err(syn::Error::new_spanned(
                input,
                "#[derive(Signature)] only supports structs with named fields",
            ));
        }
    };

    let fields = match &data.fields {
        Fields::Named(named) => &named.named,
        _ => {
            return Err(syn::Error::new_spanned(
                input,
                "#[derive(Signature)] requires named fields",
            ));
        }
    };

    let parsed = parse_signature_fields(fields, &input.attrs)?;
    generate_signature_code(input, &parsed, runtime)
}

#[derive(Clone)]
struct ParsedField {
    ident: Ident,
    ty: syn::Type,
    is_input: bool,
    is_output: bool,
    is_flatten: bool,
    description: String,
    alias: Option<String>,
    input_render: ParsedInputRender,
    constraints: Vec<ParsedConstraint>,
}

#[derive(Clone)]
enum ParsedInputRender {
    Default,
    Format(String),
    Jinja(String),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ParsedConstraintKind {
    Check,
    Assert,
}

#[derive(Clone)]
struct ParsedConstraint {
    kind: ParsedConstraintKind,
    expression: String,
    label: Option<String>,
}

struct ParsedSignature {
    input_fields: Vec<ParsedField>,
    output_fields: Vec<ParsedField>,
    all_fields: Vec<ParsedField>,
    instruction: String,
}

struct ConstraintArgs {
    expression: String,
    label: Option<String>,
}

impl Parse for ConstraintArgs {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let expression: LitStr = input.parse()?;
        let mut label = None;

        if input.parse::<Token![,]>().is_ok() {
            let ident: Ident = input.parse()?;
            if ident != "label" {
                return Err(syn::Error::new_spanned(
                    ident,
                    "expected label = \"...\" after expression",
                ));
            }
            input.parse::<Token![=]>()?;
            let label_lit: LitStr = input.parse()?;
            label = Some(label_lit.value());
        }

        if !input.is_empty() {
            return Err(syn::Error::new(
                input.span(),
                "unexpected tokens in constraint attribute",
            ));
        }

        Ok(Self {
            expression: expression.value(),
            label,
        })
    }
}

fn parse_signature_fields(
    fields: &syn::punctuated::Punctuated<syn::Field, Token![,]>,
    attrs: &[Attribute],
) -> syn::Result<ParsedSignature> {
    let mut input_fields = Vec::new();
    let mut output_fields = Vec::new();
    let mut all_fields = Vec::new();

    for field in fields {
        let parsed = parse_single_field(field)?;

        if parsed.is_input && parsed.is_output {
            return Err(syn::Error::new_spanned(
                field,
                "field cannot be both #[input] and #[output]",
            ));
        }
        if !parsed.is_input && !parsed.is_output {
            return Err(syn::Error::new_spanned(
                field,
                "field must have #[input] or #[output] attribute",
            ));
        }

        all_fields.push(parsed.clone());
        if parsed.is_input {
            input_fields.push(parsed);
        } else {
            output_fields.push(parsed);
        }
    }

    if input_fields.is_empty() {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "#[derive(Signature)] requires at least one #[input] field",
        ));
    }
    if output_fields.is_empty() {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "#[derive(Signature)] requires at least one #[output] field",
        ));
    }
    validate_unique_lm_names(&input_fields, "input")?;
    validate_unique_lm_names(&output_fields, "output")?;

    Ok(ParsedSignature {
        input_fields,
        output_fields,
        all_fields,
        instruction: collect_doc_comment(attrs),
    })
}

fn validate_unique_lm_names(fields: &[ParsedField], kind: &str) -> syn::Result<()> {
    let mut seen = HashMap::<String, String>::new();

    for field in fields {
        let rust_name = field.ident.to_string();
        let lm_name = field.alias.as_deref().unwrap_or(&rust_name).to_string();
        if let Some(previous_rust_name) = seen.insert(lm_name.clone(), rust_name.clone()) {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                format!(
                    "duplicate {kind} field name `{lm_name}` after aliasing; conflicts between `{previous_rust_name}` and `{rust_name}`"
                ),
            ));
        }
    }

    Ok(())
}

fn parse_single_field(field: &syn::Field) -> syn::Result<ParsedField> {
    let ident = field.ident.clone().ok_or_else(|| {
        syn::Error::new_spanned(field, "#[derive(Signature)] requires named fields")
    })?;

    let mut is_input = false;
    let mut is_output = false;
    let mut is_flatten = false;
    let mut saw_flatten = false;
    let mut alias = None;
    let mut format = None;
    let mut render_jinja = None;
    let mut constraints = Vec::new();
    let mut desc_override = None;

    for attr in &field.attrs {
        if attr.path().is_ident("input") {
            is_input = true;
            if let Some(desc) = parse_desc_from_attr(attr, "input")? {
                desc_override = Some(desc);
            }
        } else if attr.path().is_ident("output") {
            is_output = true;
            if let Some(desc) = parse_desc_from_attr(attr, "output")? {
                desc_override = Some(desc);
            }
        } else if attr.path().is_ident("alias") {
            alias = Some(parse_string_attr(attr, "alias")?);
        } else if attr.path().is_ident("format") {
            if format.is_some() {
                return Err(syn::Error::new_spanned(
                    attr,
                    "#[format] can only be specified once per field",
                ));
            }
            format = Some(parse_string_attr(attr, "format")?);
        } else if attr.path().is_ident("render") {
            if render_jinja.is_some() {
                return Err(syn::Error::new_spanned(
                    attr,
                    "#[render] can only be specified once per field",
                ));
            }
            let template = parse_render_jinja_attr(attr)?;
            validate_jinja_template(&template, attr.span())?;
            render_jinja = Some(template);
        } else if attr.path().is_ident("flatten") {
            if saw_flatten {
                return Err(syn::Error::new_spanned(
                    attr,
                    "#[flatten] can only be specified once per field",
                ));
            }
            saw_flatten = true;
            is_flatten = true;
        } else if attr.path().is_ident("check") {
            constraints.push(parse_constraint_attr(attr, ParsedConstraintKind::Check)?);
        } else if attr.path().is_ident("assert") {
            constraints.push(parse_constraint_attr(attr, ParsedConstraintKind::Assert)?);
        }
    }

    if format.is_some() && render_jinja.is_some() {
        return Err(syn::Error::new_spanned(
            field,
            "#[format] and #[render] cannot be combined on the same field",
        ));
    }

    if format.is_some() && !is_input {
        return Err(syn::Error::new_spanned(
            field,
            "#[format] is only supported on #[input] fields",
        ));
    }
    if render_jinja.is_some() && !is_input {
        return Err(syn::Error::new_spanned(
            field,
            "#[render] is only supported on #[input] fields",
        ));
    }

    let input_render = if let Some(template) = render_jinja {
        ParsedInputRender::Jinja(template)
    } else if let Some(format_value) = format {
        match format_value.to_ascii_lowercase().as_str() {
            "json" | "yaml" | "toon" => ParsedInputRender::Format(format_value),
            _ => {
                return Err(syn::Error::new_spanned(
                    field,
                    "unsupported #[format] value; use \"json\", \"yaml\", or \"toon\"",
                ));
            }
        }
    } else {
        ParsedInputRender::Default
    };

    if is_flatten
        && (alias.is_some()
            || !matches!(input_render, ParsedInputRender::Default)
            || !constraints.is_empty())
    {
        return Err(syn::Error::new_spanned(
            field,
            "#[flatten] cannot be combined with #[alias], #[format], #[render], #[check], or #[assert]",
        ));
    }

    validate_signature_field_type(field)?;

    let doc_comment = collect_doc_comment(&field.attrs);
    let description = desc_override.unwrap_or(doc_comment);

    Ok(ParsedField {
        ident,
        ty: field.ty.clone(),
        is_input,
        is_output,
        is_flatten,
        description,
        alias,
        input_render,
        constraints,
    })
}

fn parse_desc_from_attr(attr: &Attribute, attr_name: &str) -> syn::Result<Option<String>> {
    match &attr.meta {
        Meta::Path(_) => Ok(None),
        Meta::List(list) => {
            let metas = list.parse_args_with(
                syn::punctuated::Punctuated::<Meta, syn::Token![,]>::parse_terminated,
            )?;

            if metas.is_empty() {
                return Ok(None);
            }

            if metas.len() == 1
                && let Some(Meta::NameValue(meta)) = metas.first()
                && meta.path.is_ident("desc")
            {
                return Ok(Some(parse_string_expr(&meta.value, meta.span())?));
            }

            Err(syn::Error::new_spanned(
                attr,
                format!(
                    "unsupported arguments for #[{attr_name}(...)]; only desc = \"...\" is allowed"
                ),
            ))
        }
        _ => Err(syn::Error::new_spanned(
            attr,
            format!("expected #[{attr_name}] or #[{attr_name}(desc = \"...\")]"),
        )),
    }
}

fn parse_string_attr(attr: &Attribute, attr_name: &str) -> syn::Result<String> {
    match &attr.meta {
        Meta::NameValue(meta) => parse_string_expr(&meta.value, meta.span()),
        Meta::List(list) => {
            let lit: LitStr = list.parse_args()?;
            Ok(lit.value())
        }
        _ => Err(syn::Error::new_spanned(
            attr,
            format!("expected #[{attr_name} = \"...\"] or #[{attr_name}(\"...\")]"),
        )),
    }
}

fn parse_render_jinja_attr(attr: &Attribute) -> syn::Result<String> {
    match &attr.meta {
        Meta::List(list) => {
            let metas = list.parse_args_with(
                syn::punctuated::Punctuated::<Meta, syn::Token![,]>::parse_terminated,
            )?;

            if metas.len() != 1 {
                return Err(syn::Error::new_spanned(
                    attr,
                    "expected #[render(jinja = \"...\")]",
                ));
            }

            match metas.first() {
                Some(Meta::NameValue(meta)) if meta.path.is_ident("jinja") => {
                    parse_string_expr(&meta.value, meta.span())
                }
                _ => Err(syn::Error::new_spanned(
                    attr,
                    "expected #[render(jinja = \"...\")]",
                )),
            }
        }
        _ => Err(syn::Error::new_spanned(
            attr,
            "expected #[render(jinja = \"...\")]",
        )),
    }
}

fn validate_jinja_template(template: &str, span: proc_macro2::Span) -> syn::Result<()> {
    let mut env = minijinja::Environment::new();
    env.add_template("__input_field__", template)
        .map_err(|_| syn::Error::new(span, "invalid Jinja syntax in #[render(jinja = \"...\")]"))?;
    Ok(())
}

fn parse_constraint_attr(
    attr: &Attribute,
    kind: ParsedConstraintKind,
) -> syn::Result<ParsedConstraint> {
    let mut args: ConstraintArgs = attr.parse_args()?;
    normalize_constraint_expression(&mut args.expression);
    if kind == ParsedConstraintKind::Check && args.label.is_none() {
        return Err(syn::Error::new_spanned(
            attr,
            "#[check] requires a label: #[check(\"expr\", label = \"name\")]",
        ));
    }

    Ok(ParsedConstraint {
        kind,
        expression: args.expression,
        label: args.label,
    })
}

fn normalize_constraint_expression(expression: &mut String) {
    // Accept common Rust-style logical operators in docs/examples and normalize
    // to the Jinja expression syntax expected by downstream evaluation.
    let segments = split_constraint_segments(expression);
    let normalized: String = segments
        .into_iter()
        .map(|(segment, is_literal)| {
            if is_literal {
                segment
            } else {
                segment
                    .replace(" && ", " and ")
                    .replace(" || ", " or ")
                    .replace("&&", " and ")
                    .replace("||", " or ")
            }
        })
        .collect();
    *expression = normalized;
}

fn split_constraint_segments(expression: &str) -> Vec<(String, bool)> {
    let mut segments = Vec::new();
    let mut buf = String::new();
    let mut in_literal = false;
    let mut prev_escape = false;

    for ch in expression.chars() {
        if ch == '"' && !prev_escape {
            if in_literal {
                buf.push(ch);
                segments.push((buf.clone(), true));
                buf.clear();
                in_literal = false;
            } else {
                if !buf.is_empty() {
                    segments.push((buf.clone(), false));
                    buf.clear();
                }
                in_literal = true;
                buf.push(ch);
            }
            prev_escape = false;
            continue;
        }

        buf.push(ch);
        prev_escape = ch == '\\' && !prev_escape;
    }

    if !buf.is_empty() {
        segments.push((buf, in_literal));
    }

    segments
}

fn collect_doc_comment(attrs: &[Attribute]) -> String {
    let mut docs = Vec::new();
    for attr in attrs {
        if attr.path().is_ident("doc")
            && let Meta::NameValue(MetaNameValue {
                value:
                    Expr::Lit(ExprLit {
                        lit: Lit::Str(lit), ..
                    }),
                ..
            }) = &attr.meta
        {
            docs.push(lit.value().trim().to_string());
        }
    }
    docs.join("\n")
}

fn parse_string_expr(expr: &Expr, span: proc_macro2::Span) -> syn::Result<String> {
    match expr {
        Expr::Lit(ExprLit {
            lit: Lit::Str(s), ..
        }) => Ok(s.value()),
        _ => Err(syn::Error::new(
            span,
            "expected string literal; hint: wrap the value in quotes",
        )),
    }
}

// TODO(dsrs-derive-shared-validation): deduplicate type-validation logic with bamltype-derive.
fn validate_signature_field_type(field: &syn::Field) -> syn::Result<()> {
    if let Some(ty) = find_type_match(&field.ty, &|ty| matches!(ty, syn::Type::BareFn(_))) {
        return Err(syn::Error::new_spanned(
            ty,
            "function types are not supported in Signature fields; hint: use a concrete type",
        ));
    }

    if let Some(ty) = find_type_match(&field.ty, &|ty| matches!(ty, syn::Type::TraitObject(_))) {
        return Err(syn::Error::new_spanned(
            ty,
            "trait objects are not supported in Signature fields; hint: use a concrete type",
        ));
    }

    if let Some(ty) = find_type_match(&field.ty, &|ty| matches!(ty, syn::Type::Tuple(_))) {
        return Err(syn::Error::new_spanned(
            ty,
            "tuple types are not supported in Signature fields; hint: use a struct with named fields or a list",
        ));
    }

    if let Some(ty) = find_type_match(&field.ty, &is_serde_json_value_type) {
        return Err(syn::Error::new_spanned(
            ty,
            "serde_json::Value is not supported in Signature fields; hint: use a concrete typed value",
        ));
    }

    if let Some(ty) = find_type_match(&field.ty, &has_non_string_map_key) {
        return Err(syn::Error::new_spanned(
            ty,
            "map keys must be String in Signature fields; hint: use HashMap<String, V> or BTreeMap<String, V>",
        ));
    }

    if let Some(ty) = find_type_match(&field.ty, &is_unsupported_signature_int_type) {
        return Err(syn::Error::new_spanned(
            ty,
            "unsupported integer width in Signature fields; hint: use i64/isize/u32 or a smaller integer type",
        ));
    }

    Ok(())
}

fn find_type_match<'a, F>(ty: &'a syn::Type, predicate: &F) -> Option<&'a syn::Type>
where
    F: Fn(&syn::Type) -> bool,
{
    if predicate(ty) {
        return Some(ty);
    }

    match ty {
        syn::Type::Array(array) => find_type_match(&array.elem, predicate),
        syn::Type::Group(group) => find_type_match(&group.elem, predicate),
        syn::Type::Paren(paren) => find_type_match(&paren.elem, predicate),
        syn::Type::Ptr(ptr) => find_type_match(&ptr.elem, predicate),
        syn::Type::Reference(reference) => find_type_match(&reference.elem, predicate),
        syn::Type::Slice(slice) => find_type_match(&slice.elem, predicate),
        syn::Type::Tuple(tuple) => {
            for elem in &tuple.elems {
                if let Some(found) = find_type_match(elem, predicate) {
                    return Some(found);
                }
            }
            None
        }
        syn::Type::Path(path) => {
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

fn type_ident(ty: &syn::Type) -> Option<&syn::Ident> {
    match ty {
        syn::Type::Path(path) if path.qself.is_none() => {
            path.path.segments.last().map(|s| &s.ident)
        }
        _ => None,
    }
}

fn map_types(ty: &syn::Type) -> Option<(&syn::Type, &syn::Type)> {
    if let syn::Type::Path(path) = ty
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

fn is_string_type(ty: &syn::Type) -> bool {
    type_ident(ty)
        .map(|ident| ident == "String")
        .unwrap_or(false)
}

fn has_non_string_map_key(ty: &syn::Type) -> bool {
    map_types(ty)
        .map(|(key, _)| !is_string_type(key))
        .unwrap_or(false)
}

fn is_unsupported_signature_int_type(ty: &syn::Type) -> bool {
    match type_ident(ty).map(|ident| ident.to_string()) {
        Some(name) => matches!(name.as_str(), "u64" | "usize" | "i128" | "u128"),
        None => false,
    }
}

fn is_serde_json_value_type(ty: &syn::Type) -> bool {
    if let syn::Type::Path(path) = ty
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

fn generate_signature_code(
    input: &DeriveInput,
    parsed: &ParsedSignature,
    runtime: &syn::Path,
) -> syn::Result<proc_macro2::TokenStream> {
    let name = &input.ident;
    let vis = &input.vis;
    let generics = &input.generics;

    let helper_structs = generate_helper_structs(name, generics, parsed, vis, runtime)?;
    let input_metadata = generate_field_metadata(name, &parsed.input_fields, "INPUT", runtime)?;
    let output_metadata = generate_field_metadata(name, &parsed.output_fields, "OUTPUT", runtime)?;
    let baml_delegation = generate_baml_delegation(name, generics, parsed, runtime);
    let signature_impl = generate_signature_impl(name, generics, parsed, runtime);

    Ok(quote! {
        #helper_structs
        #input_metadata
        #output_metadata
        #baml_delegation
        #signature_impl
    })
}

fn generate_helper_structs(
    name: &Ident,
    generics: &syn::Generics,
    parsed: &ParsedSignature,
    vis: &Visibility,
    runtime: &syn::Path,
) -> syn::Result<proc_macro2::TokenStream> {
    let input_name = format_ident!("{}Input", name);
    let output_name = format_ident!("{}Output", name);
    let all_name = format_ident!("{}All", name);

    let helper_generics = unconstrained_generics(generics);
    let (helper_impl_generics, helper_ty_generics, _helper_where_clause) =
        helper_generics.split_for_impl();

    let mut input_fields: Vec<_> = parsed.input_fields.iter().map(field_tokens).collect();
    let input_marker = generic_marker_field(generics, &parsed.input_fields);
    if let Some(marker) = &input_marker {
        input_fields.push(marker.field.clone());
    }
    let input_new_args: Vec<_> = parsed
        .input_fields
        .iter()
        .map(constructor_arg_tokens)
        .collect();
    let mut input_new_fields: Vec<_> = parsed
        .input_fields
        .iter()
        .map(constructor_init_tokens)
        .collect();
    if let Some(marker) = &input_marker {
        input_new_fields.push(marker.init.clone());
    }

    let mut output_fields: Vec<_> = parsed.output_fields.iter().map(field_tokens).collect();
    let output_marker = generic_marker_field(generics, &parsed.output_fields);
    if let Some(marker) = &output_marker {
        output_fields.push(marker.field.clone());
    }
    let output_new_args: Vec<_> = parsed
        .output_fields
        .iter()
        .map(constructor_arg_tokens)
        .collect();
    let mut output_new_fields: Vec<_> = parsed
        .output_fields
        .iter()
        .map(constructor_init_tokens)
        .collect();
    if let Some(marker) = &output_marker {
        output_new_fields.push(marker.init.clone());
    }

    let mut all_fields: Vec<_> = parsed.all_fields.iter().map(field_tokens).collect();
    let all_marker = generic_marker_field(generics, &parsed.all_fields);
    if let Some(marker) = &all_marker {
        all_fields.push(marker.field.clone());
    }
    let all_new_args: Vec<_> = parsed
        .all_fields
        .iter()
        .map(constructor_arg_tokens)
        .collect();
    let mut all_new_fields: Vec<_> = parsed
        .all_fields
        .iter()
        .map(constructor_init_tokens)
        .collect();
    if let Some(marker) = &all_marker {
        all_new_fields.push(marker.init.clone());
    }

    let facet = quote! { #runtime::__macro_support::bamltype::facet };
    let schema_bundle = quote! { #runtime::__macro_support::bamltype::SchemaBundle };

    Ok(quote! {
        #[derive(Debug, Clone, #facet::Facet)]
        #[facet(crate = #facet)]
        #vis struct #input_name #helper_generics {
            #(#input_fields),*
        }

        impl #helper_impl_generics #input_name #helper_ty_generics {
            #vis fn new(#(#input_new_args),*) -> Self {
                Self {
                    #(#input_new_fields),*
                }
            }
        }

        impl #helper_impl_generics #runtime::__macro_support::bamltype::BamlSchema for #input_name #helper_ty_generics
        where
            #input_name #helper_ty_generics: for<'a> #facet::Facet<'a>,
        {
            fn baml_schema() -> &'static #schema_bundle {
                static SCHEMA: ::std::sync::OnceLock<#schema_bundle> = ::std::sync::OnceLock::new();
                SCHEMA.get_or_init(|| {
                    #schema_bundle::from_shape(<Self as #facet::Facet<'_>>::SHAPE)
                })
            }
        }

        #[derive(Debug, Clone, #facet::Facet)]
        #[facet(crate = #facet)]
        pub struct #output_name #helper_generics {
            #(#output_fields),*
        }

        impl #helper_impl_generics #output_name #helper_ty_generics {
            pub fn new(#(#output_new_args),*) -> Self {
                Self {
                    #(#output_new_fields),*
                }
            }
        }

        impl #helper_impl_generics #runtime::__macro_support::bamltype::BamlSchema for #output_name #helper_ty_generics
        where
            #output_name #helper_ty_generics: for<'a> #facet::Facet<'a>,
        {
            fn baml_schema() -> &'static #schema_bundle {
                static SCHEMA: ::std::sync::OnceLock<#schema_bundle> = ::std::sync::OnceLock::new();
                SCHEMA.get_or_init(|| {
                    #schema_bundle::from_shape(<Self as #facet::Facet<'_>>::SHAPE)
                })
            }
        }

        #[derive(Debug, Clone, #facet::Facet)]
        #[facet(crate = #facet)]
        pub struct #all_name #helper_generics {
            #(#all_fields),*
        }

        impl #helper_impl_generics #all_name #helper_ty_generics {
            pub fn new(#(#all_new_args),*) -> Self {
                Self {
                    #(#all_new_fields),*
                }
            }
        }

        impl #helper_impl_generics #runtime::__macro_support::bamltype::BamlSchema for #all_name #helper_ty_generics
        where
            #all_name #helper_ty_generics: for<'a> #facet::Facet<'a>,
        {
            fn baml_schema() -> &'static #schema_bundle {
                static SCHEMA: ::std::sync::OnceLock<#schema_bundle> = ::std::sync::OnceLock::new();
                SCHEMA.get_or_init(|| {
                    #schema_bundle::from_shape(<Self as #facet::Facet<'_>>::SHAPE)
                })
            }
        }
    })
}

fn unconstrained_generics(generics: &syn::Generics) -> syn::Generics {
    let mut helper_generics = generics.clone();

    for param in helper_generics.type_params_mut() {
        param.bounds.clear();
        param.bounds.push(syn::parse_quote!('static));
        param.eq_token = None;
        param.default = None;
    }

    helper_generics.where_clause = None;
    helper_generics
}

struct MarkerFieldTokens {
    field: proc_macro2::TokenStream,
    init: proc_macro2::TokenStream,
}

fn generic_marker_field(
    generics: &syn::Generics,
    fields: &[ParsedField],
) -> Option<MarkerFieldTokens> {
    let missing = missing_type_params_for_fields(generics, fields);
    if missing.is_empty() {
        return None;
    }

    Some(MarkerFieldTokens {
        field: quote! {
            #[doc(hidden)]
            #[facet(skip)]
            _phantom: ::std::marker::PhantomData<(#(#missing),*)>
        },
        init: quote! {
            _phantom: ::std::marker::PhantomData
        },
    })
}

fn missing_type_params_for_fields(generics: &syn::Generics, fields: &[ParsedField]) -> Vec<Ident> {
    let type_params: Vec<Ident> = generics
        .type_params()
        .map(|param| param.ident.clone())
        .collect();

    if type_params.is_empty() {
        return Vec::new();
    }

    let mut collector = TypeParamUsageCollector {
        tracked: type_params
            .iter()
            .map(|ident| ident.to_string())
            .collect::<HashSet<_>>(),
        used: HashSet::new(),
    };

    for field in fields {
        collector.visit_type(&field.ty);
    }

    type_params
        .into_iter()
        .filter(|ident| !collector.used.contains(&ident.to_string()))
        .collect()
}

struct TypeParamUsageCollector {
    tracked: HashSet<String>,
    used: HashSet<String>,
}

impl<'ast> Visit<'ast> for TypeParamUsageCollector {
    fn visit_type_path(&mut self, path: &'ast syn::TypePath) {
        if path.qself.is_none() && path.path.segments.len() == 1 {
            let ident = path.path.segments[0].ident.to_string();
            if self.tracked.contains(&ident) {
                self.used.insert(ident);
            }
        }

        syn::visit::visit_type_path(self, path);
    }
}

fn field_tokens(field: &ParsedField) -> proc_macro2::TokenStream {
    let ident = &field.ident;
    let ty = &field.ty;
    let mut attrs = Vec::new();

    if !field.description.is_empty() {
        let doc = LitStr::new(&field.description, proc_macro2::Span::call_site());
        attrs.push(quote! { #[doc = #doc] });
    }

    if field.is_flatten {
        attrs.push(quote! { #[facet(flatten)] });
    }

    // Note: aliases, input render hints, and constraints are emitted in
    // generate_field_metadata(), not as struct attributes.

    quote! {
        #(#attrs)*
        pub #ident: #ty
    }
}

fn constructor_arg_tokens(field: &ParsedField) -> proc_macro2::TokenStream {
    let ident = &field.ident;
    let ty = &field.ty;
    quote! { #ident: #ty }
}

fn constructor_init_tokens(field: &ParsedField) -> proc_macro2::TokenStream {
    let ident = &field.ident;
    quote! { #ident }
}

fn generate_field_metadata(
    name: &Ident,
    fields: &[ParsedField],
    kind: &str,
    runtime: &syn::Path,
) -> syn::Result<proc_macro2::TokenStream> {
    let metadata_array_name =
        format_ident!("__{}_{}_METADATA", name.to_string().to_uppercase(), kind);

    let mut constraint_arrays = Vec::new();
    let mut metadata_specs = Vec::new();

    for field in fields {
        let field_name = field.ident.to_string();
        let rust_name = LitStr::new(&field_name, proc_macro2::Span::call_site());
        let alias = match &field.alias {
            Some(value) => {
                let lit = LitStr::new(value, proc_macro2::Span::call_site());
                quote! { Some(#lit) }
            }
            None => quote! { None },
        };
        let input_render = match &field.input_render {
            ParsedInputRender::Default => quote! { #runtime::InputRenderSpec::Default },
            ParsedInputRender::Format(value) => {
                let lit = LitStr::new(value, proc_macro2::Span::call_site());
                quote! { #runtime::InputRenderSpec::Format(#lit) }
            }
            ParsedInputRender::Jinja(value) => {
                let lit = LitStr::new(value, proc_macro2::Span::call_site());
                quote! { #runtime::InputRenderSpec::Jinja(#lit) }
            }
        };

        let constraints_name = format_ident!(
            "__{}_{}_CONSTRAINTS",
            name.to_string().to_uppercase(),
            field_name.to_uppercase(),
        );

        if field.constraints.is_empty() {
            constraint_arrays.push(quote! {
                const #constraints_name: &[#runtime::ConstraintSpec] = &[];
            });
        } else {
            let constraint_specs: Vec<_> = field
                .constraints
                .iter()
                .map(|constraint| {
                    let kind = match constraint.kind {
                        ParsedConstraintKind::Check => quote! { #runtime::ConstraintKind::Check },
                        ParsedConstraintKind::Assert => {
                            quote! { #runtime::ConstraintKind::Assert }
                        }
                    };
                    let label = constraint.label.as_deref().unwrap_or("");
                    let label = LitStr::new(label, proc_macro2::Span::call_site());
                    let expr = LitStr::new(&constraint.expression, proc_macro2::Span::call_site());
                    quote! {
                        #runtime::ConstraintSpec {
                            kind: #kind,
                            label: #label,
                            expression: #expr,
                        }
                    }
                })
                .collect();

            constraint_arrays.push(quote! {
                const #constraints_name: &[#runtime::ConstraintSpec] = &[
                    #(#constraint_specs),*
                ];
            });
        }

        metadata_specs.push(quote! {
            #runtime::FieldMetadataSpec {
                rust_name: #rust_name,
                alias: #alias,
                constraints: #constraints_name,
                input_render: #input_render,
            }
        });
    }

    Ok(quote! {
        #(#constraint_arrays)*

        static #metadata_array_name: &[#runtime::FieldMetadataSpec] = &[
            #(#metadata_specs),*
        ];
    })
}

fn generate_baml_delegation(
    name: &Ident,
    generics: &syn::Generics,
    parsed: &ParsedSignature,
    runtime: &syn::Path,
) -> proc_macro2::TokenStream {
    let all_name = format_ident!("{}All", name);
    let field_names: Vec<_> = parsed.all_fields.iter().map(|field| &field.ident).collect();
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let mut to_value_inserts = Vec::new();
    for field in &parsed.all_fields {
        let field_name = field.ident.to_string();
        let ident = &field.ident;
        let ty = &field.ty;
        to_value_inserts.push(quote! {
            fields.insert(
                #field_name.to_string(),
                #runtime::__macro_support::bamltype::to_baml_value(&self.#ident).unwrap_or_else(|err| {
                    panic!(
                        "Signature derive failed to convert field `{}` on `{}` (type `{}`) to BamlValue: {:?}",
                        #field_name,
                        stringify!(#name),
                        ::std::any::type_name::<#ty>(),
                        err,
                    )
                }),
            );
        });
    }

    quote! {
        impl #impl_generics #runtime::BamlType for #name #ty_generics #where_clause {
            fn baml_output_format() -> &'static #runtime::OutputFormatContent {
                <#all_name #ty_generics as #runtime::BamlType>::baml_output_format()
            }

            fn baml_internal_name() -> &'static str {
                <#all_name #ty_generics as #runtime::BamlType>::baml_internal_name()
            }

            fn baml_type_ir() -> #runtime::TypeIR {
                <#all_name #ty_generics as #runtime::BamlType>::baml_type_ir()
            }

            fn try_from_baml_value(value: #runtime::BamlValue) -> Result<Self, #runtime::BamlConvertError> {
                let all = <#all_name #ty_generics as #runtime::BamlType>::try_from_baml_value(value)?;
                Ok(Self {
                    #(#field_names: all.#field_names),*
                })
            }

            fn to_baml_value(&self) -> #runtime::BamlValue {
                let mut fields = #runtime::__macro_support::bamltype::baml_types::BamlMap::new();
                #(#to_value_inserts)*
                #runtime::__macro_support::bamltype::baml_types::BamlValue::Class(
                    <Self as #runtime::BamlType>::baml_internal_name()
                        .to_string(),
                    fields,
                )
            }
        }
    }
}

fn generate_signature_impl(
    name: &Ident,
    generics: &syn::Generics,
    parsed: &ParsedSignature,
    runtime: &syn::Path,
) -> proc_macro2::TokenStream {
    let input_name = format_ident!("{}Input", name);
    let output_name = format_ident!("{}Output", name);
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let instruction = LitStr::new(&parsed.instruction, proc_macro2::Span::call_site());

    let input_metadata_static =
        format_ident!("__{}_INPUT_METADATA", name.to_string().to_uppercase());
    let output_metadata_static =
        format_ident!("__{}_OUTPUT_METADATA", name.to_string().to_uppercase());

    quote! {
        impl #impl_generics #runtime::Signature for #name #ty_generics #where_clause {
            type Input = #input_name #ty_generics;
            type Output = #output_name #ty_generics;

            fn instruction() -> &'static str {
                #instruction
            }

            fn input_shape() -> &'static #runtime::Shape {
                <#input_name #ty_generics as #runtime::__macro_support::bamltype::facet::Facet<'static>>::SHAPE
            }

            fn output_shape() -> &'static #runtime::Shape {
                <#output_name #ty_generics as #runtime::__macro_support::bamltype::facet::Facet<'static>>::SHAPE
            }

            fn input_field_metadata() -> &'static [#runtime::FieldMetadataSpec] {
                &#input_metadata_static
            }

            fn output_field_metadata() -> &'static [#runtime::FieldMetadataSpec] {
                &#output_metadata_static
            }

            fn output_format_content() -> &'static #runtime::OutputFormatContent {
                <#output_name #ty_generics as #runtime::BamlType>::baml_output_format()
            }
        }
    }
}

#[derive(Clone)]
struct AugmentField {
    ident: Ident,
    ty: syn::Type,
    description: String,
    alias: Option<String>,
}

#[derive(Default)]
struct AugmentOptions {
    prepend: bool,
}

fn expand_augmentation(
    input: &DeriveInput,
    runtime: &syn::Path,
) -> syn::Result<proc_macro2::TokenStream> {
    let data = match &input.data {
        Data::Struct(data) => data,
        _ => {
            return Err(syn::Error::new_spanned(
                input,
                "#[derive(Augmentation)] only supports structs with named fields",
            ));
        }
    };

    let fields = match &data.fields {
        Fields::Named(named) => &named.named,
        _ => {
            return Err(syn::Error::new_spanned(
                input,
                "#[derive(Augmentation)] requires named fields",
            ));
        }
    };

    let options = parse_augment_options(&input.attrs)?;
    let parsed_fields = parse_augmentation_fields(fields)?;

    if parsed_fields.is_empty() {
        return Err(syn::Error::new_spanned(
            input,
            "#[derive(Augmentation)] requires at least one #[output] field",
        ));
    }

    let struct_name = &input.ident;
    let wrapper_name = format_ident!("With{}", struct_name);

    let reasoning_fields: Vec<_> = parsed_fields
        .iter()
        .map(|field| {
            let ident = &field.ident;
            let ty = &field.ty;
            let mut attrs = Vec::new();
            if !field.description.is_empty() {
                let doc = LitStr::new(&field.description, proc_macro2::Span::call_site());
                attrs.push(quote! { #[doc = #doc] });
            }
            if let Some(alias) = &field.alias {
                let lit = LitStr::new(alias, proc_macro2::Span::call_site());
                attrs.push(quote! { #[facet(rename = #lit)] });
            }
            quote! {
                #(#attrs)*
                pub #ident: #ty
            }
        })
        .collect();

    let output_field = quote! {
        #[facet(flatten)]
        pub inner: O
    };

    let (first_fields, last_fields) = if options.prepend {
        (reasoning_fields, vec![output_field])
    } else {
        (vec![output_field], reasoning_fields)
    };

    Ok(quote! {
        #[derive(Clone, Debug, #runtime::__macro_support::bamltype::facet::Facet)]
        #[facet(crate = #runtime::__macro_support::bamltype::facet)]
        pub struct #wrapper_name<O> {
            #(#first_fields),*,
            #(#last_fields),*
        }

        impl<O> std::ops::Deref for #wrapper_name<O> {
            type Target = O;
            fn deref(&self) -> &Self::Target {
                &self.inner
            }
        }

        impl<O> #runtime::__macro_support::bamltype::BamlSchema for #wrapper_name<O>
        where
            O: for<'a> #runtime::__macro_support::bamltype::facet::Facet<'a>,
        {
            fn baml_schema(
            ) -> &'static #runtime::__macro_support::bamltype::SchemaBundle {
                static SCHEMA: ::std::sync::OnceLock<
                    #runtime::__macro_support::bamltype::SchemaBundle,
                > = ::std::sync::OnceLock::new();
                SCHEMA.get_or_init(|| {
                    #runtime::__macro_support::bamltype::SchemaBundle::from_shape(
                        <Self as #runtime::__macro_support::bamltype::facet::Facet<'_>>::SHAPE,
                    )
                })
            }
        }

        impl #runtime::augmentation::Augmentation for #struct_name {
            type Wrap<T: #runtime::BamlType + for<'a> #runtime::Facet<'a> + Send + Sync> =
                #wrapper_name<T>;
        }
    })
}

fn parse_augment_options(attrs: &[Attribute]) -> syn::Result<AugmentOptions> {
    let mut options = AugmentOptions::default();
    for attr in attrs {
        if !attr.path().is_ident("augment") {
            continue;
        }
        let meta = attr
            .parse_args_with(syn::punctuated::Punctuated::<Ident, Token![,]>::parse_terminated)?;
        for ident in meta {
            let name = ident.to_string();
            match name.as_str() {
                "output" => {}
                "prepend" => options.prepend = true,
                other => {
                    return Err(syn::Error::new_spanned(
                        ident,
                        format!("unsupported #[augment] option `{other}`"),
                    ));
                }
            }
        }
    }
    Ok(options)
}

fn parse_augmentation_fields(
    fields: &syn::punctuated::Punctuated<syn::Field, Token![,]>,
) -> syn::Result<Vec<AugmentField>> {
    let mut parsed = Vec::new();

    for field in fields {
        let ident = field.ident.clone().ok_or_else(|| {
            syn::Error::new_spanned(field, "#[derive(Augmentation)] requires named fields")
        })?;

        let mut is_output = false;
        let mut alias = None;
        let mut desc_override = None;

        for attr in &field.attrs {
            if attr.path().is_ident("output") {
                is_output = true;
                if let Some(desc) = parse_desc_from_attr(attr, "output")? {
                    desc_override = Some(desc);
                }
            } else if attr.path().is_ident("input") {
                return Err(syn::Error::new_spanned(
                    attr,
                    "#[derive(Augmentation)] does not support #[input] fields",
                ));
            } else if attr.path().is_ident("alias") {
                alias = Some(parse_string_attr(attr, "alias")?);
            } else if attr.path().is_ident("flatten") {
                return Err(syn::Error::new_spanned(
                    attr,
                    "#[derive(Augmentation)] does not support #[flatten] on fields",
                ));
            }
        }

        if !is_output {
            return Err(syn::Error::new_spanned(
                field,
                "#[derive(Augmentation)] requires fields to be marked #[output]",
            ));
        }

        let doc_comment = collect_doc_comment(&field.attrs);
        let description = desc_override.unwrap_or(doc_comment);

        parsed.push(AugmentField {
            ident,
            ty: field.ty.clone(),
            description,
            alias,
        });
    }

    Ok(parsed)
}
