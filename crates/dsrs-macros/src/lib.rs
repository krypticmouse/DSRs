use proc_macro::TokenStream;
use quote::{format_ident, quote};
use serde_json::{Value, json};
use std::collections::HashSet;
use syn::{
    Attribute, Data, DeriveInput, Expr, ExprLit, Fields, Ident, Lit, LitStr, Meta, MetaNameValue,
    Token, Visibility,
    parse::{Parse, ParseStream},
    parse_macro_input,
    spanned::Spanned,
    visit::Visit,
};

mod optim;
mod runtime_path;

use runtime_path::resolve_dspy_rs_path;

#[proc_macro_derive(Optimizable, attributes(parameter))]
pub fn derive_optimizable(input: TokenStream) -> TokenStream {
    optim::optimizable_impl(input)
}

#[proc_macro_derive(
    Signature,
    attributes(input, output, check, assert, alias, format, flatten)
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
    format: Option<String>,
    constraints: Vec<ParsedConstraint>,
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

    Ok(ParsedSignature {
        input_fields,
        output_fields,
        all_fields,
        instruction: collect_doc_comment(attrs),
    })
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

    if format.is_some() && !is_input {
        return Err(syn::Error::new_spanned(
            field,
            "#[format] is only supported on #[input] fields",
        ));
    }

    if let Some(format_value) = format.as_deref() {
        match format_value.to_ascii_lowercase().as_str() {
            "json" | "yaml" | "toon" => {}
            _ => {
                return Err(syn::Error::new_spanned(
                    field,
                    "unsupported #[format] value; use \"json\", \"yaml\", or \"toon\"",
                ));
            }
        }
    }

    if is_flatten && (alias.is_some() || format.is_some() || !constraints.is_empty()) {
        return Err(syn::Error::new_spanned(
            field,
            "#[flatten] cannot be combined with #[alias], #[format], #[check], or #[assert]",
        ));
    }

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
        format,
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
    let normalized = expression
        .replace(" && ", " and ")
        .replace(" || ", " or ")
        .replace("&&", " and ")
        .replace("||", " or ");
    *expression = normalized;
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

    // Note: aliases, formats, and constraints are emitted in
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
        let format = match &field.format {
            Some(value) => {
                let lit = LitStr::new(value, proc_macro2::Span::call_site());
                quote! { Some(#lit) }
            }
            None => quote! { None },
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
                format: #format,
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
        to_value_inserts.push(quote! {
            fields.insert(
                #field_name.to_string(),
                #runtime::__macro_support::bamltype::to_baml_value(&self.#ident).unwrap_or(#runtime::BamlValue::Null),
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

#[allow(unused_assignments, non_snake_case)]
#[proc_macro_attribute]
pub fn LegacySignature(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as DeriveInput);
    let runtime = match resolve_dspy_rs_path() {
        Ok(path) => path,
        Err(err) => return err.to_compile_error().into(),
    };

    // Parse the attributes (cot, hint, etc.)
    let attr_str = attr.to_string();
    let has_cot = attr_str.contains("cot");
    let has_hint = attr_str.contains("hint");

    let struct_name = &input.ident;

    let mut signature_instruction = String::new();
    // Store everything as serde Values
    let mut input_schema: Value = json!({});
    let mut output_schema: Value = json!({});

    // Store schema update operations to be performed at runtime
    let mut schema_updates = Vec::new();

    if has_cot {
        output_schema["reasoning"] = json!({
            "type": "String",
            "desc": "Think step by step",
            "schema": "",
            "__dsrs_field_type": "output"
        });
    }
    // Generate schema for the field

    match &input.data {
        syn::Data::Struct(s) => {
            if let syn::Fields::Named(named) = &s.fields {
                let mut found_first_input = false;

                for field in &named.named {
                    let field_name = match field.ident.as_ref() {
                        Some(name) => name.clone(),
                        None => {
                            return syn::Error::new_spanned(
                                field,
                                "LegacySignature requires named fields",
                            )
                            .to_compile_error()
                            .into();
                        }
                    };
                    let field_type = field.ty.clone();

                    // Check for #[input] or #[output] attributes
                    let (is_input, desc) = has_io_attribute(&field.attrs, "input");
                    let (is_output, desc2) = has_io_attribute(&field.attrs, "output");

                    if is_input && is_output {
                        return syn::Error::new_spanned(
                            field,
                            format!("Field `{field_name}` cannot be both input and output"),
                        )
                        .to_compile_error()
                        .into();
                    }

                    if !is_input && !is_output {
                        return syn::Error::new_spanned(
                            field,
                            format!(
                                "Field `{field_name}` must have either #[input] or #[output] attribute"
                            ),
                        )
                        .to_compile_error()
                        .into();
                    }

                    let field_desc = if is_input { desc } else { desc2 };

                    // Collect doc comments from first input field as instruction
                    if is_input && !found_first_input {
                        signature_instruction = field
                            .attrs
                            .iter()
                            .filter(|a| a.path().is_ident("doc"))
                            .filter_map(|a| match &a.meta {
                                syn::Meta::NameValue(nv) => match &nv.value {
                                    syn::Expr::Lit(syn::ExprLit {
                                        lit: syn::Lit::Str(s),
                                        ..
                                    }) => Some(s.value()),
                                    _ => None,
                                },
                                _ => None,
                            })
                            .map(|s| s.trim().to_string())
                            .collect::<Vec<_>>()
                            .join("\n");
                        found_first_input = true;
                    }

                    // Create the field metadata as a serde Value
                    let type_str = quote!(#field_type).to_string();

                    let field_metadata = json!({
                        "type": type_str,
                        "desc": field_desc,
                        "schema": "",
                        "__dsrs_field_type": if is_input { "input" } else { "output" }
                    });

                    if is_input {
                        input_schema[field_name.to_string()] = field_metadata;
                        // Check if type needs schema generation (not primitive types)
                        if !is_primitive_type(&type_str) {
                            let field_name_str = field_name.to_string();
                            schema_updates.push(quote! {
                                {
                                    let schema = #runtime::__macro_support::schemars::schema_for!(#field_type);
                                    let schema_json = #runtime::__macro_support::serde_json::to_value(schema).unwrap();
                                    // Extract just the properties if it's an object schema
                                    if let Some(obj) = schema_json.as_object() {
                                        if obj.contains_key("properties") {
                                            input_fields[#field_name_str]["schema"] = schema_json["properties"].clone();
                                        } else {
                                            input_fields[#field_name_str]["schema"] = schema_json;
                                        }
                                    } else {
                                        input_fields[#field_name_str]["schema"] = schema_json;
                                    }
                                }
                            });
                        }
                    } else if is_output {
                        output_schema[field_name.to_string()] = field_metadata;
                        // Check if type needs schema generation (not primitive types)
                        if !is_primitive_type(&type_str) {
                            let field_name_str = field_name.to_string();
                            schema_updates.push(quote! {
                                {
                                    let schema = #runtime::__macro_support::schemars::schema_for!(#field_type);
                                    let schema_json = #runtime::__macro_support::serde_json::to_value(schema).unwrap();
                                    // Extract just the properties if it's an object schema
                                    if let Some(obj) = schema_json.as_object() {
                                        if obj.contains_key("properties") {
                                            output_fields[#field_name_str]["schema"] = schema_json["properties"].clone();
                                        } else {
                                            output_fields[#field_name_str]["schema"] = schema_json;
                                        }
                                    } else {
                                        output_fields[#field_name_str]["schema"] = schema_json;
                                    }
                                }
                            });
                        }
                    }
                }
            }
        }
        _ => {
            return syn::Error::new_spanned(
                &input,
                "LegacySignature can only be applied to structs with named fields",
            )
            .to_compile_error()
            .into();
        }
    }

    if has_hint {
        input_schema["hint"] = json!({
            "type": "String",
            "desc": "Hint for the query",
            "schema": "",
            "__dsrs_field_type": "input"
        });
    }

    // Serialize the schemas to strings so we can embed them in the generated code
    let input_schema_str = serde_json::to_string(&input_schema).unwrap();
    let output_schema_str = serde_json::to_string(&output_schema).unwrap();

    let generated = quote! {
        #[derive(Default, Debug, Clone, #runtime::__macro_support::serde::Serialize, #runtime::__macro_support::serde::Deserialize)]
        struct #struct_name {
            instruction: String,
            input_fields: #runtime::__macro_support::serde_json::Value,
            output_fields: #runtime::__macro_support::serde_json::Value,
            demos: Vec<#runtime::Example>,
        }

        impl #struct_name {
            pub fn new() -> Self {
                let mut input_fields: #runtime::__macro_support::serde_json::Value = #runtime::__macro_support::serde_json::from_str(#input_schema_str).unwrap();
                let mut output_fields: #runtime::__macro_support::serde_json::Value = #runtime::__macro_support::serde_json::from_str(#output_schema_str).unwrap();

                // Update schemas for complex types
                #(#schema_updates)*

                Self {
                    instruction: #signature_instruction.to_string(),
                    input_fields: input_fields,
                    output_fields: output_fields,
                    demos: vec![],
                }
            }

            pub fn input_fields_len(&self) -> usize {
                self.input_fields.as_object().map_or(0, |obj| obj.len())
            }

            pub fn output_fields_len(&self) -> usize {
                self.output_fields.as_object().map_or(0, |obj| obj.len())
            }
        }

        impl #runtime::core::MetaSignature for #struct_name {
            fn demos(&self) -> Vec<#runtime::Example> {
                self.demos.clone()
            }

            fn set_demos(&mut self, demos: Vec<#runtime::Example>) -> #runtime::__macro_support::anyhow::Result<()> {
                self.demos = demos;
                Ok(())
            }

            fn instruction(&self) -> String {
                self.instruction.clone()
            }

            fn input_fields(&self) -> #runtime::__macro_support::serde_json::Value {
                self.input_fields.clone()
            }

            fn output_fields(&self) -> #runtime::__macro_support::serde_json::Value {
                self.output_fields.clone()
            }

            fn update_instruction(&mut self, instruction: String) -> #runtime::__macro_support::anyhow::Result<()> {
                self.instruction = instruction;
                Ok(())
            }

            fn append(&mut self, name: &str, field_value: #runtime::__macro_support::serde_json::Value) -> #runtime::__macro_support::anyhow::Result<()> {
                match field_value["__dsrs_field_type"].as_str() {
                    Some("input") => {
                        self.input_fields[name] = field_value;
                    }
                    Some("output") => {
                        self.output_fields[name] = field_value;
                    }
                    _ => {
                        return Err(#runtime::__macro_support::anyhow::anyhow!("Invalid field type: {:?}", field_value["__dsrs_field_type"].as_str()));
                    }
                }
                Ok(())
            }
        }
    };

    generated.into()
}

fn has_io_attribute(attrs: &[Attribute], attr_name: &str) -> (bool, String) {
    for attr in attrs {
        if attr.path().is_ident(attr_name) {
            // Try to parse desc parameter
            if let Ok(list) = attr.meta.require_list() {
                let desc = parse_desc_from_tokens(list.tokens.clone());
                return (true, desc);
            }

            // Just #[input] or #[output] without parameters.
            return (true, String::new());
        }
    }
    (false, String::new())
}

fn parse_desc_from_tokens(tokens: proc_macro2::TokenStream) -> String {
    if let Ok(nv) = syn::parse2::<MetaNameValue>(tokens)
        && nv.path.is_ident("desc")
        && let syn::Expr::Lit(syn::ExprLit {
            lit: Lit::Str(s), ..
        }) = nv.value
    {
        return s.value();
    }
    String::new()
}

fn is_primitive_type(type_str: &str) -> bool {
    matches!(
        type_str,
        "String"
            | "str"
            | "bool"
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
    )
}
