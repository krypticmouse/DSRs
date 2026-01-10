extern crate self as dsrs_macros;

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use serde_json::{Value, json};
use syn::{
    Attribute, Data, DeriveInput, Expr, ExprLit, Fields, Ident, Lit, LitStr, Meta, MetaNameValue,
    Token, Visibility,
    parse::{Parse, ParseStream},
    parse_macro_input,
    spanned::Spanned,
};

mod optim;

#[proc_macro_derive(Optimizable, attributes(parameter))]
pub fn derive_optimizable(input: TokenStream) -> TokenStream {
    optim::optimizable_impl(input)
}

#[proc_macro_derive(Signature, attributes(input, output, check, assert, alias))]
pub fn derive_signature(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match expand_signature(&input) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

fn expand_signature(input: &DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
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
    generate_signature_code(input, &parsed)
}

#[derive(Clone)]
struct ParsedField {
    ident: Ident,
    ty: syn::Type,
    is_input: bool,
    is_output: bool,
    description: String,
    alias: Option<String>,
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
    let mut alias = None;
    let mut constraints = Vec::new();
    let mut desc_override = None;

    for attr in &field.attrs {
        if attr.path().is_ident("input") {
            is_input = true;
            if let Some(desc) = parse_desc_from_attr(attr) {
                desc_override = Some(desc);
            }
        } else if attr.path().is_ident("output") {
            is_output = true;
            if let Some(desc) = parse_desc_from_attr(attr) {
                desc_override = Some(desc);
            }
        } else if attr.path().is_ident("alias") {
            alias = Some(parse_string_attr(attr)?);
        } else if attr.path().is_ident("check") {
            constraints.push(parse_constraint_attr(attr, ParsedConstraintKind::Check)?);
        } else if attr.path().is_ident("assert") {
            constraints.push(parse_constraint_attr(attr, ParsedConstraintKind::Assert)?);
        }
    }

    let doc_comment = collect_doc_comment(&field.attrs);
    let description = desc_override.unwrap_or(doc_comment);

    Ok(ParsedField {
        ident,
        ty: field.ty.clone(),
        is_input,
        is_output,
        description,
        alias,
        constraints,
    })
}

fn parse_desc_from_attr(attr: &Attribute) -> Option<String> {
    let list = attr.meta.require_list().ok()?;
    let desc = parse_desc_from_tokens(list.tokens.clone());
    if desc.is_empty() { None } else { Some(desc) }
}

fn parse_string_attr(attr: &Attribute) -> syn::Result<String> {
    match &attr.meta {
        Meta::NameValue(meta) => parse_string_expr(&meta.value, meta.span()),
        Meta::List(list) => {
            let lit: LitStr = list.parse_args()?;
            Ok(lit.value())
        }
        _ => Err(syn::Error::new_spanned(
            attr,
            "expected #[alias = \"...\"] or #[alias(\"...\")]",
        )),
    }
}

fn parse_constraint_attr(
    attr: &Attribute,
    kind: ParsedConstraintKind,
) -> syn::Result<ParsedConstraint> {
    let args: ConstraintArgs = attr.parse_args()?;
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
) -> syn::Result<proc_macro2::TokenStream> {
    let name = &input.ident;
    let vis = &input.vis;

    let helper_structs = generate_helper_structs(name, parsed, vis)?;
    let input_fields = generate_field_specs(name, &parsed.input_fields, "INPUT")?;
    let output_fields = generate_field_specs(name, &parsed.output_fields, "OUTPUT")?;
    let baml_delegation = generate_baml_delegation(name, parsed);
    let signature_impl = generate_signature_impl(name, parsed);

    Ok(quote! {
        #helper_structs
        #input_fields
        #output_fields
        #baml_delegation
        #signature_impl
    })
}

fn generate_helper_structs(
    name: &Ident,
    parsed: &ParsedSignature,
    vis: &Visibility,
) -> syn::Result<proc_macro2::TokenStream> {
    let input_name = format_ident!("{}Input", name);
    let output_name = format_ident!("__{}Output", name);
    let all_name = format_ident!("__{}All", name);

    let input_fields: Vec<_> = parsed.input_fields.iter().map(field_tokens).collect();
    let output_fields: Vec<_> = parsed.output_fields.iter().map(field_tokens).collect();
    let all_fields: Vec<_> = parsed.all_fields.iter().map(field_tokens).collect();

    Ok(quote! {
        #[derive(Debug, Clone, ::dspy_rs::BamlType)]
        #vis struct #input_name {
            #(#input_fields),*
        }

        #[derive(Debug, Clone, ::dspy_rs::BamlType)]
        pub struct #output_name {
            #(#output_fields),*
        }

        #[derive(Debug, Clone, ::dspy_rs::BamlType)]
        pub struct #all_name {
            #(#all_fields),*
        }
    })
}

fn field_tokens(field: &ParsedField) -> proc_macro2::TokenStream {
    let ident = &field.ident;
    let ty = &field.ty;
    let mut attrs = Vec::new();

    if !field.description.is_empty() {
        let doc = LitStr::new(&field.description, proc_macro2::Span::call_site());
        attrs.push(quote! { #[doc = #doc] });
    }

    if let Some(alias) = &field.alias {
        let alias = LitStr::new(alias, proc_macro2::Span::call_site());
        attrs.push(quote! { #[baml(alias = #alias)] });
    }

    for constraint in &field.constraints {
        let expr = LitStr::new(&constraint.expression, proc_macro2::Span::call_site());
        let label = constraint.label.as_deref().unwrap_or("");
        let label = LitStr::new(label, proc_macro2::Span::call_site());
        match constraint.kind {
            ParsedConstraintKind::Check => {
                attrs.push(quote! { #[baml(check(label = #label, expr = #expr))] });
            }
            ParsedConstraintKind::Assert => {
                attrs.push(quote! { #[baml(assert(label = #label, expr = #expr))] });
            }
        }
    }

    quote! {
        #(#attrs)*
        pub #ident: #ty
    }
}

fn generate_field_specs(
    name: &Ident,
    fields: &[ParsedField],
    kind: &str,
) -> syn::Result<proc_macro2::TokenStream> {
    let prefix = name.to_string().to_lowercase();
    let array_name = format_ident!("__{}_{}_FIELDS", name.to_string().to_uppercase(), kind);

    let mut type_ir_fns = Vec::new();
    let mut constraint_arrays = Vec::new();
    let mut field_specs = Vec::new();

    for field in fields {
        let field_name = field.ident.to_string();
        let field_name_ident = &field.ident;
        let ty = &field.ty;

        let llm_name = field.alias.as_ref().unwrap_or(&field_name);
        let llm_name = LitStr::new(llm_name, proc_macro2::Span::call_site());
        let rust_name = LitStr::new(&field_name, proc_macro2::Span::call_site());
        let description = LitStr::new(&field.description, proc_macro2::Span::call_site());

        let type_ir_fn_name = format_ident!("__{}_{}_type_ir", prefix, field_name_ident);

        if field.constraints.is_empty() {
            type_ir_fns.push(quote! {
                fn #type_ir_fn_name() -> ::dspy_rs::TypeIR {
                    <#ty as ::dspy_rs::baml_bridge::BamlTypeInternal>::baml_type_ir()
                }
            });
        } else {
            let constraint_tokens: Vec<_> = field
                .constraints
                .iter()
                .map(|constraint| {
                    let expr = LitStr::new(&constraint.expression, proc_macro2::Span::call_site());
                    let label = constraint.label.as_deref().unwrap_or("");
                    let label = LitStr::new(label, proc_macro2::Span::call_site());
                    match constraint.kind {
                        ParsedConstraintKind::Check => {
                            quote! { ::dspy_rs::Constraint::new_check(#label, #expr) }
                        }
                        ParsedConstraintKind::Assert => {
                            quote! { ::dspy_rs::Constraint::new_assert(#label, #expr) }
                        }
                    }
                })
                .collect();

            type_ir_fns.push(quote! {
                fn #type_ir_fn_name() -> ::dspy_rs::TypeIR {
                    let base = <#ty as ::dspy_rs::baml_bridge::BamlTypeInternal>::baml_type_ir();
                    ::dspy_rs::baml_bridge::with_constraints(base, vec![#(#constraint_tokens),*])
                }
            });
        }

        let constraints_name = format_ident!(
            "__{}_{}_CONSTRAINTS",
            name.to_string().to_uppercase(),
            field_name.to_uppercase(),
        );

        if field.constraints.is_empty() {
            constraint_arrays.push(quote! {
                const #constraints_name: &[::dspy_rs::ConstraintSpec] = &[];
            });
        } else {
            let constraint_specs: Vec<_> = field
                .constraints
                .iter()
                .map(|constraint| {
                    let kind = match constraint.kind {
                        ParsedConstraintKind::Check => quote! { ::dspy_rs::ConstraintKind::Check },
                        ParsedConstraintKind::Assert => {
                            quote! { ::dspy_rs::ConstraintKind::Assert }
                        }
                    };
                    let label = constraint.label.as_deref().unwrap_or("");
                    let label = LitStr::new(label, proc_macro2::Span::call_site());
                    let expr = LitStr::new(&constraint.expression, proc_macro2::Span::call_site());
                    quote! {
                        ::dspy_rs::ConstraintSpec {
                            kind: #kind,
                            label: #label,
                            expression: #expr,
                        }
                    }
                })
                .collect();

            constraint_arrays.push(quote! {
                const #constraints_name: &[::dspy_rs::ConstraintSpec] = &[
                    #(#constraint_specs),*
                ];
            });
        }

        field_specs.push(quote! {
            ::dspy_rs::FieldSpec {
                name: #llm_name,
                rust_name: #rust_name,
                description: #description,
                type_ir: #type_ir_fn_name,
                constraints: #constraints_name,
            }
        });
    }

    Ok(quote! {
        #(#type_ir_fns)*
        #(#constraint_arrays)*

        static #array_name: &[::dspy_rs::FieldSpec] = &[
            #(#field_specs),*
        ];
    })
}

fn generate_baml_delegation(name: &Ident, parsed: &ParsedSignature) -> proc_macro2::TokenStream {
    let all_name = format_ident!("__{}All", name);
    let field_names: Vec<_> = parsed.all_fields.iter().map(|field| &field.ident).collect();

    let mut to_value_inserts = Vec::new();
    for field in &parsed.all_fields {
        let field_name = field.ident.to_string();
        let ident = &field.ident;
        to_value_inserts.push(quote! {
            fields.insert(
                #field_name.to_string(),
                ::dspy_rs::baml_bridge::ToBamlValue::to_baml_value(&self.#ident),
            );
        });
    }

    quote! {
        impl ::dspy_rs::baml_bridge::BamlTypeInternal for #name {
            fn baml_internal_name() -> &'static str {
                <#all_name as ::dspy_rs::baml_bridge::BamlTypeInternal>::baml_internal_name()
            }

            fn baml_type_ir() -> ::dspy_rs::TypeIR {
                <#all_name as ::dspy_rs::baml_bridge::BamlTypeInternal>::baml_type_ir()
            }

            fn register(reg: &mut ::dspy_rs::baml_bridge::Registry) {
                <#all_name as ::dspy_rs::baml_bridge::BamlTypeInternal>::register(reg)
            }
        }

        impl ::dspy_rs::baml_bridge::BamlValueConvert for #name {
            fn try_from_baml_value(
                value: ::dspy_rs::BamlValue,
                path: Vec<String>,
            ) -> Result<Self, ::dspy_rs::BamlConvertError> {
                let all = <#all_name as ::dspy_rs::baml_bridge::BamlValueConvert>
                    ::try_from_baml_value(value, path)?;
                Ok(Self {
                    #(#field_names: all.#field_names),*
                })
            }
        }

        impl ::dspy_rs::baml_bridge::BamlType for #name {
            fn baml_output_format() -> &'static ::dspy_rs::OutputFormatContent {
                <#all_name as ::dspy_rs::baml_bridge::BamlType>::baml_output_format()
            }
        }

        impl ::dspy_rs::baml_bridge::ToBamlValue for #name {
            fn to_baml_value(&self) -> ::dspy_rs::BamlValue {
                let mut fields = ::dspy_rs::baml_bridge::baml_types::BamlMap::new();
                #(#to_value_inserts)*
                ::dspy_rs::baml_bridge::baml_types::BamlValue::Class(
                    <Self as ::dspy_rs::baml_bridge::BamlTypeInternal>::baml_internal_name().to_string(),
                    fields,
                )
            }
        }
    }
}

fn generate_signature_impl(name: &Ident, parsed: &ParsedSignature) -> proc_macro2::TokenStream {
    let input_name = format_ident!("{}Input", name);
    let output_name = format_ident!("__{}Output", name);

    let instruction = LitStr::new(&parsed.instruction, proc_macro2::Span::call_site());

    let input_field_names: Vec<_> = parsed
        .input_fields
        .iter()
        .map(|field| &field.ident)
        .collect();
    let output_field_names: Vec<_> = parsed
        .output_fields
        .iter()
        .map(|field| &field.ident)
        .collect();

    let input_fields_static = format_ident!("__{}_INPUT_FIELDS", name.to_string().to_uppercase());
    let output_fields_static = format_ident!("__{}_OUTPUT_FIELDS", name.to_string().to_uppercase());

    quote! {
        impl ::dspy_rs::Signature for #name {
            type Input = #input_name;
            type Output = #output_name;

            fn instruction() -> &'static str {
                #instruction
            }

            fn input_fields() -> &'static [::dspy_rs::FieldSpec] {
                &#input_fields_static
            }

            fn output_fields() -> &'static [::dspy_rs::FieldSpec] {
                &#output_fields_static
            }

            fn output_format_content() -> &'static ::dspy_rs::OutputFormatContent {
                <#output_name as ::dspy_rs::baml_bridge::BamlType>::baml_output_format()
            }

            fn from_parts(input: Self::Input, output: Self::Output) -> Self {
                Self {
                    #(#input_field_names: input.#input_field_names),*,
                    #(#output_field_names: output.#output_field_names),*
                }
            }

            fn into_parts(self) -> (Self::Input, Self::Output) {
                (
                    #input_name {
                        #(#input_field_names: self.#input_field_names),*
                    },
                    #output_name {
                        #(#output_field_names: self.#output_field_names),*
                    },
                )
            }
        }
    }
}

#[allow(unused_assignments, non_snake_case)]
#[proc_macro_attribute]
pub fn LegacySignature(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as DeriveInput);

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
                    let field_name = field.ident.as_ref().unwrap().clone();
                    let field_type = field.ty.clone();

                    // Check for #[input] or #[output] attributes
                    let (is_input, desc) = has_in_attribute(&field.attrs);
                    let (is_output, desc2) = has_out_attribute(&field.attrs);

                    if is_input && is_output {
                        panic!("Field {field_name} cannot be both input and output");
                    }

                    if !is_input && !is_output {
                        panic!(
                            "Field {field_name} must have either #[input] or #[output] attribute"
                        );
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
                                    let schema = schemars::schema_for!(#field_type);
                                    let schema_json = serde_json::to_value(schema).unwrap();
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
                                    let schema = schemars::schema_for!(#field_type);
                                    let schema_json = serde_json::to_value(schema).unwrap();
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
        _ => panic!("Signature can only be applied to structs"),
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
        #[derive(Default, Debug, Clone, serde::Serialize, serde::Deserialize)]
        struct #struct_name {
            instruction: String,
            input_fields: serde_json::Value,
            output_fields: serde_json::Value,
            demos: Vec<dspy_rs::Example>,
        }

        impl #struct_name {
            pub fn new() -> Self {
                let mut input_fields: serde_json::Value = serde_json::from_str(#input_schema_str).unwrap();
                let mut output_fields: serde_json::Value = serde_json::from_str(#output_schema_str).unwrap();

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

        impl dspy_rs::core::MetaSignature for #struct_name {
            fn demos(&self) -> Vec<dspy_rs::Example> {
                self.demos.clone()
            }

            fn set_demos(&mut self, demos: Vec<dspy_rs::Example>) -> anyhow::Result<()> {
                self.demos = demos;
                Ok(())
            }

            fn instruction(&self) -> String {
                self.instruction.clone()
            }

            fn input_fields(&self) -> serde_json::Value {
                self.input_fields.clone()
            }

            fn output_fields(&self) -> serde_json::Value {
                self.output_fields.clone()
            }

            fn update_instruction(&mut self, instruction: String) -> anyhow::Result<()> {
                self.instruction = instruction;
                Ok(())
            }

            fn append(&mut self, name: &str, field_value: serde_json::Value) -> anyhow::Result<()> {
                match field_value["__dsrs_field_type"].as_str() {
                    Some("input") => {
                        self.input_fields[name] = field_value;
                    }
                    Some("output") => {
                        self.output_fields[name] = field_value;
                    }
                    _ => {
                        return Err(anyhow::anyhow!("Invalid field type: {:?}", field_value["__dsrs_field_type"].as_str()));
                    }
                }
                Ok(())
            }
        }
    };

    generated.into()
}

fn has_in_attribute(attrs: &[Attribute]) -> (bool, String) {
    for attr in attrs {
        if attr.path().is_ident("input") {
            // Try to parse desc parameter
            if let Ok(list) = attr.meta.require_list() {
                let desc = parse_desc_from_tokens(list.tokens.clone());
                return (true, desc);
            } else {
                // Just #[input] without parameters
                return (true, String::new());
            }
        }
    }
    (false, String::new())
}

fn has_out_attribute(attrs: &[Attribute]) -> (bool, String) {
    for attr in attrs {
        if attr.path().is_ident("output") {
            // Try to parse desc parameter
            if let Ok(list) = attr.meta.require_list() {
                let desc = parse_desc_from_tokens(list.tokens.clone());
                return (true, desc);
            } else {
                // Just #[output] without parameters
                return (true, String::new());
            }
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
