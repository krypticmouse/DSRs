use proc_macro::TokenStream;
use std::collections::BTreeSet;

use minijinja::machinery as jinja_machinery;
use minijinja::machinery::ast as jinja_ast;
use quote::quote;
use syn::{
    parse_macro_input, spanned::Spanned, Attribute, Data, DataEnum, DataStruct, DeriveInput, Expr,
    ExprLit, Fields, Lit, LitStr, Meta, Path, Type,
};

#[proc_macro_derive(BamlType, attributes(baml, serde, render))]
pub fn derive_baml_type(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match expand_derive(&input) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

fn expand_derive(input: &DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    match &input.data {
        Data::Struct(data) => derive_struct(input, data),
        Data::Enum(data) => derive_enum(input, data),
        Data::Union(_) => Err(syn::Error::new_spanned(
            input,
            "BamlType does not support `union` items; hint: use a struct or enum instead",
        )),
    }
}

#[derive(Default)]
struct ContainerAttrs {
    name: Option<String>,
    internal_name: Option<String>,
    rename_all: Option<RenameRule>,
    constraints: Vec<ConstraintSpec>,
    tag: Option<String>,
    as_union: bool,
    as_enum: bool,
    description: Option<String>,
    render_attrs: Vec<RenderAttr>,
}

#[derive(Default)]
struct FieldAttrs {
    alias: Option<String>,
    skip: bool,
    default: bool,
    with: Option<Path>,
    constraints: Vec<ConstraintSpec>,
    description: Option<String>,
    int_repr: Option<IntRepr>,
    map_key_repr: Option<MapKeyRepr>,
}

#[derive(Default)]
struct VariantAttrs {
    alias: Option<String>,
    description: Option<String>,
}

struct RenderAttr {
    default: Option<LitStr>,
    style: Option<LitStr>,
    template: Option<LitStr>,
    func: Option<Path>,
    allow_dynamic: bool,
    span: proc_macro2::Span,
}

impl Default for RenderAttr {
    fn default() -> Self {
        Self {
            default: None,
            style: None,
            template: None,
            func: None,
            allow_dynamic: false,
            span: proc_macro2::Span::call_site(),
        }
    }
}

enum RenderTarget {
    Class,
    Enum,
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

#[derive(Clone, Copy)]
enum IntRepr {
    String,
    I64,
}

#[derive(Clone, Copy)]
enum MapKeyRepr {
    String,
    Pairs,
}

struct ConstraintSpec {
    level: ConstraintLevelSpec,
    label: String,
    expr: String,
}

struct MapEntryInfo {
    internal_name_expr: proc_macro2::TokenStream,
    rendered_name: Option<String>,
}

#[derive(Clone, Copy)]
enum ConstraintLevelSpec {
    Check,
    Assert,
}

fn derive_struct(input: &DeriveInput, data: &DataStruct) -> syn::Result<proc_macro2::TokenStream> {
    let name = &input.ident;
    let container_attrs = parse_container_attrs(&input.attrs)?;

    let internal_name_expr = container_attrs
        .internal_name
        .as_ref()
        .map(|name| quote! { #name })
        .unwrap_or_else(|| quote! { concat!(module_path!(), "::", stringify!(#name)) });
    let rendered_name = container_attrs
        .name
        .clone()
        .unwrap_or_else(|| name.to_string());

    let rename_all = container_attrs.rename_all;

    let fields = match &data.fields {
        Fields::Named(fields) => fields,
        Fields::Unit => {
            return Err(syn::Error::new_spanned(
                input,
                "Unit structs are not supported for BAML outputs; hint: use a named-field struct or enum",
            ))
        }
        Fields::Unnamed(_) => {
            return Err(syn::Error::new_spanned(
                input,
                "Tuple structs are not supported for BAML outputs; hint: use a named-field struct",
            ))
        }
    };

    validate_render_templates(&container_attrs.render_attrs, Some(&fields.named))?;

    let mut field_defs = Vec::new();
    let mut register_calls = Vec::new();
    let mut field_inits = Vec::new();
    let mut to_value_inserts = Vec::new();
    let mut field_idents = Vec::new();

    for field in &fields.named {
        let ident = field.ident.as_ref().unwrap();
        let field_name = ident.to_string();
        let field_attrs = parse_field_attrs(&field.attrs)?;

        if field_attrs.skip {
            field_inits.push(quote! { #ident: ::std::default::Default::default() });
            field_idents.push(ident);
            continue;
        }

        let field_desc = field_description(&field_attrs, &field.attrs);
        let field_alias = field_alias(&field_attrs, rename_all, &field_name);
        let rendered_field = field_alias.clone().unwrap_or_else(|| field_name.clone());
        let map_entry_info =
            map_entry_info(&field_name, &rendered_field, None, None, &field_attrs)?;

        let field_type_ir = type_ir_for_field(&field.ty, &field_attrs, map_entry_info.as_ref())?;
        let field_type_ir = with_constraints_tokens(field_type_ir, &field_attrs.constraints);

        let name_expr = name_tokens(&field_name, field_alias.as_deref());
        let desc_expr = match field_desc {
            Some(desc) => quote! { Some(#desc.to_string()) },
            None => quote! { None },
        };

        field_defs.push(quote! {
            (#name_expr, #field_type_ir, #desc_expr, false)
        });

        register_calls.push(register_call_tokens(
            &field.ty,
            &field_attrs,
            map_entry_info.as_ref(),
        )?);

        let conversion =
            field_conversion_tokens(&field.ty, &field_attrs, &field_name, field_alias.as_deref())?;
        field_inits.push(quote! { #ident: #conversion });
        let to_value = field_to_value_tokens(&field.ty, &field_attrs, quote! { &self.#ident })?;
        to_value_inserts.push(quote! {
            fields.insert(#field_name.to_string(), #to_value);
        });
        field_idents.push(ident);
    }

    let render_calls = render_registration_tokens(
        &container_attrs.render_attrs,
        RenderTarget::Class,
        &internal_name_expr,
    )?;
    register_calls.extend(render_calls);

    let type_constraints = constraints_tokens(&container_attrs.constraints);
    let type_ir = quote! {{
        let mut r#type = ::dspy_rs::baml_bridge::baml_types::TypeIR::class(
            <Self as ::dspy_rs::baml_bridge::BamlTypeInternal>::baml_internal_name(),
        );
        r#type.meta_mut().constraints.extend(#type_constraints);
        r#type
    }};

    let class_name_expr = name_tokens_expr(
        quote! { <Self as ::dspy_rs::baml_bridge::BamlTypeInternal>::baml_internal_name().to_string() },
        Some(&rendered_name),
    );
    let class_desc = container_description(&container_attrs, &input.attrs);
    let class_desc_expr = match class_desc {
        Some(desc) => quote! { Some(#desc.to_string()) },
        None => quote! { None },
    };

    let class_def = quote! {
        ::dspy_rs::baml_bridge::internal_baml_jinja::types::Class {
            name: #class_name_expr,
            description: #class_desc_expr,
            namespace: ::dspy_rs::baml_bridge::baml_types::StreamingMode::NonStreaming,
            fields: vec![#(#field_defs),*],
            constraints: #type_constraints,
            streaming_behavior: ::dspy_rs::baml_bridge::default_streaming_behavior(),
        }
    };

    let register_impl = quote! {
        impl ::dspy_rs::baml_bridge::BamlTypeInternal for #name {
            fn baml_internal_name() -> &'static str {
                #internal_name_expr
            }

            fn baml_type_ir() -> ::dspy_rs::baml_bridge::baml_types::TypeIR {
                #type_ir
            }

            fn register(reg: &mut ::dspy_rs::baml_bridge::Registry) {
                if !reg.mark_type(<Self as ::dspy_rs::baml_bridge::BamlTypeInternal>::baml_internal_name()) {
                    return;
                }
                reg.register_class(#class_def);
                #(#register_calls;)*
            }
        }
    };

    let value_convert_impl = quote! {
        impl ::dspy_rs::baml_bridge::BamlValueConvert for #name {
            fn try_from_baml_value(
                value: ::dspy_rs::baml_bridge::baml_types::BamlValue,
                path: Vec<String>,
            ) -> Result<Self, ::dspy_rs::baml_bridge::BamlConvertError> {
                let map = match value {
                    ::dspy_rs::baml_bridge::baml_types::BamlValue::Class(_, map)
                    | ::dspy_rs::baml_bridge::baml_types::BamlValue::Map(map) => map,
                    other => {
                        return Err(::dspy_rs::baml_bridge::BamlConvertError::new(
                            path,
                            "object",
                            format!("{other:?}"),
                            "expected an object",
                        ))
                    }
                };

                Ok(Self {
                    #(#field_inits),*
                })
            }
        }
    };

    let to_value_impl = quote! {
        impl ::dspy_rs::baml_bridge::ToBamlValue for #name {
            fn to_baml_value(&self) -> ::dspy_rs::baml_bridge::baml_types::BamlValue {
                let mut fields = ::dspy_rs::baml_bridge::baml_types::BamlMap::new();
                #(#to_value_inserts)*
                ::dspy_rs::baml_bridge::baml_types::BamlValue::Class(
                    <Self as ::dspy_rs::baml_bridge::BamlTypeInternal>::baml_internal_name().to_string(),
                    fields,
                )
            }
        }
    };

    let output_format_impl = quote! {
        impl ::dspy_rs::baml_bridge::BamlType for #name {
            fn baml_output_format() -> &'static ::dspy_rs::baml_bridge::internal_baml_jinja::types::OutputFormatContent {
                static OUTPUT_FORMAT: ::std::sync::OnceLock<::dspy_rs::baml_bridge::internal_baml_jinja::types::OutputFormatContent> = ::std::sync::OnceLock::new();
                OUTPUT_FORMAT.get_or_init(|| {
                    let mut reg = ::dspy_rs::baml_bridge::Registry::new();
                    <Self as ::dspy_rs::baml_bridge::BamlTypeInternal>::register(&mut reg);
                    reg.build(<Self as ::dspy_rs::baml_bridge::BamlTypeInternal>::baml_type_ir())
                })
            }
        }
    };

    Ok(quote! {
        #register_impl
        #value_convert_impl
        #to_value_impl
        #output_format_impl
    })
}

fn derive_enum(input: &DeriveInput, data: &DataEnum) -> syn::Result<proc_macro2::TokenStream> {
    let name = &input.ident;
    let container_attrs = parse_container_attrs(&input.attrs)?;

    let internal_name_expr = container_attrs
        .internal_name
        .as_ref()
        .map(|name| quote! { #name })
        .unwrap_or_else(|| quote! { concat!(module_path!(), "::", stringify!(#name)) });
    let rendered_name = container_attrs
        .name
        .clone()
        .unwrap_or_else(|| name.to_string());

    let rename_all = container_attrs.rename_all;

    validate_render_templates(&container_attrs.render_attrs, None)?;

    let mut has_data_variant = false;
    for variant in &data.variants {
        match &variant.fields {
            Fields::Unit => {}
            Fields::Named(_) => has_data_variant = true,
            Fields::Unnamed(_) => return Err(syn::Error::new_spanned(
                variant,
                "Tuple enum variants are not supported; hint: use a unit or struct-like variant",
            )),
        }
    }

    if container_attrs.as_enum && has_data_variant {
        return Err(syn::Error::new_spanned(
            input,
            "as_enum is only valid for unit enums; hint: remove #[baml(as_enum)] or convert variants to unit",
        ));
    }

    let type_constraints = constraints_tokens(&container_attrs.constraints);

    if !has_data_variant {
        derive_unit_enum(
            name,
            data,
            &internal_name_expr,
            &rendered_name,
            rename_all,
            &container_attrs,
            &input.attrs,
            type_constraints,
        )
    } else {
        derive_data_enum(
            name,
            data,
            &internal_name_expr,
            &rendered_name,
            rename_all,
            &container_attrs,
            type_constraints,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn derive_unit_enum(
    name: &syn::Ident,
    data: &DataEnum,
    internal_name_expr: &proc_macro2::TokenStream,
    rendered_name: &str,
    rename_all: Option<RenameRule>,
    container_attrs: &ContainerAttrs,
    raw_attrs: &[Attribute],
    type_constraints: proc_macro2::TokenStream,
) -> syn::Result<proc_macro2::TokenStream> {
    if container_attrs.as_union {
        return derive_unit_enum_as_union(
            name,
            data,
            internal_name_expr,
            rendered_name,
            rename_all,
            container_attrs,
            type_constraints,
        );
    }

    let mut values = Vec::new();
    let mut match_arms = Vec::new();
    let mut to_value_match_arms = Vec::new();

    for variant in &data.variants {
        let variant_ident = &variant.ident;
        let variant_name = variant_ident.to_string();
        let variant_attrs = parse_variant_attrs(&variant.attrs)?;
        let rendered_variant = variant_alias(&variant_attrs, rename_all, &variant_name);
        let variant_desc = variant_description(&variant_attrs, &variant.attrs);

        let name_expr = name_tokens(&variant_name, rendered_variant.as_deref());
        let desc_expr = match variant_desc {
            Some(desc) => quote! { Some(#desc.to_string()) },
            None => quote! { None },
        };
        values.push(quote! { (#name_expr, #desc_expr) });

        let match_values = match_strings_for_variant(&variant_name, rendered_variant.as_deref());
        match_arms.push(quote! { #(#match_values)|* => Ok(Self::#variant_ident) });
        let rendered_variant_value = rendered_variant
            .clone()
            .unwrap_or_else(|| variant_name.clone());
        to_value_match_arms.push(quote! {
            Self::#variant_ident => ::dspy_rs::baml_bridge::baml_types::BamlValue::Enum(
                <Self as ::dspy_rs::baml_bridge::BamlTypeInternal>::baml_internal_name().to_string(),
                #rendered_variant_value.to_string(),
            )
        });
    }

    let enum_name_expr = name_tokens_expr(
        quote! { #internal_name_expr.to_string() },
        Some(rendered_name),
    );
    let enum_desc = container_description(container_attrs, raw_attrs);
    let enum_desc_expr = match enum_desc {
        Some(desc) => quote! { Some(#desc.to_string()) },
        None => quote! { None },
    };

    let type_ir = quote! {{
        let mut r#type = ::dspy_rs::baml_bridge::baml_types::TypeIR::r#enum(
            <Self as ::dspy_rs::baml_bridge::BamlTypeInternal>::baml_internal_name(),
        );
        r#type.meta_mut().constraints.extend(#type_constraints);
        r#type
    }};

    let render_calls = render_registration_tokens(
        &container_attrs.render_attrs,
        RenderTarget::Enum,
        internal_name_expr,
    )?;

    let register_impl = quote! {
        impl ::dspy_rs::baml_bridge::BamlTypeInternal for #name {
            fn baml_internal_name() -> &'static str {
                #internal_name_expr
            }

            fn baml_type_ir() -> ::dspy_rs::baml_bridge::baml_types::TypeIR {
                #type_ir
            }

            fn register(reg: &mut ::dspy_rs::baml_bridge::Registry) {
                if !reg.mark_type(<Self as ::dspy_rs::baml_bridge::BamlTypeInternal>::baml_internal_name()) {
                    return;
                }
                reg.register_enum(::dspy_rs::baml_bridge::internal_baml_jinja::types::Enum {
                    name: #enum_name_expr,
                    description: #enum_desc_expr,
                    values: vec![#(#values),*],
                    constraints: #type_constraints,
                });
                #(#render_calls;)*
            }
        }
    };

    let value_convert_impl = quote! {
        impl ::dspy_rs::baml_bridge::BamlValueConvert for #name {
            fn try_from_baml_value(
                value: ::dspy_rs::baml_bridge::baml_types::BamlValue,
                path: Vec<String>,
            ) -> Result<Self, ::dspy_rs::baml_bridge::BamlConvertError> {
                let value = match value {
                    ::dspy_rs::baml_bridge::baml_types::BamlValue::Enum(_, value)
                    | ::dspy_rs::baml_bridge::baml_types::BamlValue::String(value) => value,
                    other => {
                        return Err(::dspy_rs::baml_bridge::BamlConvertError::new(
                            path,
                            "enum",
                            format!("{other:?}"),
                            "expected an enum value",
                        ))
                    }
                };

                match value.as_str() {
                    #(#match_arms),*,
                    _ => Err(::dspy_rs::baml_bridge::BamlConvertError::new(
                        path,
                        "enum",
                        value,
                        "unknown enum variant",
                    )),
                }
            }
        }
    };

    let to_value_impl = quote! {
        impl ::dspy_rs::baml_bridge::ToBamlValue for #name {
            fn to_baml_value(&self) -> ::dspy_rs::baml_bridge::baml_types::BamlValue {
                match self {
                    #(#to_value_match_arms),*
                }
            }
        }
    };

    let output_format_impl = quote! {
        impl ::dspy_rs::baml_bridge::BamlType for #name {
            fn baml_output_format() -> &'static ::dspy_rs::baml_bridge::internal_baml_jinja::types::OutputFormatContent {
                static OUTPUT_FORMAT: ::std::sync::OnceLock<::dspy_rs::baml_bridge::internal_baml_jinja::types::OutputFormatContent> = ::std::sync::OnceLock::new();
                OUTPUT_FORMAT.get_or_init(|| {
                    let mut reg = ::dspy_rs::baml_bridge::Registry::new();
                    <Self as ::dspy_rs::baml_bridge::BamlTypeInternal>::register(&mut reg);
                    reg.build(<Self as ::dspy_rs::baml_bridge::BamlTypeInternal>::baml_type_ir())
                })
            }
        }
    };

    Ok(quote! {
        #register_impl
        #value_convert_impl
        #to_value_impl
        #output_format_impl
    })
}

fn derive_unit_enum_as_union(
    name: &syn::Ident,
    data: &DataEnum,
    internal_name_expr: &proc_macro2::TokenStream,
    _rendered_name: &str,
    rename_all: Option<RenameRule>,
    container_attrs: &ContainerAttrs,
    type_constraints: proc_macro2::TokenStream,
) -> syn::Result<proc_macro2::TokenStream> {
    let mut literals = Vec::new();
    let mut match_arms = Vec::new();
    let mut to_value_match_arms = Vec::new();

    for variant in &data.variants {
        let variant_ident = &variant.ident;
        let variant_name = variant_ident.to_string();
        let variant_attrs = parse_variant_attrs(&variant.attrs)?;
        let rendered_variant = variant_alias(&variant_attrs, rename_all, &variant_name);
        let literal = rendered_variant
            .clone()
            .unwrap_or_else(|| variant_name.clone());

        literals.push(
            quote! { ::dspy_rs::baml_bridge::baml_types::TypeIR::literal_string(#literal.to_string()) },
        );
        let match_values = match_strings_for_variant(&variant_name, rendered_variant.as_deref());
        match_arms.push(quote! { #(#match_values)|* => Ok(Self::#variant_ident) });
        to_value_match_arms.push(quote! {
            Self::#variant_ident => ::dspy_rs::baml_bridge::baml_types::BamlValue::Enum(
                <Self as ::dspy_rs::baml_bridge::BamlTypeInternal>::baml_internal_name().to_string(),
                #literal.to_string(),
            )
        });
    }

    let type_ir = quote! {{
        let mut r#type = ::dspy_rs::baml_bridge::baml_types::TypeIR::union_with_meta(
            vec![#(#literals),*],
            ::dspy_rs::baml_bridge::baml_types::type_meta::IR::default(),
        );
        r#type.meta_mut().constraints.extend(#type_constraints);
        r#type
    }};

    let render_calls = render_registration_tokens(
        &container_attrs.render_attrs,
        RenderTarget::Enum,
        internal_name_expr,
    )?;

    let register_impl = quote! {
        impl ::dspy_rs::baml_bridge::BamlTypeInternal for #name {
            fn baml_internal_name() -> &'static str {
                #internal_name_expr
            }

            fn baml_type_ir() -> ::dspy_rs::baml_bridge::baml_types::TypeIR {
                #type_ir
            }

            fn register(reg: &mut ::dspy_rs::baml_bridge::Registry) {
                if !reg.mark_type(<Self as ::dspy_rs::baml_bridge::BamlTypeInternal>::baml_internal_name()) {
                    return;
                }
                #(#render_calls;)*
            }
        }
    };

    let value_convert_impl = quote! {
        impl ::dspy_rs::baml_bridge::BamlValueConvert for #name {
            fn try_from_baml_value(
                value: ::dspy_rs::baml_bridge::baml_types::BamlValue,
                path: Vec<String>,
            ) -> Result<Self, ::dspy_rs::baml_bridge::BamlConvertError> {
                let value = match value {
                    ::dspy_rs::baml_bridge::baml_types::BamlValue::Enum(_, value)
                    | ::dspy_rs::baml_bridge::baml_types::BamlValue::String(value) => value,
                    other => {
                        return Err(::dspy_rs::baml_bridge::BamlConvertError::new(
                            path,
                            "enum",
                            format!("{other:?}"),
                            "expected an enum value",
                        ))
                    }
                };

                match value.as_str() {
                    #(#match_arms),*,
                    _ => Err(::dspy_rs::baml_bridge::BamlConvertError::new(
                        path,
                        "enum",
                        value,
                        "unknown enum variant",
                    )),
                }
            }
        }
    };

    let to_value_impl = quote! {
        impl ::dspy_rs::baml_bridge::ToBamlValue for #name {
            fn to_baml_value(&self) -> ::dspy_rs::baml_bridge::baml_types::BamlValue {
                match self {
                    #(#to_value_match_arms),*
                }
            }
        }
    };

    let output_format_impl = quote! {
        impl ::dspy_rs::baml_bridge::BamlType for #name {
            fn baml_output_format() -> &'static ::dspy_rs::baml_bridge::internal_baml_jinja::types::OutputFormatContent {
                static OUTPUT_FORMAT: ::std::sync::OnceLock<::dspy_rs::baml_bridge::internal_baml_jinja::types::OutputFormatContent> = ::std::sync::OnceLock::new();
                OUTPUT_FORMAT.get_or_init(|| {
                    let mut reg = ::dspy_rs::baml_bridge::Registry::new();
                    <Self as ::dspy_rs::baml_bridge::BamlTypeInternal>::register(&mut reg);
                    reg.build(<Self as ::dspy_rs::baml_bridge::BamlTypeInternal>::baml_type_ir())
                })
            }
        }
    };

    Ok(quote! {
        #register_impl
        #value_convert_impl
        #to_value_impl
        #output_format_impl
    })
}

fn derive_data_enum(
    name: &syn::Ident,
    data: &DataEnum,
    internal_name_expr: &proc_macro2::TokenStream,
    rendered_name: &str,
    rename_all: Option<RenameRule>,
    container_attrs: &ContainerAttrs,
    type_constraints: proc_macro2::TokenStream,
) -> syn::Result<proc_macro2::TokenStream> {
    let tag = container_attrs
        .tag
        .clone()
        .unwrap_or_else(|| "type".to_string());

    let mut variant_classes = Vec::new();
    let mut register_calls = Vec::new();
    let mut match_arms = Vec::new();
    let mut to_value_match_arms = Vec::new();
    let mut union_variants = Vec::new();

    for variant in &data.variants {
        let variant_ident = &variant.ident;
        let variant_name = variant_ident.to_string();
        let variant_attrs = parse_variant_attrs(&variant.attrs)?;
        let rendered_variant = variant_alias(&variant_attrs, rename_all, &variant_name)
            .unwrap_or_else(|| variant_name.clone());

        let variant_rendered_name = format!("{}_{}", rendered_name, rendered_variant);

        let variant_desc = variant_description(&variant_attrs, &variant.attrs);
        let variant_desc_expr = match variant_desc {
            Some(desc) => quote! { Some(#desc.to_string()) },
            None => quote! { None },
        };

        let mut fields = Vec::new();
        let mut to_value_inserts = Vec::new();
        let mut to_value_bindings = Vec::new();
        let tag_field_name = name_tokens(&tag, None);
        let tag_field_type = quote! { ::dspy_rs::baml_bridge::baml_types::TypeIR::literal_string(#rendered_variant.to_string()) };
        fields.push(quote! { (#tag_field_name, #tag_field_type, None, false) });

        match &variant.fields {
            Fields::Unit => {}
            Fields::Named(variant_fields) => {
                for field in &variant_fields.named {
                    let ident = field.ident.as_ref().unwrap();
                    let field_name = ident.to_string();
                    let field_attrs = parse_field_attrs(&field.attrs)?;

                    if field_attrs.skip {
                        continue;
                    }

                    let field_desc = field_description(&field_attrs, &field.attrs);
                    let field_alias = field_alias(&field_attrs, rename_all, &field_name);
                    let rendered_field = field_alias.clone().unwrap_or_else(|| field_name.clone());
                    let map_entry_info = map_entry_info(
                        &field_name,
                        &rendered_field,
                        Some(&variant_name),
                        Some(&rendered_variant),
                        &field_attrs,
                    )?;

                    let field_type_ir =
                        type_ir_for_field(&field.ty, &field_attrs, map_entry_info.as_ref())?;
                    let field_type_ir =
                        with_constraints_tokens(field_type_ir, &field_attrs.constraints);

                    let name_expr = name_tokens(&field_name, field_alias.as_deref());
                    let desc_expr = match field_desc {
                        Some(desc) => quote! { Some(#desc.to_string()) },
                        None => quote! { None },
                    };

                    fields.push(quote! { (#name_expr, #field_type_ir, #desc_expr, false) });
                    register_calls.push(register_call_tokens(
                        &field.ty,
                        &field_attrs,
                        map_entry_info.as_ref(),
                    )?);
                    to_value_bindings.push(ident);
                    let to_value =
                        field_to_value_tokens(&field.ty, &field_attrs, quote! { &#ident })?;
                    to_value_inserts.push(quote! {
                        fields.insert(#field_name.to_string(), #to_value);
                    });
                }
            }
            Fields::Unnamed(_) => return Err(syn::Error::new_spanned(
                variant,
                "Tuple enum variants are not supported; hint: use a unit or struct-like variant",
            )),
        }

        let class_name_expr = name_tokens_expr(
            quote! {
                format!(
                    "{}__{}",
                    <Self as ::dspy_rs::baml_bridge::BamlTypeInternal>::baml_internal_name(),
                    #variant_name
                )
            },
            Some(&variant_rendered_name),
        );
        let class_def = quote! {
            ::dspy_rs::baml_bridge::internal_baml_jinja::types::Class {
                name: #class_name_expr,
                description: #variant_desc_expr,
                namespace: ::dspy_rs::baml_bridge::baml_types::StreamingMode::NonStreaming,
                fields: vec![#(#fields),*],
                constraints: Vec::new(),
                streaming_behavior: ::dspy_rs::baml_bridge::default_streaming_behavior(),
            }
        };

        variant_classes.push(class_def);
        union_variants.push(quote! {
            ::dspy_rs::baml_bridge::baml_types::TypeIR::class(format!(
                "{}__{}",
                <Self as ::dspy_rs::baml_bridge::BamlTypeInternal>::baml_internal_name(),
                #variant_name
            ))
        });

        let match_values = match_strings_for_variant(&variant_name, Some(&rendered_variant));
        let parse_variant =
            variant_parse_tokens(variant, &variant_name, Some(&rendered_variant), rename_all)?;
        match_arms.push(quote! { #(#match_values)|* => #parse_variant });

        let tag_literal = rendered_variant.clone();
        let pattern = match &variant.fields {
            Fields::Unit => quote! { Self::#variant_ident },
            Fields::Named(_) if to_value_bindings.is_empty() => {
                quote! { Self::#variant_ident { .. } }
            }
            Fields::Named(_) => quote! { Self::#variant_ident { #(#to_value_bindings),*, .. } },
            Fields::Unnamed(_) => return Err(syn::Error::new_spanned(
                variant,
                "Tuple enum variants are not supported; hint: use a unit or struct-like variant",
            )),
        };

        to_value_match_arms.push(quote! {
            #pattern => {
                let mut fields = ::dspy_rs::baml_bridge::baml_types::BamlMap::new();
                fields.insert(
                    #tag.to_string(),
                    ::dspy_rs::baml_bridge::baml_types::BamlValue::String(#tag_literal.to_string()),
                );
                #(#to_value_inserts)*
                ::dspy_rs::baml_bridge::baml_types::BamlValue::Class(
                    <Self as ::dspy_rs::baml_bridge::BamlTypeInternal>::baml_internal_name().to_string(),
                    fields,
                )
            }
        });
    }

    let type_ir = quote! {{
        let mut r#type = ::dspy_rs::baml_bridge::baml_types::TypeIR::union_with_meta(
            vec![#(#union_variants),*],
            ::dspy_rs::baml_bridge::baml_types::type_meta::IR::default(),
        );
        r#type.meta_mut().constraints.extend(#type_constraints);
        r#type
    }};

    let render_calls = render_registration_tokens(
        &container_attrs.render_attrs,
        RenderTarget::Enum,
        internal_name_expr,
    )?;

    let register_impl = quote! {
        impl ::dspy_rs::baml_bridge::BamlTypeInternal for #name {
            fn baml_internal_name() -> &'static str {
                #internal_name_expr
            }

            fn baml_type_ir() -> ::dspy_rs::baml_bridge::baml_types::TypeIR {
                #type_ir
            }

            fn register(reg: &mut ::dspy_rs::baml_bridge::Registry) {
                if !reg.mark_type(<Self as ::dspy_rs::baml_bridge::BamlTypeInternal>::baml_internal_name()) {
                    return;
                }
                #(reg.register_class(#variant_classes);)*
                #(#register_calls;)*
                #(#render_calls;)*
            }
        }
    };

    let value_convert_impl = quote! {
        impl ::dspy_rs::baml_bridge::BamlValueConvert for #name {
            fn try_from_baml_value(
                value: ::dspy_rs::baml_bridge::baml_types::BamlValue,
                path: Vec<String>,
            ) -> Result<Self, ::dspy_rs::baml_bridge::BamlConvertError> {
                let map = match value {
                    ::dspy_rs::baml_bridge::baml_types::BamlValue::Class(_, map)
                    | ::dspy_rs::baml_bridge::baml_types::BamlValue::Map(map) => map,
                    other => {
                        return Err(::dspy_rs::baml_bridge::BamlConvertError::new(
                            path,
                            "object",
                            format!("{other:?}"),
                            "expected an object",
                        ))
                    }
                };

                let tag_value = match map.get(#tag) {
                    Some(::dspy_rs::baml_bridge::baml_types::BamlValue::String(v)) => v.clone(),
                    Some(::dspy_rs::baml_bridge::baml_types::BamlValue::Enum(_, v)) => v.clone(),
                    Some(other) => {
                        return Err(::dspy_rs::baml_bridge::BamlConvertError::new(
                            path,
                            "string",
                            format!("{other:?}"),
                            "expected tag field to be a string",
                        ))
                    }
                    None => {
                        return Err(::dspy_rs::baml_bridge::BamlConvertError::new(
                            path,
                            "string",
                            "<missing>",
                            "missing enum tag",
                        ))
                    }
                };

                match tag_value.as_str() {
                    #(#match_arms),*,
                    _ => Err(::dspy_rs::baml_bridge::BamlConvertError::new(
                        path,
                        "enum",
                        tag_value,
                        "unknown enum variant",
                    )),
                }
            }
        }
    };

    let to_value_impl = quote! {
        impl ::dspy_rs::baml_bridge::ToBamlValue for #name {
            fn to_baml_value(&self) -> ::dspy_rs::baml_bridge::baml_types::BamlValue {
                match self {
                    #(#to_value_match_arms),*
                }
            }
        }
    };

    let output_format_impl = quote! {
        impl ::dspy_rs::baml_bridge::BamlType for #name {
            fn baml_output_format() -> &'static ::dspy_rs::baml_bridge::internal_baml_jinja::types::OutputFormatContent {
                static OUTPUT_FORMAT: ::std::sync::OnceLock<::dspy_rs::baml_bridge::internal_baml_jinja::types::OutputFormatContent> = ::std::sync::OnceLock::new();
                OUTPUT_FORMAT.get_or_init(|| {
                    let mut reg = ::dspy_rs::baml_bridge::Registry::new();
                    <Self as ::dspy_rs::baml_bridge::BamlTypeInternal>::register(&mut reg);
                    reg.build(<Self as ::dspy_rs::baml_bridge::BamlTypeInternal>::baml_type_ir())
                })
            }
        }
    };

    Ok(quote! {
        #register_impl
        #value_convert_impl
        #to_value_impl
        #output_format_impl
    })
}

fn variant_parse_tokens(
    variant: &syn::Variant,
    _variant_name: &str,
    _rendered_variant: Option<&str>,
    rename_all: Option<RenameRule>,
) -> syn::Result<proc_macro2::TokenStream> {
    let variant_ident = &variant.ident;
    match &variant.fields {
        Fields::Unit => Ok(quote! { Ok(Self::#variant_ident) }),
        Fields::Named(fields) => {
            let mut field_inits = Vec::new();
            for field in &fields.named {
                let ident = field.ident.as_ref().unwrap();
                let field_name = ident.to_string();
                let field_attrs = parse_field_attrs(&field.attrs)?;

                if field_attrs.skip {
                    field_inits.push(quote! { #ident: ::std::default::Default::default() });
                    continue;
                }

                let field_alias = field_alias(&field_attrs, rename_all, &field_name);
                let conversion = field_conversion_tokens(
                    &field.ty,
                    &field_attrs,
                    &field_name,
                    field_alias.as_deref(),
                )?;
                field_inits.push(quote! { #ident: #conversion });
            }

            Ok(quote! { Ok(Self::#variant_ident { #(#field_inits),* }) })
        }
        Fields::Unnamed(_) => Err(syn::Error::new_spanned(
            variant,
            "Tuple enum variants are not supported; hint: use a unit or struct-like variant",
        )),
    }
}

fn field_conversion_tokens(
    ty: &Type,
    attrs: &FieldAttrs,
    field_name: &str,
    alias: Option<&str>,
) -> syn::Result<proc_macro2::TokenStream> {
    let missing_value = if attrs.default {
        quote! { ::std::default::Default::default() }
    } else if is_option_type(ty).is_some() {
        quote! { None }
    } else {
        quote! {
            return Err(::dspy_rs::baml_bridge::BamlConvertError::new(
                path,
                "value",
                "<missing>",
                "missing required field",
            ));
        }
    };

    if attrs.skip {
        return Ok(quote! { ::std::default::Default::default() });
    }

    let alias_token = match alias {
        Some(alias) => quote! { Some(#alias) },
        None => quote! { None },
    };

    let get_value = quote! {
        ::dspy_rs::baml_bridge::get_field(&map, #field_name, #alias_token)
    };

    let conversion = if let Some(adapter) = &attrs.with {
        quote! {
            <#adapter as ::dspy_rs::baml_bridge::BamlAdapter<#ty>>::try_from_baml(value.clone(), field_path)?
        }
    } else if let Some(int_repr) = attrs.int_repr {
        let conv = int_repr_conversion_tokens(ty, int_repr)?;
        quote! {{
            let value = value.clone();
            #conv
        }}
    } else if let Some(map_repr) = attrs.map_key_repr {
        let conv = map_key_repr_conversion_tokens(ty, map_repr)?;
        quote! {{
            let value = value.clone();
            #conv
        }}
    } else {
        quote! {
            <#ty as ::dspy_rs::baml_bridge::BamlValueConvert>::try_from_baml_value(value.clone(), field_path)?
        }
    };

    Ok(quote! {{
        match #get_value {
            Some(value) => {
                let mut field_path = path.clone();
                field_path.push(#field_name.to_string());
                #conversion
            }
            None => { #missing_value }
        }
    }})
}

fn field_to_value_tokens(
    ty: &Type,
    attrs: &FieldAttrs,
    value_expr: proc_macro2::TokenStream,
) -> syn::Result<proc_macro2::TokenStream> {
    if let Some(int_repr) = attrs.int_repr {
        let conv = int_repr_to_value_tokens(ty, int_repr)?;
        return Ok(quote! {{
            let value = #value_expr;
            #conv
        }});
    }

    if let Some(map_repr) = attrs.map_key_repr {
        let conv = map_key_repr_to_value_tokens(ty, map_repr)?;
        return Ok(quote! {{
            let value = #value_expr;
            #conv
        }});
    }

    Ok(quote! {
        ::dspy_rs::baml_bridge::ToBamlValue::to_baml_value(#value_expr)
    })
}

fn int_repr_to_value_tokens(ty: &Type, repr: IntRepr) -> syn::Result<proc_macro2::TokenStream> {
    if let Some(inner) = is_option_type(ty) {
        let inner_conv = int_repr_to_value_tokens(inner, repr)?;
        return Ok(quote! {
            match value.as_ref() {
                Some(inner) => {
                    let value = inner;
                    #inner_conv
                }
                None => ::dspy_rs::baml_bridge::baml_types::BamlValue::Null,
            }
        });
    }
    if let Some(inner) = is_vec_type(ty) {
        let inner_conv = int_repr_to_value_tokens(inner, repr)?;
        return Ok(quote! {
            ::dspy_rs::baml_bridge::baml_types::BamlValue::List(
                value
                    .iter()
                    .map(|item| {
                        let value = item;
                        #inner_conv
                    })
                    .collect()
            )
        });
    }
    if let Some(inner) = is_box_type(ty) {
        let inner_conv = int_repr_to_value_tokens(inner, repr)?;
        return Ok(quote! {{
            let value = value.as_ref();
            #inner_conv
        }});
    }
    if let Some(inner) = is_arc_type(ty) {
        let inner_conv = int_repr_to_value_tokens(inner, repr)?;
        return Ok(quote! {{
            let value = value.as_ref();
            #inner_conv
        }});
    }
    if let Some(inner) = is_rc_type(ty) {
        let inner_conv = int_repr_to_value_tokens(inner, repr)?;
        return Ok(quote! {{
            let value = value.as_ref();
            #inner_conv
        }});
    }

    if !is_large_int_type(ty) {
        return Err(syn::Error::new_spanned(
            ty,
            "int_repr is only supported for u64/usize/i128/u128 (optionally wrapped in Option/Vec/Box/Arc/Rc); hint: remove int_repr or change the field type",
        ));
    }

    match repr {
        IntRepr::String => Ok(quote! {
            ::dspy_rs::baml_bridge::baml_types::BamlValue::String(value.to_string())
        }),
        IntRepr::I64 => Ok(quote! {{
            let value = *value as i128;
            let min = i64::MIN as i128;
            let max = i64::MAX as i128;
            if value < min || value > max {
                panic!("integer out of range for i64 representation");
            }
            ::dspy_rs::baml_bridge::baml_types::BamlValue::Int(value as i64)
        }}),
    }
}

fn map_key_repr_to_value_tokens(
    ty: &Type,
    repr: MapKeyRepr,
) -> syn::Result<proc_macro2::TokenStream> {
    if let Some(inner) = is_option_type(ty) {
        let inner_conv = map_key_repr_to_value_tokens(inner, repr)?;
        return Ok(quote! {
            match value.as_ref() {
                Some(inner) => {
                    let value = inner;
                    #inner_conv
                }
                None => ::dspy_rs::baml_bridge::baml_types::BamlValue::Null,
            }
        });
    }
    if let Some(inner) = is_vec_type(ty) {
        let inner_conv = map_key_repr_to_value_tokens(inner, repr)?;
        return Ok(quote! {
            ::dspy_rs::baml_bridge::baml_types::BamlValue::List(
                value
                    .iter()
                    .map(|item| {
                        let value = item;
                        #inner_conv
                    })
                    .collect()
            )
        });
    }
    if let Some(inner) = is_box_type(ty) {
        let inner_conv = map_key_repr_to_value_tokens(inner, repr)?;
        return Ok(quote! {{
            let value = value.as_ref();
            #inner_conv
        }});
    }
    if let Some(inner) = is_arc_type(ty) {
        let inner_conv = map_key_repr_to_value_tokens(inner, repr)?;
        return Ok(quote! {{
            let value = value.as_ref();
            #inner_conv
        }});
    }
    if let Some(inner) = is_rc_type(ty) {
        let inner_conv = map_key_repr_to_value_tokens(inner, repr)?;
        return Ok(quote! {{
            let value = value.as_ref();
            #inner_conv
        }});
    }

    let (_key_ty, _value_ty) = map_types(ty).ok_or_else(|| {
        syn::Error::new_spanned(
            ty,
            "map_key_repr only applies to map fields (HashMap/BTreeMap), optionally wrapped in Option/Vec/Box/Arc/Rc; hint: remove the attribute or change the field type",
        )
    })?;

    match repr {
        MapKeyRepr::String => Ok(quote! {{
            let mut map = ::dspy_rs::baml_bridge::baml_types::BamlMap::new();
            for (key, value) in value.iter() {
                map.insert(
                    key.to_string(),
                    ::dspy_rs::baml_bridge::ToBamlValue::to_baml_value(value),
                );
            }
            ::dspy_rs::baml_bridge::baml_types::BamlValue::Map(map)
        }}),
        MapKeyRepr::Pairs => Ok(quote! {{
            let mut entries = Vec::with_capacity(value.len());
            for (key, value) in value.iter() {
                let mut entry = ::dspy_rs::baml_bridge::baml_types::BamlMap::new();
                entry.insert(
                    "key".to_string(),
                    ::dspy_rs::baml_bridge::ToBamlValue::to_baml_value(key),
                );
                entry.insert(
                    "value".to_string(),
                    ::dspy_rs::baml_bridge::ToBamlValue::to_baml_value(value),
                );
                entries.push(::dspy_rs::baml_bridge::baml_types::BamlValue::Map(entry));
            }
            ::dspy_rs::baml_bridge::baml_types::BamlValue::List(entries)
        }}),
    }
}

fn int_repr_conversion_tokens(ty: &Type, repr: IntRepr) -> syn::Result<proc_macro2::TokenStream> {
    if let Some(inner) = is_option_type(ty) {
        let inner_conv = int_repr_conversion_tokens(inner, repr)?;
        return Ok(quote! {
            match value {
                ::dspy_rs::baml_bridge::baml_types::BamlValue::Null => None,
                other => {
                    let value = other;
                    let field_path = field_path.clone();
                    Some({ #inner_conv })
                }
            }
        });
    }
    if let Some(inner) = is_vec_type(ty) {
        let inner_conv = int_repr_conversion_tokens(inner, repr)?;
        return Ok(quote! {
            match value {
                ::dspy_rs::baml_bridge::baml_types::BamlValue::List(items) => {
                    let mut out = Vec::with_capacity(items.len());
                    for (idx, item) in items.into_iter().enumerate() {
                        let mut item_path = field_path.clone();
                        item_path.push(idx.to_string());
                        let value = item;
                        let field_path = item_path;
                        out.push({ #inner_conv });
                    }
                    out
                }
                other => {
                    return Err(::dspy_rs::baml_bridge::BamlConvertError::new(
                        field_path,
                        "list",
                        format!("{other:?}"),
                        "expected a list",
                    ))
                }
            }
        });
    }
    if let Some(inner) = is_box_type(ty) {
        let inner_conv = int_repr_conversion_tokens(inner, repr)?;
        return Ok(quote! { ::std::boxed::Box::new({ #inner_conv }) });
    }
    if let Some(inner) = is_arc_type(ty) {
        let inner_conv = int_repr_conversion_tokens(inner, repr)?;
        return Ok(quote! { ::std::sync::Arc::new({ #inner_conv }) });
    }
    if let Some(inner) = is_rc_type(ty) {
        let inner_conv = int_repr_conversion_tokens(inner, repr)?;
        return Ok(quote! { ::std::rc::Rc::new({ #inner_conv }) });
    }

    if !is_large_int_type(ty) {
        return Err(syn::Error::new_spanned(
            ty,
            "int_repr is only supported for u64/usize/i128/u128 (optionally wrapped in Option/Vec/Box/Arc/Rc); hint: remove int_repr or change the field type",
        ));
    }

    match repr {
        IntRepr::String => Ok(quote! {
            match value {
                ::dspy_rs::baml_bridge::baml_types::BamlValue::String(s) => {
                    s.parse::<#ty>().map_err(|_| ::dspy_rs::baml_bridge::BamlConvertError::new(
                        field_path,
                        stringify!(#ty),
                        s,
                        "failed to parse string as integer",
                    ))?
                }
                other => {
                    return Err(::dspy_rs::baml_bridge::BamlConvertError::new(
                        field_path,
                        "string",
                        format!("{other:?}"),
                        "expected string for integer representation",
                    ))
                }
            }
        }),
        IntRepr::I64 => Ok(quote! {
            match value {
                ::dspy_rs::baml_bridge::baml_types::BamlValue::Int(v) => {
                    let min = <#ty>::MIN as i128;
                    let max = <#ty>::MAX as i128;
                    let v = v as i128;
                    if v < min || v > max {
                        return Err(::dspy_rs::baml_bridge::BamlConvertError::new(
                            field_path,
                            stringify!(#ty),
                            v.to_string(),
                            "integer out of range",
                        ));
                    }
                    v as #ty
                }
                other => {
                    return Err(::dspy_rs::baml_bridge::BamlConvertError::new(
                        field_path,
                        "int",
                        format!("{other:?}"),
                        "expected integer",
                    ))
                }
            }
        }),
    }
}

fn map_key_repr_conversion_tokens(
    ty: &Type,
    repr: MapKeyRepr,
) -> syn::Result<proc_macro2::TokenStream> {
    if let Some(inner) = is_option_type(ty) {
        let inner_conv = map_key_repr_conversion_tokens(inner, repr)?;
        return Ok(quote! {
            match value {
                ::dspy_rs::baml_bridge::baml_types::BamlValue::Null => None,
                other => {
                    let value = other;
                    let field_path = field_path.clone();
                    Some({ #inner_conv })
                }
            }
        });
    }
    if let Some(inner) = is_vec_type(ty) {
        let inner_conv = map_key_repr_conversion_tokens(inner, repr)?;
        return Ok(quote! {
            match value {
                ::dspy_rs::baml_bridge::baml_types::BamlValue::List(items) => {
                    let mut out = Vec::with_capacity(items.len());
                    for (idx, item) in items.into_iter().enumerate() {
                        let mut item_path = field_path.clone();
                        item_path.push(idx.to_string());
                        let value = item;
                        let field_path = item_path;
                        out.push({ #inner_conv });
                    }
                    out
                }
                other => {
                    return Err(::dspy_rs::baml_bridge::BamlConvertError::new(
                        field_path,
                        "list",
                        format!("{other:?}"),
                        "expected a list",
                    ))
                }
            }
        });
    }
    if let Some(inner) = is_box_type(ty) {
        let inner_conv = map_key_repr_conversion_tokens(inner, repr)?;
        return Ok(quote! { ::std::boxed::Box::new({ #inner_conv }) });
    }
    if let Some(inner) = is_arc_type(ty) {
        let inner_conv = map_key_repr_conversion_tokens(inner, repr)?;
        return Ok(quote! { ::std::sync::Arc::new({ #inner_conv }) });
    }
    if let Some(inner) = is_rc_type(ty) {
        let inner_conv = map_key_repr_conversion_tokens(inner, repr)?;
        return Ok(quote! { ::std::rc::Rc::new({ #inner_conv }) });
    }

    let (key_ty, value_ty) = map_types(ty).ok_or_else(|| {
        syn::Error::new_spanned(
            ty,
            "map_key_repr only applies to map fields (HashMap/BTreeMap), optionally wrapped in Option/Vec/Box/Arc/Rc; hint: remove the attribute or change the field type",
        )
    })?;

    match repr {
        MapKeyRepr::String => Ok(quote! {
            match value {
                ::dspy_rs::baml_bridge::baml_types::BamlValue::Map(map) => {
                    let mut out: #ty = ::std::default::Default::default();
                    for (key, value) in map.into_iter() {
                        let parsed_key = key.parse::<#key_ty>().map_err(|_| {
                            ::dspy_rs::baml_bridge::BamlConvertError::new(
                                field_path.clone(),
                                stringify!(#key_ty),
                                key.clone(),
                                "failed to parse map key",
                            )
                        })?;
                        let mut item_path = field_path.clone();
                        item_path.push(key);
                        let parsed_value = <#value_ty as ::dspy_rs::baml_bridge::BamlValueConvert>::try_from_baml_value(value, item_path)?;
                        out.insert(parsed_key, parsed_value);
                    }
                    out
                }
                other => {
                    return Err(::dspy_rs::baml_bridge::BamlConvertError::new(
                        field_path,
                        "map",
                        format!("{other:?}"),
                        "expected map",
                    ))
                }
            }
        }),
        MapKeyRepr::Pairs => Ok(quote! {
            match value {
                ::dspy_rs::baml_bridge::baml_types::BamlValue::List(items) => {
                    let mut out: #ty = ::std::default::Default::default();
                    for (idx, item) in items.into_iter().enumerate() {
                        let entry_map = match item {
                            ::dspy_rs::baml_bridge::baml_types::BamlValue::Class(_, map)
                            | ::dspy_rs::baml_bridge::baml_types::BamlValue::Map(map) => map,
                            other => {
                                return Err(::dspy_rs::baml_bridge::BamlConvertError::new(
                                    field_path.clone(),
                                    "object",
                                    format!("{other:?}"),
                                    "expected map entry object",
                                ))
                            }
                        };

                        let key_value = ::dspy_rs::baml_bridge::get_field(&entry_map, "key", None)
                            .cloned()
                            .ok_or_else(|| {
                                ::dspy_rs::baml_bridge::BamlConvertError::new(
                                    field_path.clone(),
                                    "key",
                                    "<missing>",
                                    "missing map entry key",
                                )
                            })?;
                        let value_value = ::dspy_rs::baml_bridge::get_field(&entry_map, "value", None)
                            .cloned()
                            .ok_or_else(|| {
                                ::dspy_rs::baml_bridge::BamlConvertError::new(
                                    field_path.clone(),
                                    "value",
                                    "<missing>",
                                    "missing map entry value",
                                )
                            })?;

                        let mut key_path = field_path.clone();
                        key_path.push(idx.to_string());
                        key_path.push("key".to_string());
                        let parsed_key =
                            <#key_ty as ::dspy_rs::baml_bridge::BamlValueConvert>::try_from_baml_value(
                                key_value,
                                key_path,
                            )?;

                        let mut value_path = field_path.clone();
                        value_path.push(idx.to_string());
                        value_path.push("value".to_string());
                        let parsed_value =
                            <#value_ty as ::dspy_rs::baml_bridge::BamlValueConvert>::try_from_baml_value(
                                value_value,
                                value_path,
                            )?;

                        out.insert(parsed_key, parsed_value);
                    }
                    out
                }
                other => {
                    return Err(::dspy_rs::baml_bridge::BamlConvertError::new(
                        field_path,
                        "list",
                        format!("{other:?}"),
                        "expected list of map entries",
                    ))
                }
            }
        }),
    }
}

fn register_call_tokens(
    ty: &Type,
    attrs: &FieldAttrs,
    map_entry: Option<&MapEntryInfo>,
) -> syn::Result<proc_macro2::TokenStream> {
    if let Some(adapter) = &attrs.with {
        return Ok(quote! { <#adapter as ::dspy_rs::baml_bridge::BamlAdapter<#ty>>::register(reg) });
    }

    if attrs.int_repr.is_some() {
        return Ok(quote! {});
    }

    if let Some(map_repr) = attrs.map_key_repr {
        if let Some((key_ty, value_ty)) = map_types_for_repr(ty) {
            if matches!(map_repr, MapKeyRepr::Pairs) {
                let entry_info = map_entry.ok_or_else(|| {
                    syn::Error::new_spanned(
                        ty,
                        "internal error: missing map entry metadata for map_key_repr=\"pairs\"; hint: please report a bug",
                    )
                })?;
                let entry_class = map_entry_class_tokens(entry_info, key_ty, value_ty)?;
                return Ok(quote! {{
                    #entry_class
                    <#key_ty as ::dspy_rs::baml_bridge::BamlTypeInternal>::register(reg);
                    <#value_ty as ::dspy_rs::baml_bridge::BamlTypeInternal>::register(reg);
                }});
            }

            return Ok(quote! { <#value_ty as ::dspy_rs::baml_bridge::BamlTypeInternal>::register(reg) });
        }
    }

    Ok(quote! { <#ty as ::dspy_rs::baml_bridge::BamlTypeInternal>::register(reg) })
}

fn type_ir_for_field(
    ty: &Type,
    attrs: &FieldAttrs,
    map_entry: Option<&MapEntryInfo>,
) -> syn::Result<proc_macro2::TokenStream> {
    if let Some(adapter) = attrs.with.as_ref() {
        return Ok(quote! { <#adapter as ::dspy_rs::baml_bridge::BamlAdapter<#ty>>::type_ir() });
    }

    if let Some(int_repr) = attrs.int_repr {
        return int_repr_type_ir(ty, int_repr);
    }

    if let Some(map_repr) = attrs.map_key_repr {
        return map_key_repr_type_ir(ty, map_repr, map_entry);
    }

    match_type_ir(ty)
}

fn int_repr_type_ir(ty: &Type, repr: IntRepr) -> syn::Result<proc_macro2::TokenStream> {
    if let Some(inner) = is_option_type(ty) {
        let inner_ir = int_repr_type_ir(inner, repr)?;
        return Ok(quote! { ::dspy_rs::baml_bridge::baml_types::TypeIR::optional(#inner_ir) });
    }
    if let Some(inner) = is_vec_type(ty) {
        let inner_ir = int_repr_type_ir(inner, repr)?;
        return Ok(quote! { ::dspy_rs::baml_bridge::baml_types::TypeIR::list(#inner_ir) });
    }
    if let Some(inner) = is_box_type(ty) {
        return int_repr_type_ir(inner, repr);
    }
    if let Some(inner) = is_arc_type(ty) {
        return int_repr_type_ir(inner, repr);
    }
    if let Some(inner) = is_rc_type(ty) {
        return int_repr_type_ir(inner, repr);
    }
    if !is_large_int_type(ty) {
        return Err(syn::Error::new_spanned(
            ty,
            "int_repr is only supported for u64/usize/i128/u128 (optionally wrapped in Option/Vec/Box/Arc/Rc); hint: remove int_repr or change the field type",
        ));
    }

    match repr {
        IntRepr::String => Ok(quote! { ::dspy_rs::baml_bridge::baml_types::TypeIR::string() }),
        IntRepr::I64 => Ok(quote! { ::dspy_rs::baml_bridge::baml_types::TypeIR::int() }),
    }
}

fn map_key_repr_type_ir(
    ty: &Type,
    repr: MapKeyRepr,
    map_entry: Option<&MapEntryInfo>,
) -> syn::Result<proc_macro2::TokenStream> {
    if let Some(inner) = is_option_type(ty) {
        let inner_ir = map_key_repr_type_ir(inner, repr, map_entry)?;
        return Ok(quote! { ::dspy_rs::baml_bridge::baml_types::TypeIR::optional(#inner_ir) });
    }
    if let Some(inner) = is_vec_type(ty) {
        let inner_ir = map_key_repr_type_ir(inner, repr, map_entry)?;
        return Ok(quote! { ::dspy_rs::baml_bridge::baml_types::TypeIR::list(#inner_ir) });
    }
    if let Some(inner) = is_box_type(ty) {
        return map_key_repr_type_ir(inner, repr, map_entry);
    }
    if let Some(inner) = is_arc_type(ty) {
        return map_key_repr_type_ir(inner, repr, map_entry);
    }
    if let Some(inner) = is_rc_type(ty) {
        return map_key_repr_type_ir(inner, repr, map_entry);
    }

    let (_key_ty, value_ty) = map_types(ty).ok_or_else(|| {
        syn::Error::new_spanned(
            ty,
            "map_key_repr only applies to map fields (HashMap/BTreeMap), optionally wrapped in Option/Vec/Box/Arc/Rc; hint: remove the attribute or change the field type",
        )
    })?;

    match repr {
        MapKeyRepr::String => Ok(quote! {
            ::dspy_rs::baml_bridge::baml_types::TypeIR::map(
                ::dspy_rs::baml_bridge::baml_types::TypeIR::string(),
                <#value_ty as ::dspy_rs::baml_bridge::BamlTypeInternal>::baml_type_ir(),
            )
        }),
        MapKeyRepr::Pairs => {
            let entry = map_entry.ok_or_else(|| {
                syn::Error::new_spanned(
                    ty,
                    "internal error: missing map entry metadata for map_key_repr=\"pairs\"; hint: please report a bug",
                )
            })?;
            let entry_name_expr = entry.internal_name_expr.clone();
            Ok(quote! {
                ::dspy_rs::baml_bridge::baml_types::TypeIR::list(
                    ::dspy_rs::baml_bridge::baml_types::TypeIR::class(#entry_name_expr)
                )
            })
        }
    }
}

fn match_type_ir(ty: &Type) -> syn::Result<proc_macro2::TokenStream> {
    match ty {
        Type::Tuple(_) => {
            return Err(syn::Error::new_spanned(
                ty,
                "tuple types are not supported in BAML outputs; hint: use a struct with named fields or a list",
            ))
        }
        Type::BareFn(_) => {
            return Err(syn::Error::new_spanned(
                ty,
                "function types are not supported in BAML outputs; hint: remove the field or use #[baml(with = \"...\")] to adapt it",
            ))
        }
        Type::TraitObject(_) => {
            return Err(syn::Error::new_spanned(
                ty,
                "trait objects are not supported in BAML outputs; hint: use a concrete type or a custom adapter",
            ))
        }
        _ => {}
    }

    if let Some(inner) = is_option_type(ty) {
        let inner_ir = match_type_ir(inner)?;
        return Ok(quote! { ::dspy_rs::baml_bridge::baml_types::TypeIR::optional(#inner_ir) });
    }
    if let Some(inner) = is_vec_type(ty) {
        let inner_ir = match_type_ir(inner)?;
        return Ok(quote! { ::dspy_rs::baml_bridge::baml_types::TypeIR::list(#inner_ir) });
    }
    if let Some(inner) = is_box_type(ty) {
        return match_type_ir(inner);
    }
    if let Some(inner) = is_arc_type(ty) {
        return match_type_ir(inner);
    }
    if let Some(inner) = is_rc_type(ty) {
        return match_type_ir(inner);
    }
    if let Some((_key, value)) = is_string_map_type(ty) {
        let value_ir = match_type_ir(value)?;
        return Ok(
            quote! { ::dspy_rs::baml_bridge::baml_types::TypeIR::map(::dspy_rs::baml_bridge::baml_types::TypeIR::string(), #value_ir) },
        );
    }
    if let Some((key, _)) = map_types(ty) {
        if !is_string_type(key) {
            return Err(syn::Error::new_spanned(
                ty,
                "map keys must be String for object maps; hint: use HashMap<String, V> or add #[baml(map_key_repr = \"string\"|\"pairs\")], or use a custom adapter",
            ));
        }
    }

    if is_serde_json_value(ty) {
        return Err(syn::Error::new_spanned(
            ty,
            "serde_json::Value is not supported without a #[baml(with = \"...\")] adapter; hint: use a concrete type or provide a custom adapter",
        ));
    }

    let ident = type_ident(ty).ok_or_else(|| {
        syn::Error::new_spanned(
            ty,
            "unsupported type for BamlType; hint: derive BamlType for it or use #[baml(with = \"...\")]",
        )
    })?;

    let ident_str = ident.to_string();
    let ir = match ident_str.as_str() {
        "String" => quote! { ::dspy_rs::baml_bridge::baml_types::TypeIR::string() },
        "bool" => quote! { ::dspy_rs::baml_bridge::baml_types::TypeIR::bool() },
        "f32" | "f64" => quote! { ::dspy_rs::baml_bridge::baml_types::TypeIR::float() },
        "i8" | "i16" | "i32" | "i64" | "isize" => quote! { ::dspy_rs::baml_bridge::baml_types::TypeIR::int() },
        "u8" | "u16" | "u32" => quote! { ::dspy_rs::baml_bridge::baml_types::TypeIR::int() },
        "u64" | "usize" | "i128" | "u128" => {
            return Err(syn::Error::new_spanned(
                ty,
                "unsupported integer width for BAML outputs; hint: use #[baml(int_repr = \"string\"|\"i64\")] or a smaller integer type",
            ))
        }
        _ => quote! { <#ty as ::dspy_rs::baml_bridge::BamlTypeInternal>::baml_type_ir() },
    };

    Ok(ir)
}

fn with_constraints_tokens(
    r#type: proc_macro2::TokenStream,
    constraints: &[ConstraintSpec],
) -> proc_macro2::TokenStream {
    if constraints.is_empty() {
        return r#type;
    }
    let constraint_tokens = constraints_tokens(constraints);
    quote! { ::dspy_rs::baml_bridge::with_constraints(#r#type, #constraint_tokens) }
}

fn constraints_tokens(constraints: &[ConstraintSpec]) -> proc_macro2::TokenStream {
    let tokens = constraints.iter().map(|spec| {
        let label = &spec.label;
        let expr = &spec.expr;
        match spec.level {
            ConstraintLevelSpec::Check => {
                quote! { ::dspy_rs::baml_bridge::baml_types::Constraint::new_check(#label, #expr) }
            }
            ConstraintLevelSpec::Assert => {
                quote! { ::dspy_rs::baml_bridge::baml_types::Constraint::new_assert(#label, #expr) }
            }
        }
    });
    quote! { vec![#(#tokens),*] }
}

fn name_tokens(name: &str, alias: Option<&str>) -> proc_macro2::TokenStream {
    name_tokens_expr(quote! { #name.to_string() }, alias)
}

fn name_tokens_expr(
    real_expr: proc_macro2::TokenStream,
    alias: Option<&str>,
) -> proc_macro2::TokenStream {
    match alias {
        Some(alias) => quote! {
            ::dspy_rs::baml_bridge::internal_baml_jinja::types::Name::new_with_alias(#real_expr, Some(#alias.to_string()))
        },
        None => quote! {
            ::dspy_rs::baml_bridge::internal_baml_jinja::types::Name::new(#real_expr)
        },
    }
}

fn field_alias(
    attrs: &FieldAttrs,
    rename_all: Option<RenameRule>,
    field_name: &str,
) -> Option<String> {
    if let Some(alias) = &attrs.alias {
        return Some(alias.clone());
    }
    rename_all.map(|rule| apply_rename(rule, field_name))
}

fn map_entry_info(
    field_name: &str,
    rendered_field: &str,
    variant_name: Option<&str>,
    variant_rendered: Option<&str>,
    attrs: &FieldAttrs,
) -> syn::Result<Option<MapEntryInfo>> {
    if !matches!(attrs.map_key_repr, Some(MapKeyRepr::Pairs)) {
        return Ok(None);
    }

    let suffix = match variant_name {
        Some(variant) => format!("{variant}__{field_name}__Entry"),
        None => format!("{field_name}__Entry"),
    };

    let internal_name_expr = quote! {
        format!(
            "{}::{}",
            <Self as ::dspy_rs::baml_bridge::BamlTypeInternal>::baml_internal_name(),
            #suffix
        )
    };

    let rendered_entry = match variant_rendered {
        Some(variant) => format!("{variant}{rendered_field}Entry"),
        None => format!("{rendered_field}Entry"),
    };

    Ok(Some(MapEntryInfo {
        internal_name_expr,
        rendered_name: Some(rendered_entry),
    }))
}

fn map_entry_class_tokens(
    entry: &MapEntryInfo,
    key_ty: &Type,
    value_ty: &Type,
) -> syn::Result<proc_macro2::TokenStream> {
    let key_ir = match_type_ir(key_ty)?;
    let value_ir = match_type_ir(value_ty)?;
    let name_expr = name_tokens_expr(
        entry.internal_name_expr.clone(),
        entry.rendered_name.as_deref(),
    );
    let key_name = name_tokens("key", None);
    let value_name = name_tokens("value", None);

    Ok(quote! {
        reg.register_class(::dspy_rs::baml_bridge::internal_baml_jinja::types::Class {
            name: #name_expr,
            description: None,
            namespace: ::dspy_rs::baml_bridge::baml_types::StreamingMode::NonStreaming,
            fields: vec![
                (#key_name, #key_ir, None, false),
                (#value_name, #value_ir, None, false),
            ],
            constraints: Vec::new(),
            streaming_behavior: ::dspy_rs::baml_bridge::default_streaming_behavior(),
        });
    })
}

fn variant_alias(
    attrs: &VariantAttrs,
    rename_all: Option<RenameRule>,
    name: &str,
) -> Option<String> {
    if let Some(alias) = &attrs.alias {
        return Some(alias.clone());
    }
    rename_all.map(|rule| apply_rename(rule, name))
}

fn match_strings_for_variant(name: &str, alias: Option<&str>) -> Vec<proc_macro2::TokenStream> {
    let mut values = vec![quote! { #name }];
    if let Some(alias) = alias {
        if alias != name {
            values.push(quote! { #alias });
        }
    }
    values
}

fn field_description(attrs: &FieldAttrs, raw_attrs: &[Attribute]) -> Option<String> {
    if let Some(desc) = &attrs.description {
        return Some(desc.clone());
    }
    extract_doc(raw_attrs)
}

fn variant_description(attrs: &VariantAttrs, raw_attrs: &[Attribute]) -> Option<String> {
    if let Some(desc) = &attrs.description {
        return Some(desc.clone());
    }
    extract_doc(raw_attrs)
}

fn container_description(attrs: &ContainerAttrs, raw_attrs: &[Attribute]) -> Option<String> {
    if let Some(desc) = &attrs.description {
        return Some(desc.clone());
    }
    extract_doc(raw_attrs)
}

fn parse_container_attrs(attrs: &[Attribute]) -> syn::Result<ContainerAttrs> {
    let mut out = ContainerAttrs::default();
    for attr in attrs {
        if attr.path().is_ident("render") {
            out.render_attrs.push(parse_render_attr(attr)?);
        }
        if attr.path().is_ident("baml") {
            parse_baml_meta(attr, |meta| {
                match meta {
                Meta::NameValue(meta) if meta.path.is_ident("name") => {
                    out.name = Some(parse_string_expr(&meta.value, meta.span())?);
                    Ok(())
                }
                Meta::NameValue(meta) if meta.path.is_ident("internal_name") => {
                    out.internal_name = Some(parse_string_expr(&meta.value, meta.span())?);
                    Ok(())
                }
                Meta::NameValue(meta) if meta.path.is_ident("rename_all") => {
                    out.rename_all = Some(parse_rename_rule(&meta.value, meta.span())?);
                    Ok(())
                }
                Meta::NameValue(meta) if meta.path.is_ident("tag") => {
                    out.tag = Some(parse_string_expr(&meta.value, meta.span())?);
                    Ok(())
                }
                Meta::NameValue(meta) if meta.path.is_ident("description") => {
                    out.description = parse_optional_string(&meta.value, meta.span())?;
                    Ok(())
                }
                Meta::List(meta) if meta.path.is_ident("check") => {
                    out.constraints.push(parse_constraint(meta, ConstraintLevelSpec::Check)?);
                    Ok(())
                }
                Meta::List(meta) if meta.path.is_ident("assert") => {
                    out.constraints.push(parse_constraint(meta, ConstraintLevelSpec::Assert)?);
                    Ok(())
                }
                Meta::Path(path) if path.is_ident("as_union") => {
                    out.as_union = true;
                    Ok(())
                }
                Meta::Path(path) if path.is_ident("as_enum") => {
                    out.as_enum = true;
                    Ok(())
                }
                _ => Err(syn::Error::new_spanned(
                    meta,
                    "unsupported #[baml(...)] attribute; hint: check the supported keys in the bridge docs",
                )),
            }
            })?;
        }

        if attr.path().is_ident("serde") {
            parse_serde_meta(attr, |meta| {
                match meta {
                Meta::NameValue(meta) if meta.path.is_ident("rename") => {
                    if out.name.is_none() {
                        out.name = Some(parse_string_expr(&meta.value, meta.span())?);
                    }
                    Ok(())
                }
                Meta::NameValue(meta) if meta.path.is_ident("rename_all") => {
                    if out.rename_all.is_none() {
                        out.rename_all = Some(parse_rename_rule(&meta.value, meta.span())?);
                    }
                    Ok(())
                }
                Meta::NameValue(meta) if meta.path.is_ident("tag") => {
                    if out.tag.is_none() {
                        out.tag = Some(parse_string_expr(&meta.value, meta.span())?);
                    }
                    Ok(())
                }
                Meta::Path(path) if path.is_ident("untagged") => Err(syn::Error::new_spanned(
                    path,
                    "serde(untagged) is not supported; hint: use #[baml(tag = \"...\")] for data enums",
                )),
                Meta::Path(path) if path.is_ident("flatten") => Err(syn::Error::new_spanned(
                    path,
                    "serde(flatten) is not supported; hint: model fields explicitly",
                )),
                _ => Ok(()),
            }
            })?;
        }
    }
    Ok(out)
}

fn parse_render_attr(attr: &Attribute) -> syn::Result<RenderAttr> {
    let mut out = RenderAttr {
        span: attr.span(),
        ..RenderAttr::default()
    };
    let metas = parse_meta_list(attr)?;
    for meta in metas {
        match meta {
            Meta::NameValue(meta) if meta.path.is_ident("default") => {
                if out.default.is_some() {
                    return Err(syn::Error::new_spanned(
                        meta,
                        "duplicate render attribute: default",
                    ));
                }
                out.default = Some(parse_lit_str_expr(&meta.value, meta.span())?);
            }
            Meta::NameValue(meta) if meta.path.is_ident("style") => {
                if out.style.is_some() {
                    return Err(syn::Error::new_spanned(
                        meta,
                        "duplicate render attribute: style",
                    ));
                }
                out.style = Some(parse_lit_str_expr(&meta.value, meta.span())?);
            }
            Meta::NameValue(meta) if meta.path.is_ident("template") => {
                if out.template.is_some() {
                    return Err(syn::Error::new_spanned(
                        meta,
                        "duplicate render attribute: template",
                    ));
                }
                out.template = Some(parse_lit_str_expr(&meta.value, meta.span())?);
            }
            Meta::NameValue(meta) if meta.path.is_ident("fn") => {
                if out.func.is_some() {
                    return Err(syn::Error::new_spanned(
                        meta,
                        "duplicate render attribute: fn",
                    ));
                }
                out.func = Some(parse_path_expr(&meta.value, meta.span())?);
            }
            Meta::NameValue(meta) if meta.path.is_ident("allow_dynamic") => {
                out.allow_dynamic = parse_bool_expr(&meta.value, meta.span())?;
            }
            Meta::Path(path) if path.is_ident("allow_dynamic") => {
                out.allow_dynamic = true;
            }
            _ => {
                return Err(syn::Error::new_spanned(
                    meta,
                    "unsupported #[render(...)] attribute; hint: use default/style/template/fn/allow_dynamic",
                ));
            }
        }
    }

    validate_render_attr(&out)?;
    Ok(out)
}

fn validate_render_attr(attr: &RenderAttr) -> syn::Result<()> {
    if attr.default.is_some() {
        if attr.style.is_some() || attr.template.is_some() || attr.func.is_some() {
            return Err(syn::Error::new(
                attr.span,
                "render(default = ...) cannot be combined with style/template/fn",
            ));
        }
        return Ok(());
    }

    if attr.template.is_some() && attr.func.is_some() {
        return Err(syn::Error::new(
            attr.span,
            "render attribute must specify either template or fn, not both",
        ));
    }

    if attr.template.is_none() && attr.func.is_none() {
        return Err(syn::Error::new(
            attr.span,
            "render attribute missing template or fn",
        ));
    }

    Ok(())
}

fn validate_render_templates(
    attrs: &[RenderAttr],
    struct_fields: Option<&syn::punctuated::Punctuated<syn::Field, syn::Token![,]>>,
) -> syn::Result<()> {
    for attr in attrs {
        let Some(template) = &attr.template else {
            continue;
        };
        validate_template(template, struct_fields, attr.allow_dynamic, attr.span)?;
    }
    Ok(())
}

fn validate_template(
    template: &LitStr,
    struct_fields: Option<&syn::punctuated::Punctuated<syn::Field, syn::Token![,]>>,
    allow_dynamic: bool,
    span: proc_macro2::Span,
) -> syn::Result<()> {
    let source = template.value();
    let ast = jinja_machinery::parse(
        &source,
        "<render>",
        Default::default(),
        Default::default(),
    )
    .map_err(|err| syn::Error::new(span, format!("Jinja syntax error: {}", err)))?;

    let mut field_refs = BTreeSet::new();
    let mut filter_refs = BTreeSet::new();
    collect_from_stmt(&ast, &mut field_refs, &mut filter_refs);

    // TODO(dsrs-ocy): Enum render templates (struct_fields == None) don't validate field access yet.
    // Design per-variant field access checks + compile-fail coverage for enum templates.
    if let Some(fields) = struct_fields {
        if !allow_dynamic {
            let known_fields: BTreeSet<String> = fields
                .iter()
                .filter_map(|field| field.ident.as_ref().map(|ident| ident.to_string()))
                .collect();
            for field in field_refs {
                if !known_fields.contains(&field) {
                    return Err(syn::Error::new(
                        span,
                        format!(
                            "Template references unknown field 'value.{field}'. Use #[render(allow_dynamic = true)] for dynamic access."
                        ),
                    ));
                }
            }
        }
    }

    let known_filters = [
        "truncate",
        "slice_chars",
        "format_count",
        "length",
        "sum",
        "regex_match",
    ];
    for filter in filter_refs {
        if !known_filters.contains(&filter.as_str()) {
            return Err(syn::Error::new(
                span,
                format!("Template uses unknown filter '{filter}'"),
            ));
        }
    }

    Ok(())
}

fn collect_from_stmt(
    stmt: &jinja_ast::Stmt<'_>,
    field_refs: &mut BTreeSet<String>,
    filter_refs: &mut BTreeSet<String>,
) {
    match stmt {
        jinja_ast::Stmt::Template(template) => {
            for child in &template.children {
                collect_from_stmt(child, field_refs, filter_refs);
            }
        }
        jinja_ast::Stmt::EmitExpr(expr) => {
            collect_from_expr(&expr.expr, field_refs, filter_refs);
        }
        jinja_ast::Stmt::EmitRaw(_) => {}
        jinja_ast::Stmt::ForLoop(loop_stmt) => {
            collect_from_expr(&loop_stmt.target, field_refs, filter_refs);
            collect_from_expr(&loop_stmt.iter, field_refs, filter_refs);
            if let Some(filter_expr) = &loop_stmt.filter_expr {
                collect_from_expr(filter_expr, field_refs, filter_refs);
            }
            for child in &loop_stmt.body {
                collect_from_stmt(child, field_refs, filter_refs);
            }
            for child in &loop_stmt.else_body {
                collect_from_stmt(child, field_refs, filter_refs);
            }
        }
        jinja_ast::Stmt::IfCond(cond) => {
            collect_from_expr(&cond.expr, field_refs, filter_refs);
            for child in &cond.true_body {
                collect_from_stmt(child, field_refs, filter_refs);
            }
            for child in &cond.false_body {
                collect_from_stmt(child, field_refs, filter_refs);
            }
        }
        jinja_ast::Stmt::WithBlock(block) => {
            for (target, expr) in &block.assignments {
                collect_from_expr(target, field_refs, filter_refs);
                collect_from_expr(expr, field_refs, filter_refs);
            }
            for child in &block.body {
                collect_from_stmt(child, field_refs, filter_refs);
            }
        }
        jinja_ast::Stmt::Set(stmt) => {
            collect_from_expr(&stmt.target, field_refs, filter_refs);
            collect_from_expr(&stmt.expr, field_refs, filter_refs);
        }
        jinja_ast::Stmt::SetBlock(stmt) => {
            collect_from_expr(&stmt.target, field_refs, filter_refs);
            if let Some(filter_expr) = &stmt.filter {
                collect_from_expr(filter_expr, field_refs, filter_refs);
            }
            for child in &stmt.body {
                collect_from_stmt(child, field_refs, filter_refs);
            }
        }
        jinja_ast::Stmt::AutoEscape(stmt) => {
            collect_from_expr(&stmt.enabled, field_refs, filter_refs);
            for child in &stmt.body {
                collect_from_stmt(child, field_refs, filter_refs);
            }
        }
        jinja_ast::Stmt::FilterBlock(stmt) => {
            collect_from_expr(&stmt.filter, field_refs, filter_refs);
            for child in &stmt.body {
                collect_from_stmt(child, field_refs, filter_refs);
            }
        }
        jinja_ast::Stmt::Do(stmt) => {
            collect_from_expr(&stmt.call.expr, field_refs, filter_refs);
            for arg in &stmt.call.args {
                collect_from_call_arg(arg, field_refs, filter_refs);
            }
        }
        _ => {}
    }
}

fn collect_from_expr(
    expr: &jinja_ast::Expr<'_>,
    field_refs: &mut BTreeSet<String>,
    filter_refs: &mut BTreeSet<String>,
) {
    match expr {
        jinja_ast::Expr::Var(_) | jinja_ast::Expr::Const(_) => {}
        jinja_ast::Expr::Slice(expr) => {
            collect_from_expr(&expr.expr, field_refs, filter_refs);
            if let Some(start) = &expr.start {
                collect_from_expr(start, field_refs, filter_refs);
            }
            if let Some(stop) = &expr.stop {
                collect_from_expr(stop, field_refs, filter_refs);
            }
            if let Some(step) = &expr.step {
                collect_from_expr(step, field_refs, filter_refs);
            }
        }
        jinja_ast::Expr::UnaryOp(expr) => {
            collect_from_expr(&expr.expr, field_refs, filter_refs);
        }
        jinja_ast::Expr::BinOp(expr) => {
            collect_from_expr(&expr.left, field_refs, filter_refs);
            collect_from_expr(&expr.right, field_refs, filter_refs);
        }
        jinja_ast::Expr::IfExpr(expr) => {
            collect_from_expr(&expr.test_expr, field_refs, filter_refs);
            collect_from_expr(&expr.true_expr, field_refs, filter_refs);
            if let Some(false_expr) = &expr.false_expr {
                collect_from_expr(false_expr, field_refs, filter_refs);
            }
        }
        jinja_ast::Expr::Filter(expr) => {
            filter_refs.insert(expr.name.to_string());
            if let Some(inner) = &expr.expr {
                collect_from_expr(inner, field_refs, filter_refs);
            }
            for arg in &expr.args {
                collect_from_call_arg(arg, field_refs, filter_refs);
            }
        }
        jinja_ast::Expr::Test(expr) => {
            collect_from_expr(&expr.expr, field_refs, filter_refs);
            for arg in &expr.args {
                collect_from_call_arg(arg, field_refs, filter_refs);
            }
        }
        jinja_ast::Expr::GetAttr(expr) => {
            if let jinja_ast::Expr::Var(var) = &expr.expr {
                if var.id == "value" {
                    field_refs.insert(expr.name.to_string());
                }
            }
            collect_from_expr(&expr.expr, field_refs, filter_refs);
        }
        jinja_ast::Expr::GetItem(expr) => {
            if let jinja_ast::Expr::Var(var) = &expr.expr {
                if var.id == "value" {
                    if let jinja_ast::Expr::Const(constant) = &expr.subscript_expr {
                        if let Some(key) = constant.value.as_str() {
                            field_refs.insert(key.to_string());
                        }
                    }
                }
            }
            collect_from_expr(&expr.expr, field_refs, filter_refs);
            collect_from_expr(&expr.subscript_expr, field_refs, filter_refs);
        }
        jinja_ast::Expr::Call(expr) => {
            collect_from_expr(&expr.expr, field_refs, filter_refs);
            for arg in &expr.args {
                collect_from_call_arg(arg, field_refs, filter_refs);
            }
        }
        jinja_ast::Expr::List(expr) => {
            for item in &expr.items {
                collect_from_expr(item, field_refs, filter_refs);
            }
        }
        jinja_ast::Expr::Map(expr) => {
            for key in &expr.keys {
                collect_from_expr(key, field_refs, filter_refs);
            }
            for value in &expr.values {
                collect_from_expr(value, field_refs, filter_refs);
            }
        }
    }
}

fn collect_from_call_arg(
    arg: &jinja_ast::CallArg<'_>,
    field_refs: &mut BTreeSet<String>,
    filter_refs: &mut BTreeSet<String>,
) {
    match arg {
        jinja_ast::CallArg::Pos(expr)
        | jinja_ast::CallArg::PosSplat(expr)
        | jinja_ast::CallArg::Kwarg(_, expr)
        | jinja_ast::CallArg::KwargSplat(expr) => {
            collect_from_expr(expr, field_refs, filter_refs);
        }
    }
}

fn render_registration_tokens(
    attrs: &[RenderAttr],
    target: RenderTarget,
    internal_name_expr: &proc_macro2::TokenStream,
) -> syn::Result<Vec<proc_macro2::TokenStream>> {
    let mut calls = Vec::new();
    for attr in attrs {
        let (style, spec) = render_spec_tokens(attr)?;
        let key = match target {
            RenderTarget::Class => quote! {
                ::dspy_rs::baml_bridge::RendererKey::for_class(
                    #internal_name_expr.to_string(),
                    ::dspy_rs::baml_bridge::baml_types::StreamingMode::NonStreaming,
                    #style,
                )
            },
            RenderTarget::Enum => quote! {
                ::dspy_rs::baml_bridge::RendererKey::for_enum(
                    #internal_name_expr.to_string(),
                    #style,
                )
            },
        };
        calls.push(quote! {
            reg.register_renderer(#key, #spec)
        });
    }
    Ok(calls)
}

fn render_spec_tokens(
    attr: &RenderAttr,
) -> syn::Result<(LitStr, proc_macro2::TokenStream)> {
    if let Some(default) = &attr.default {
        let style = LitStr::new("default", default.span());
        return Ok((
            style,
            quote! { ::dspy_rs::baml_bridge::RendererSpec::Jinja { source: #default } },
        ));
    }

    let style = attr
        .style
        .clone()
        .unwrap_or_else(|| LitStr::new("default", attr.span));

    match (&attr.template, &attr.func) {
        (Some(template), None) => Ok((
            style,
            quote! { ::dspy_rs::baml_bridge::RendererSpec::Jinja { source: #template } },
        )),
        (None, Some(func)) => Ok((
            style,
            quote! { ::dspy_rs::baml_bridge::RendererSpec::Func { f: #func } },
        )),
        (None, None) => Err(syn::Error::new(
            attr.span,
            "render attribute missing template or fn",
        )),
        (Some(_), Some(_)) => Err(syn::Error::new(
            attr.span,
            "render attribute must specify either template or fn, not both",
        )),
    }
}

fn parse_field_attrs(attrs: &[Attribute]) -> syn::Result<FieldAttrs> {
    let mut out = FieldAttrs::default();
    for attr in attrs {
        if attr.path().is_ident("baml") {
            parse_baml_meta(attr, |meta| {
                match meta {
                Meta::NameValue(meta) if meta.path.is_ident("alias") => {
                    out.alias = Some(parse_string_expr(&meta.value, meta.span())?);
                    Ok(())
                }
                Meta::NameValue(meta) if meta.path.is_ident("description") => {
                    out.description = parse_optional_string(&meta.value, meta.span())?;
                    Ok(())
                }
                Meta::NameValue(meta) if meta.path.is_ident("with") => {
                    let path_str = parse_string_expr(&meta.value, meta.span())?;
                    out.with = Some(syn::parse_str::<Path>(&path_str)?);
                    Ok(())
                }
                Meta::NameValue(meta) if meta.path.is_ident("int_repr") => {
                    out.int_repr = Some(parse_int_repr(&meta.value, meta.span())?);
                    Ok(())
                }
                Meta::NameValue(meta) if meta.path.is_ident("map_key_repr") => {
                    out.map_key_repr = Some(parse_map_key_repr(&meta.value, meta.span())?);
                    Ok(())
                }
                Meta::List(meta) if meta.path.is_ident("check") => {
                    out.constraints.push(parse_constraint(meta, ConstraintLevelSpec::Check)?);
                    Ok(())
                }
                Meta::List(meta) if meta.path.is_ident("assert") => {
                    out.constraints.push(parse_constraint(meta, ConstraintLevelSpec::Assert)?);
                    Ok(())
                }
                Meta::Path(path) if path.is_ident("skip") => {
                    out.skip = true;
                    Ok(())
                }
                Meta::Path(path) if path.is_ident("default") => {
                    out.default = true;
                    Ok(())
                }
                _ => Err(syn::Error::new_spanned(
                    meta,
                    "unsupported #[baml(...)] attribute; hint: check the supported keys in the bridge docs",
                )),
            }
            })?;
        }

        if attr.path().is_ident("serde") {
            parse_serde_meta(attr, |meta| {
                match meta {
                Meta::NameValue(meta) if meta.path.is_ident("rename") => {
                    if out.alias.is_none() {
                        out.alias = Some(parse_string_expr(&meta.value, meta.span())?);
                    }
                    Ok(())
                }
                Meta::Path(path) if path.is_ident("skip") => {
                    out.skip = true;
                    Ok(())
                }
                Meta::Path(path) if path.is_ident("default") => {
                    out.default = true;
                    Ok(())
                }
                Meta::NameValue(meta) if meta.path.is_ident("default") => Err(syn::Error::new_spanned(
                    meta,
                    "serde(default = \"path\") is not supported; hint: use #[baml(default)] or Default::default",
                )),
                Meta::Path(path) if path.is_ident("flatten") => Err(syn::Error::new_spanned(
                    path,
                    "serde(flatten) is not supported; hint: model fields explicitly",
                )),
                _ => Ok(()),
            }
            })?;
        }
    }
    Ok(out)
}

fn parse_variant_attrs(attrs: &[Attribute]) -> syn::Result<VariantAttrs> {
    let mut out = VariantAttrs::default();
    for attr in attrs {
        if attr.path().is_ident("baml") {
            parse_baml_meta(attr, |meta| {
                match meta {
                Meta::NameValue(meta) if meta.path.is_ident("alias") => {
                    out.alias = Some(parse_string_expr(&meta.value, meta.span())?);
                    Ok(())
                }
                Meta::NameValue(meta) if meta.path.is_ident("description") => {
                    out.description = parse_optional_string(&meta.value, meta.span())?;
                    Ok(())
                }
                _ => Err(syn::Error::new_spanned(
                    meta,
                    "unsupported #[baml(...)] attribute; hint: check the supported keys in the bridge docs",
                )),
            }
            })?;
        }

        if attr.path().is_ident("serde") {
            parse_serde_meta(attr, |meta| {
                match meta {
                Meta::NameValue(meta) if meta.path.is_ident("rename") => {
                    if out.alias.is_none() {
                        out.alias = Some(parse_string_expr(&meta.value, meta.span())?);
                    }
                    Ok(())
                }
                Meta::Path(path) if path.is_ident("skip") => Err(syn::Error::new_spanned(
                    path,
                    "serde(skip) is not supported on enum variants; hint: remove the variant or use a separate enum",
                )),
                _ => Ok(()),
            }
            })?;
        }
    }
    Ok(out)
}

fn parse_baml_meta(
    attr: &Attribute,
    mut handle: impl FnMut(Meta) -> syn::Result<()>,
) -> syn::Result<()> {
    let metas = parse_meta_list(attr)?;
    for meta in metas {
        handle(meta)?;
    }
    Ok(())
}

fn parse_serde_meta(
    attr: &Attribute,
    mut handle: impl FnMut(Meta) -> syn::Result<()>,
) -> syn::Result<()> {
    let metas = parse_meta_list(attr)?;
    for meta in metas {
        handle(meta)?;
    }
    Ok(())
}

fn parse_meta_list(attr: &Attribute) -> syn::Result<Vec<Meta>> {
    match attr
        .parse_args_with(syn::punctuated::Punctuated::<Meta, syn::Token![,]>::parse_terminated)
    {
        Ok(list) => Ok(list.into_iter().collect()),
        Err(err) => Err(err),
    }
}

fn parse_constraint(
    meta: syn::MetaList,
    level: ConstraintLevelSpec,
) -> syn::Result<ConstraintSpec> {
    let nested = meta
        .parse_args_with(syn::punctuated::Punctuated::<Meta, syn::Token![,]>::parse_terminated)?;
    let mut label = None;
    let mut expr = None;

    for meta in nested {
        match meta {
            Meta::NameValue(meta) if meta.path.is_ident("label") => {
                label = Some(parse_string_expr(&meta.value, meta.span())?);
            }
            Meta::NameValue(meta) if meta.path.is_ident("expr") => {
                expr = Some(parse_string_expr(&meta.value, meta.span())?);
            }
            _ => {
                return Err(syn::Error::new_spanned(
                    meta,
                    "unsupported constraint attribute; hint: use #[baml(check(...))] or #[baml(assert(...))]",
                ));
            }
        }
    }

    let label = label.ok_or_else(|| {
        syn::Error::new(
            meta.span(),
            "constraint missing label; hint: use label = \"...\"",
        )
    })?;
    let expr = expr.ok_or_else(|| {
        syn::Error::new(
            meta.span(),
            "constraint missing expr; hint: use expr = \"...\"",
        )
    })?;

    Ok(ConstraintSpec { level, label, expr })
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

fn parse_lit_str_expr(expr: &Expr, span: proc_macro2::Span) -> syn::Result<LitStr> {
    match expr {
        Expr::Lit(ExprLit {
            lit: Lit::Str(s), ..
        }) => Ok(s.clone()),
        _ => Err(syn::Error::new(
            span,
            "expected string literal; hint: wrap the value in quotes",
        )),
    }
}

fn parse_path_expr(expr: &Expr, span: proc_macro2::Span) -> syn::Result<Path> {
    match expr {
        Expr::Path(path) => Ok(path.path.clone()),
        _ => Err(syn::Error::new(
            span,
            "expected path; hint: use a function path like crate::render",
        )),
    }
}

fn parse_bool_expr(expr: &Expr, span: proc_macro2::Span) -> syn::Result<bool> {
    match expr {
        Expr::Lit(ExprLit {
            lit: Lit::Bool(b), ..
        }) => Ok(b.value()),
        _ => Err(syn::Error::new(
            span,
            "expected boolean literal; hint: use true or false",
        )),
    }
}

fn parse_optional_string(expr: &Expr, span: proc_macro2::Span) -> syn::Result<Option<String>> {
    if let Expr::Path(path) = expr {
        if path.path.is_ident("None") {
            return Ok(None);
        }
    }
    Ok(Some(parse_string_expr(expr, span)?))
}

fn parse_rename_rule(expr: &Expr, span: proc_macro2::Span) -> syn::Result<RenameRule> {
    let value = parse_string_expr(expr, span)?;
    let rule = match value.as_str() {
        "camelCase" => RenameRule::Camel,
        "snake_case" => RenameRule::Snake,
        "PascalCase" => RenameRule::Pascal,
        "kebab-case" => RenameRule::Kebab,
        "SCREAMING_SNAKE_CASE" => RenameRule::ScreamingSnake,
        "lowercase" => RenameRule::Lower,
        "UPPERCASE" => RenameRule::Upper,
        "SCREAMING-KEBAB-CASE" => RenameRule::ScreamingKebab,
        _ => {
            return Err(syn::Error::new(span, "unsupported rename_all value"));
        }
    };
    Ok(rule)
}

fn parse_int_repr(expr: &Expr, span: proc_macro2::Span) -> syn::Result<IntRepr> {
    let value = parse_string_expr(expr, span)?;
    match value.as_str() {
        "string" => Ok(IntRepr::String),
        "i64" => Ok(IntRepr::I64),
        _ => Err(syn::Error::new(
            span,
            "int_repr must be \"string\" or \"i64\"",
        )),
    }
}

fn parse_map_key_repr(expr: &Expr, span: proc_macro2::Span) -> syn::Result<MapKeyRepr> {
    let value = parse_string_expr(expr, span)?;
    match value.as_str() {
        "string" => Ok(MapKeyRepr::String),
        "pairs" => Ok(MapKeyRepr::Pairs),
        _ => Err(syn::Error::new(
            span,
            "map_key_repr must be \"string\" or \"pairs\"",
        )),
    }
}

fn type_ident(ty: &Type) -> Option<&syn::Ident> {
    match ty {
        Type::Path(path) if path.qself.is_none() => path.path.segments.last().map(|s| &s.ident),
        _ => None,
    }
}

fn is_option_type(ty: &Type) -> Option<&Type> {
    if let Type::Path(path) = ty {
        if let Some(segment) = path.path.segments.last() {
            if segment.ident == "Option" {
                if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                    if let Some(syn::GenericArgument::Type(inner)) = args.args.first() {
                        return Some(inner);
                    }
                }
            }
        }
    }
    None
}

fn is_vec_type(ty: &Type) -> Option<&Type> {
    if let Type::Path(path) = ty {
        if let Some(segment) = path.path.segments.last() {
            if segment.ident == "Vec" {
                if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                    if let Some(syn::GenericArgument::Type(inner)) = args.args.first() {
                        return Some(inner);
                    }
                }
            }
        }
    }
    None
}

fn is_box_type(ty: &Type) -> Option<&Type> {
    extract_single_arg(ty, "Box")
}

fn is_arc_type(ty: &Type) -> Option<&Type> {
    extract_single_arg(ty, "Arc")
}

fn is_rc_type(ty: &Type) -> Option<&Type> {
    extract_single_arg(ty, "Rc")
}

fn extract_single_arg<'a>(ty: &'a Type, ident: &str) -> Option<&'a Type> {
    if let Type::Path(path) = ty {
        if let Some(segment) = path.path.segments.last() {
            if segment.ident == ident {
                if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                    if let Some(syn::GenericArgument::Type(inner)) = args.args.first() {
                        return Some(inner);
                    }
                }
            }
        }
    }
    None
}

fn is_string_map_type(ty: &Type) -> Option<(&Type, &Type)> {
    if let Some((key, value)) = map_types(ty) {
        if is_string_type(key) {
            return Some((key, value));
        }
    }
    None
}

fn map_types(ty: &Type) -> Option<(&Type, &Type)> {
    if let Type::Path(path) = ty {
        if let Some(segment) = path.path.segments.last() {
            if segment.ident == "HashMap" || segment.ident == "BTreeMap" {
                if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
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
            }
        }
    }
    None
}

fn map_types_for_repr(ty: &Type) -> Option<(&Type, &Type)> {
    if let Some((key, value)) = map_types(ty) {
        return Some((key, value));
    }
    if let Some(inner) = is_option_type(ty) {
        return map_types_for_repr(inner);
    }
    if let Some(inner) = is_vec_type(ty) {
        return map_types_for_repr(inner);
    }
    if let Some(inner) = is_box_type(ty) {
        return map_types_for_repr(inner);
    }
    if let Some(inner) = is_arc_type(ty) {
        return map_types_for_repr(inner);
    }
    if let Some(inner) = is_rc_type(ty) {
        return map_types_for_repr(inner);
    }
    None
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
    if let Type::Path(path) = ty {
        if let Some(segment) = path.path.segments.last() {
            if segment.ident == "Value" {
                return path
                    .path
                    .segments
                    .iter()
                    .any(|seg| seg.ident == "serde_json");
            }
        }
    }
    false
}

fn apply_rename(rule: RenameRule, name: &str) -> String {
    use convert_case::{Case, Casing};
    let case = match rule {
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

fn extract_doc(attrs: &[Attribute]) -> Option<String> {
    let mut lines = Vec::new();
    for attr in attrs {
        if !attr.path().is_ident("doc") {
            continue;
        }
        if let Meta::NameValue(meta) = &attr.meta {
            if let Expr::Lit(ExprLit {
                lit: Lit::Str(s), ..
            }) = &meta.value
            {
                let mut value = s.value();
                if value.starts_with(' ') {
                    value.remove(0);
                }
                lines.push(value);
            }
        }
    }

    if lines.is_empty() {
        return None;
    }

    let joined = lines.join("\n");
    let trimmed = joined.trim_end().to_string();
    if trimmed.trim().is_empty() {
        None
    } else {
        Some(trimmed)
    }
}
