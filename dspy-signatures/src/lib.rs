extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{DeriveInput, parse_macro_input};

#[allow(unused_assignments)]
#[proc_macro_derive(Signature)]
pub fn struct_signature(item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as DeriveInput);
    let ident = &input.ident;

    // gather each field name + concatenated doc text
    let mut first_field_doc = String::new();
    let mut input_field_insertions = vec![];
    let mut output_field_insertions = vec![];

    match &input.data {
        syn::Data::Struct(s) => {
            // Get doc comments for the first field only
            let mut is_first = true;
            if let syn::Fields::Named(named) = &s.fields {
                for f in &named.named {
                    if is_first {
                        first_field_doc = f
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
                        is_first = false;
                        break;
                    }
                }
            }

            for field in &s.fields {
                let identifier = field.ident.as_ref().unwrap();
                let ty = &field.ty;

                match ty {
                    syn::Type::Path(type_path) => {
                        if let Some(first_segment) = type_path.path.segments.first() {
                            if first_segment.ident == "In" {
                                // Get the inner type
                                if let syn::PathArguments::AngleBracketed(args) =
                                    &first_segment.arguments
                                {
                                    if let Some(syn::GenericArgument::Type(inner_ty)) =
                                        args.args.first()
                                    {
                                        input_field_insertions.push(quote! {
                                            let json_value = serde_json::to_value(&schemars::schema_for!(#inner_ty)).unwrap();

                                            let schema = if let Some(properties) = json_value.as_object()
                                                .and_then(|obj| obj.get("properties"))
                                                .and_then(|props| props.as_object()) {
                                                serde_json::to_string(&properties).unwrap_or_else(|_| "".to_string())
                                            } else {
                                                "".to_string()
                                            };

                                            input_fields.insert(stringify!(#identifier).to_string(), dspy_rs::internal::MetaField {
                                                desc: String::new(),
                                                schema: schema,
                                                data_type: stringify!(#inner_ty).to_string(),
                                                __dsrs_field_type: "Input".to_string(),
                                            });
                                        });
                                    }
                                }
                            } else if first_segment.ident == "Out" {
                                // Get the inner type
                                if let syn::PathArguments::AngleBracketed(args) =
                                    &first_segment.arguments
                                {
                                    if let Some(syn::GenericArgument::Type(inner_ty)) =
                                        args.args.first()
                                    {
                                        output_field_insertions.push(quote! {
                                            let json_value = serde_json::to_value(&schemars::schema_for!(#inner_ty)).unwrap();

                                            let schema = if let Some(properties) = json_value.as_object()
                                                .and_then(|obj| obj.get("properties"))
                                                .and_then(|props| props.as_object()) {
                                                serde_json::to_string(&properties).unwrap_or_else(|_| "".to_string())
                                            } else {
                                                "".to_string()
                                            };

                                            output_fields.insert(stringify!(#identifier).to_string(), dspy_rs::internal::MetaField {
                                                desc: String::new(),
                                                schema: schema,
                                                data_type: stringify!(#inner_ty).to_string(),
                                                __dsrs_field_type: "Output".to_string(),
                                            });
                                        });
                                    }
                                }
                            }
                        }
                    }
                    _ => unimplemented!("Only structs and inlines are supported"),
                }
            }
        }
        _ => {}
    }

    let generated = quote! {
        impl #ident {
            pub fn new() -> dspy_rs::internal::MetaSignature {
                let mut input_fields = indexmap::IndexMap::new();
                let mut output_fields = indexmap::IndexMap::new();

                #(#input_field_insertions)*
                #(#output_field_insertions)*

                dspy_rs::internal::MetaSignature {
                    name: stringify!(#ident).to_string(),
                    instruction: #first_field_doc.to_string(),
                    input_fields: input_fields,
                    output_fields: output_fields,
                }
            }
        }
    };
    generated.into()
}
