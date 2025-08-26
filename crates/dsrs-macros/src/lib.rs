extern crate self as dsrs_macros;

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Attribute, MetaNameValue, Lit};
use serde_json::{json, Value};

#[allow(unused_assignments, non_snake_case)]
#[proc_macro_attribute]
pub fn Signature(attr: TokenStream, item: TokenStream) -> TokenStream {
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
    
    if has_hint {
        input_schema["hint"] = json!({
            "type": "String",
            "desc": "Hint for the query"
        });
    }

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
                        panic!("Field {} cannot be both input and output", field_name);
                    }
                    
                    if !is_input && !is_output {
                        panic!("Field {} must have either #[input] or #[output] attribute", field_name);
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
                        "desc": field_desc
                    });
                    
                    if is_input {
                        input_schema[field_name.to_string()] = field_metadata;
                    } else if is_output {
                        output_schema[field_name.to_string()] = field_metadata;
                    }
                }
            }
        }
        _ => panic!("Signature can only be applied to structs"),
    }

    if has_cot {
        output_schema["reasoning"] = json!({
            "type": "String",
            "desc": "Think step by step"
        });
    }

    // Serialize the schemas to strings so we can embed them in the generated code
    let input_schema_str = serde_json::to_string(&input_schema).unwrap();
    let output_schema_str = serde_json::to_string(&output_schema).unwrap();
    
    let generated = quote! {
        #[derive(Default, Debug)]
        struct #struct_name {
            instruction: String,
            input_schema: serde_json::Value,
            output_schema: serde_json::Value,
        }

        impl #struct_name {
            pub fn new() -> Self {
                Self {
                    instruction: #signature_instruction.to_string(),
                    input_schema: serde_json::from_str(#input_schema_str).unwrap(),
                    output_schema: serde_json::from_str(#output_schema_str).unwrap(),
                }
            }
        }

        impl dspy_rs::core::MetaSignature for #struct_name {
            fn update_instruction(&mut self, instruction: String) -> anyhow::Result<()> {
                self.instruction = instruction;
                Ok(())
            }
        
            fn append(&mut self, name: &str, t: &str, desc: Option<String>, field_type: &str) -> anyhow::Result<()> {
                match field_type {
                    "in" | "input" => {
                        self.input_schema[name] = serde_json::json!({
                            "type": t,
                            "desc": desc
                        });
                    }
                    "out" | "output" => {
                        self.output_schema[name] = serde_json::json!({
                            "type": t,
                            "desc": desc
                        });
                    }
                    _ => {
                        return Err(anyhow::anyhow!("Invalid field type: {}", field_type));
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
    if let Ok(nv) = syn::parse2::<MetaNameValue>(tokens) {
        if nv.path.is_ident("desc") {
            if let syn::Expr::Lit(syn::ExprLit { lit: Lit::Str(s), .. }) = nv.value {
                return s.value();
            }
        }
    }
    String::new()
}
