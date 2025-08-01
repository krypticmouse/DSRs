use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    parse_macro_input, DeriveInput, Fields
};

#[proc_macro_derive(Signature)]
pub fn Signature(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let ast: DeriveInput = parse_macro_input!(input);
    let ident = &ast.ident;

    // gather each field name + concatenated doc text
    let mut first_field_doc = String::new();
    let mut input_field_inits = Vec::<proc_macro2::TokenStream>::new();
    let mut output_field_inits = Vec::<proc_macro2::TokenStream>::new();
    
    if let syn::Data::Struct(s) = ast.data {
        if let Fields::Named(named) = s.fields {
            let mut is_first = true;
            for f in named.named {
                let name = f.ident.unwrap();
                
                // Get doc comments for the first field only
                if is_first {
                    first_field_doc = f.attrs.iter()
                        .filter(|a| a.path().is_ident("doc"))
                        .filter_map(|a| match &a.meta {
                            syn::Meta::NameValue(nv) => match &nv.value {
                                syn::Expr::Lit(syn::ExprLit { lit: syn::Lit::Str(s), .. }) => Some(s.value()),
                                _ => None
                            },
                            _ => None
                        })
                        .map(|s| s.trim().to_string())
                        .collect::<Vec<_>>()
                        .join("\n");
                    is_first = false;
                }

                match &f.ty {
                    syn::Type::Path(type_path) if type_path.path.is_ident("In") => {
                        input_field_inits.push(quote! { (stringify!(#name).to_string(), dspy_rs::field::In::default()) });
                    }
                    syn::Type::Path(type_path) if type_path.path.is_ident("Out") => {
                        output_field_inits.push(quote! { (stringify!(#name).to_string(), dspy_rs::field::Out::default()) });
                    }
                    _ => {}
                }
            }
        }
    }

    let generated = quote! {
        impl #ident {
            pub fn new() -> dspy_rs::signature::Signature {
                dspy_rs::signature::Signature {
                    name: stringify!(#ident).to_string(),
                    instruction: #first_field_doc.to_string(),
                    input_fields: indexmap::IndexMap::from([#(#input_field_inits),*]),
                    output_fields: indexmap::IndexMap::from([#(#output_field_inits),*]),
                }
            }
        }
    };
    generated.into()
}