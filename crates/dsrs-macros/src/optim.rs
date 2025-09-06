use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Field, Fields, parse_macro_input, parse_str};

pub fn optimizable_impl(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    // Define trait path as a constant - easy to change in one place
    let trait_path = parse_str::<syn::Path>("::dspy_rs::core::module::Optimizable").unwrap();

    // Extract parameter field names
    let parameter_fields = extract_parameter_fields(&input);

    let name = &input.ident;
    let generics = &input.generics;
    let (impl_generics, type_generics, where_clause) = generics.split_for_impl();
    let parameter_names: Vec<_> = parameter_fields
        .iter()
        .map(|field| field.ident.as_ref().unwrap())
        .collect();

    // Generate the Optimizable implementation
    let expanded = quote! {
        impl #impl_generics #trait_path for #name #type_generics #where_clause {
            fn parameters(
                &mut self,
            ) -> ::std::collections::HashMap<::std::string::String, &mut dyn #trait_path> {
                let mut params = ::std::collections::HashMap::new();
                #(
                    params.insert(stringify!(#parameter_names).to_string(), &mut self.#parameter_names as &mut dyn #trait_path);
                )*
                params
            }
        }
    };

    TokenStream::from(expanded)
}

fn extract_parameter_fields(input: &DeriveInput) -> Vec<&Field> {
    match &input.data {
        Data::Struct(data_struct) => match &data_struct.fields {
            Fields::Named(fields_named) => fields_named
                .named
                .iter()
                .filter(|field| has_parameter_attribute(field))
                .collect(),
            _ => {
                panic!("Optimizable can only be derived for structs with named fields");
            }
        },
        _ => {
            panic!("Optimizable can only be derived for structs");
        }
    }
}

fn has_parameter_attribute(field: &Field) -> bool {
    field
        .attrs
        .iter()
        .any(|attr| attr.path().is_ident("parameter"))
}

#[test]
fn trybuild() {
    let t = trybuild::TestCases::new();
    t.pass("tests/optim/*.rs");
}
