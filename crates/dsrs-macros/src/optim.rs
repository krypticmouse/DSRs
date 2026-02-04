use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Field, Fields, parse_macro_input};

use crate::runtime_path::resolve_dspy_rs_path;

pub fn optimizable_impl(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let runtime = match resolve_dspy_rs_path() {
        Ok(path) => path,
        Err(err) => return err.to_compile_error().into(),
    };
    let trait_path: syn::Path = syn::parse_quote!(#runtime::core::module::Optimizable);

    // Extract parameter field names
    let parameter_fields = match extract_parameter_fields(&input) {
        Ok(fields) => fields,
        Err(err) => return err.to_compile_error().into(),
    };

    let name = &input.ident;
    let generics = &input.generics;
    let (impl_generics, type_generics, where_clause) = generics.split_for_impl();
    let mut parameter_names = Vec::with_capacity(parameter_fields.len());
    for field in &parameter_fields {
        let Some(ident) = field.ident.as_ref() else {
            return syn::Error::new_spanned(
                field,
                "Optimizable can only be derived for structs with named fields",
            )
            .to_compile_error()
            .into();
        };
        parameter_names.push(ident);
    }

    // Generate the Optimizable implementation (flatten nested parameters with compound names)
    let expanded = quote! {
        impl #impl_generics #trait_path for #name #type_generics #where_clause {
            fn parameters(
                &mut self,
            ) -> #runtime::indexmap::IndexMap<::std::string::String, &mut dyn #trait_path> {
                let mut params: #runtime::indexmap::IndexMap<::std::string::String, &mut dyn #trait_path> = #runtime::indexmap::IndexMap::new();
                #(
                {
                    let __field_name = stringify!(#parameter_names).to_string();
                    // SAFETY: We only create disjoint mutable borrows to distinct struct fields
                    let __field_ptr: *mut dyn #trait_path = &mut self.#parameter_names as *mut dyn #trait_path;
                    let __child_params: #runtime::indexmap::IndexMap<::std::string::String, &mut dyn #trait_path> = unsafe { (&mut *__field_ptr).parameters() };
                    if __child_params.is_empty() {
                        // Leaf: insert the field itself
                        unsafe {
                            params.insert(__field_name, &mut *__field_ptr);
                        }
                    } else {
                        // Composite: flatten children with compound names
                        for (grand_name, grand_param) in __child_params.into_iter() {
                            params.insert(format!("{}.{}", __field_name, grand_name), grand_param);
                        }
                    }
                }
                )*
                params
            }
        }
    };

    TokenStream::from(expanded)
}

fn extract_parameter_fields(input: &DeriveInput) -> syn::Result<Vec<&Field>> {
    match &input.data {
        Data::Struct(data_struct) => match &data_struct.fields {
            Fields::Named(fields_named) => Ok(fields_named
                .named
                .iter()
                .filter(|field| has_parameter_attribute(field))
                .collect()),
            _ => Err(syn::Error::new_spanned(
                input,
                "Optimizable can only be derived for structs with named fields",
            )),
        },
        _ => Err(syn::Error::new_spanned(
            input,
            "Optimizable can only be derived for structs",
        )),
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
