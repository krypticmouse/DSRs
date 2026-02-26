use proc_macro::TokenStream;
use syn::parse_macro_input;

mod rlm_attr;
mod rlm_type;
mod runtime_path;

#[proc_macro_attribute]
pub fn rlm_type(attr: TokenStream, item: TokenStream) -> TokenStream {
    rlm_attr::expand(attr, item)
}

#[proc_macro_derive(RlmType, attributes(rlm))]
pub fn derive_rlm_type(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::DeriveInput);
    match rlm_type::derive(input) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}
