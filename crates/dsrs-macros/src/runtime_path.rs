use proc_macro_crate::{FoundCrate, crate_name};
use proc_macro2::Span;

pub(crate) fn resolve_dspy_rs_path() -> syn::Result<syn::Path> {
    match crate_name("dspy-rs") {
        // `crate` fails in examples/binaries inside the dspy-rs package because
        // there it points at the example crate, not the library. Use the crate
        // alias (`extern crate self as dspy_rs`) for a stable path.
        Ok(FoundCrate::Itself) => Ok(syn::parse_quote!(::dspy_rs)),
        Ok(FoundCrate::Name(name)) => {
            let ident = syn::Ident::new(&name.replace('-', "_"), Span::call_site());
            Ok(syn::parse_quote!(::#ident))
        }
        Err(_) => Err(syn::Error::new(
            Span::call_site(),
            "could not resolve `dspy-rs`; add it as a dependency (renamed dependencies are supported)",
        )),
    }
}
