use proc_macro_crate::{FoundCrate, crate_name};
use proc_macro2::Span;

pub(crate) fn resolve_dsrs_core_path() -> syn::Result<syn::Path> {
    match crate_name("dsrs-core") {
        // `crate` fails in examples/binaries inside the dsrs-core package because
        // there it points at the example crate, not the library. Use the crate
        // alias (`extern crate self as dsrs_core`) for a stable path.
        Ok(FoundCrate::Itself) => Ok(syn::parse_quote!(::dsrs_core)),
        Ok(FoundCrate::Name(name)) => {
            let ident = syn::Ident::new(&name.replace('-', "_"), Span::call_site());
            Ok(syn::parse_quote!(::#ident))
        }
        Err(_) => Err(syn::Error::new(
            Span::call_site(),
            "could not resolve `dsrs-core`; add it as a dependency (renamed dependencies are supported)",
        )),
    }
}
