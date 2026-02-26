use proc_macro_crate::{FoundCrate, crate_name};
use proc_macro2::Span;
use syn::Path;

pub(crate) fn resolve_dspy_rs_path() -> syn::Result<Path> {
    match crate_name("dspy-rs") {
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

pub(crate) fn ensure_facet_resolvable() -> syn::Result<()> {
    match crate_name("facet") {
        Ok(_) => Ok(()),
        Err(_) => Err(syn::Error::new(
            Span::call_site(),
            "rlm_type requires `facet` to be resolvable; add `facet` as a dependency or use a dspy-rs workspace crate that already depends on facet",
        )),
    }
}
