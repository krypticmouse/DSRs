//! Facet extension attributes used by bamltype.

use baml_types::{BamlValue, TypeIR};

use crate::adapters::FieldCodecRegisterContext;
use crate::runtime::BamlConvertError;

/// Field-level adapter application function.
pub type AdapterApplyFn = fn(
    facet_reflect::Partial<'static>,
    BamlValue,
    Vec<String>,
) -> Result<facet_reflect::Partial<'static>, BamlConvertError>;

/// Field-level adapter schema registration function.
pub type AdapterRegisterFn = for<'a> fn(FieldCodecRegisterContext<'a>);

/// Runtime hooks for `#[baml(with = "...")]`.
#[derive(Clone, Copy, Debug, facet::Facet)]
#[facet(opaque)]
pub struct WithAdapterFns {
    /// Schema type representation callback.
    pub type_ir: fn() -> TypeIR,
    /// Schema registration callback.
    pub register: AdapterRegisterFn,
    /// Value conversion callback.
    pub apply: AdapterApplyFn,
}

impl PartialEq for WithAdapterFns {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::fn_addr_eq(self.type_ir, other.type_ir)
            && std::ptr::fn_addr_eq(self.register, other.register)
            && std::ptr::fn_addr_eq(self.apply, other.apply)
    }
}

impl Eq for WithAdapterFns {}

/// Resolve `#[facet(bamltype::with = ...)]` payload from a field attribute list.
pub fn with_adapter_fns(attrs: &'static [facet::Attr]) -> Option<&'static WithAdapterFns> {
    for attr in attrs {
        if attr.ns != Some("bamltype") || attr.key != "with" {
            continue;
        }

        let Some(ext_attr) = attr.get_as::<Attr>() else {
            continue;
        };

        if let Attr::With(Some(fns)) = ext_attr {
            return Some(*fns);
        }
    }

    None
}

facet::define_attr_grammar! {
    ns "bamltype";
    crate_path $crate::facet_ext;

    /// Constraint payload for BAML-compatible `check` / `assert` attributes.
    pub struct BamlConstraint {
        /// Constraint label.
        pub label: &'static str,
        /// Constraint expression.
        pub expr: &'static str,
    }

    /// bamltype extension attrs consumed by derive and runtime conversion/schema.
    pub enum Attr {
        /// Container-level override for internal type name.
        InternalName(&'static str),
        /// Field-level integer representation strategy.
        IntRepr(&'static str),
        /// Field-level map key representation strategy.
        MapKeyRepr(&'static str),
        /// Field-level adapter payload (`#[facet(bamltype::with = ...)]`).
        With(Option<&'static WithAdapterFns>),
        /// BAML-compatible check constraint (`#[facet(bamltype::check(...))]`).
        Check(BamlConstraint),
        /// BAML-compatible assert constraint (`#[facet(bamltype::assert(...))]`).
        Assert(BamlConstraint),
    }
}
