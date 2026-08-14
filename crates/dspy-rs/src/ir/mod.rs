//! The intermediate representation (RFC 0002).
//!
//! Stage IR-1: value-level signatures. [`SignatureDef`] is a signature as an owned
//! runtime value — constructible without macros via [`SignatureDef::build`], bridged
//! from `#[derive(Signature)]` types via [`SignatureDef::of`], and serde-derivable
//! for the program artifact. The type model is [`typesys::FieldType`](crate::typesys::FieldType)
//! unchanged; class/enum definitions live in a program-owned [`TypeTable`].
//!
//! Later stages add parameters/overlays (IR-2), the graph core and interpreter
//! (IR-3+), and the text format (IR-5).

pub mod sig;

pub use crate::typesys::{ClassDef, EnumDef, EnumValueDef, FieldType, TypeTable};
pub use sig::{
    ConstraintDef, FieldDef, RenderSpec, SigError, SigMismatch, SignatureBuilder, SignatureDef,
};
