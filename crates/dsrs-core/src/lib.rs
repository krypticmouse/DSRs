//! Core typed substrate for DSRs.

// TODO(dsrs-facet-lint-scope): remove this crate-level allow once Facet's generated
// extension-attr dispatch no longer triggers rust-lang/rust#52234 on in-crate usage.
#![allow(macro_expanded_macro_exports_accessed_by_absolute_paths)]

mod augmentation;
mod demo;
pub mod dyn_predictor;
mod errors;
mod example;
mod module;
mod module_ext;
mod predicted;
mod prediction;
mod schema;
mod signature;
mod specials;
mod usage;

pub use augmentation::*;
pub use demo::*;
pub use dyn_predictor::*;
pub use errors::{ConversionError, ErrorClass, JsonishError, LmError, ParseError, PredictError};
pub use example::Example as RawExample;
pub use module::*;
pub use module_ext::*;
pub use predicted::{CallMetadata, ConstraintResult, FieldMeta, Predicted};
pub use prediction::*;
pub use schema::{FieldMetadataSpec, FieldPath, FieldSchema, InputRenderSpec, SignatureSchema};
pub use signature::*;
pub use specials::*;
pub use usage::*;

pub use bamltype::BamlConvertError;
pub use bamltype::BamlType;
pub use bamltype::Shape;
pub use bamltype::baml_types::{
    BamlValue, Constraint, ConstraintLevel, ResponseCheck, StreamingMode, TypeIR,
};
pub use bamltype::internal_baml_jinja::types::{OutputFormatContent, RenderOptions};
pub use bamltype::jsonish::deserializer::deserialize_flags::Flag;
pub use facet::Facet;

#[derive(Clone, Debug, serde::Serialize)]
pub struct TrackedValue {
    pub value: serde_json::Value,
    #[serde(skip)]
    pub source: Option<(usize, String)>,
}

impl TrackedValue {
    pub fn new(value: serde_json::Value, source: Option<(usize, String)>) -> Self {
        Self { value, source }
    }
}
