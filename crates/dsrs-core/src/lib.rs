//! Core typed substrate for DSRs.

// TODO(dsrs-facet-lint-scope): remove this crate-level allow once Facet's generated
// extension-attr dispatch no longer triggers rust-lang/rust#52234 on in-crate usage.
#![allow(macro_expanded_macro_exports_accessed_by_absolute_paths)]

pub mod augmentation;
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
pub use dsrs_macros::*;
pub use facet::Facet;

#[doc(hidden)]
pub mod __macro_support {
    pub use anyhow;
    pub use bamltype;
    pub use indexmap;
    pub use schemars;
    pub use serde;
    pub use serde_json;
}

#[macro_export]
macro_rules! field {
    { $($field_type:ident[$desc:literal] => $field_name:ident : $field_ty:ty),* $(,)? } => {{
        use $crate::__macro_support::serde_json::json;

        let mut result = $crate::__macro_support::serde_json::Map::new();

        $(
            let type_str = stringify!($field_ty);
            let schema = {
                let schema = $crate::__macro_support::schemars::schema_for!($field_ty);
                let schema_json = $crate::__macro_support::serde_json::to_value(schema).unwrap();
                if let Some(obj) = schema_json.as_object() {
                    if obj.contains_key("properties") {
                        schema_json["properties"].clone()
                    } else {
                        "".to_string().into()
                    }
                } else {
                    "".to_string().into()
                }
            };
            result.insert(
                stringify!($field_name).to_string(),
                json!({
                    "type": type_str,
                    "desc": $desc,
                    "schema": schema,
                    "__dsrs_field_type": stringify!($field_type)
                })
            );
        )*

        $crate::__macro_support::serde_json::Value::Object(result)
    }};

    { $($field_type:ident => $field_name:ident : $field_ty:ty),* $(,)? } => {{
        use $crate::__macro_support::serde_json::json;

        let mut result = $crate::__macro_support::serde_json::Map::new();

        $(
            let type_str = stringify!($field_ty);
            let schema = {
                let schema = $crate::__macro_support::schemars::schema_for!($field_ty);
                let schema_json = $crate::__macro_support::serde_json::to_value(schema).unwrap();
                if let Some(obj) = schema_json.as_object() {
                    if obj.contains_key("properties") {
                        schema_json["properties"].clone()
                    } else {
                        "".to_string().into()
                    }
                } else {
                    "".to_string().into()
                }
            };
            result.insert(
                stringify!($field_name).to_string(),
                json!({
                    "type": type_str,
                    "desc": "",
                    "schema": schema,
                    "__dsrs_field_type": stringify!($field_type)
                })
            );
        )*

        $crate::__macro_support::serde_json::Value::Object(result)
    }};
}

#[macro_export]
macro_rules! hashmap {
    () => {
        ::std::collections::HashMap::new()
    };

    ($($key:expr => $value:expr),+ $(,)?) => {
        ::std::collections::HashMap::from([ $(($key, $value)),* ])
    };
}

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
