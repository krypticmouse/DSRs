//! Facet runtime helpers and conversion glue.

use baml_types::{BamlMap, BamlValue, TypeIR, type_meta};
use facet::Facet;

use crate::convert;
use crate::schema_builder::build_type_ir_from_shape;

/// Error during BamlValue ↔ Rust conversion.
#[derive(Debug, Clone)]
pub struct BamlConvertError {
    pub path: Vec<String>,
    pub expected: &'static str,
    pub got: String,
    pub message: String,
}

impl BamlConvertError {
    pub fn new(
        path: Vec<String>,
        expected: &'static str,
        got: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            path,
            expected,
            got: got.into(),
            message: message.into(),
        }
    }

    pub fn with_path(mut self, segment: impl Into<String>) -> Self {
        self.path.push(segment.into());
        self
    }

    pub fn path_string(&self) -> String {
        if self.path.is_empty() {
            "<root>".to_string()
        } else {
            self.path.join(".")
        }
    }
}

impl std::fmt::Display for BamlConvertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} (expected {}, got {}) at {}",
            self.message,
            self.expected,
            self.got,
            self.path_string()
        )
    }
}

impl std::error::Error for BamlConvertError {}

impl From<convert::ConvertError> for BamlConvertError {
    fn from(err: convert::ConvertError) -> Self {
        match err {
            convert::ConvertError::Adapter(inner) => inner,
            other => Self {
                path: Vec::new(),
                expected: "compatible type",
                got: other.to_string(),
                message: other.to_string(),
            },
        }
    }
}

/// Compute the BAML internal name for a type.
pub fn baml_internal_name<T: Facet<'static>>() -> &'static str {
    for attr in T::SHAPE.attributes {
        if attr.ns != Some("bamltype") || attr.key != "internal_name" {
            continue;
        }

        if let Some(name) = attr.get_as::<&'static str>() {
            return name;
        }
    }

    if let Some(name) = T::SHAPE.get_builtin_attr_value::<&'static str>("internal_name") {
        name
    } else {
        std::any::type_name::<T>()
    }
}

/// Compute TypeIR for a type from facet shape data.
pub fn baml_type_ir<T: Facet<'static>>() -> TypeIR {
    build_type_ir_from_shape(T::SHAPE)
}

/// Convert a BamlValue into a concrete Rust type with stable BamlConvertError shape.
pub fn try_from_baml_value<T: Facet<'static>>(value: BamlValue) -> Result<T, BamlConvertError> {
    convert::from_baml_value(value).map_err(BamlConvertError::from)
}

/// Convert a Rust value into BamlValue, preserving legacy null-fallback behavior.
pub fn to_baml_value_lossy<T: Facet<'static>>(value: &T) -> BamlValue {
    convert::to_baml_value(value).unwrap_or(BamlValue::Null)
}

/// Default streaming behavior helper.
pub fn default_streaming_behavior() -> type_meta::base::StreamingBehavior {
    type_meta::base::StreamingBehavior::default()
}

/// Lookup helper matching legacy bridge semantics (`name` then optional alias).
pub fn get_field<'a>(
    map: &'a BamlMap<String, BamlValue>,
    name: &str,
    alias: Option<&str>,
) -> Option<&'a BamlValue> {
    map.get(name)
        .or_else(|| alias.and_then(|alias| map.get(alias)))
}
