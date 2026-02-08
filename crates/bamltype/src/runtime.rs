//! Facet runtime traits and conversion helpers.

use baml_types::{BamlMap, BamlValue, TypeIR, type_meta};
use facet::Facet;
use internal_baml_jinja::types::OutputFormatContent;

use crate::BamlSchema;
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

/// Internal type metadata.
pub trait BamlTypeInternal {
    fn baml_internal_name() -> &'static str;
    fn baml_type_ir() -> TypeIR;
}

/// Convert from BamlValue to Rust.
pub trait BamlValueConvert: Sized {
    fn try_from_baml_value(value: BamlValue, path: Vec<String>) -> Result<Self, BamlConvertError>;
}

/// Convert from Rust to BamlValue.
pub trait ToBamlValue {
    fn to_baml_value(&self) -> BamlValue;
}

/// Runtime trait used by schema rendering and parsing.
///
/// Named `BamlTypeTrait` to avoid collision with the `#[BamlType]` attribute macro.
pub trait BamlTypeTrait: BamlTypeInternal + BamlValueConvert + Sized + 'static {
    fn baml_output_format() -> &'static OutputFormatContent;

    fn baml_internal_name() -> &'static str {
        <Self as BamlTypeInternal>::baml_internal_name()
    }

    fn baml_type_ir() -> TypeIR {
        <Self as BamlTypeInternal>::baml_type_ir()
    }
}

impl<T: Facet<'static>> BamlTypeInternal for T {
    fn baml_internal_name() -> &'static str {
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

    fn baml_type_ir() -> TypeIR {
        build_type_ir_from_shape(T::SHAPE)
    }
}

impl<T: Facet<'static>> BamlValueConvert for T {
    fn try_from_baml_value(value: BamlValue, _path: Vec<String>) -> Result<Self, BamlConvertError> {
        convert::from_baml_value(value).map_err(BamlConvertError::from)
    }
}

impl<T: Facet<'static>> ToBamlValue for T {
    fn to_baml_value(&self) -> BamlValue {
        convert::to_baml_value(self).unwrap_or(BamlValue::Null)
    }
}

impl<T: BamlSchema> BamlTypeTrait for T {
    fn baml_output_format() -> &'static OutputFormatContent {
        &T::baml_schema().output_format
    }
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
