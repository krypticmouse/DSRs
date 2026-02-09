//! Adapter extension points for advanced field-level conversions.

use baml_types::{BamlValue, TypeIR};

use crate::BamlConvertError;

/// Registry type used by field codecs to register custom schema artifacts.
///
/// This is the canonical registry surface for facet-level codec extensions.
pub type AdapterSchemaRegistry = crate::schema_registry::SchemaRegistry;

/// Backward-friendly alias clarifying registry ownership at the codec boundary.
pub type FieldCodecSchemaRegistry = AdapterSchemaRegistry;

/// Context passed to `FieldCodec::register`.
#[derive(Debug)]
pub struct FieldCodecRegisterContext<'a> {
    pub registry: &'a mut AdapterSchemaRegistry,
    pub owner_internal_name: Option<&'a str>,
    pub field_name: Option<&'a str>,
    pub rendered_field_name: Option<&'a str>,
    pub variant_name: Option<&'a str>,
    pub rendered_variant_name: Option<&'a str>,
}

impl<'a> FieldCodecRegisterContext<'a> {
    pub fn new(registry: &'a mut AdapterSchemaRegistry) -> Self {
        Self {
            registry,
            owner_internal_name: None,
            field_name: None,
            rendered_field_name: None,
            variant_name: None,
            rendered_variant_name: None,
        }
    }
}

/// Field-level conversion + schema representation hook.
pub trait FieldCodec<T> {
    fn type_ir() -> TypeIR;
    fn register(_ctx: FieldCodecRegisterContext<'_>) {}
    fn try_from_baml(value: BamlValue, path: Vec<String>) -> Result<T, BamlConvertError>;
}
