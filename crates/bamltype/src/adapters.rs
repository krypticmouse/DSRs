//! Adapter extension points for advanced field-level conversions.

use baml_types::{BamlValue, TypeIR};

use crate::BamlConvertError;

/// Registry type used by field codecs to register custom schema artifacts.
pub type AdapterSchemaRegistry = crate::schema_registry::SchemaRegistry;

/// Field-level conversion + schema representation hook.
pub trait FieldCodec<T> {
    fn type_ir() -> TypeIR;
    fn register(_reg: &mut AdapterSchemaRegistry) {}
    fn try_from_baml(value: BamlValue, path: Vec<String>) -> Result<T, BamlConvertError>;
}
