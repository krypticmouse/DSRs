//! BamlValue ↔ Rust conversion using direct facet reflection
//!
//! This module provides bidirectional conversion between BamlValue and any
//! Rust type that implements Facet, using facet's Partial (for building)
//! and Peek (for reading) APIs.

use baml_types::{BamlMap, BamlValue};
use facet::{Def, Facet, Shape, Type, UserType};
use facet_reflect::{HasFields, HeapValue, Partial, Peek, ReflectError, ScalarType, VariantError};
use indexmap::IndexMap;

use crate::BamlValueWithFlags;
use crate::runtime::BamlConvertError;
use crate::schema_builder::internal_name_for_shape;

/// Error during BamlValue conversion.
#[derive(Debug, thiserror::Error)]
pub enum ConvertError {
    /// Error from facet reflection operations.
    #[error("Reflection error: {0}")]
    Reflect(#[from] ReflectError),

    /// Error from enum variant access.
    #[error("Variant error: {0}")]
    Variant(#[from] VariantError),

    /// Type mismatch during conversion.
    #[error("Type mismatch: expected {expected}, got {actual}")]
    TypeMismatch {
        expected: &'static str,
        actual: String,
    },

    /// Missing required field.
    #[error("Missing field: {0}")]
    MissingField(String),

    /// Unknown enum variant.
    #[error("Unknown variant: {0}")]
    UnknownVariant(String),

    /// Unsupported type for conversion.
    #[error("Unsupported type: {0}")]
    Unsupported(String),

    /// Adapter conversion error.
    #[error("{0}")]
    Adapter(#[from] BamlConvertError),
}

impl ConvertError {
    fn with_path_prefix(self, segment: impl Into<String>) -> Self {
        let segment = segment.into();
        match self {
            ConvertError::Adapter(mut inner) => {
                inner.path.insert(0, segment);
                ConvertError::Adapter(inner)
            }
            other => {
                let message = other.to_string();
                let mut inner =
                    BamlConvertError::new(Vec::new(), "compatible type", message.clone(), message);
                inner.path.insert(0, segment);
                ConvertError::Adapter(inner)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct ReprHints {
    int_repr: Option<IntReprHint>,
    map_key_repr: Option<MapKeyReprHint>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IntReprHint {
    String,
    I64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MapKeyReprHint {
    String,
    Pairs,
}

// ============================================================================
// BamlValue → Rust (using Partial API)
// ============================================================================

/// Convert a BamlValue to a Rust type using facet reflection.
pub fn from_baml_value<T: Facet<'static>>(value: BamlValue) -> Result<T, ConvertError> {
    let partial = Partial::alloc::<T>().map_err(|err| ConvertError::Reflect(err.into()))?;
    let partial = build_from_baml_value(partial, &value)?;
    let heap_value: HeapValue<'static> = partial.build()?;
    heap_value
        .materialize::<T>()
        .map_err(|err| ConvertError::TypeMismatch {
            expected: std::any::type_name::<T>(),
            actual: err.to_string(),
        })
}

/// Convert a BamlValueWithFlags to a Rust type.
///
/// This is the primary entry point for converting parsed LLM output to Rust types.
pub fn from_baml_value_with_flags<T: Facet<'static>>(
    value: &BamlValueWithFlags,
) -> Result<T, ConvertError> {
    let baml_value: BamlValue = value.clone().into();
    from_baml_value(baml_value)
}

fn build_from_baml_value(
    partial: Partial<'static>,
    value: &BamlValue,
) -> Result<Partial<'static>, ConvertError> {
    build_from_baml_value_with_hints(partial, value, ReprHints::default())
}

/// Recursive helper to build a Partial from BamlValue.
fn build_from_baml_value_with_hints(
    partial: Partial<'static>,
    value: &BamlValue,
    hints: ReprHints,
) -> Result<Partial<'static>, ConvertError> {
    let target_shape = partial.shape();

    // Smart pointers (Box, Arc, Rc) - enter the pointer, build inner, exit.
    if let Def::Pointer(_) = target_shape.def {
        let p = partial.begin_smart_ptr()?;
        let p = build_from_baml_value_with_hints(p, value, hints)?;
        return Ok(p.end()?);
    }

    // Option<T> - wrap non-null values in Some.
    if let Def::Option(_) = target_shape.def {
        if matches!(value, BamlValue::Null) {
            return Ok(partial.set_default()?);
        }
        let p = partial.begin_some()?;
        let p = build_from_baml_value_with_hints(p, value, hints)?;
        return Ok(p.end()?);
    }

    match value {
        // Strings map to either textual primitives or parseable scalar types.
        BamlValue::String(s) => {
            if matches!(partial.shape().ty, Type::User(UserType::Enum(_))) {
                return select_enum_variant(partial, s);
            }
            if is_string_target_shape(partial.shape()) {
                Ok(partial.set(s.clone())?)
            } else if hints.int_repr == Some(IntReprHint::String) {
                Ok(partial.parse_from_str(s)?)
            } else {
                Err(ConvertError::Adapter(BamlConvertError::new(
                    Vec::new(),
                    expected_kind_for_shape(partial.shape()),
                    format!("{value:?}"),
                    format!("expected a {}", expected_kind_for_shape(partial.shape())),
                )))
            }
        }
        BamlValue::Int(i) => {
            if let Some((expected, min, max)) = integer_expected_and_bounds(partial.shape()) {
                let value = *i as i128;
                if value < min || value > max {
                    return Err(ConvertError::Adapter(BamlConvertError::new(
                        Vec::new(),
                        expected,
                        i.to_string(),
                        "integer out of range",
                    )));
                }
            }
            Ok(partial.parse_from_str(&i.to_string())?)
        }
        BamlValue::Float(f) => Ok(partial.parse_from_str(&f.to_string())?),
        BamlValue::Bool(b) => Ok(partial.set(*b)?),
        BamlValue::Null => {
            let message = format!(
                "null provided for required {}",
                shape_diagnostics(partial.shape())
            );
            Err(ConvertError::Adapter(BamlConvertError::new(
                Vec::new(),
                expected_kind_for_shape(partial.shape()),
                "null",
                message,
            )))
        }

        // Class input: either enum object form, struct object form, or map object form.
        BamlValue::Class(_type_name, fields) => {
            if matches!(partial.shape().ty, Type::User(UserType::Enum(_))) {
                return build_enum_from_object(partial, fields);
            }

            match partial.shape().def {
                Def::Map(_) => build_map_from_object_pairs(partial, fields),
                _ => build_struct_from_object(partial, fields),
            }
        }

        // List: either list-like values or map-entry pairs.
        BamlValue::List(items) => {
            if matches!(partial.shape().def, Def::Map(_)) {
                if hints.map_key_repr == Some(MapKeyReprHint::Pairs) {
                    return build_map_from_pair_entries(partial, items);
                }
                return Err(ConvertError::Adapter(BamlConvertError::new(
                    Vec::new(),
                    "map",
                    format!("{value:?}"),
                    "expected a map",
                )));
            }

            let mut p = partial.init_list()?;
            for (idx, item) in items.iter().enumerate() {
                p = p.begin_list_item()?;
                p = build_from_baml_value_with_hints(p, item, hints)
                    .map_err(|err| err.with_path_prefix(idx.to_string()))?;
                p = p.end()?;
            }
            Ok(p)
        }

        // Object map.
        BamlValue::Map(map) => {
            if matches!(partial.shape().ty, Type::User(UserType::Enum(_))) {
                return build_enum_from_object(partial, map);
            }

            if matches!(partial.shape().ty, Type::User(UserType::Struct(_))) {
                return build_struct_from_object(partial, map);
            }

            match hints.map_key_repr {
                Some(MapKeyReprHint::Pairs) => Err(ConvertError::TypeMismatch {
                    expected: "list",
                    actual: "map".to_string(),
                }),
                Some(MapKeyReprHint::String) | None => build_map_from_object_pairs(partial, map),
            }
        }

        // Enum variant (unit-like representation).
        BamlValue::Enum(_type_name, variant_name) => select_enum_variant(partial, variant_name),

        // Media - intentionally unsupported for now.
        // TODO(dsrs-media): define typed media contract and implement BamlValue::Media conversions end-to-end.
        BamlValue::Media(_media) => Err(ConvertError::Unsupported(format!(
            "TODO(dsrs-media): BamlValue::Media -> Rust conversion is deferred; failed to convert into target shape ({})",
            shape_diagnostics(partial.shape())
        ))),
    }
}

fn build_struct_from_object(
    partial: Partial<'static>,
    fields: &BamlMap<String, BamlValue>,
) -> Result<Partial<'static>, ConvertError> {
    build_object_fields(partial, fields, None, true)
}

fn build_enum_from_object(
    partial: Partial<'static>,
    fields: &BamlMap<String, BamlValue>,
) -> Result<Partial<'static>, ConvertError> {
    let tag_name = partial.shape().get_tag_attr().unwrap_or("type");

    let tag_value = fields
        .get(tag_name)
        .ok_or_else(|| ConvertError::MissingField(tag_name.to_string()))?;

    let variant_name = match tag_value {
        BamlValue::String(v) => v.as_str(),
        BamlValue::Enum(_, v) => v.as_str(),
        other => {
            return Err(ConvertError::TypeMismatch {
                expected: "string",
                actual: baml_value_kind(other),
            });
        }
    };

    let p = select_enum_variant(partial, variant_name)?;
    build_object_fields(p, fields, Some(tag_name), false)
}

fn build_object_fields(
    mut partial: Partial<'static>,
    fields: &BamlMap<String, BamlValue>,
    skip_field_name: Option<&str>,
    skip_struct_deserialize_skips: bool,
) -> Result<Partial<'static>, ConvertError> {
    for (field_name, field_value) in fields {
        if skip_field_name == Some(field_name.as_str()) {
            continue;
        }

        if skip_struct_deserialize_skips
            && should_skip_struct_field_deserializing(partial.shape(), field_name)
        {
            continue;
        }

        // Unknown fields are ignored for compatibility.
        let Some(index) = resolve_field_index(&partial, field_name) else {
            continue;
        };

        let field = current_field(&partial, index);
        if let Some(field) = field {
            // Preserve facet(default) semantics when parsers materialize missing
            // fields as explicit nulls.
            if matches!(field_value, BamlValue::Null) && field.has_default() {
                continue;
            }
        }

        if let Some(field) = field
            && let Some(with) = crate::facet_ext::with_adapter_fns(field.attributes)
        {
            let field_path = vec![field_name.to_string()];
            partial = partial.begin_nth_field(index)?;
            partial = (with.apply)(partial, field_value.clone(), field_path)
                .map_err(ConvertError::Adapter)?;
            partial = partial.end()?;
            continue;
        }

        let hints = field.map(field_hints).unwrap_or_default();
        partial = partial.begin_nth_field(index)?;
        partial = build_from_baml_value_with_hints(partial, field_value, hints)
            .map_err(|err| err.with_path_prefix(field_name.to_string()))?;
        partial = partial.end()?;
    }

    Ok(partial)
}

fn build_map_from_object_pairs(
    partial: Partial<'static>,
    map: &BamlMap<String, BamlValue>,
) -> Result<Partial<'static>, ConvertError> {
    let mut p = partial.init_map()?;
    for (key, value) in map {
        p = p.begin_key()?;
        if is_string_target_shape(p.shape()) {
            p = p.set(key.clone())?;
        } else {
            p = p
                .parse_from_str(key)
                .map_err(ConvertError::from)
                .map_err(|err| err.with_path_prefix(key.clone()))?;
        }
        p = p.end()?;

        p = p.begin_value()?;
        p = build_from_baml_value_with_hints(p, value, ReprHints::default())
            .map_err(|err| err.with_path_prefix(key.clone()))?;
        p = p.end()?;
    }
    Ok(p)
}

fn build_map_from_pair_entries(
    partial: Partial<'static>,
    items: &[BamlValue],
) -> Result<Partial<'static>, ConvertError> {
    let mut p = partial.init_map()?;

    for (idx, item) in items.iter().enumerate() {
        let entry_map = match item {
            BamlValue::Class(_, map) | BamlValue::Map(map) => map,
            other => {
                return Err(ConvertError::TypeMismatch {
                    expected: "object",
                    actual: baml_value_kind(other),
                });
            }
        };

        let key_value = entry_map
            .get("key")
            .ok_or_else(|| ConvertError::MissingField("key".to_string()))?;
        let value_value = entry_map
            .get("value")
            .ok_or_else(|| ConvertError::MissingField("value".to_string()))?;

        p = p.begin_key()?;
        p = build_from_baml_value_with_hints(p, key_value, ReprHints::default()).map_err(
            |err| {
                err.with_path_prefix("key")
                    .with_path_prefix(idx.to_string())
            },
        )?;
        p = p.end()?;

        p = p.begin_value()?;
        p = build_from_baml_value_with_hints(p, value_value, ReprHints::default()).map_err(
            |err| {
                err.with_path_prefix("value")
                    .with_path_prefix(idx.to_string())
            },
        )?;
        p = p.end()?;
    }

    Ok(p)
}

fn resolve_field_index(partial: &Partial<'static>, field_name: &str) -> Option<usize> {
    if let Some(index) = partial.field_index(field_name) {
        return Some(index);
    }

    let shape = partial.shape();
    if let Type::User(UserType::Struct(struct_type)) = &shape.ty {
        return struct_type
            .fields
            .iter()
            .enumerate()
            .find_map(|(i, field)| field_matches_name(field, field_name).then_some(i));
    }

    if let Type::User(UserType::Enum(_)) = &shape.ty
        && let Some(variant) = partial.selected_variant()
    {
        return variant
            .data
            .fields
            .iter()
            .enumerate()
            .find_map(|(i, field)| field_matches_name(field, field_name).then_some(i));
    }

    None
}

fn current_field(partial: &Partial<'static>, index: usize) -> Option<facet::Field> {
    let shape = partial.shape();
    if let Type::User(UserType::Struct(struct_type)) = &shape.ty {
        return struct_type.fields.get(index).copied();
    }

    if let Type::User(UserType::Enum(_)) = &shape.ty
        && let Some(variant) = partial.selected_variant()
    {
        return variant.data.fields.get(index).copied();
    }

    None
}

fn field_matches_name(field: &facet::Field, input_name: &str) -> bool {
    field.name == input_name
        || field.effective_name() == input_name
        || field.alias == Some(input_name)
}

fn field_hints(field: facet::Field) -> ReprHints {
    ReprHints {
        int_repr: bamltype_attr_static_str(field.attributes, "int_repr")
            .and_then(parse_int_repr_hint),
        map_key_repr: bamltype_attr_static_str(field.attributes, "map_key_repr")
            .and_then(parse_map_key_repr_hint),
    }
}

fn bamltype_attr_static_str(attrs: &'static [facet::Attr], key: &str) -> Option<&'static str> {
    for attr in attrs {
        if attr.ns != Some("bamltype") || attr.key != key {
            continue;
        }

        if let Some(value) = attr.get_as::<&'static str>() {
            return Some(*value);
        }
    }

    None
}

fn parse_int_repr_hint(value: &'static str) -> Option<IntReprHint> {
    match value {
        "string" => Some(IntReprHint::String),
        "i64" => Some(IntReprHint::I64),
        _ => None,
    }
}

fn parse_map_key_repr_hint(value: &'static str) -> Option<MapKeyReprHint> {
    match value {
        "string" => Some(MapKeyReprHint::String),
        "pairs" => Some(MapKeyReprHint::Pairs),
        _ => None,
    }
}

fn integer_expected_and_bounds(shape: &'static Shape) -> Option<(&'static str, i128, i128)> {
    use facet::{NumericType, PrimitiveType};

    let Type::Primitive(PrimitiveType::Numeric(NumericType::Integer { .. })) = shape.ty else {
        return None;
    };

    match shape.type_identifier {
        "i8" => Some(("i8", i8::MIN as i128, i8::MAX as i128)),
        "i16" => Some(("i16", i16::MIN as i128, i16::MAX as i128)),
        "i32" => Some(("i32", i32::MIN as i128, i32::MAX as i128)),
        "i64" => Some(("i64", i64::MIN as i128, i64::MAX as i128)),
        "isize" => Some(("isize", isize::MIN as i128, isize::MAX as i128)),
        "u8" => Some(("u8", 0, u8::MAX as i128)),
        "u16" => Some(("u16", 0, u16::MAX as i128)),
        "u32" => Some(("u32", 0, u32::MAX as i128)),
        "u64" => Some(("u64", 0, u64::MAX as i128)),
        "usize" => Some(("usize", 0, usize::MAX as i128)),
        _ => None,
    }
}

fn baml_value_kind(value: &BamlValue) -> String {
    match value {
        BamlValue::String(_) => "string",
        BamlValue::Int(_) => "int",
        BamlValue::Float(_) => "float",
        BamlValue::Bool(_) => "bool",
        BamlValue::Map(_) => "map",
        BamlValue::List(_) => "list",
        BamlValue::Class(_, _) => "class",
        BamlValue::Enum(_, _) => "enum",
        BamlValue::Null => "null",
        BamlValue::Media(_) => "media",
    }
    .to_string()
}

fn shape_diagnostics(shape: &'static Shape) -> String {
    format!(
        "shape_id={:?}, type_identifier={}, def={:?}",
        shape.id, shape.type_identifier, shape.def
    )
}

// ============================================================================
// Rust → BamlValue (using Peek API)
// ============================================================================

/// Convert a Rust value to BamlValue using facet reflection.
pub fn to_baml_value<T: Facet<'static>>(value: &T) -> Result<BamlValue, ConvertError> {
    let peek = Peek::new(value);
    peek_to_baml_value(peek)
}

/// Recursive helper to convert Peek to BamlValue.
fn peek_to_baml_value(peek: Peek<'_, '_>) -> Result<BamlValue, ConvertError> {
    peek_to_baml_value_with_hints(peek, ReprHints::default())
}

fn peek_to_baml_value_with_hints(
    peek: Peek<'_, '_>,
    hints: ReprHints,
) -> Result<BamlValue, ConvertError> {
    let peek = peek.innermost_peek(); // unwrap transparent wrappers

    // Option - check before list/map/struct.
    if let Ok(opt) = peek.into_option() {
        return match opt.value() {
            Some(inner) => peek_to_baml_value_with_hints(inner, hints),
            None => Ok(BamlValue::Null),
        };
    }

    // Handle explicit int_repr first.
    if let Some(int_repr) = hints.int_repr {
        if let Some(value) = peek_signed_i128(peek) {
            return match int_repr {
                IntReprHint::String => Ok(BamlValue::String(value.to_string())),
                IntReprHint::I64 => i64::try_from(value)
                    .map(BamlValue::Int)
                    .map_err(|_| ConvertError::Unsupported("integer out of range for i64".into())),
            };
        }
        if let Some(value) = peek_unsigned_u128(peek) {
            return match int_repr {
                IntReprHint::String => Ok(BamlValue::String(value.to_string())),
                IntReprHint::I64 => i64::try_from(value)
                    .map(BamlValue::Int)
                    .map_err(|_| ConvertError::Unsupported("integer out of range for i64".into())),
            };
        }
    }

    // Try scalar types first.
    if let Some(s) = peek.as_str() {
        return Ok(BamlValue::String(s.to_string()));
    }
    if let Ok(b) = peek.get::<bool>() {
        return Ok(BamlValue::Bool(*b));
    }

    if let Some(value) = peek_signed_i128(peek) {
        return i64::try_from(value)
            .map(BamlValue::Int)
            .map_err(|_| ConvertError::Unsupported("integer out of range for i64".into()));
    }
    if let Some(value) = peek_unsigned_u128(peek) {
        return i64::try_from(value)
            .map(BamlValue::Int)
            .map_err(|_| ConvertError::Unsupported("integer out of range for i64".into()));
    }

    if let Ok(f) = peek.get::<f64>() {
        return Ok(BamlValue::Float(*f));
    }
    if let Ok(f) = peek.get::<f32>() {
        return Ok(BamlValue::Float(*f as f64));
    }

    // List/Array.
    if let Ok(list) = peek.into_list_like() {
        let items: Result<Vec<_>, _> = list
            .iter()
            .map(|item| peek_to_baml_value_with_hints(item, hints))
            .collect();
        return Ok(BamlValue::List(items?));
    }

    // Map.
    if let Ok(map) = peek.into_map() {
        return map_to_baml_value(map, hints);
    }

    // Struct.
    if let Ok(struct_peek) = peek.into_struct() {
        let type_name = internal_name_for_shape(peek.shape());
        let mut fields = IndexMap::new();

        for (field_item, field_peek) in struct_peek.fields_for_serialize() {
            let field_name = field_item.effective_name().to_string();
            let field_hints = field_item.field.map(field_hints).unwrap_or_default();
            fields.insert(
                field_name,
                peek_to_baml_value_with_hints(field_peek, field_hints)?,
            );
        }
        return Ok(BamlValue::Class(type_name, fields));
    }

    // Enum.
    if let Ok(enum_peek) = peek.into_enum() {
        let type_name = internal_name_for_shape(peek.shape());
        let variant = enum_peek.active_variant()?;

        if enum_has_data_variants(peek.shape()) {
            let tag_name = peek.shape().get_tag_attr().unwrap_or("type");
            let mut fields = IndexMap::new();
            fields.insert(
                tag_name.to_string(),
                BamlValue::String(variant.effective_name().to_string()),
            );

            for (field_item, field_peek) in enum_peek.fields_for_serialize() {
                let field_name = field_item.effective_name().to_string();
                let field_hints = field_item.field.map(field_hints).unwrap_or_default();
                fields.insert(
                    field_name,
                    peek_to_baml_value_with_hints(field_peek, field_hints)?,
                );
            }

            return Ok(BamlValue::Class(type_name, fields));
        }

        return Ok(BamlValue::Enum(
            type_name,
            variant.effective_name().to_string(),
        ));
    }

    Err(ConvertError::Unsupported(format!(
        "Cannot convert type {} to BamlValue",
        peek.shape()
    )))
}

fn map_to_baml_value(
    map: facet_reflect::PeekMap<'_, '_>,
    hints: ReprHints,
) -> Result<BamlValue, ConvertError> {
    match hints.map_key_repr {
        Some(MapKeyReprHint::Pairs) => {
            let mut entries = Vec::with_capacity(map.len());
            for (key, value) in map.iter() {
                let mut entry = IndexMap::new();
                entry.insert(
                    "key".to_string(),
                    peek_to_baml_value_with_hints(key, ReprHints::default())?,
                );
                entry.insert(
                    "value".to_string(),
                    peek_to_baml_value_with_hints(value, ReprHints::default())?,
                );
                entries.push(BamlValue::Map(entry));
            }
            Ok(BamlValue::List(entries))
        }
        Some(MapKeyReprHint::String) | None => {
            let mut result = IndexMap::new();
            for (key, value) in map.iter() {
                let key_str = key_to_string(key)?;
                result.insert(
                    key_str,
                    peek_to_baml_value_with_hints(value, ReprHints::default())?,
                );
            }
            Ok(BamlValue::Map(result))
        }
    }
}

fn key_to_string(key: Peek<'_, '_>) -> Result<String, ConvertError> {
    if let Some(s) = key.as_str() {
        return Ok(s.to_string());
    }

    if let Some(value) = peek_signed_i128(key) {
        return Ok(value.to_string());
    }

    if let Some(value) = peek_unsigned_u128(key) {
        return Ok(value.to_string());
    }

    if let Ok(value) = key.get::<bool>() {
        return Ok(value.to_string());
    }

    Ok(format!("{}", key))
}

fn enum_has_data_variants(shape: &'static Shape) -> bool {
    let Type::User(UserType::Enum(enum_type)) = &shape.ty else {
        return false;
    };
    enum_type
        .variants
        .iter()
        .any(|variant| !variant.data.fields.is_empty())
}

fn peek_signed_i128(peek: Peek<'_, '_>) -> Option<i128> {
    if let Ok(v) = peek.get::<i128>() {
        return Some(*v);
    }
    if let Ok(v) = peek.get::<i64>() {
        return Some(*v as i128);
    }
    if let Ok(v) = peek.get::<i32>() {
        return Some(*v as i128);
    }
    if let Ok(v) = peek.get::<i16>() {
        return Some(*v as i128);
    }
    if let Ok(v) = peek.get::<i8>() {
        return Some(*v as i128);
    }
    if let Ok(v) = peek.get::<isize>() {
        return Some(*v as i128);
    }
    None
}

fn peek_unsigned_u128(peek: Peek<'_, '_>) -> Option<u128> {
    if let Ok(v) = peek.get::<u128>() {
        return Some(*v);
    }
    if let Ok(v) = peek.get::<u64>() {
        return Some(*v as u128);
    }
    if let Ok(v) = peek.get::<u32>() {
        return Some(*v as u128);
    }
    if let Ok(v) = peek.get::<u16>() {
        return Some(*v as u128);
    }
    if let Ok(v) = peek.get::<u8>() {
        return Some(*v as u128);
    }
    if let Ok(v) = peek.get::<usize>() {
        return Some(*v as u128);
    }
    None
}

fn is_string_target_shape(shape: &'static Shape) -> bool {
    matches!(
        shape.scalar_type(),
        Some(ScalarType::Str | ScalarType::String | ScalarType::CowStr | ScalarType::Char)
    )
}

fn expected_kind_for_shape(shape: &'static Shape) -> &'static str {
    use facet::{NumericType, PrimitiveType, TextualType};

    match &shape.ty {
        Type::Primitive(PrimitiveType::Boolean) => "bool",
        Type::Primitive(PrimitiveType::Numeric(NumericType::Integer { .. })) => "int",
        Type::Primitive(PrimitiveType::Numeric(NumericType::Float)) => "float",
        Type::Primitive(PrimitiveType::Textual(TextualType::Str | TextualType::Char)) => "string",
        Type::User(UserType::Struct(_)) | Type::User(UserType::Enum(_)) => "object",
        _ => match shape.def {
            Def::Map(_) => "map",
            Def::List(_) | Def::Array(_) | Def::Set(_) => "list",
            _ => "compatible type",
        },
    }
}

fn should_skip_struct_field_deserializing(shape: &'static facet::Shape, input_name: &str) -> bool {
    let Type::User(UserType::Struct(struct_type)) = &shape.ty else {
        return false;
    };

    struct_type.fields.iter().any(|field| {
        let matches_name = field_matches_name(field, input_name);
        matches_name && field.should_skip_deserializing()
    })
}

fn select_enum_variant(
    partial: Partial<'static>,
    variant_name: &str,
) -> Result<Partial<'static>, ConvertError> {
    let shape = partial.shape();
    let Type::User(UserType::Enum(enum_type)) = &shape.ty else {
        return Ok(partial.select_variant_named(variant_name)?);
    };

    if let Some((index, _)) = enum_type.variants.iter().enumerate().find(|(_, variant)| {
        variant.effective_name() == variant_name || variant.name == variant_name
    }) {
        return Ok(partial.select_nth_variant(index)?);
    }

    Err(ConvertError::UnknownVariant(variant_name.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use baml_types::{BamlMedia, BamlMediaType};

    #[test]
    fn test_primitives_to_baml() {
        let pi = std::f64::consts::PI;
        assert_eq!(
            to_baml_value(&"hello".to_string()).unwrap(),
            BamlValue::String("hello".into())
        );
        assert_eq!(to_baml_value(&42i64).unwrap(), BamlValue::Int(42));
        assert_eq!(to_baml_value(&pi).unwrap(), BamlValue::Float(pi));
        assert_eq!(to_baml_value(&true).unwrap(), BamlValue::Bool(true));
    }

    #[test]
    fn test_primitives_from_baml() {
        let s: String = from_baml_value(BamlValue::String("world".into())).unwrap();
        assert_eq!(s, "world");

        let i: i64 = from_baml_value(BamlValue::Int(100)).unwrap();
        assert_eq!(i, 100);

        let e = std::f64::consts::E;
        let f: f64 = from_baml_value(BamlValue::Float(e)).unwrap();
        assert!((f - e).abs() < 0.001);

        let b: bool = from_baml_value(BamlValue::Bool(false)).unwrap();
        assert!(!b);
    }

    #[test]
    fn test_list_round_trip() {
        let original = vec![1i64, 2, 3];
        let baml = to_baml_value(&original).unwrap();
        assert!(matches!(baml, BamlValue::List(_)));

        let restored: Vec<i64> = from_baml_value(baml).unwrap();
        assert_eq!(original, restored);
    }

    #[test]
    fn test_option_to_baml() {
        let some_val: Option<i64> = Some(42);
        let none_val: Option<i64> = None;

        assert_eq!(to_baml_value(&some_val).unwrap(), BamlValue::Int(42));
        assert_eq!(to_baml_value(&none_val).unwrap(), BamlValue::Null);
    }

    #[test]
    fn null_to_required_errs() {
        let err = from_baml_value::<i32>(BamlValue::Null).unwrap_err();
        match err {
            ConvertError::Adapter(inner) => {
                assert_eq!(inner.expected, "int");
                assert_eq!(inner.got, "null");
                assert!(inner.message.starts_with("null provided for required"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn null_into_option_succeeds() {
        let value: Option<i32> = from_baml_value(BamlValue::Null).unwrap();
        assert_eq!(value, None);
    }

    #[test]
    fn media_conversion_error_includes_todo() {
        let media = BamlMedia::url(
            BamlMediaType::Image,
            "https://example.com/img.png".to_string(),
            Some("image/png".to_string()),
        );
        let err = from_baml_value::<i32>(BamlValue::Media(media)).unwrap_err();
        match err {
            ConvertError::Unsupported(message) => {
                assert!(message.contains("TODO(dsrs-media)"));
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }
}
