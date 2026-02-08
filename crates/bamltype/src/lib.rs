//! BamlType - Facet-based BAML type generation
//!
//! This crate provides automatic BAML schema generation and LLM output parsing
//! for Rust types using facet's compile-time reflection.
//!
//! # Usage
//!
//! ```ignore
//! use bamltype::BamlType;
//!
//! #[BamlType]
//! struct Response {
//!     /// The user's name
//!     name: String,
//!     /// Age in years
//!     age: u32,
//! }
//!
//! // Render schema for LLM prompt
//! let schema = bamltype::render_schema::<Response>(bamltype::RenderOptions::default())?;
//!
//! // Parse LLM output
//! let parsed = bamltype::parse_llm_output::<Response>(llm_output, true)?;
//! ```

use std::collections::HashSet;

use baml_types::{BamlValue, Constraint, ConstraintLevel, ResponseCheck, TypeIR};
use internal_baml_jinja::types::OutputFormatContent;
use jsonish::deserializer::{
    coercer::{ParsingError, run_user_checks},
    deserialize_flags::{DeserializerConditions, Flag},
};
use sha2::{Digest, Sha256};

// Re-export underlying crates for consumers (replaces legacy bridge re-exports)
pub use baml_types;
pub use internal_baml_jinja;
pub use internal_baml_jinja::types::{HoistClasses, MapStyle, RenderOptions};
pub use jsonish;
pub use jsonish::BamlValueWithFlags;

// Re-export facet for users
pub use facet;
pub use facet::Shape;
pub use facet_reflect;

// Re-export the attribute macro
#[cfg(feature = "derive")]
pub use bamltype_derive::BamlType;

mod schema_builder;
pub use schema_builder::*;

mod convert;
pub use convert::{ConvertError, from_baml_value, from_baml_value_with_flags, to_baml_value};

mod runtime;
pub use runtime::{
    BamlConvertError, baml_internal_name, baml_type_ir, default_streaming_behavior, get_field,
    to_baml_value_lossy, try_from_baml_value,
};

pub mod adapters;
mod schema_registry;

pub mod facet_ext;

/// A bundle containing everything needed to render schemas and parse LLM output.
#[derive(Debug, Clone)]
pub struct SchemaBundle {
    /// The TypeIR for the root type
    pub target: TypeIR,
    /// The output format content (classes, enums, etc.)
    pub output_format: OutputFormatContent,
}

impl SchemaBundle {
    /// Build a SchemaBundle from a facet Shape.
    ///
    /// This walks the type graph starting from the given shape,
    /// building BAML class/enum definitions for all reachable types.
    pub fn from_shape(shape: &'static facet::Shape) -> Self {
        schema_builder::build_schema_bundle(shape)
    }
}

/// Trait for types that can generate BAML schemas.
///
/// Implemented automatically by `#[BamlType]`.
pub trait BamlSchema: for<'a> facet::Facet<'a> {
    /// Get the schema bundle for this type.
    ///
    /// This is lazily initialized and cached.
    fn baml_schema() -> &'static SchemaBundle;
}

/// Runtime trait for types that expose BAML schema + conversion entry points.
pub trait BamlType: Sized + 'static {
    fn baml_output_format() -> &'static OutputFormatContent;
    fn baml_internal_name() -> &'static str;
    fn baml_type_ir() -> TypeIR;
    fn try_from_baml_value(value: BamlValue) -> Result<Self, BamlConvertError>;
    fn to_baml_value(&self) -> BamlValue;
}

impl<T: BamlSchema> BamlType for T {
    fn baml_output_format() -> &'static OutputFormatContent {
        &T::baml_schema().output_format
    }

    fn baml_internal_name() -> &'static str {
        runtime::baml_internal_name::<T>()
    }

    fn baml_type_ir() -> TypeIR {
        runtime::baml_type_ir::<T>()
    }

    fn try_from_baml_value(value: BamlValue) -> Result<Self, BamlConvertError> {
        runtime::try_from_baml_value(value)
    }

    fn to_baml_value(&self) -> BamlValue {
        runtime::to_baml_value_lossy(self)
    }
}

/// Parsed output bundle matching legacy bridge behavior.
#[derive(Debug, Clone)]
pub struct Parsed<T> {
    pub value: T,
    pub baml_value: BamlValue,
    pub flags: Vec<Flag>,
    pub checks: Vec<ResponseCheck>,
    pub explanations: Vec<ParsingError>,
}

/// Error type for parse + conversion failures.
#[derive(Debug, thiserror::Error)]
pub enum BamlParseError {
    #[error("jsonish parse error: {0}")]
    Jsonish(#[from] anyhow::Error),

    #[error("constraint asserts failed")]
    ConstraintAssertsFailed { failed: Vec<ResponseCheck> },

    #[error("conversion error: {0}")]
    Convert(#[from] BamlConvertError),
}

/// Error type for parsing failures.
#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("Schema rendering failed: {0}")]
    RenderError(String),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Coercion error: {0}")]
    CoercionError(String),
}

/// Render the BAML schema for a type, matching legacy bridge signature.
pub fn render_schema<T: BamlType>(
    options: RenderOptions,
) -> Result<Option<String>, minijinja::Error> {
    T::baml_output_format().render(options)
}

/// Convenience helper equivalent to `render_schema(RenderOptions::default())`.
pub fn render_schema_default<T: BamlType>() -> Result<String, ParseError> {
    render_schema_string::<T>(RenderOptions::default())
}

/// Parse LLM output into a BamlValue.
///
/// This uses jsonish for flexible JSON parsing that handles
/// common LLM output quirks (markdown code blocks, trailing commas, etc).
pub fn parse<T: BamlType>(raw: &str) -> Result<BamlValueWithFlags, ParseError> {
    parse_with_mode::<T>(raw, true)
}

/// Parse streaming LLM output (partial results).
pub fn parse_partial<T: BamlType>(raw: &str) -> Result<BamlValueWithFlags, ParseError> {
    parse_with_mode::<T>(raw, false)
}

/// Parse raw LLM output and convert into `T`, collecting flags/checks/explanations.
pub fn parse_llm_output<T: BamlType>(
    raw: &str,
    is_done: bool,
) -> Result<Parsed<T>, BamlParseError> {
    let output_format = T::baml_output_format();
    let parsed = match parse_raw::<T>(raw, is_done) {
        Ok(parsed) => parsed,
        Err(err) => {
            if has_assert_failure(&err) {
                let failed = collect_assert_constraints(output_format);
                return Err(BamlParseError::ConstraintAssertsFailed { failed });
            }
            return Err(BamlParseError::Jsonish(err));
        }
    };

    let baml_value_with_meta: baml_types::BamlValueWithMeta<baml_types::TypeIR> =
        parsed.clone().into();
    let baml_value: BamlValue = baml_value_with_meta.into();

    let value = <T as BamlType>::try_from_baml_value(baml_value.clone())?;

    let mut flags = Vec::new();
    collect_flags(&parsed, &mut flags);

    let mut checks = Vec::new();
    let mut failed_asserts = Vec::new();
    collect_checks_and_assert_failures(&parsed, &mut checks, &mut failed_asserts)?;

    let mut explanations = Vec::new();
    parsed.explanation_impl(vec!["<root>".to_string()], &mut explanations);

    if !failed_asserts.is_empty() {
        return Err(BamlParseError::ConstraintAssertsFailed {
            failed: failed_asserts,
        });
    }

    Ok(Parsed {
        value,
        baml_value,
        flags,
        checks,
        explanations,
    })
}

/// Stable schema fingerprint used by cache keys and regression tests.
pub fn schema_fingerprint(
    output_format: &OutputFormatContent,
    options: RenderOptions,
) -> Result<String, minijinja::Error> {
    let rendered = output_format.render(options)?.unwrap_or_default();
    let mut hasher = Sha256::new();
    hasher.update(rendered.as_bytes());
    hasher.update(output_format.target.to_string().as_bytes());
    Ok(format!("{:x}", hasher.finalize()))
}

fn parse_raw<T: BamlType>(raw: &str, is_done: bool) -> Result<BamlValueWithFlags, anyhow::Error> {
    let output_format = T::baml_output_format();
    jsonish::from_str(output_format, &output_format.target, raw, is_done)
}

fn render_schema_string<T: BamlType>(options: RenderOptions) -> Result<String, ParseError> {
    render_schema::<T>(options)
        .map(Option::unwrap_or_default)
        .map_err(|e| ParseError::RenderError(e.to_string()))
}

fn parse_with_mode<T: BamlType>(
    raw: &str,
    is_done: bool,
) -> Result<BamlValueWithFlags, ParseError> {
    parse_raw::<T>(raw, is_done).map_err(|e| ParseError::ParseError(e.to_string()))
}

fn collect_flags_recursive(value: &BamlValueWithFlags, flags: &mut Vec<Flag>) {
    match value {
        BamlValueWithFlags::String(v) => collect_from_conditions(&v.flags, flags),
        BamlValueWithFlags::Int(v) => collect_from_conditions(&v.flags, flags),
        BamlValueWithFlags::Float(v) => collect_from_conditions(&v.flags, flags),
        BamlValueWithFlags::Bool(v) => collect_from_conditions(&v.flags, flags),
        BamlValueWithFlags::Enum(_, _, v) => collect_from_conditions(&v.flags, flags),
        BamlValueWithFlags::Media(_, v) => collect_from_conditions(&v.flags, flags),
        BamlValueWithFlags::List(conds, _, items) => {
            collect_from_conditions(conds, flags);
            for item in items {
                collect_flags_recursive(item, flags);
            }
        }
        BamlValueWithFlags::Map(conds, _, items) => {
            collect_from_conditions(conds, flags);
            for (_, (entry_flags, entry_value)) in items {
                collect_from_conditions(entry_flags, flags);
                collect_flags_recursive(entry_value, flags);
            }
        }
        BamlValueWithFlags::Class(_, conds, _, fields) => {
            collect_from_conditions(conds, flags);
            for (_, field_value) in fields {
                collect_flags_recursive(field_value, flags);
            }
        }
        BamlValueWithFlags::Null(_, conds) => collect_from_conditions(conds, flags),
    }
}

fn collect_from_conditions(conditions: &DeserializerConditions, flags: &mut Vec<Flag>) {
    flags.extend(conditions.flags.iter().cloned());
}

fn collect_flags(value: &BamlValueWithFlags, flags: &mut Vec<Flag>) {
    collect_flags_recursive(value, flags);
}

fn collect_checks_and_assert_failures(
    value: &BamlValueWithFlags,
    checks: &mut Vec<ResponseCheck>,
    failed_asserts: &mut Vec<ResponseCheck>,
) -> Result<(), BamlParseError> {
    let baml_value = baml_value_from_flags(value);
    let results = run_user_checks(&baml_value, value.field_type()).map_err(BamlParseError::from)?;

    for (constraint, ok) in results {
        if constraint.level == ConstraintLevel::Assert {
            if !ok {
                failed_asserts.push(ResponseCheck {
                    name: constraint.label.unwrap_or_else(|| "assert".to_string()),
                    expression: constraint.expression.0,
                    status: "failed".to_string(),
                });
            }
            continue;
        }

        if let Some(check) = ResponseCheck::from_check_result((constraint, ok)) {
            checks.push(check);
        }
    }

    match value {
        BamlValueWithFlags::List(_, _, items) => {
            for item in items {
                collect_checks_and_assert_failures(item, checks, failed_asserts)?;
            }
        }
        BamlValueWithFlags::Map(_, _, items) => {
            for (_, (_, entry_value)) in items {
                collect_checks_and_assert_failures(entry_value, checks, failed_asserts)?;
            }
        }
        BamlValueWithFlags::Class(_, _, _, fields) => {
            for (_, field_value) in fields {
                collect_checks_and_assert_failures(field_value, checks, failed_asserts)?;
            }
        }
        _ => {}
    }

    Ok(())
}

fn collect_assert_constraints(output_format: &OutputFormatContent) -> Vec<ResponseCheck> {
    let mut failed = Vec::new();
    let mut seen = HashSet::new();

    collect_assert_constraints_in_type(&output_format.target, &mut failed, &mut seen);

    for class in output_format.classes.values() {
        for constraint in &class.constraints {
            push_assert_constraint(constraint, &mut failed, &mut seen);
        }
        for (_, field_type, _, _) in &class.fields {
            collect_assert_constraints_in_type(field_type, &mut failed, &mut seen);
        }
    }

    for r#enum in output_format.enums.values() {
        for constraint in &r#enum.constraints {
            push_assert_constraint(constraint, &mut failed, &mut seen);
        }
    }

    failed
}

fn collect_assert_constraints_in_type(
    r#type: &TypeIR,
    failed: &mut Vec<ResponseCheck>,
    seen: &mut HashSet<(String, String)>,
) {
    for constraint in &r#type.meta().constraints {
        push_assert_constraint(constraint, failed, seen);
    }

    match r#type {
        TypeIR::List(inner, _) => collect_assert_constraints_in_type(inner, failed, seen),
        TypeIR::Map(key, value, _) => {
            collect_assert_constraints_in_type(key, failed, seen);
            collect_assert_constraints_in_type(value, failed, seen);
        }
        TypeIR::Union(union, _) => {
            for item in union.iter_include_null() {
                collect_assert_constraints_in_type(item, failed, seen);
            }
        }
        TypeIR::Tuple(items, _) => {
            for item in items {
                collect_assert_constraints_in_type(item, failed, seen);
            }
        }
        TypeIR::Arrow(arrow, _) => {
            for param in &arrow.param_types {
                collect_assert_constraints_in_type(param, failed, seen);
            }
            collect_assert_constraints_in_type(&arrow.return_type, failed, seen);
        }
        TypeIR::Top(_)
        | TypeIR::Primitive(..)
        | TypeIR::Enum { .. }
        | TypeIR::Literal(..)
        | TypeIR::Class { .. }
        | TypeIR::RecursiveTypeAlias { .. } => {}
    }
}

fn push_assert_constraint(
    constraint: &Constraint,
    failed: &mut Vec<ResponseCheck>,
    seen: &mut HashSet<(String, String)>,
) {
    if constraint.level != ConstraintLevel::Assert {
        return;
    }

    let name = constraint
        .label
        .clone()
        .unwrap_or_else(|| "assert".to_string());
    let expr = constraint.expression.0.clone();
    if seen.insert((name.clone(), expr.clone())) {
        failed.push(ResponseCheck {
            name,
            expression: expr,
            status: "failed".to_string(),
        });
    }
}

fn has_assert_failure(err: &anyhow::Error) -> bool {
    err.to_string().contains("Assertions failed.")
}

fn baml_value_from_flags(value: &BamlValueWithFlags) -> BamlValue {
    match value {
        BamlValueWithFlags::String(v) => BamlValue::String(v.value.clone()),
        BamlValueWithFlags::Int(v) => BamlValue::Int(v.value),
        BamlValueWithFlags::Float(v) => BamlValue::Float(v.value),
        BamlValueWithFlags::Bool(v) => BamlValue::Bool(v.value),
        BamlValueWithFlags::Enum(name, _, v) => BamlValue::Enum(name.clone(), v.value.clone()),
        BamlValueWithFlags::Media(_, v) => BamlValue::Media(v.value.clone()),
        BamlValueWithFlags::List(_, _, items) => {
            BamlValue::List(items.iter().map(baml_value_from_flags).collect())
        }
        BamlValueWithFlags::Map(_, _, items) => BamlValue::Map(
            items
                .iter()
                .map(|(k, (_, v))| (k.clone(), baml_value_from_flags(v)))
                .collect(),
        ),
        BamlValueWithFlags::Class(name, _, _, fields) => BamlValue::Class(
            name.clone(),
            fields
                .iter()
                .map(|(k, v)| (k.clone(), baml_value_from_flags(v)))
                .collect(),
        ),
        BamlValueWithFlags::Null(_, _) => BamlValue::Null,
    }
}
