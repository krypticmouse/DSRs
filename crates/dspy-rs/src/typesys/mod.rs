//! In-house type system that replaces the vendored BAML stack (`bamltype`, `baml_types`,
//! `internal_baml_jinja`, `jsonish`).
//!
//! - [`schema`] — the [`FieldType`]/[`OutputSchema`] type model + the [`Schema`] trait,
//!   built from facet `Shape` metadata.
//! - [`render`] — prompt rendering (type labels + expanded schema blocks).
//! - [`coerce`] — tolerant parsing of raw LM text into `serde_json::Value`.
//! - [`constraint`] — `#[check]`/`#[assert]` evaluation via minijinja.

pub mod coerce;
pub mod constraint;
pub mod render;
pub mod schema;

pub use coerce::{Coerced, Flag, coerce};
pub use constraint::{
    Constraint, ConstraintKind, ConstraintLevel, ConstraintOutcome, ResponseCheck,
    evaluate_constraints,
};
pub use render::{schema_block, type_name};
pub use schema::{
    ClassDef, EnumDef, EnumValueDef, FieldDef, FieldType, OutputSchema, Schema, TypeTable,
    field_type_from_shape, internal_name_for_shape,
};

use serde_json::Value;

/// Renders a value for an input field, honoring an explicit `#[format(...)]` hint.
///
/// `json` produces compact JSON. `yaml`/`toon` currently fall back to JSON (see roadmap:
/// BAML-only input formats were dropped in the de-BAML migration).
pub fn format_value(value: &Value, format: &str) -> String {
    match format {
        "json" => serde_json::to_string(value).unwrap_or_else(|_| "null".to_string()),
        _ => serde_json::to_string(value).unwrap_or_else(|_| "null".to_string()),
    }
}
