use std::collections::HashMap;

use bamltype::RenderOptions;
use bamltype::jsonish::deserializer::deserialize_flags::Flag;
use bamltype::{BamlParseError, parse_llm_output, render_schema, schema_fingerprint};

/// A user profile returned by the model.
///
/// ## Notes
/// - `fullName` should be the display name.
#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
struct DocUser {
    /// Full name as displayed in the UI.
    #[baml(alias = "fullName")]
    name: String,
    age: i64,
}

/// This should be ignored.
#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(description = "Override description.")]
struct OverrideUser {
    value: String,
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
enum Color {
    /// Red hot.
    Red,
    #[baml(alias = "green")]
    Green,
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
struct CheckedValue {
    #[baml(check(label = "positive", expr = "this > 0"))]
    value: i64,
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
struct AssertedValue {
    #[baml(assert(label = "positive", expr = "this > 0"))]
    value: i64,
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
struct Unsigned32 {
    value: u32,
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
struct FuzzUser {
    name: String,
    age: u32,
    nickname: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
struct FuzzExplain {
    values: HashMap<String, i64>,
}

#[test]
fn doc_comments_render() {
    let schema = render_schema::<DocUser>(RenderOptions::default())
        .expect("render failed")
        .unwrap_or_default();

    assert!(schema.contains("A user profile returned by the model."));
    assert!(schema.contains("## Notes"));
    assert!(schema.contains("Full name as displayed in the UI."));
}

#[test]
fn description_override_wins() {
    let schema = render_schema::<OverrideUser>(RenderOptions::default())
        .expect("render failed")
        .unwrap_or_default();

    assert!(schema.contains("Override description."));
    assert!(!schema.contains("This should be ignored."));
}

#[test]
fn enum_variant_descriptions_render() {
    let schema = render_schema::<Color>(RenderOptions::default())
        .expect("render failed")
        .unwrap_or_default();

    assert!(schema.contains("Red: Red hot."));
}

#[test]
fn constraint_checks_are_reported() {
    let raw = r#"{ "value": 3 }"#;
    let parsed = parse_llm_output::<CheckedValue>(raw, true).expect("parse failed");
    assert!(
        parsed
            .checks
            .iter()
            .any(|check| check.name == "positive" && check.status == "succeeded")
    );
}

#[test]
fn constraint_asserts_fail() {
    let raw = r#"{ "value": -1 }"#;
    let err = parse_llm_output::<AssertedValue>(raw, true).expect_err("expected assert failure");
    match err {
        BamlParseError::ConstraintAssertsFailed { failed } => {
            assert!(failed.iter().any(|check| check.name == "positive"));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn unsigned_bounds_enforced() {
    let raw = r#"{ "value": -1 }"#;
    let err = parse_llm_output::<Unsigned32>(raw, true).expect_err("expected range error");
    assert!(matches!(err, BamlParseError::Convert(_)));

    let raw = r#"{ "value": 4294967296 }"#;
    let err = parse_llm_output::<Unsigned32>(raw, true).expect_err("expected range error");
    assert!(matches!(err, BamlParseError::Convert(_)));
}

#[test]
fn markdown_fence_parses_and_sets_flag() {
    let raw = "```json\n{ \"name\": \"Ada\", \"age\": 36 }\n```";
    let parsed = parse_llm_output::<FuzzUser>(raw, true).expect("parse");

    assert_eq!(parsed.value.name, "Ada");
    assert!(
        parsed
            .flags
            .iter()
            .any(|flag| matches!(flag, Flag::ObjectFromMarkdown(_)))
    );
}

#[test]
fn explanations_surface_on_map_parse_error() {
    let raw = r#"{ "values": { "ok": 1, "bad": "oops" } }"#;
    let parsed = parse_llm_output::<FuzzExplain>(raw, true).expect("parse");
    assert_eq!(parsed.value.values.get("ok"), Some(&1));
    assert!(!parsed.explanations.is_empty());
}

#[test]
fn schema_fingerprint_is_stable() {
    let output = <DocUser as bamltype::BamlType>::baml_output_format();
    let a = schema_fingerprint(output, RenderOptions::default())
        .expect("schema fingerprint should work");
    let b = schema_fingerprint(output, RenderOptions::default())
        .expect("schema fingerprint should work");
    assert_eq!(a, b);
}
