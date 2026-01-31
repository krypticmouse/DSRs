use std::collections::HashMap;

use baml_bridge::baml_types::StreamingMode;
use baml_bridge::RenderOptions;
use baml_bridge::{parse_llm_output, render_schema, BamlParseError, BamlType};

/// A user profile returned by the model.
///
/// ## Notes
/// - `fullName` should be the display name.
#[derive(Debug, Clone, PartialEq, BamlType)]
struct DocUser {
    /// Full name as displayed in the UI.
    #[baml(alias = "fullName")]
    name: String,
    age: i64,
}

/// This should be ignored.
#[derive(Debug, Clone, PartialEq, BamlType)]
#[baml(description = "Override description.")]
struct OverrideUser {
    value: String,
}

#[derive(Debug, Clone, PartialEq, BamlType)]
enum Color {
    /// Red hot.
    Red,
    #[baml(alias = "green")]
    Green,
}

#[derive(Debug, Clone, PartialEq, BamlType)]
#[baml(tag = "type")]
enum Shape {
    /// A circle, defined by its radius.
    Circle {
        /// Radius in meters.
        radius: f64,
    },
    Rectangle {
        width: f64,
        height: f64,
    },
}

#[derive(Debug, Clone, PartialEq, BamlType)]
struct BigIntString {
    #[baml(int_repr = "string")]
    id: u64,
}

#[derive(Debug, Clone, PartialEq, BamlType)]
struct BigIntOption {
    #[baml(int_repr = "i64")]
    id: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, BamlType)]
struct MapKeys {
    #[baml(map_key_repr = "string")]
    values: HashMap<u32, String>,
}

#[derive(Debug, Clone, PartialEq, BamlType)]
struct MapKeysOption {
    #[baml(map_key_repr = "string")]
    values: Option<HashMap<u32, String>>,
}

#[derive(Debug, Clone, PartialEq, BamlType)]
struct MapKeyPairs {
    #[baml(map_key_repr = "pairs")]
    values: HashMap<u32, String>,
}

#[derive(Debug, Clone, PartialEq, BamlType)]
struct Node {
    value: i64,
    next: Option<Box<Node>>,
}

#[derive(Debug, Clone, PartialEq, BamlType)]
#[baml(rename_all = "camelCase")]
struct RenameAllUser {
    full_name: String,
}

#[derive(Debug, Clone, PartialEq, BamlType)]
struct CheckedValue {
    #[baml(check(label = "positive", expr = "this > 0"))]
    value: i64,
}

#[derive(Debug, Clone, PartialEq, BamlType)]
struct AssertedValue {
    #[baml(assert(label = "positive", expr = "this > 0"))]
    value: i64,
}

#[derive(Debug, Clone, PartialEq, BamlType)]
struct Unsigned32 {
    value: u32,
}

mod collision_a {
    use super::*;

    #[derive(Debug, Clone, PartialEq, BamlType)]
    pub struct User {
        pub name: String,
    }
}

mod collision_b {
    use super::*;

    #[derive(Debug, Clone, PartialEq, BamlType)]
    pub struct User {
        pub name: String,
    }
}

#[test]
fn doc_comments_render() {
    let schema = render_schema::<DocUser>(RenderOptions::default())
        .expect("render failed")
        .unwrap_or_default();

    assert!(schema.contains("  // A user profile returned by the model."));
    assert!(schema.contains("  // \n  // ## Notes"));
    assert!(schema.contains("  // Full name as displayed in the UI."));
}

#[test]
fn description_override_wins() {
    let schema = render_schema::<OverrideUser>(RenderOptions::default())
        .expect("render failed")
        .unwrap_or_default();

    assert!(schema.contains("  // Override description."));
    assert!(!schema.contains("  // This should be ignored."));
}

#[test]
fn enum_variant_descriptions_render() {
    let schema = render_schema::<Color>(RenderOptions::default())
        .expect("render failed")
        .unwrap_or_default();

    assert!(schema.contains("Red: Red hot."));
}

#[test]
fn unit_enum_alias_parses() {
    let parsed = parse_llm_output::<Color>(r#""green""#, true).expect("parse failed");
    assert_eq!(parsed.value, Color::Green);
}

#[test]
fn data_enum_tagged_parses() {
    let raw = r#"{ "type": "Circle", "radius": 2.5 }"#;
    let parsed = parse_llm_output::<Shape>(raw, true).expect("parse failed");
    assert_eq!(parsed.value, Shape::Circle { radius: 2.5 });
}

#[test]
fn data_enum_docs_render() {
    let schema = render_schema::<Shape>(RenderOptions::default())
        .expect("render failed")
        .unwrap_or_default();

    assert!(schema.contains("// A circle, defined by its radius."));
    assert!(schema.contains("// Radius in meters."));
}

#[test]
fn int_repr_string_parses() {
    let raw = r#"{ "id": "18446744073709551615" }"#;
    let parsed = parse_llm_output::<BigIntString>(raw, true).expect("parse failed");
    assert_eq!(parsed.value.id, u64::MAX);
}

#[test]
fn int_repr_option_parses() {
    let raw = r#"{ "id": 42 }"#;
    let parsed = parse_llm_output::<BigIntOption>(raw, true).expect("parse failed");
    assert_eq!(parsed.value.id, Some(42));

    let raw = r#"{ "id": null }"#;
    let parsed = parse_llm_output::<BigIntOption>(raw, true).expect("parse failed");
    assert_eq!(parsed.value.id, None);
}

#[test]
fn map_key_repr_string_parses() {
    let raw = r#"{ "values": { "1": "a", "2": "b" } }"#;
    let parsed = parse_llm_output::<MapKeys>(raw, true).expect("parse failed");
    let mut expected = HashMap::new();
    expected.insert(1_u32, "a".to_string());
    expected.insert(2_u32, "b".to_string());
    assert_eq!(parsed.value.values, expected);
}

#[test]
fn map_key_repr_option_parses() {
    let raw = r#"{ "values": { "10": "x" } }"#;
    let parsed = parse_llm_output::<MapKeysOption>(raw, true).expect("parse failed");
    let mut expected = HashMap::new();
    expected.insert(10_u32, "x".to_string());
    assert_eq!(parsed.value.values, Some(expected));
}

#[test]
fn map_key_repr_pairs_parses() {
    let raw = r#"{ "values": [ { "key": 1, "value": "a" }, { "key": 2, "value": "b" } ] }"#;
    let parsed = parse_llm_output::<MapKeyPairs>(raw, true).expect("parse failed");
    let mut expected = HashMap::new();
    expected.insert(1_u32, "a".to_string());
    expected.insert(2_u32, "b".to_string());
    assert_eq!(parsed.value.values, expected);
}

#[test]
fn map_key_repr_pairs_registers_entry_class() {
    let entry_name = format!("{}::values__Entry", MapKeyPairs::baml_internal_name());
    let of = MapKeyPairs::baml_output_format();
    let class = of
        .classes
        .get(&(entry_name, StreamingMode::NonStreaming))
        .expect("entry class missing");

    assert_eq!(class.name.rendered_name(), "valuesEntry");
}

#[test]
fn recursion_is_detected() {
    let of = Node::baml_output_format();
    assert!(of.recursive_classes.contains(Node::baml_internal_name()));
}

#[test]
fn rename_all_applies() {
    let raw = r#"{ "fullName": "Ada" }"#;
    let parsed = parse_llm_output::<RenameAllUser>(raw, true).expect("parse failed");
    assert_eq!(parsed.value.full_name, "Ada");
}

#[test]
fn constraint_checks_are_reported() {
    let raw = r#"{ "value": 3 }"#;
    let parsed = parse_llm_output::<CheckedValue>(raw, true).expect("parse failed");
    assert!(parsed
        .checks
        .iter()
        .any(|check| check.name == "positive" && check.status == "succeeded"));
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
fn internal_names_are_unique() {
    let a_name = collision_a::User::baml_internal_name();
    let b_name = collision_b::User::baml_internal_name();
    assert_ne!(a_name, b_name);

    let a_class = collision_a::User::baml_output_format()
        .classes
        .get(&(a_name.to_string(), StreamingMode::NonStreaming))
        .expect("class missing");
    let b_class = collision_b::User::baml_output_format()
        .classes
        .get(&(b_name.to_string(), StreamingMode::NonStreaming))
        .expect("class missing");

    assert_eq!(a_class.name.rendered_name(), "User");
    assert_eq!(b_class.name.rendered_name(), "User");
}

// Test nested #[render(...)] on fields
#[derive(Debug, Clone, PartialEq, BamlType)]
struct TruncatedMessage {
    /// This field should be truncated to 5 chars.
    #[render(max_string_chars = 5)]
    content: String,
    /// This field uses default settings.
    title: String,
}

#[test]
fn nested_render_spec_truncates_field() {
    use baml_bridge::prompt::{PromptPath, PromptValue, PromptWorld};
    use baml_bridge::prompt::renderer::{RenderSession, RenderSettings};
    use baml_bridge::baml_types::{BamlValue, TypeIR};
    use baml_bridge::{BamlTypeInternal, Registry};
    use indexmap::IndexMap;
    use std::sync::Arc;

    let mut reg = Registry::new();
    TruncatedMessage::register(&mut reg);

    // Get the internal name (may differ due to module prefixing)
    let internal_name = <TruncatedMessage as BamlTypeInternal>::baml_internal_name();

    let (output_format, renderer_seed) = reg.build_with_renderers(TypeIR::class(internal_name));

    let world = Arc::new(
        PromptWorld::from_registry(output_format, renderer_seed, RenderSettings::default())
            .expect("prompt world"),
    );

    // Create a value with a long content string
    let value = BamlValue::Class(
        internal_name.to_string(),
        IndexMap::from([
            ("content".to_string(), BamlValue::String("hello world this is a long message".to_string())),
            ("title".to_string(), BamlValue::String("Short".to_string())),
        ]),
    );

    let pv = PromptValue::new(
        value,
        TypeIR::class(internal_name),
        world.clone(),
        Arc::new(RenderSession::new(RenderSettings::default())),
        PromptPath::new(),
    );

    let rendered = world.render_prompt_value(&pv, None).expect("render");

    // content should be truncated to 5 chars + "... (truncated)"
    // title should be rendered normally
    assert!(rendered.contains("hello... (truncated)"), "content should be truncated, got: {}", rendered);
    assert!(rendered.contains("Short"), "title should be rendered normally, got: {}", rendered);
}

// Test enum variant #[render(...)] applies everywhere (top-level and in lists)
#[derive(Debug, Clone, PartialEq, BamlType)]
enum Mood {
    /// Excited!
    #[render(template = "{{ value.raw }}!!!")]
    Happy,
    Sad,
}

#[test]
fn enum_variant_render_spec_applies_everywhere() {
    use baml_bridge::prompt::{PromptPath, PromptValue, PromptWorld};
    use baml_bridge::prompt::renderer::{RenderSession, RenderSettings};
    use baml_bridge::baml_types::{BamlValue, TypeIR};
    use baml_bridge::{BamlTypeInternal, Registry};
    use std::sync::Arc;

    let mut reg = Registry::new();
    Mood::register(&mut reg);

    let internal_name = <Mood as BamlTypeInternal>::baml_internal_name();
    let (output_format, renderer_seed) = reg.build_with_renderers(TypeIR::r#enum(internal_name));

    let world = Arc::new(
        PromptWorld::from_registry(output_format, renderer_seed, RenderSettings::default())
            .expect("prompt world"),
    );

    // Test 1: Top-level enum value - Happy variant should get "!!!" appended
    let happy_value = BamlValue::Enum(internal_name.to_string(), "Happy".to_string());
    let pv = PromptValue::new(
        happy_value.clone(),
        TypeIR::r#enum(internal_name),
        world.clone(),
        Arc::new(RenderSession::new(RenderSettings::default())),
        PromptPath::new(),
    );
    let rendered = world.render_prompt_value(&pv, None).expect("render");
    assert_eq!(rendered, "Happy!!!", "top-level Happy should have template applied, got: {}", rendered);

    // Test 2: Sad variant should NOT have template (no spec)
    let sad_value = BamlValue::Enum(internal_name.to_string(), "Sad".to_string());
    let pv = PromptValue::new(
        sad_value,
        TypeIR::r#enum(internal_name),
        world.clone(),
        Arc::new(RenderSession::new(RenderSettings::default())),
        PromptPath::new(),
    );
    let rendered = world.render_prompt_value(&pv, None).expect("render");
    assert_eq!(rendered, "Sad", "Sad should render normally, got: {}", rendered);

    // Test 3: Happy inside a list - should ALSO get "!!!" (proves "everywhere")
    let list_value = BamlValue::List(vec![happy_value]);
    let pv = PromptValue::new(
        list_value,
        TypeIR::list(TypeIR::r#enum(internal_name)),
        world.clone(),
        Arc::new(RenderSession::new(RenderSettings::default())),
        PromptPath::new(),
    );
    let rendered = world.render_prompt_value(&pv, None).expect("render");
    assert_eq!(rendered, "[Happy!!!]", "Happy in list should have template applied, got: {}", rendered);
}
