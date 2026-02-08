//! Integration tests for bamltype.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use baml_types::{BamlValue, ConstraintLevel, LiteralValue, StreamingMode, TypeIR};
use bamltype::adapters::{AdapterSchemaRegistry, FieldCodec};
use bamltype::{
    BamlSchema, BamlTypeInternal, from_baml_value, from_baml_value_with_flags, parse,
    render_schema_default, to_baml_value,
};
use indexmap::IndexMap;

/// A simple response struct for testing.
#[bamltype::BamlType]
struct Response {
    /// The user's name
    name: String,
    /// Age in years
    age: u32,
    /// Whether the user is active
    active: bool,
}

/// A nested struct for testing.
#[bamltype::BamlType]
struct UserProfile {
    /// User information
    user: Response,
    /// Optional email address
    email: Option<String>,
    /// List of tags
    tags: Vec<String>,
}

#[test]
fn test_simple_struct_schema() {
    let schema = render_schema_default::<Response>().expect("Should render schema");

    // The schema should contain the field names
    assert!(
        schema.contains("name"),
        "Schema should mention 'name' field"
    );
    assert!(schema.contains("age"), "Schema should mention 'age' field");
    assert!(
        schema.contains("active"),
        "Schema should mention 'active' field"
    );
}

#[test]
fn test_nested_struct_schema() {
    let schema = render_schema_default::<UserProfile>().expect("Should render schema");

    println!("UserProfile schema:\n{}", schema);

    // The schema should contain nested type information
    assert!(
        schema.contains("user"),
        "Schema should mention 'user' field"
    );
    assert!(
        schema.contains("email"),
        "Schema should mention 'email' field"
    );
    assert!(
        schema.contains("tags"),
        "Schema should mention 'tags' field"
    );
}

#[test]
fn test_schema_bundle_caching() {
    // Get schema bundle twice - should return the same static reference
    let bundle1 = Response::baml_schema();
    let bundle2 = Response::baml_schema();

    // These should be the same pointer (cached)
    assert!(
        std::ptr::eq(bundle1, bundle2),
        "Schema bundle should be cached"
    );
}

#[test]
fn test_schema_output_format() {
    let schema = render_schema_default::<Response>().expect("Should render schema");

    // Print the schema for debugging
    println!("Generated schema:\n{}", schema);

    // Schema should not be empty
    assert!(!schema.is_empty(), "Schema should not be empty");
}

#[test]
fn test_parse_llm_output() {
    // Simulate LLM output (with markdown code block, which jsonish handles)
    let llm_output = r#"```json
{
    "name": "Alice",
    "age": 30,
    "active": true
}
```"#;

    let parsed = parse::<Response>(llm_output).expect("Should parse LLM output");

    // BamlValueWithFlags should have the right structure
    println!("Parsed: {:?}", parsed);
}

#[test]
fn test_parse_nested_struct() {
    let llm_output = r#"{
        "user": {
            "name": "Bob",
            "age": 25,
            "active": false
        },
        "email": "bob@example.com",
        "tags": ["admin", "user"]
    }"#;

    let parsed = parse::<UserProfile>(llm_output).expect("Should parse nested struct");
    println!("Parsed nested: {:?}", parsed);
}

#[test]
fn test_parse_with_optional_null() {
    // First check the schema to see how email is typed
    let schema = render_schema_default::<UserProfile>().expect("render");
    println!("Schema for optional test:\n{}", schema);

    let llm_output = r#"{
        "user": {"name": "Charlie", "age": 35, "active": true},
        "email": null,
        "tags": []
    }"#;

    let parsed = parse::<UserProfile>(llm_output).expect("Should handle null optional");
    println!("Parsed with null: {:?}", parsed);
}

// ============================================================================
// BamlValue conversion tests
// ============================================================================

#[test]
fn test_from_baml_value_simple_struct() {
    let mut fields = IndexMap::new();
    fields.insert("name".to_string(), BamlValue::String("Alice".into()));
    fields.insert("age".to_string(), BamlValue::Int(30));
    fields.insert("active".to_string(), BamlValue::Bool(true));

    let baml_value = BamlValue::Class("Response".into(), fields);

    let response: Response = from_baml_value(baml_value).expect("Should convert to Response");

    assert_eq!(response.name, "Alice");
    assert_eq!(response.age, 30);
    assert!(response.active);
}

#[test]
fn test_to_baml_value_simple_struct() {
    let name = "Bob".to_string();
    let baml = to_baml_value(&name).expect("Should convert String");
    assert_eq!(baml, BamlValue::String("Bob".into()));

    let num: i64 = 42;
    let baml = to_baml_value(&num).expect("Should convert i64");
    assert_eq!(baml, BamlValue::Int(42));

    let flag: bool = true;
    let baml = to_baml_value(&flag).expect("Should convert bool");
    assert_eq!(baml, BamlValue::Bool(true));
}

#[test]
fn test_from_baml_value_with_list() {
    let items = vec![BamlValue::Int(1), BamlValue::Int(2), BamlValue::Int(3)];
    let baml_value = BamlValue::List(items);

    let result: Vec<i64> = from_baml_value(baml_value).expect("Should convert to Vec<i64>");
    assert_eq!(result, vec![1i64, 2, 3]);
}

#[test]
fn test_from_baml_value_nested_struct() {
    let mut user_fields = IndexMap::new();
    user_fields.insert("name".to_string(), BamlValue::String("Test".into()));
    user_fields.insert("age".to_string(), BamlValue::Int(25));
    user_fields.insert("active".to_string(), BamlValue::Bool(false));

    let mut profile_fields = IndexMap::new();
    profile_fields.insert(
        "user".to_string(),
        BamlValue::Class("Response".into(), user_fields),
    );
    profile_fields.insert(
        "email".to_string(),
        BamlValue::String("test@example.com".into()),
    );
    profile_fields.insert(
        "tags".to_string(),
        BamlValue::List(vec![
            BamlValue::String("tag1".into()),
            BamlValue::String("tag2".into()),
        ]),
    );

    let baml_value = BamlValue::Class("UserProfile".into(), profile_fields);

    let profile: UserProfile = from_baml_value(baml_value).expect("Should convert nested struct");

    assert_eq!(profile.user.name, "Test");
    assert_eq!(profile.user.age, 25);
    assert!(!profile.user.active);
    assert_eq!(profile.email, Some("test@example.com".into()));
    assert_eq!(profile.tags, vec!["tag1", "tag2"]);
}

#[test]
fn test_from_baml_value_with_null_optional() {
    let mut user_fields = IndexMap::new();
    user_fields.insert("name".to_string(), BamlValue::String("NoEmail".into()));
    user_fields.insert("age".to_string(), BamlValue::Int(20));
    user_fields.insert("active".to_string(), BamlValue::Bool(true));

    let mut profile_fields = IndexMap::new();
    profile_fields.insert(
        "user".to_string(),
        BamlValue::Class("Response".into(), user_fields),
    );
    profile_fields.insert("email".to_string(), BamlValue::Null);
    profile_fields.insert("tags".to_string(), BamlValue::List(vec![]));

    let baml_value = BamlValue::Class("UserProfile".into(), profile_fields);

    let profile: UserProfile = from_baml_value(baml_value).expect("Should handle null optional");

    assert_eq!(profile.email, None);
}

#[test]
fn test_round_trip_list() {
    let original: Vec<String> = vec!["a".into(), "b".into(), "c".into()];
    let baml = to_baml_value(&original).expect("to_baml_value");
    let restored: Vec<String> = from_baml_value(baml).expect("from_baml_value");
    assert_eq!(original, restored);
}

#[test]
fn test_from_baml_value_with_flags() {
    let llm_output = r#"{"name": "FlagsTest", "age": 40, "active": false}"#;

    let parsed = parse::<Response>(llm_output).expect("Should parse");
    let response: Response =
        from_baml_value_with_flags(&parsed).expect("Should convert from flags");

    assert_eq!(response.name, "FlagsTest");
    assert_eq!(response.age, 40);
    assert!(!response.active);
}

#[test]
fn test_parse_and_convert_nested() {
    let llm_output = r#"{
        "user": {"name": "Integration", "age": 50, "active": true},
        "email": "int@test.com",
        "tags": ["rust", "baml"]
    }"#;

    let parsed = parse::<UserProfile>(llm_output).expect("Should parse");
    let profile: UserProfile = from_baml_value_with_flags(&parsed).expect("Should convert");

    assert_eq!(profile.user.name, "Integration");
    assert_eq!(profile.user.age, 50);
    assert!(profile.user.active);
    assert_eq!(profile.email, Some("int@test.com".into()));
    assert_eq!(profile.tags, vec!["rust", "baml"]);
}

// ============================================================================
// Smart pointer tests
// ============================================================================

#[test]
fn test_from_baml_value_box() {
    let val: Box<String> = from_baml_value(BamlValue::String("boxed".into())).unwrap();
    assert_eq!(*val, "boxed");
}

#[test]
fn test_from_baml_value_arc_struct() {
    let mut fields = IndexMap::new();
    fields.insert("name".to_string(), BamlValue::String("ArcUser".into()));
    fields.insert("age".to_string(), BamlValue::Int(28));
    fields.insert("active".to_string(), BamlValue::Bool(true));

    let val: Arc<Response> = from_baml_value(BamlValue::Class("Response".into(), fields)).unwrap();
    assert_eq!(val.name, "ArcUser");
    assert_eq!(val.age, 28);
    assert!(val.active);
}

#[test]
fn test_to_baml_value_box() {
    let val = Box::new(42i64);
    let baml = to_baml_value(&val).unwrap();
    assert_eq!(baml, BamlValue::Int(42));
}

#[test]
fn test_to_baml_value_arc() {
    let val = Arc::new("hello".to_string());
    let baml = to_baml_value(&val).unwrap();
    assert_eq!(baml, BamlValue::String("hello".into()));
}

#[test]
fn test_round_trip_box() {
    let mut fields = IndexMap::new();
    fields.insert("name".to_string(), BamlValue::String("Boxed".into()));
    fields.insert("age".to_string(), BamlValue::Int(33));
    fields.insert("active".to_string(), BamlValue::Bool(false));

    let boxed: Box<Response> =
        from_baml_value(BamlValue::Class("Response".into(), fields)).unwrap();

    let baml = to_baml_value(&boxed).unwrap();
    match &baml {
        BamlValue::Class(name, fields) => {
            assert_eq!(name, <Response as BamlTypeInternal>::baml_internal_name());
            assert_eq!(fields.get("name"), Some(&BamlValue::String("Boxed".into())));
            assert_eq!(fields.get("age"), Some(&BamlValue::Int(33)));
            assert_eq!(fields.get("active"), Some(&BamlValue::Bool(false)));
        }
        other => panic!("Expected Class, got {:?}", other),
    }
}

// ============================================================================
// Integer narrowing tests
// ============================================================================

#[test]
fn test_from_baml_int_to_u32() {
    let val: u32 = from_baml_value(BamlValue::Int(30)).unwrap();
    assert_eq!(val, 30);
}

#[test]
fn test_from_baml_int_to_i32() {
    let val: i32 = from_baml_value(BamlValue::Int(-5)).unwrap();
    assert_eq!(val, -5);
}

#[test]
fn test_from_baml_int_overflow_u32() {
    let result = from_baml_value::<u32>(BamlValue::Int(-1));
    assert!(
        result.is_err(),
        "Expected error for -1 → u32, got {:?}",
        result
    );
}

#[test]
fn test_from_baml_int_overflow_i32() {
    let result = from_baml_value::<i32>(BamlValue::Int(i64::MAX));
    assert!(
        result.is_err(),
        "Expected error for i64::MAX → i32, got {:?}",
        result
    );
}

#[test]
fn test_to_baml_value_u32() {
    let baml = to_baml_value(&42u32).unwrap();
    assert_eq!(baml, BamlValue::Int(42));
}

#[test]
fn test_to_baml_value_i32() {
    let baml = to_baml_value(&-7i32).unwrap();
    assert_eq!(baml, BamlValue::Int(-7));
}

// ============================================================================
// Float coercion tests
// ============================================================================

#[test]
fn test_from_baml_int_to_f64() {
    let val: f64 = from_baml_value(BamlValue::Int(42)).unwrap();
    assert!((val - 42.0).abs() < f64::EPSILON);
}

#[test]
fn test_from_baml_int_to_f32() {
    let val: f32 = from_baml_value(BamlValue::Int(42)).unwrap();
    assert!((val - 42.0).abs() < f32::EPSILON);
}

#[test]
fn test_to_baml_value_f32() {
    let pi = std::f32::consts::PI;
    let baml = to_baml_value(&pi).unwrap();
    match baml {
        BamlValue::Float(f) => assert!((f - f64::from(pi)).abs() < 0.01),
        other => panic!("Expected Float, got {:?}", other),
    }
}

// ============================================================================
// Map tests
// ============================================================================

#[test]
fn test_from_baml_map_to_hashmap() {
    let mut map = IndexMap::new();
    map.insert("a".to_string(), BamlValue::Int(1));
    map.insert("b".to_string(), BamlValue::Int(2));

    let result: HashMap<String, i64> = from_baml_value(BamlValue::Map(map)).unwrap();
    assert_eq!(result.get("a"), Some(&1i64));
    assert_eq!(result.get("b"), Some(&2i64));
}

#[test]
fn test_from_baml_map_to_btreemap() {
    let mut map = IndexMap::new();
    map.insert("x".to_string(), BamlValue::String("hello".into()));
    map.insert("y".to_string(), BamlValue::String("world".into()));

    let result: BTreeMap<String, String> = from_baml_value(BamlValue::Map(map)).unwrap();
    assert_eq!(result.get("x"), Some(&"hello".to_string()));
    assert_eq!(result.get("y"), Some(&"world".to_string()));
}

#[test]
fn test_to_baml_value_hashmap() {
    let mut map = HashMap::new();
    map.insert("key".to_string(), 99i64);

    let baml = to_baml_value(&map).unwrap();
    match baml {
        BamlValue::Map(m) => {
            assert_eq!(m.get("key"), Some(&BamlValue::Int(99)));
        }
        other => panic!("Expected Map, got {:?}", other),
    }
}

#[test]
fn test_round_trip_hashmap() {
    let mut original = HashMap::new();
    original.insert("one".to_string(), 1i64);
    original.insert("two".to_string(), 2i64);

    let baml = to_baml_value(&original).unwrap();
    let restored: HashMap<String, i64> = from_baml_value(baml).unwrap();
    assert_eq!(original, restored);
}

// ============================================================================
// Nested container tests
// ============================================================================

#[test]
fn test_option_vec_some() {
    let items = vec![BamlValue::String("a".into()), BamlValue::String("b".into())];
    let baml = BamlValue::List(items);

    let result: Option<Vec<String>> = from_baml_value(baml).unwrap();
    assert_eq!(result, Some(vec!["a".to_string(), "b".to_string()]));
}

#[test]
fn test_option_vec_none() {
    let result: Option<Vec<String>> = from_baml_value(BamlValue::Null).unwrap();
    assert_eq!(result, None);
}

#[test]
fn test_vec_option() {
    let items = vec![BamlValue::Int(1), BamlValue::Null, BamlValue::Int(3)];
    let baml = BamlValue::List(items);

    let result: Vec<Option<i64>> = from_baml_value(baml).unwrap();
    assert_eq!(result, vec![Some(1i64), None, Some(3i64)]);
}

#[test]
fn test_nested_option_some_some() {
    let result: Option<Option<String>> = from_baml_value(BamlValue::String("x".into())).unwrap();
    assert_eq!(result, Some(Some("x".to_string())));
}

#[test]
fn test_nested_option_none() {
    let result: Option<Option<String>> = from_baml_value(BamlValue::Null).unwrap();
    assert_eq!(result, None);
}

// ============================================================================
// Enum tests
// ============================================================================

#[derive(Debug, PartialEq)]
#[bamltype::BamlType]
enum Color {
    Red,
    Green,
    Blue,
}

#[test]
fn test_from_baml_enum() {
    let val: Color = from_baml_value(BamlValue::Enum("Color".into(), "Red".into())).unwrap();
    assert_eq!(val, Color::Red);
}

#[test]
fn test_to_baml_enum() {
    let baml = to_baml_value(&Color::Green).unwrap();
    assert_eq!(
        baml,
        BamlValue::Enum(
            <Color as BamlTypeInternal>::baml_internal_name().into(),
            "Green".into()
        )
    );
}

#[test]
fn test_round_trip_enum() {
    let original = Color::Blue;
    let baml = to_baml_value(&original).unwrap();
    let restored: Color = from_baml_value(baml).unwrap();
    assert_eq!(original, restored);
}

#[test]
fn test_unknown_variant_errors() {
    let result = from_baml_value::<Color>(BamlValue::Enum("Color".into(), "Yellow".into()));
    assert!(
        result.is_err(),
        "Expected error for unknown variant 'Yellow', got {:?}",
        result
    );
}

// ============================================================================
// Edge cases
// ============================================================================

#[test]
fn test_extra_fields_ignored() {
    let mut fields = IndexMap::new();
    fields.insert("name".to_string(), BamlValue::String("Alice".into()));
    fields.insert("age".to_string(), BamlValue::Int(30));
    fields.insert("active".to_string(), BamlValue::Bool(true));
    fields.insert(
        "extra_field".to_string(),
        BamlValue::String("ignored".into()),
    );

    let response: Response = from_baml_value(BamlValue::Class("Response".into(), fields)).unwrap();
    assert_eq!(response.name, "Alice");
    assert_eq!(response.age, 30);
    assert!(response.active);
}

#[derive(Debug, PartialEq)]
#[bamltype::BamlType]
struct Empty {}

#[test]
fn test_empty_struct() {
    let fields = IndexMap::new();
    let val: Empty = from_baml_value(BamlValue::Class("Empty".into(), fields)).unwrap();
    let baml = to_baml_value(&val).unwrap();
    match baml {
        BamlValue::Class(name, fields) => {
            assert_eq!(name, <Empty as BamlTypeInternal>::baml_internal_name());
            assert!(fields.is_empty());
        }
        other => panic!("Expected Class, got {:?}", other),
    }
}

// ============================================================================
// #[baml(...)] compatibility tests
// ============================================================================

#[derive(Debug, PartialEq)]
#[bamltype::BamlType]
struct CompatStruct {
    #[baml(alias = "fullName")]
    full_name: String,
    #[baml(skip)]
    internal: String,
    #[baml(default)]
    note: Option<String>,
}

#[derive(Debug, PartialEq)]
#[bamltype::BamlType]
enum CompatEnum {
    #[baml(alias = "go")]
    Start,
    Stop,
}

#[derive(Debug, PartialEq)]
#[bamltype::BamlType]
struct SerdeRenameCompat {
    #[serde(rename = "nickName")]
    nickname: String,
}

#[test]
fn test_baml_alias_and_skip_to_baml() {
    let value = CompatStruct {
        full_name: "Alice".into(),
        internal: "secret".into(),
        note: None,
    };

    let baml = to_baml_value(&value).unwrap();
    match baml {
        BamlValue::Class(_, fields) => {
            assert_eq!(
                fields.get("fullName"),
                Some(&BamlValue::String("Alice".into()))
            );
            assert!(!fields.contains_key("full_name"));
            assert!(!fields.contains_key("internal"));
        }
        other => panic!("Expected Class, got {:?}", other),
    }
}

#[test]
fn test_baml_alias_and_skip_from_baml() {
    let mut aliased = IndexMap::new();
    aliased.insert("fullName".to_string(), BamlValue::String("Bob".into()));
    aliased.insert(
        "internal".to_string(),
        BamlValue::String("should_skip".into()),
    );
    let parsed_alias: CompatStruct =
        from_baml_value(BamlValue::Class("CompatStruct".into(), aliased)).unwrap();
    assert_eq!(parsed_alias.full_name, "Bob");
    assert_eq!(parsed_alias.internal, "");
    assert_eq!(parsed_alias.note, None);

    let mut original = IndexMap::new();
    original.insert("full_name".to_string(), BamlValue::String("Charlie".into()));
    let parsed_original: CompatStruct =
        from_baml_value(BamlValue::Class("CompatStruct".into(), original)).unwrap();
    assert_eq!(parsed_original.full_name, "Charlie");
    assert_eq!(parsed_original.internal, "");
}

#[test]
fn test_baml_skip_field_excluded_from_schema() {
    let schema = render_schema_default::<CompatStruct>().expect("schema");
    assert!(schema.contains("fullName"));
    assert!(!schema.contains("internal"));
}

#[test]
fn test_baml_enum_alias_round_trip() {
    let as_baml = to_baml_value(&CompatEnum::Start).unwrap();
    assert_eq!(
        as_baml,
        BamlValue::Enum(
            <CompatEnum as BamlTypeInternal>::baml_internal_name().into(),
            "go".into()
        )
    );

    let from_alias: CompatEnum =
        from_baml_value(BamlValue::Enum("CompatEnum".into(), "go".into())).unwrap();
    assert_eq!(from_alias, CompatEnum::Start);

    let from_original: CompatEnum =
        from_baml_value(BamlValue::Enum("CompatEnum".into(), "Start".into())).unwrap();
    assert_eq!(from_original, CompatEnum::Start);
}

#[test]
fn test_serde_rename_is_accepted_by_bamltype() {
    let value = SerdeRenameCompat {
        nickname: "D".into(),
    };

    let baml = to_baml_value(&value).unwrap();
    match baml {
        BamlValue::Class(_, fields) => {
            assert_eq!(fields.get("nickName"), Some(&BamlValue::String("D".into())));
        }
        other => panic!("Expected Class, got {:?}", other),
    }

    let mut aliased = IndexMap::new();
    aliased.insert("nickName".to_string(), BamlValue::String("E".into()));
    let parsed_alias: SerdeRenameCompat =
        from_baml_value(BamlValue::Class("SerdeRenameCompat".into(), aliased)).unwrap();
    assert_eq!(parsed_alias.nickname, "E");

    let mut original = IndexMap::new();
    original.insert("nickname".to_string(), BamlValue::String("F".into()));
    let parsed_original: SerdeRenameCompat =
        from_baml_value(BamlValue::Class("SerdeRenameCompat".into(), original)).unwrap();
    assert_eq!(parsed_original.nickname, "F");
}

// ============================================================================
// Bridge parity tests
// ============================================================================

#[derive(Debug, PartialEq)]
#[bamltype::BamlType]
#[baml(tag = "type")]
enum TaggedShapeParity {
    /// A circle, defined by its radius.
    Circle {
        /// Radius in meters.
        radius: f64,
    },
    Rectangle {
        width: f64,
        height: f64,
    },
    Empty,
}

#[derive(Debug, PartialEq)]
#[bamltype::BamlType]
#[baml(as_union)]
enum UnitAsUnionParity {
    Red,
    #[baml(alias = "green")]
    Green,
}

#[derive(Debug, PartialEq)]
#[bamltype::BamlType]
#[baml(rename_all = "lowercase")]
enum LowercaseEnumParity {
    Done,
}

#[derive(Debug, PartialEq)]
#[bamltype::BamlType]
#[baml(rename_all = "UPPERCASE")]
struct UppercaseFieldParity {
    value: i64,
}

#[derive(Debug, PartialEq)]
#[bamltype::BamlType]
struct BigIntStringParity {
    #[baml(int_repr = "string")]
    id: u64,
}

#[derive(Debug, PartialEq)]
#[bamltype::BamlType]
struct MapKeysStringParity {
    #[baml(map_key_repr = "string")]
    values: HashMap<u32, String>,
}

#[derive(Debug, PartialEq)]
#[bamltype::BamlType]
struct MapKeysPairsParity {
    #[baml(map_key_repr = "pairs")]
    values: HashMap<u32, String>,
}

#[derive(Debug, PartialEq)]
#[bamltype::BamlType]
struct BigIntOptionParity {
    #[baml(int_repr = "i64")]
    id: Option<u64>,
}

#[derive(Debug, PartialEq)]
#[bamltype::BamlType]
struct MapKeysOptionParity {
    #[baml(map_key_repr = "string")]
    values: Option<HashMap<u32, String>>,
}

#[derive(Debug, PartialEq)]
#[bamltype::BamlType]
struct RecursiveNodeParity {
    value: i64,
    next: Option<Box<RecursiveNodeParity>>,
}

#[derive(Debug, PartialEq)]
#[bamltype::BamlType]
struct CheckedValueParity {
    #[baml(check(label = "positive", expr = "this > 0"))]
    value: i64,
}

#[derive(Debug, PartialEq)]
#[bamltype::BamlType]
struct AssertedValueParity {
    #[baml(assert(label = "positive", expr = "this > 0"))]
    value: i64,
}

#[derive(Debug, PartialEq)]
#[bamltype::BamlType]
#[baml(rename_all = "camelCase")]
struct RenameAllParity {
    full_name: String,
}

struct U64ObjectAdapter;

impl FieldCodec<u64> for U64ObjectAdapter {
    fn type_ir() -> TypeIR {
        TypeIR::class("AdapterU64Wrapper")
    }

    fn register(reg: &mut AdapterSchemaRegistry) {
        use bamltype::internal_baml_jinja::types::{Class, Name};

        if !reg.mark_type("AdapterU64Wrapper") {
            return;
        }

        reg.register_class(Class {
            name: Name::new("AdapterU64Wrapper".to_string()),
            description: None,
            namespace: StreamingMode::NonStreaming,
            fields: vec![(
                Name::new("value".to_string()),
                TypeIR::string(),
                None,
                false,
            )],
            constraints: Vec::new(),
            streaming_behavior: Default::default(),
        });
    }

    fn try_from_baml(
        value: BamlValue,
        path: Vec<String>,
    ) -> Result<u64, bamltype::BamlConvertError> {
        let map = match value {
            BamlValue::Class(_, map) | BamlValue::Map(map) => map,
            other => {
                return Err(bamltype::BamlConvertError::new(
                    path,
                    "object",
                    format!("{other:?}"),
                    "expected adapter payload object",
                ));
            }
        };

        let value = bamltype::get_field(&map, "value", None).ok_or_else(|| {
            bamltype::BamlConvertError::new(
                path,
                "value",
                "<missing>",
                "missing required adapter field",
            )
        })?;

        match value {
            BamlValue::String(raw) => raw.parse::<u64>().map_err(|err| {
                bamltype::BamlConvertError::new(
                    Vec::new(),
                    "u64",
                    raw.clone(),
                    format!("invalid integer string: {err}"),
                )
            }),
            other => Err(bamltype::BamlConvertError::new(
                Vec::new(),
                "string",
                format!("{other:?}"),
                "adapter value must be a string",
            )),
        }
    }
}

#[derive(Debug, PartialEq)]
#[bamltype::BamlType]
struct WithAdapterParity {
    #[baml(with = "U64ObjectAdapter")]
    id: u64,
}

mod parity_collision_a {
    #[derive(Debug, PartialEq)]
    #[bamltype::BamlType]
    pub struct User {
        pub name: String,
    }
}

mod parity_collision_b {
    #[derive(Debug, PartialEq)]
    #[bamltype::BamlType]
    pub struct User {
        pub name: String,
    }
}

#[test]
fn data_enum_tagged_parses() {
    let raw = r#"{ "type": "Circle", "radius": 2.5 }"#;
    let parsed = parse::<TaggedShapeParity>(raw).expect("parse failed");
    let value: TaggedShapeParity = from_baml_value_with_flags(&parsed).expect("convert failed");
    assert_eq!(value, TaggedShapeParity::Circle { radius: 2.5 });
}

#[test]
fn as_union_unit_enum_type_ir_is_literal_union() {
    let type_ir = <UnitAsUnionParity as BamlTypeInternal>::baml_type_ir();
    let TypeIR::Union(union, _) = type_ir else {
        panic!("expected union type IR");
    };

    let mut literals = union
        .iter_skip_null()
        .into_iter()
        .map(|item| match item {
            TypeIR::Literal(LiteralValue::String(value), _) => value.clone(),
            other => panic!("expected string literal in union, got {other:?}"),
        })
        .collect::<Vec<_>>();
    literals.sort();

    assert_eq!(literals, vec!["Red".to_string(), "green".to_string()]);
}

#[test]
fn as_union_unit_enum_alias_parses() {
    let parsed = parse::<UnitAsUnionParity>(r#""green""#).expect("parse failed");
    let value: UnitAsUnionParity = from_baml_value_with_flags(&parsed).expect("convert failed");
    assert_eq!(value, UnitAsUnionParity::Green);
}

#[test]
fn rename_all_lowercase_variant_parses() {
    let parsed = parse::<LowercaseEnumParity>(r#""done""#).expect("parse failed");
    let value: LowercaseEnumParity = from_baml_value_with_flags(&parsed).expect("convert failed");
    assert_eq!(value, LowercaseEnumParity::Done);
}

#[test]
fn rename_all_uppercase_field_parses() {
    let parsed = parse::<UppercaseFieldParity>(r#"{ "VALUE": 7 }"#).expect("parse failed");
    let value: UppercaseFieldParity = from_baml_value_with_flags(&parsed).expect("convert failed");
    assert_eq!(value.value, 7);

    let roundtrip =
        to_baml_value(&UppercaseFieldParity { value: 9 }).expect("to_baml_value failed");
    match roundtrip {
        BamlValue::Class(_, fields) | BamlValue::Map(fields) => {
            assert_eq!(fields.get("VALUE"), Some(&BamlValue::Int(9)));
        }
        other => panic!("expected object-like value, got {other:?}"),
    }
}

#[test]
fn to_baml_data_enum_shape() {
    let value = TaggedShapeParity::Circle { radius: 1.25 };
    let baml = to_baml_value(&value).expect("to_baml_value failed");
    match baml {
        BamlValue::Class(_, fields) | BamlValue::Map(fields) => {
            assert_eq!(
                fields.get("type"),
                Some(&BamlValue::String("Circle".into()))
            );
            assert_eq!(fields.get("radius"), Some(&BamlValue::Float(1.25)));
        }
        other => panic!("expected class-like value, got {other:?}"),
    }
}

#[test]
fn int_repr_string_parses() {
    let raw = r#"{ "id": "18446744073709551615" }"#;
    let parsed = parse::<BigIntStringParity>(raw).expect("parse failed");
    let value: BigIntStringParity = from_baml_value_with_flags(&parsed).expect("convert failed");
    assert_eq!(value.id, u64::MAX);
}

#[test]
fn int_repr_option_parses() {
    let raw = r#"{ "id": 42 }"#;
    let parsed = parse::<BigIntOptionParity>(raw).expect("parse failed");
    let value: BigIntOptionParity = from_baml_value_with_flags(&parsed).expect("convert failed");
    assert_eq!(value.id, Some(42));

    let raw = r#"{ "id": null }"#;
    let parsed = parse::<BigIntOptionParity>(raw).expect("parse failed");
    let value: BigIntOptionParity = from_baml_value_with_flags(&parsed).expect("convert failed");
    assert_eq!(value.id, None);
}

#[test]
fn map_key_repr_string_parses() {
    let raw = r#"{ "values": { "1": "a", "2": "b" } }"#;
    let parsed = parse::<MapKeysStringParity>(raw).expect("parse failed");
    let value: MapKeysStringParity = from_baml_value_with_flags(&parsed).expect("convert failed");

    let mut expected = HashMap::new();
    expected.insert(1_u32, "a".to_string());
    expected.insert(2_u32, "b".to_string());
    assert_eq!(value.values, expected);
}

#[test]
fn map_key_repr_option_parses() {
    let raw = r#"{ "values": { "10": "x" } }"#;
    let parsed = parse::<MapKeysOptionParity>(raw).expect("parse failed");
    let value: MapKeysOptionParity = from_baml_value_with_flags(&parsed).expect("convert failed");

    let mut expected = HashMap::new();
    expected.insert(10_u32, "x".to_string());
    assert_eq!(value.values, Some(expected));
}

#[test]
fn map_key_repr_pairs_parses() {
    let raw = r#"{ "values": [ { "key": 1, "value": "a" }, { "key": 2, "value": "b" } ] }"#;
    let parsed = parse::<MapKeysPairsParity>(raw).expect("parse failed");
    let value: MapKeysPairsParity = from_baml_value_with_flags(&parsed).expect("convert failed");

    let mut expected = HashMap::new();
    expected.insert(1_u32, "a".to_string());
    expected.insert(2_u32, "b".to_string());
    assert_eq!(value.values, expected);
}

#[test]
fn map_key_repr_pairs_registers_entry_class() {
    let entry_name = format!(
        "{}::values__Entry",
        <MapKeysPairsParity as BamlTypeInternal>::baml_internal_name()
    );
    let of = &MapKeysPairsParity::baml_schema().output_format;
    let class = of
        .classes
        .get(&(entry_name, StreamingMode::NonStreaming))
        .expect("entry class missing");

    assert_eq!(class.name.rendered_name(), "valuesEntry");
}

#[test]
fn recursion_is_detected() {
    let of = &RecursiveNodeParity::baml_schema().output_format;
    assert!(
        of.recursive_classes
            .contains(<RecursiveNodeParity as BamlTypeInternal>::baml_internal_name())
    );
}

#[test]
fn rename_all_applies() {
    let raw = r#"{ "fullName": "Ada" }"#;
    let parsed = parse::<RenameAllParity>(raw).expect("parse failed");
    let value: RenameAllParity = from_baml_value_with_flags(&parsed).expect("convert failed");
    assert_eq!(value.full_name, "Ada");
}

#[test]
fn field_constraints_are_registered() {
    let of = &CheckedValueParity::baml_schema().output_format;
    let internal_name = <CheckedValueParity as BamlTypeInternal>::baml_internal_name().to_string();
    let class = of
        .classes
        .get(&(internal_name, StreamingMode::NonStreaming))
        .expect("class missing");

    let (_, field_type, _, _) = class
        .fields
        .iter()
        .find(|(name, _, _, _)| name.real_name() == "value")
        .expect("value field missing");

    assert!(field_type.meta().constraints.iter().any(|constraint| {
        constraint.level == ConstraintLevel::Check
            && constraint.label.as_deref() == Some("positive")
    }));
}

#[test]
fn field_assert_constraints_are_registered() {
    let of = &AssertedValueParity::baml_schema().output_format;
    let internal_name = <AssertedValueParity as BamlTypeInternal>::baml_internal_name().to_string();
    let class = of
        .classes
        .get(&(internal_name, StreamingMode::NonStreaming))
        .expect("class missing");

    let (_, field_type, _, _) = class
        .fields
        .iter()
        .find(|(name, _, _, _)| name.real_name() == "value")
        .expect("value field missing");

    assert!(field_type.meta().constraints.iter().any(|constraint| {
        constraint.level == ConstraintLevel::Assert
            && constraint.label.as_deref() == Some("positive")
    }));
}

#[test]
fn internal_names_are_unique() {
    let a_name = <parity_collision_a::User as BamlTypeInternal>::baml_internal_name();
    let b_name = <parity_collision_b::User as BamlTypeInternal>::baml_internal_name();
    assert_ne!(a_name, b_name);

    let a_class = parity_collision_a::User::baml_schema()
        .output_format
        .classes
        .get(&(a_name.to_string(), StreamingMode::NonStreaming))
        .expect("class missing");
    let b_class = parity_collision_b::User::baml_schema()
        .output_format
        .classes
        .get(&(b_name.to_string(), StreamingMode::NonStreaming))
        .expect("class missing");

    assert_eq!(a_class.name.rendered_name(), "User");
    assert_eq!(b_class.name.rendered_name(), "User");
}

#[test]
fn with_adapter_schema_and_registration_are_used() {
    let of = &WithAdapterParity::baml_schema().output_format;

    let adapter_class = of
        .classes
        .get(&("AdapterU64Wrapper".to_string(), StreamingMode::NonStreaming))
        .expect("adapter class missing");
    assert_eq!(adapter_class.name.real_name(), "AdapterU64Wrapper");
    assert!(
        adapter_class
            .fields
            .iter()
            .any(|(name, _, _, _)| name.real_name() == "value")
    );

    let owner_name = <WithAdapterParity as BamlTypeInternal>::baml_internal_name().to_string();
    let owner = of
        .classes
        .get(&(owner_name, StreamingMode::NonStreaming))
        .expect("owner class missing");
    let (_, field_ir, _, _) = owner
        .fields
        .iter()
        .find(|(name, _, _, _)| name.real_name() == "id")
        .expect("id field missing");
    assert!(matches!(
        field_ir,
        TypeIR::Class { name, .. } if name == "AdapterU64Wrapper"
    ));
}

#[test]
fn with_adapter_parse_uses_custom_converter() {
    let raw = r#"{ "id": { "value": "18446744073709551615" } }"#;
    let parsed = parse::<WithAdapterParity>(raw).expect("parse failed");
    let value: WithAdapterParity = from_baml_value_with_flags(&parsed).expect("convert failed");
    assert_eq!(value.id, u64::MAX);
}
