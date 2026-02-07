use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;
use std::sync::Arc;

use baml_bridge as legacy;
use baml_bridge::baml_types::{BamlValue, LiteralValue, StreamingMode, TypeIR};
use bamltype as facet_runtime;

#[derive(Debug, Clone, PartialEq, legacy::BamlType)]
#[baml(name = "ContractUser")]
#[baml(internal_name = "contract::User")]
struct BridgeUser {
    #[baml(alias = "fullName")]
    name: String,
    age: i64,
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(name = "ContractUser")]
#[baml(internal_name = "contract::User")]
struct FacetUser {
    #[baml(alias = "fullName")]
    name: String,
    age: i64,
}

#[derive(Debug, Clone, PartialEq, legacy::BamlType)]
#[baml(name = "ContractShape")]
#[baml(internal_name = "contract::Shape")]
#[baml(tag = "kind")]
enum BridgeShape {
    Circle { radius: f64 },
    Square { side: f64 },
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(name = "ContractShape")]
#[baml(internal_name = "contract::Shape")]
#[baml(tag = "kind")]
enum FacetShape {
    Circle { radius: f64 },
    Square { side: f64 },
}

#[derive(Debug, Clone, PartialEq, legacy::BamlType)]
#[baml(internal_name = "contract::Checked")]
struct BridgeChecked {
    #[baml(check(label = "positive", expr = "this > 0"))]
    value: i64,
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(internal_name = "contract::Checked")]
struct FacetChecked {
    #[baml(check(label = "positive", expr = "this > 0"))]
    value: i64,
}

#[derive(Debug, Clone, PartialEq, legacy::BamlType)]
#[baml(internal_name = "contract::Asserted")]
struct BridgeAsserted {
    #[baml(assert(label = "positive", expr = "this > 0"))]
    value: i64,
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(internal_name = "contract::Asserted")]
struct FacetAsserted {
    #[baml(assert(label = "positive", expr = "this > 0"))]
    value: i64,
}

#[derive(Debug, Clone, PartialEq, legacy::BamlType)]
#[baml(internal_name = "contract::MapPairs")]
struct BridgeMapPairs {
    #[baml(map_key_repr = "pairs")]
    values: HashMap<u32, String>,
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(internal_name = "contract::MapPairs")]
struct FacetMapPairs {
    #[baml(map_key_repr = "pairs")]
    values: HashMap<u32, String>,
}

#[derive(Debug, Clone, PartialEq, legacy::BamlType)]
#[baml(name = "OrderB")]
#[baml(internal_name = "contract::OrderB")]
struct BridgeOrderB {
    value: i64,
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(name = "OrderB")]
#[baml(internal_name = "contract::OrderB")]
struct FacetOrderB {
    value: i64,
}

#[derive(Debug, Clone, PartialEq, legacy::BamlType)]
#[baml(name = "OrderA")]
#[baml(internal_name = "contract::OrderA")]
struct BridgeOrderA {
    value: i64,
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(name = "OrderA")]
#[baml(internal_name = "contract::OrderA")]
struct FacetOrderA {
    value: i64,
}

#[derive(Debug, Clone, PartialEq, legacy::BamlType)]
#[baml(name = "OrderRoot")]
#[baml(internal_name = "contract::OrderRoot")]
struct BridgeOrderRoot {
    b: BridgeOrderB,
    a: BridgeOrderA,
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(name = "OrderRoot")]
#[baml(internal_name = "contract::OrderRoot")]
struct FacetOrderRoot {
    b: FacetOrderB,
    a: FacetOrderA,
}

#[derive(Debug, Clone, PartialEq, legacy::BamlType)]
#[baml(internal_name = "contract::BigIntString")]
struct BridgeBigIntString {
    #[baml(int_repr = "string")]
    id: u64,
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(internal_name = "contract::BigIntString")]
struct FacetBigIntString {
    #[baml(int_repr = "string")]
    id: u64,
}

#[derive(Debug, Clone, PartialEq, legacy::BamlType)]
#[baml(internal_name = "contract::Unsigned32")]
struct BridgeUnsigned32 {
    value: u32,
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(internal_name = "contract::Unsigned32")]
struct FacetUnsigned32 {
    value: u32,
}

#[derive(Debug, Clone, PartialEq, legacy::BamlType)]
#[baml(internal_name = "contract::FuzzUser")]
struct BridgeFuzzUser {
    name: String,
    age: u32,
    nickname: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(internal_name = "contract::FuzzUser")]
struct FacetFuzzUser {
    name: String,
    age: u32,
    nickname: Option<String>,
}

#[derive(Debug, Clone, PartialEq, legacy::BamlType)]
#[baml(internal_name = "contract::FuzzExplain")]
struct BridgeFuzzExplain {
    values: HashMap<String, i64>,
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(internal_name = "contract::FuzzExplain")]
struct FacetFuzzExplain {
    values: HashMap<String, i64>,
}

#[derive(Debug, Clone, PartialEq, legacy::BamlType)]
#[baml(internal_name = "contract::ColorAlias")]
enum BridgeColorAlias {
    Red,
    #[baml(alias = "green")]
    Green,
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(internal_name = "contract::ColorAlias")]
enum FacetColorAlias {
    Red,
    #[baml(alias = "green")]
    Green,
}

#[derive(Debug, Clone, PartialEq, legacy::BamlType)]
#[baml(name = "ContractDocShape")]
#[baml(internal_name = "contract::DocShape")]
#[baml(tag = "type")]
enum BridgeDocShape {
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

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(name = "ContractDocShape")]
#[baml(internal_name = "contract::DocShape")]
#[baml(tag = "type")]
enum FacetDocShape {
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

#[derive(Debug, Clone, PartialEq, legacy::BamlType)]
#[baml(internal_name = "contract::BigIntOption")]
struct BridgeBigIntOption {
    #[baml(int_repr = "i64")]
    id: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(internal_name = "contract::BigIntOption")]
struct FacetBigIntOption {
    #[baml(int_repr = "i64")]
    id: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, legacy::BamlType)]
#[baml(internal_name = "contract::MapKeys")]
struct BridgeMapKeys {
    #[baml(map_key_repr = "string")]
    values: HashMap<u32, String>,
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(internal_name = "contract::MapKeys")]
struct FacetMapKeys {
    #[baml(map_key_repr = "string")]
    values: HashMap<u32, String>,
}

#[derive(Debug, Clone, PartialEq, legacy::BamlType)]
#[baml(internal_name = "contract::MapKeysOption")]
struct BridgeMapKeysOption {
    #[baml(map_key_repr = "string")]
    values: Option<HashMap<u32, String>>,
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(internal_name = "contract::MapKeysOption")]
struct FacetMapKeysOption {
    #[baml(map_key_repr = "string")]
    values: Option<HashMap<u32, String>>,
}

#[derive(Debug, Clone, PartialEq, legacy::BamlType)]
#[baml(internal_name = "contract::Node")]
struct BridgeNode {
    value: i64,
    next: Option<Box<BridgeNode>>,
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(internal_name = "contract::Node")]
struct FacetNode {
    value: i64,
    next: Option<Box<FacetNode>>,
}

#[derive(Debug, Clone, PartialEq, legacy::BamlType)]
#[baml(rename_all = "camelCase")]
#[baml(internal_name = "contract::RenameAllUser")]
struct BridgeRenameAllUser {
    full_name: String,
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(rename_all = "camelCase")]
#[baml(internal_name = "contract::RenameAllUser")]
struct FacetRenameAllUser {
    full_name: String,
}

#[derive(Debug, Clone, PartialEq, legacy::BamlType)]
#[baml(internal_name = "contract::RoundtripStruct")]
struct BridgeRoundtripStruct {
    name: String,
    count: i32,
    tags: Vec<String>,
    meta: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(internal_name = "contract::RoundtripStruct")]
struct FacetRoundtripStruct {
    name: String,
    count: i32,
    tags: Vec<String>,
    meta: Option<String>,
}

#[derive(Debug, Clone, PartialEq, legacy::BamlType)]
#[baml(internal_name = "contract::RoundtripUnitEnum")]
enum BridgeRoundtripUnitEnum {
    Alpha,
    #[baml(alias = "beta")]
    Beta,
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(internal_name = "contract::RoundtripUnitEnum")]
enum FacetRoundtripUnitEnum {
    Alpha,
    #[baml(alias = "beta")]
    Beta,
}

#[derive(Debug, Clone, PartialEq, legacy::BamlType)]
#[baml(internal_name = "contract::RoundtripDataEnum")]
#[baml(tag = "kind")]
enum BridgeRoundtripDataEnum {
    Message { body: String, count: i64 },
    Empty,
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(internal_name = "contract::RoundtripDataEnum")]
#[baml(tag = "kind")]
enum FacetRoundtripDataEnum {
    Message { body: String, count: i64 },
    Empty,
}

#[derive(Debug, Clone, PartialEq, legacy::BamlType)]
#[baml(internal_name = "contract::NestedStruct")]
struct BridgeNestedStruct {
    title: String,
    items: Option<Vec<BridgeRoundtripStruct>>,
    metadata: HashMap<String, Option<i32>>,
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(internal_name = "contract::NestedStruct")]
struct FacetNestedStruct {
    title: String,
    items: Option<Vec<FacetRoundtripStruct>>,
    metadata: HashMap<String, Option<i32>>,
}

#[derive(Debug, Clone, PartialEq, legacy::BamlType)]
#[baml(internal_name = "contract::AsUnionColor")]
#[baml(as_union)]
enum BridgeAsUnionColor {
    Red,
    #[baml(alias = "green")]
    Green,
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(internal_name = "contract::AsUnionColor")]
#[baml(as_union)]
enum FacetAsUnionColor {
    Red,
    #[baml(alias = "green")]
    Green,
}

struct LegacyI64ObjectAdapter;

impl legacy::BamlAdapter<i64> for LegacyI64ObjectAdapter {
    fn type_ir() -> TypeIR {
        TypeIR::class("AdapterI64Wrapper")
    }

    fn register(reg: &mut legacy::Registry) {
        use legacy::internal_baml_jinja::types::{Class, Name};

        if !reg.mark_type("AdapterI64Wrapper") {
            return;
        }

        reg.register_class(Class {
            name: Name::new("AdapterI64Wrapper".to_string()),
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
        mut path: Vec<String>,
    ) -> Result<i64, legacy::BamlConvertError> {
        let map = match value {
            BamlValue::Class(_, fields) | BamlValue::Map(fields) => fields,
            other => {
                return Err(legacy::BamlConvertError::new(
                    path,
                    "object",
                    format!("{other:?}"),
                    "expected object adapter payload",
                ));
            }
        };

        let raw_value = map.get("value").ok_or_else(|| {
            legacy::BamlConvertError::new(
                path.clone(),
                "value",
                "<missing>",
                "missing required field",
            )
        })?;
        path.push("value".to_string());

        match raw_value {
            BamlValue::String(s) => s.parse::<i64>().map_err(|_| {
                legacy::BamlConvertError::new(path.clone(), "i64", s.clone(), "failed to parse i64")
            }),
            BamlValue::Int(i) => Ok(*i),
            other => Err(legacy::BamlConvertError::new(
                path,
                "i64",
                format!("{other:?}"),
                "expected value field to be string or int",
            )),
        }
    }
}

struct FacetI64ObjectAdapter;

impl facet_runtime::BamlAdapter<i64> for FacetI64ObjectAdapter {
    fn type_ir() -> TypeIR {
        TypeIR::class("AdapterI64Wrapper")
    }

    fn register(reg: &mut facet_runtime::Registry) {
        use facet_runtime::internal_baml_jinja::types::{Class, Name};

        if !reg.mark_type("AdapterI64Wrapper") {
            return;
        }

        reg.register_class(Class {
            name: Name::new("AdapterI64Wrapper".to_string()),
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
        mut path: Vec<String>,
    ) -> Result<i64, facet_runtime::BamlConvertError> {
        let map = match value {
            BamlValue::Class(_, fields) | BamlValue::Map(fields) => fields,
            other => {
                return Err(facet_runtime::BamlConvertError::new(
                    path,
                    "object",
                    format!("{other:?}"),
                    "expected object adapter payload",
                ));
            }
        };

        let raw_value = map.get("value").ok_or_else(|| {
            facet_runtime::BamlConvertError::new(
                path.clone(),
                "value",
                "<missing>",
                "missing required field",
            )
        })?;
        path.push("value".to_string());

        match raw_value {
            BamlValue::String(s) => s.parse::<i64>().map_err(|_| {
                facet_runtime::BamlConvertError::new(
                    path.clone(),
                    "i64",
                    s.clone(),
                    "failed to parse i64",
                )
            }),
            BamlValue::Int(i) => Ok(*i),
            other => Err(facet_runtime::BamlConvertError::new(
                path,
                "i64",
                format!("{other:?}"),
                "expected value field to be string or int",
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, legacy::BamlType)]
#[baml(internal_name = "contract::WithAdapter")]
struct BridgeWithAdapter {
    #[baml(with = "LegacyI64ObjectAdapter")]
    id: i64,
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(internal_name = "contract::WithAdapter")]
struct FacetWithAdapter {
    #[baml(with = "FacetI64ObjectAdapter")]
    id: i64,
}

mod legacy_collision_a {
    use super::legacy;

    #[derive(Debug, Clone, PartialEq, legacy::BamlType)]
    pub struct User {
        pub name: String,
    }
}

mod legacy_collision_b {
    use super::legacy;

    #[derive(Debug, Clone, PartialEq, legacy::BamlType)]
    pub struct User {
        pub name: String,
    }
}

mod facet_collision_a {
    #[derive(Debug, Clone, PartialEq)]
    #[bamltype::BamlType]
    pub struct User {
        pub name: String,
    }
}

mod facet_collision_b {
    #[derive(Debug, Clone, PartialEq)]
    #[bamltype::BamlType]
    pub struct User {
        pub name: String,
    }
}

fn sorted_union_string_literals(type_ir: TypeIR) -> Vec<String> {
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
    literals
}

#[test]
fn contract_render_schema_default_matches_legacy() {
    let old = legacy::render_schema::<BridgeUser>(legacy::RenderOptions::default())
        .expect("legacy render")
        .unwrap_or_default();
    let new = facet_runtime::render_schema::<FacetUser>(facet_runtime::RenderOptions::default())
        .expect("facet render")
        .unwrap_or_default();

    assert_eq!(old, new);
}

#[test]
fn contract_render_schema_hoisted_matches_legacy() {
    let opts = legacy::RenderOptions::hoist_classes(legacy::HoistClasses::All);
    let old = legacy::render_schema::<BridgeShape>(opts)
        .expect("legacy render")
        .unwrap_or_default();

    let opts = facet_runtime::RenderOptions::hoist_classes(facet_runtime::HoistClasses::All);
    let new = facet_runtime::render_schema::<FacetShape>(opts)
        .expect("facet render")
        .unwrap_or_default();

    assert_eq!(old, new);
}

#[test]
fn contract_render_schema_ordering_matches_legacy() {
    let opts = legacy::RenderOptions::hoist_classes(legacy::HoistClasses::All);
    let old = legacy::render_schema::<BridgeOrderRoot>(opts)
        .expect("legacy render")
        .unwrap_or_default();

    let opts = facet_runtime::RenderOptions::hoist_classes(facet_runtime::HoistClasses::All);
    let new = facet_runtime::render_schema::<FacetOrderRoot>(opts)
        .expect("facet render")
        .unwrap_or_default();

    assert_eq!(old, new);
}

#[test]
fn contract_parse_envelope_matches_legacy() {
    let raw = r#"{ "kind": "Circle", "radius": 2.5 }"#;
    let old = legacy::parse_llm_output::<BridgeShape>(raw, true).expect("legacy parse");
    let new = facet_runtime::parse_llm_output::<FacetShape>(raw, true).expect("facet parse");

    assert_eq!(old.baml_value, new.baml_value);
    assert_eq!(format!("{:?}", old.flags), format!("{:?}", new.flags));
    assert_eq!(format!("{:?}", old.checks), format!("{:?}", new.checks));
    assert_eq!(
        old.explanations
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>(),
        new.explanations
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
    );
}

#[test]
fn contract_parse_streaming_mode_matches_legacy() {
    let raw = r#"{ "name": "Ada", "age": 36 }"#;
    let old = legacy::parse_llm_output::<BridgeFuzzUser>(raw, false).expect("legacy parse");
    let new = facet_runtime::parse_llm_output::<FacetFuzzUser>(raw, false).expect("facet parse");

    assert_eq!(old.value.name, new.value.name);
    assert_eq!(old.value.age, new.value.age);
    assert_eq!(old.value.nickname, new.value.nickname);
    assert_eq!(old.baml_value, new.baml_value);
    assert_eq!(format!("{:?}", old.flags), format!("{:?}", new.flags));
    assert_eq!(format!("{:?}", old.checks), format!("{:?}", new.checks));
    assert_eq!(
        old.explanations
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>(),
        new.explanations
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
    );
}

#[test]
fn contract_int_repr_string_matches_legacy() {
    let raw = r#"{ "id": "18446744073709551615" }"#;
    let old = legacy::parse_llm_output::<BridgeBigIntString>(raw, true).expect("legacy parse");
    let new = facet_runtime::parse_llm_output::<FacetBigIntString>(raw, true).expect("facet parse");

    assert_eq!(old.value.id, new.value.id);
    assert_eq!(old.baml_value, new.baml_value);
}

#[test]
fn contract_map_key_pairs_parse_and_registration_match_legacy() {
    let raw = r#"{ "values": [ { "key": 1, "value": "a" }, { "key": 2, "value": "b" } ] }"#;
    let old = legacy::parse_llm_output::<BridgeMapPairs>(raw, true).expect("legacy parse");
    let new = facet_runtime::parse_llm_output::<FacetMapPairs>(raw, true).expect("facet parse");
    assert_eq!(old.baml_value, new.baml_value);

    let old_entry = format!(
        "{}::values__Entry",
        <BridgeMapPairs as legacy::BamlType>::baml_internal_name()
    );
    let old_class = <BridgeMapPairs as legacy::BamlType>::baml_output_format()
        .classes
        .get(&(old_entry, StreamingMode::NonStreaming))
        .expect("legacy entry class");

    let new_entry = format!(
        "{}::values__Entry",
        <FacetMapPairs as facet_runtime::BamlType>::baml_internal_name()
    );
    let new_class = <FacetMapPairs as facet_runtime::BamlType>::baml_output_format()
        .classes
        .get(&(new_entry, StreamingMode::NonStreaming))
        .expect("facet entry class");

    assert_eq!(
        old_class.name.rendered_name(),
        new_class.name.rendered_name()
    );
}

#[test]
fn contract_check_results_match_legacy() {
    let raw = r#"{ "value": 3 }"#;
    let old = legacy::parse_llm_output::<BridgeChecked>(raw, true).expect("legacy parse");
    let new = facet_runtime::parse_llm_output::<FacetChecked>(raw, true).expect("facet parse");

    assert_eq!(format!("{:?}", old.checks), format!("{:?}", new.checks));
}

#[test]
fn contract_assert_error_shape_matches_legacy() {
    let raw = r#"{ "value": -1 }"#;
    let old = legacy::parse_llm_output::<BridgeAsserted>(raw, true).expect_err("legacy err");
    let new = facet_runtime::parse_llm_output::<FacetAsserted>(raw, true).expect_err("facet err");

    let old_failed = match old {
        legacy::BamlParseError::ConstraintAssertsFailed { failed } => failed,
        other => panic!("legacy expected assert failure, got {other:?}"),
    };
    let new_failed = match new {
        facet_runtime::BamlParseError::ConstraintAssertsFailed { failed } => failed,
        other => panic!("facet expected assert failure, got {other:?}"),
    };

    assert_eq!(format!("{:?}", old_failed), format!("{:?}", new_failed));
}

#[test]
fn contract_markdown_flags_and_explanations_match_legacy() {
    let raw = "```json\n{ \"name\": \"Ada\", \"age\": 36 }\n```";
    let old = legacy::parse_llm_output::<BridgeFuzzUser>(raw, true).expect("legacy parse");
    let new = facet_runtime::parse_llm_output::<FacetFuzzUser>(raw, true).expect("facet parse");
    assert_eq!(format!("{:?}", old.flags), format!("{:?}", new.flags));

    let raw = r#"{ "values": { "ok": 1, "bad": "oops" } }"#;
    let old = legacy::parse_llm_output::<BridgeFuzzExplain>(raw, true).expect("legacy parse");
    let new = facet_runtime::parse_llm_output::<FacetFuzzExplain>(raw, true).expect("facet parse");
    assert_eq!(
        old.explanations
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>(),
        new.explanations
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
    );
}

#[test]
fn contract_unit_enum_alias_parse_matches_legacy() {
    let raw = r#""green""#;
    let old = legacy::parse_llm_output::<BridgeColorAlias>(raw, true).expect("legacy parse");
    let new = facet_runtime::parse_llm_output::<FacetColorAlias>(raw, true).expect("facet parse");

    assert_eq!(old.baml_value, new.baml_value);
    assert!(matches!(old.value, BridgeColorAlias::Green));
    assert!(matches!(new.value, FacetColorAlias::Green));
}

#[test]
fn contract_as_union_type_ir_and_parse_match_legacy() {
    let old_literals =
        sorted_union_string_literals(<BridgeAsUnionColor as legacy::BamlType>::baml_type_ir());
    let new_literals = sorted_union_string_literals(
        <FacetAsUnionColor as facet_runtime::BamlType>::baml_type_ir(),
    );
    assert_eq!(old_literals, new_literals);

    let raw = r#""green""#;
    let old = legacy::parse_llm_output::<BridgeAsUnionColor>(raw, true).expect("legacy parse");
    let new = facet_runtime::parse_llm_output::<FacetAsUnionColor>(raw, true).expect("facet parse");

    assert_eq!(old.baml_value, new.baml_value);
    assert!(matches!(old.value, BridgeAsUnionColor::Green));
    assert!(matches!(new.value, FacetAsUnionColor::Green));
}

#[test]
fn contract_data_enum_docs_render_match_legacy() {
    let old = legacy::render_schema::<BridgeDocShape>(legacy::RenderOptions::default())
        .expect("legacy render")
        .unwrap_or_default();
    let new =
        facet_runtime::render_schema::<FacetDocShape>(facet_runtime::RenderOptions::default())
            .expect("facet render")
            .unwrap_or_default();

    assert_eq!(old, new);
}

#[test]
fn contract_int_repr_option_matches_legacy() {
    let raw = r#"{ "id": 42 }"#;
    let old = legacy::parse_llm_output::<BridgeBigIntOption>(raw, true).expect("legacy parse");
    let new = facet_runtime::parse_llm_output::<FacetBigIntOption>(raw, true).expect("facet parse");
    assert_eq!(old.baml_value, new.baml_value);
    assert_eq!(old.value.id, new.value.id);

    let raw = r#"{ "id": null }"#;
    let old = legacy::parse_llm_output::<BridgeBigIntOption>(raw, true).expect("legacy parse");
    let new = facet_runtime::parse_llm_output::<FacetBigIntOption>(raw, true).expect("facet parse");
    assert_eq!(old.baml_value, new.baml_value);
    assert_eq!(old.value.id, new.value.id);
}

#[test]
fn contract_map_key_repr_string_and_option_match_legacy() {
    let raw = r#"{ "values": { "1": "a", "2": "b" } }"#;
    let old = legacy::parse_llm_output::<BridgeMapKeys>(raw, true).expect("legacy parse");
    let new = facet_runtime::parse_llm_output::<FacetMapKeys>(raw, true).expect("facet parse");
    assert_eq!(old.baml_value, new.baml_value);
    assert_eq!(old.value.values, new.value.values);

    let raw = r#"{ "values": { "10": "x" } }"#;
    let old = legacy::parse_llm_output::<BridgeMapKeysOption>(raw, true).expect("legacy parse");
    let new =
        facet_runtime::parse_llm_output::<FacetMapKeysOption>(raw, true).expect("facet parse");
    assert_eq!(old.baml_value, new.baml_value);
    assert_eq!(old.value.values, new.value.values);
}

#[test]
fn contract_rename_all_matches_legacy() {
    let raw = r#"{ "fullName": "Ada" }"#;
    let old = legacy::parse_llm_output::<BridgeRenameAllUser>(raw, true).expect("legacy parse");
    let new =
        facet_runtime::parse_llm_output::<FacetRenameAllUser>(raw, true).expect("facet parse");

    assert_eq!(old.baml_value, new.baml_value);
    assert_eq!(old.value.full_name, new.value.full_name);
}

#[test]
fn contract_recursion_class_detection_matches_legacy() {
    let old_of = <BridgeNode as legacy::BamlType>::baml_output_format();
    let new_of = <FacetNode as facet_runtime::BamlType>::baml_output_format();

    assert_eq!(old_of.recursive_classes, new_of.recursive_classes);
    assert!(
        old_of
            .recursive_classes
            .contains(<BridgeNode as legacy::BamlType>::baml_internal_name())
    );
    assert!(
        new_of
            .recursive_classes
            .contains(<FacetNode as facet_runtime::BamlType>::baml_internal_name())
    );
}

#[test]
fn contract_internal_name_collision_behavior_matches_legacy() {
    let old_a = <legacy_collision_a::User as legacy::BamlType>::baml_internal_name();
    let old_b = <legacy_collision_b::User as legacy::BamlType>::baml_internal_name();
    let new_a = <facet_collision_a::User as facet_runtime::BamlType>::baml_internal_name();
    let new_b = <facet_collision_b::User as facet_runtime::BamlType>::baml_internal_name();

    assert_ne!(old_a, old_b);
    assert_ne!(new_a, new_b);

    let old_a_class = <legacy_collision_a::User as legacy::BamlType>::baml_output_format()
        .classes
        .get(&(old_a.to_string(), StreamingMode::NonStreaming))
        .expect("legacy class a missing");
    let old_b_class = <legacy_collision_b::User as legacy::BamlType>::baml_output_format()
        .classes
        .get(&(old_b.to_string(), StreamingMode::NonStreaming))
        .expect("legacy class b missing");
    let new_a_class = <facet_collision_a::User as facet_runtime::BamlType>::baml_output_format()
        .classes
        .get(&(new_a.to_string(), StreamingMode::NonStreaming))
        .expect("facet class a missing");
    let new_b_class = <facet_collision_b::User as facet_runtime::BamlType>::baml_output_format()
        .classes
        .get(&(new_b.to_string(), StreamingMode::NonStreaming))
        .expect("facet class b missing");

    assert_eq!(old_a_class.name.rendered_name(), "User");
    assert_eq!(old_b_class.name.rendered_name(), "User");
    assert_eq!(new_a_class.name.rendered_name(), "User");
    assert_eq!(new_b_class.name.rendered_name(), "User");
}

#[test]
fn contract_with_adapter_schema_and_parse_match_legacy() {
    let old_of = <BridgeWithAdapter as legacy::BamlType>::baml_output_format();
    let new_of = <FacetWithAdapter as facet_runtime::BamlType>::baml_output_format();

    let old_adapter = old_of
        .classes
        .get(&("AdapterI64Wrapper".to_string(), StreamingMode::NonStreaming))
        .expect("legacy adapter class missing");
    let new_adapter = new_of
        .classes
        .get(&("AdapterI64Wrapper".to_string(), StreamingMode::NonStreaming))
        .expect("facet adapter class missing");
    assert_eq!(old_adapter.name.real_name(), new_adapter.name.real_name());
    assert_eq!(
        old_adapter
            .fields
            .iter()
            .map(|(name, _, _, _)| name.real_name())
            .collect::<Vec<_>>(),
        new_adapter
            .fields
            .iter()
            .map(|(name, _, _, _)| name.real_name())
            .collect::<Vec<_>>()
    );

    let old_owner_name = <BridgeWithAdapter as legacy::BamlType>::baml_internal_name().to_string();
    let new_owner_name =
        <FacetWithAdapter as facet_runtime::BamlType>::baml_internal_name().to_string();
    let old_owner = old_of
        .classes
        .get(&(old_owner_name, StreamingMode::NonStreaming))
        .expect("legacy owner class missing");
    let new_owner = new_of
        .classes
        .get(&(new_owner_name, StreamingMode::NonStreaming))
        .expect("facet owner class missing");
    let (_, old_field_ir, _, _) = old_owner
        .fields
        .iter()
        .find(|(name, _, _, _)| name.real_name() == "id")
        .expect("legacy id field missing");
    let (_, new_field_ir, _, _) = new_owner
        .fields
        .iter()
        .find(|(name, _, _, _)| name.real_name() == "id")
        .expect("facet id field missing");
    assert_eq!(format!("{old_field_ir:?}"), format!("{new_field_ir:?}"));

    let raw = r#"{ "id": { "value": "9223372036854775807" } }"#;
    let old = legacy::parse_llm_output::<BridgeWithAdapter>(raw, true).expect("legacy parse");
    let new = facet_runtime::parse_llm_output::<FacetWithAdapter>(raw, true).expect("facet parse");
    assert_eq!(old.value.id, new.value.id);
    assert_eq!(old.baml_value, new.baml_value);
}

fn assert_cross_runtime_roundtrip<Old, New>(old_value: Old, new_value: New)
where
    Old: Clone + std::fmt::Debug + PartialEq + legacy::ToBamlValue + legacy::BamlValueConvert,
    New: Clone
        + std::fmt::Debug
        + PartialEq
        + facet_runtime::compat::ToBamlValue
        + facet_runtime::compat::BamlValueConvert,
{
    let old_baml = legacy::ToBamlValue::to_baml_value(&old_value);
    let new_baml = facet_runtime::compat::ToBamlValue::to_baml_value(&new_value);
    assert_eq!(old_baml, new_baml);

    let old_back = <Old as legacy::BamlValueConvert>::try_from_baml_value(old_baml, Vec::new())
        .expect("legacy roundtrip");
    let new_back =
        <New as facet_runtime::compat::BamlValueConvert>::try_from_baml_value(new_baml, Vec::new())
            .expect("facet roundtrip");
    assert_eq!(old_back, old_value);
    assert_eq!(new_back, new_value);
}

fn assert_default_adapter_parity<T>(value: T)
where
    T: Clone
        + std::fmt::Debug
        + PartialEq
        + legacy::ToBamlValue
        + legacy::BamlValueConvert
        + facet_runtime::compat::ToBamlValue
        + facet_runtime::compat::BamlValueConvert,
{
    let old_baml = legacy::ToBamlValue::to_baml_value(&value);
    let new_baml = facet_runtime::compat::ToBamlValue::to_baml_value(&value);
    assert_eq!(old_baml, new_baml);

    let old_back = <T as legacy::BamlValueConvert>::try_from_baml_value(old_baml, Vec::new())
        .expect("legacy convert");
    let new_back =
        <T as facet_runtime::compat::BamlValueConvert>::try_from_baml_value(new_baml, Vec::new())
            .expect("facet convert");
    assert_eq!(old_back, new_back);
    assert_eq!(old_back, value);
}

#[test]
fn contract_default_adapter_complex_roundtrips_match_legacy() {
    let old_struct = BridgeRoundtripStruct {
        name: "example".to_string(),
        count: 7,
        tags: vec!["tag".to_string()],
        meta: Some("meta".to_string()),
    };
    let new_struct = FacetRoundtripStruct {
        name: "example".to_string(),
        count: 7,
        tags: vec!["tag".to_string()],
        meta: Some("meta".to_string()),
    };
    assert_cross_runtime_roundtrip(old_struct, new_struct);

    assert_cross_runtime_roundtrip(BridgeRoundtripUnitEnum::Beta, FacetRoundtripUnitEnum::Beta);

    assert_cross_runtime_roundtrip(
        BridgeRoundtripDataEnum::Message {
            body: "hello".to_string(),
            count: 3,
        },
        FacetRoundtripDataEnum::Message {
            body: "hello".to_string(),
            count: 3,
        },
    );

    let mut old_metadata = HashMap::new();
    old_metadata.insert("alpha".to_string(), Some(1));
    old_metadata.insert("beta".to_string(), None);

    let mut new_metadata = HashMap::new();
    new_metadata.insert("alpha".to_string(), Some(1));
    new_metadata.insert("beta".to_string(), None);

    let old_nested = BridgeNestedStruct {
        title: "nested".to_string(),
        items: Some(vec![BridgeRoundtripStruct {
            name: "child".to_string(),
            count: 2,
            tags: vec!["x".to_string(), "y".to_string()],
            meta: None,
        }]),
        metadata: old_metadata,
    };
    let new_nested = FacetNestedStruct {
        title: "nested".to_string(),
        items: Some(vec![FacetRoundtripStruct {
            name: "child".to_string(),
            count: 2,
            tags: vec!["x".to_string(), "y".to_string()],
            meta: None,
        }]),
        metadata: new_metadata,
    };
    assert_cross_runtime_roundtrip(old_nested, new_nested);
}

#[test]
fn contract_default_adapter_roundtrips_match_legacy() {
    assert_default_adapter_parity("hello".to_string());
    assert_default_adapter_parity(true);
    assert_default_adapter_parity(123i32);
    assert_default_adapter_parity(-99i64);
    assert_default_adapter_parity(3.5f32);
    assert_default_adapter_parity(9.75f64);

    assert_default_adapter_parity(Some(42i32));
    assert_default_adapter_parity(None::<i32>);
    assert_default_adapter_parity(vec!["a".to_string(), "b".to_string()]);
    assert_default_adapter_parity(Box::new("boxed".to_string()));
    assert_default_adapter_parity(Arc::new("arc".to_string()));
    assert_default_adapter_parity(Rc::new("rc".to_string()));

    let mut hm = HashMap::new();
    hm.insert("answer".to_string(), 42i32);
    assert_default_adapter_parity(hm);

    let mut bt = BTreeMap::new();
    bt.insert("left".to_string(), 1i64);
    bt.insert("right".to_string(), 2i64);
    assert_default_adapter_parity(bt);
}

#[test]
fn contract_default_adapter_error_messages_match_legacy() {
    let invalid = BamlValue::Class(
        "contract::Unsigned32".to_string(),
        [("value".to_string(), BamlValue::Int(-1))]
            .into_iter()
            .collect(),
    );

    let old_err = <BridgeUnsigned32 as legacy::BamlValueConvert>::try_from_baml_value(
        invalid.clone(),
        Vec::new(),
    )
    .expect_err("legacy should fail");
    let new_err =
        <FacetUnsigned32 as facet_runtime::compat::BamlValueConvert>::try_from_baml_value(
            invalid,
            Vec::new(),
        )
        .expect_err("facet should fail");

    assert_eq!(old_err.expected, new_err.expected);
    assert_eq!(old_err.path, new_err.path);
    assert_eq!(old_err.to_string(), new_err.to_string());
}

#[test]
fn contract_direct_integer_string_conversion_matches_legacy() {
    let old_err = <i64 as legacy::BamlValueConvert>::try_from_baml_value(
        BamlValue::String("123".into()),
        Vec::new(),
    )
    .expect_err("legacy should reject string->int direct conversion");
    let new_err = <i64 as facet_runtime::compat::BamlValueConvert>::try_from_baml_value(
        BamlValue::String("123".into()),
        Vec::new(),
    )
    .expect_err("facet should reject string->int direct conversion");

    assert_eq!(old_err.expected, new_err.expected);
}

#[test]
fn contract_direct_map_pairs_conversion_matches_legacy() {
    let pair_entry = BamlValue::Map(
        [
            ("key".to_string(), BamlValue::String("k".to_string())),
            ("value".to_string(), BamlValue::Int(1)),
        ]
        .into_iter()
        .collect(),
    );
    let raw = BamlValue::List(vec![pair_entry]);

    let old_err = <HashMap<String, i64> as legacy::BamlValueConvert>::try_from_baml_value(
        raw.clone(),
        Vec::new(),
    )
    .expect_err("legacy should reject map pair-list direct conversion");
    let new_err =
        <HashMap<String, i64> as facet_runtime::compat::BamlValueConvert>::try_from_baml_value(
            raw,
            Vec::new(),
        )
        .expect_err("facet should reject map pair-list direct conversion");

    assert_eq!(old_err.expected, new_err.expected);
}

#[test]
fn contract_schema_fingerprint_matches_legacy() {
    let old_of = <BridgeUser as legacy::BamlType>::baml_output_format();
    let new_of = <FacetUser as facet_runtime::BamlType>::baml_output_format();

    let old_fp = legacy::schema_fingerprint(old_of, legacy::RenderOptions::default())
        .expect("legacy fingerprint");
    let new_fp = facet_runtime::schema_fingerprint(new_of, facet_runtime::RenderOptions::default())
        .expect("facet fingerprint");

    assert_eq!(old_fp, new_fp);
}
