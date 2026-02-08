use std::collections::HashMap;

use bamltype as facet_runtime;
use bamltype::baml_types::{BamlValue, StreamingMode, TypeIR};
use expect_test::expect;
use proptest::collection::{hash_map, vec};
use proptest::prelude::*;
use proptest::string::string_regex;
use proptest::test_runner::RngSeed;
use serde_json::json;

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[bamltype::BamlType]
#[baml(internal_name = "contract::property::User")]
struct RoundTripUser {
    name: String,
    age: u32,
    active: bool,
    tags: Vec<String>,
    meta: HashMap<String, i64>,
    nickname: Option<String>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
struct RecursiveSeed {
    value: i64,
    note: Option<String>,
    next: Option<Box<RecursiveSeed>>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[bamltype::BamlType]
#[baml(internal_name = "contract::property::RecursiveNode")]
struct RecursiveNode {
    value: i64,
    note: Option<String>,
    next: Option<Box<RecursiveNode>>,
}

struct PropertyI64Adapter;

impl facet_runtime::adapters::FieldCodec<i64> for PropertyI64Adapter {
    fn type_ir() -> TypeIR {
        TypeIR::class("PropertyAdapterI64Wrapper")
    }

    fn register(reg: &mut facet_runtime::adapters::AdapterSchemaRegistry) {
        use facet_runtime::internal_baml_jinja::types::{Class, Name};

        if !reg.mark_type("PropertyAdapterI64Wrapper") {
            return;
        }

        reg.register_class(Class {
            name: Name::new("PropertyAdapterI64Wrapper".to_string()),
            description: None,
            namespace: facet_runtime::baml_types::StreamingMode::NonStreaming,
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

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[bamltype::BamlType]
#[baml(internal_name = "contract::property::AdapterHeavy")]
struct AdapterHeavy {
    #[baml(with = "PropertyI64Adapter")]
    primary: i64,
    #[baml(with = "PropertyI64Adapter")]
    secondary: i64,
    #[baml(with = "PropertyI64Adapter")]
    tertiary: i64,
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(internal_name = "contract::property::FuzzUser")]
struct FuzzUser {
    name: String,
    age: u32,
    nickname: Option<String>,
}

fn arb_string() -> impl Strategy<Value = String> {
    string_regex("[a-zA-Z0-9 _-]{0,12}").expect("valid regex")
}

fn arb_user_fields() -> impl Strategy<
    Value = (
        String,
        u32,
        bool,
        Vec<String>,
        HashMap<String, i64>,
        Option<String>,
    ),
> {
    (
        arb_string(),
        0_u32..1000_u32,
        any::<bool>(),
        vec(arb_string(), 0..5),
        hash_map(arb_string(), -1000_i64..1000_i64, 0..5),
        proptest::option::of(arb_string()),
    )
}

fn arb_recursive_seed() -> impl Strategy<Value = RecursiveSeed> {
    let leaf =
        (-1000_i64..1000_i64, proptest::option::of(arb_string())).prop_map(|(value, note)| {
            RecursiveSeed {
                value,
                note,
                next: None,
            }
        });

    leaf.prop_recursive(6, 128, 2, |inner| {
        (
            -1000_i64..1000_i64,
            proptest::option::of(arb_string()),
            proptest::option::of(inner),
        )
            .prop_map(|(value, note, next)| RecursiveSeed {
                value,
                note,
                next: next.map(Box::new),
            })
    })
}

fn seed_to_node(seed: &RecursiveSeed) -> RecursiveNode {
    RecursiveNode {
        value: seed.value,
        note: seed.note.clone(),
        next: seed.next.as_ref().map(|next| Box::new(seed_to_node(next))),
    }
}

fn canonical_baml(value: &BamlValue) -> serde_json::Value {
    match value {
        BamlValue::String(v) => json!({ "kind": "string", "value": v }),
        BamlValue::Int(v) => json!({ "kind": "int", "value": v }),
        BamlValue::Float(v) => json!({ "kind": "float", "value": v }),
        BamlValue::Bool(v) => json!({ "kind": "bool", "value": v }),
        BamlValue::Null => json!({ "kind": "null" }),
        BamlValue::Enum(name, variant) => {
            json!({ "kind": "enum", "name": name, "variant": variant })
        }
        BamlValue::List(items) => {
            let canonical_items = items.iter().map(canonical_baml).collect::<Vec<_>>();
            json!({ "kind": "list", "value": canonical_items })
        }
        BamlValue::Map(map) => {
            let mut entries = map
                .iter()
                .map(|(k, v)| (k.clone(), canonical_baml(v)))
                .collect::<Vec<_>>();
            entries.sort_by(|(left, _), (right, _)| left.cmp(right));

            let mut object = serde_json::Map::new();
            for (key, value) in entries {
                object.insert(key, value);
            }
            json!({ "kind": "map", "value": object })
        }
        BamlValue::Class(name, fields) => {
            let mut entries = fields
                .iter()
                .map(|(k, v)| (k.clone(), canonical_baml(v)))
                .collect::<Vec<_>>();
            entries.sort_by(|(left, _), (right, _)| left.cmp(right));

            let mut object = serde_json::Map::new();
            for (key, value) in entries {
                object.insert(key, value);
            }

            json!({ "kind": "class", "name": name, "fields": object })
        }
        BamlValue::Media(media) => json!({ "kind": "media", "value": format!("{media:?}") }),
    }
}

fn parsed_fixture<T>(parsed: &facet_runtime::Parsed<T>) -> String {
    let fixture = json!({
        "baml_value": canonical_baml(&parsed.baml_value),
        "flags": format!("{:?}", parsed.flags),
        "checks": format!("{:?}", parsed.checks),
        "explanations": parsed
            .explanations
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>(),
    });

    serde_json::to_string_pretty(&fixture).expect("serialize parsed fixture")
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 96,
        rng_seed: RngSeed::Fixed(0x5eed_f00d),
        .. ProptestConfig::default()
    })]

    #[test]
    fn property_round_trip_user((name, age, active, tags, meta, nickname) in arb_user_fields()) {
        let expected = RoundTripUser {
            name,
            age,
            active,
            tags,
            meta,
            nickname,
        };

        let json = serde_json::to_string(&expected).expect("serialize");
        let parsed = facet_runtime::parse_llm_output::<RoundTripUser>(&json, true).expect("facet parse");

        prop_assert_eq!(parsed.value, expected);
    }

    #[test]
    fn property_recursive_parse(seed in arb_recursive_seed()) {
        let expected = seed_to_node(&seed);
        let json = serde_json::to_string(&seed).expect("serialize");
        let parsed = facet_runtime::parse_llm_output::<RecursiveNode>(&json, true).expect("facet parse");

        prop_assert_eq!(parsed.value, expected);
    }

    #[test]
    fn property_adapter_heavy_parse(
        primary in -1_000_000_i64..1_000_000_i64,
        secondary in -1_000_000_i64..1_000_000_i64,
        tertiary in -1_000_000_i64..1_000_000_i64,
        primary_as_int in any::<bool>(),
        secondary_as_int in any::<bool>(),
        tertiary_as_int in any::<bool>(),
    ) {
        let expected = AdapterHeavy {
            primary,
            secondary,
            tertiary,
        };

        let payload = serde_json::json!({
            "primary": { "value": if primary_as_int { serde_json::json!(primary) } else { serde_json::json!(primary.to_string()) } },
            "secondary": { "value": if secondary_as_int { serde_json::json!(secondary) } else { serde_json::json!(secondary.to_string()) } },
            "tertiary": { "value": if tertiary_as_int { serde_json::json!(tertiary) } else { serde_json::json!(tertiary.to_string()) } },
        })
        .to_string();

        let parsed = facet_runtime::parse_llm_output::<AdapterHeavy>(&payload, true).expect("facet parse");

        prop_assert_eq!(parsed.value, expected);
    }
}

#[test]
fn trailing_comma_parses_fixture() {
    let raw = r#"{ "name": "Ada", "age": 36, }"#;
    let parsed = facet_runtime::parse_llm_output::<FuzzUser>(raw, true).expect("facet parse");

    assert_eq!(parsed.value.age, 36);
    expect![[r#"
        {
          "baml_value": {
            "kind": "class",
            "name": "contract::property::FuzzUser",
            "fields": {
              "age": {
                "kind": "int",
                "value": 36
              },
              "name": {
                "kind": "string",
                "value": "Ada"
              },
              "nickname": {
                "kind": "null"
              }
            }
          },
          "flags": "[ObjectFromFixedJson([]), FirstMatch(0, []), ObjectFromFixedJson([GreppedForJSON]), FirstMatch(0, []), OptionalDefaultFromNoValue, Pending]",
          "checks": "[]",
          "explanations": []
        }"#]].assert_eq(&parsed_fixture(&parsed));
}

#[test]
fn extra_keys_are_ignored_fixture() {
    let raw = r#"{ "name": "Ada", "age": 36, "extra": "ignored" }"#;
    let parsed = facet_runtime::parse_llm_output::<FuzzUser>(raw, true).expect("facet parse");

    assert_eq!(parsed.value.name, "Ada");
    assert_eq!(parsed.value.nickname, None);
    expect![[r#"
        {
          "baml_value": {
            "kind": "class",
            "name": "contract::property::FuzzUser",
            "fields": {
              "age": {
                "kind": "int",
                "value": 36
              },
              "name": {
                "kind": "string",
                "value": "Ada"
              },
              "nickname": {
                "kind": "null"
              }
            }
          },
          "flags": "[ExtraKey(\"extra\", String(\"ignored\", Complete)), FirstMatch(0, []), OptionalDefaultFromNoValue, Pending]",
          "checks": "[]",
          "explanations": []
        }"#]].assert_eq(&parsed_fixture(&parsed));
}

#[test]
fn adapter_heavy_schema_registration_fixture() {
    let output_format = <AdapterHeavy as facet_runtime::BamlType>::baml_output_format();

    let adapter = output_format
        .classes
        .get(&(
            "PropertyAdapterI64Wrapper".to_string(),
            StreamingMode::NonStreaming,
        ))
        .expect("adapter class missing");
    let adapter_fields = adapter
        .fields
        .iter()
        .map(|(name, type_ir, _, _)| format!("{}: {type_ir:?}", name.real_name()))
        .collect::<Vec<_>>();

    let owner_name = <AdapterHeavy as facet_runtime::BamlType>::baml_internal_name().to_string();
    let owner = output_format
        .classes
        .get(&(owner_name, StreamingMode::NonStreaming))
        .expect("owner class missing");
    let owner_fields = owner
        .fields
        .iter()
        .map(|(name, type_ir, _, _)| format!("{}: {type_ir:?}", name.real_name()))
        .collect::<Vec<_>>();

    let snapshot = serde_json::to_string_pretty(&json!({
        "adapter_name": adapter.name.real_name(),
        "adapter_fields": adapter_fields,
        "owner_fields": owner_fields,
    }))
    .expect("serialize adapter schema fixture");

    expect![[r#"
        {
          "adapter_name": "PropertyAdapterI64Wrapper",
          "adapter_fields": [
            "value: Primitive(String, TypeMeta { constraints: [], streaming_behavior: StreamingBehavior { needed: false, done: false, state: false } })"
          ],
          "owner_fields": [
            "primary: Class { name: \"PropertyAdapterI64Wrapper\", mode: NonStreaming, dynamic: false, meta: TypeMeta { constraints: [], streaming_behavior: StreamingBehavior { needed: false, done: false, state: false } } }",
            "secondary: Class { name: \"PropertyAdapterI64Wrapper\", mode: NonStreaming, dynamic: false, meta: TypeMeta { constraints: [], streaming_behavior: StreamingBehavior { needed: false, done: false, state: false } } }",
            "tertiary: Class { name: \"PropertyAdapterI64Wrapper\", mode: NonStreaming, dynamic: false, meta: TypeMeta { constraints: [], streaming_behavior: StreamingBehavior { needed: false, done: false, state: false } } }"
          ]
        }"#]].assert_eq(&snapshot);
}

#[test]
fn adapter_heavy_error_shape_fixture() {
    let raw = serde_json::json!({
        "primary": { "value": "11" },
        "secondary": { "value": "oops" },
        "tertiary": { "value": "22" }
    })
    .to_string();

    let err = facet_runtime::parse_llm_output::<AdapterHeavy>(&raw, true).expect_err("facet err");
    let facet_runtime::BamlParseError::Convert(convert) = err else {
        panic!("expected conversion error for adapter parse failure");
    };

    let snapshot = serde_json::to_string_pretty(&json!({
        "expected": convert.expected,
        "path": convert.path,
        "display": convert.to_string(),
    }))
    .expect("serialize adapter error fixture");

    expect![[r#"
        {
          "expected": "i64",
          "path": [
            "secondary",
            "value"
          ],
          "display": "failed to parse i64 (expected i64, got oops) at secondary.value"
        }"#]]
    .assert_eq(&snapshot);
}
