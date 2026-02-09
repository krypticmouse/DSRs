use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;
use std::sync::Arc;

use bamltype as facet_runtime;
use bamltype::baml_types::{BamlValue, LiteralValue, StreamingMode, TypeIR};
use expect_test::expect;
use serde_json::json;

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(name = "ContractUser")]
#[baml(internal_name = "contract::User")]
struct FacetUser {
    #[baml(alias = "fullName")]
    name: String,
    age: i64,
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

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(internal_name = "contract::Checked")]
struct FacetChecked {
    #[baml(check(label = "positive", expr = "this > 0"))]
    value: i64,
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(internal_name = "contract::Asserted")]
struct FacetAsserted {
    #[baml(assert(label = "positive", expr = "this > 0"))]
    value: i64,
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(internal_name = "contract::MapPairs")]
struct FacetMapPairs {
    #[baml(map_key_repr = "pairs")]
    values: HashMap<u32, String>,
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(name = "OrderB")]
#[baml(internal_name = "contract::OrderB")]
struct FacetOrderB {
    value: i64,
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(name = "OrderA")]
#[baml(internal_name = "contract::OrderA")]
struct FacetOrderA {
    value: i64,
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(name = "OrderRoot")]
#[baml(internal_name = "contract::OrderRoot")]
struct FacetOrderRoot {
    b: FacetOrderB,
    a: FacetOrderA,
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(internal_name = "contract::BigIntString")]
struct FacetBigIntString {
    #[baml(int_repr = "string")]
    id: u64,
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(internal_name = "contract::Unsigned32")]
struct FacetUnsigned32 {
    value: u32,
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(internal_name = "contract::FuzzUser")]
struct FacetFuzzUser {
    name: String,
    age: u32,
    nickname: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(internal_name = "contract::FuzzExplain")]
struct FacetFuzzExplain {
    values: HashMap<String, i64>,
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(internal_name = "contract::ColorAlias")]
enum FacetColorAlias {
    Red,
    #[baml(alias = "green")]
    Green,
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

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(internal_name = "contract::BigIntOption")]
struct FacetBigIntOption {
    #[baml(int_repr = "i64")]
    id: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(internal_name = "contract::MapKeys")]
struct FacetMapKeys {
    #[baml(map_key_repr = "string")]
    values: HashMap<u32, String>,
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(internal_name = "contract::MapKeysOption")]
struct FacetMapKeysOption {
    #[baml(map_key_repr = "string")]
    values: Option<HashMap<u32, String>>,
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(internal_name = "contract::Node")]
struct FacetNode {
    value: i64,
    next: Option<Box<FacetNode>>,
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(rename_all = "camelCase")]
#[baml(internal_name = "contract::RenameAllUser")]
struct FacetRenameAllUser {
    full_name: String,
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

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(internal_name = "contract::RoundtripUnitEnum")]
enum FacetRoundtripUnitEnum {
    Alpha,
    #[baml(alias = "beta")]
    Beta,
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(internal_name = "contract::RoundtripDataEnum")]
#[baml(tag = "kind")]
enum FacetRoundtripDataEnum {
    Message { body: String, count: i64 },
    Empty,
}

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(internal_name = "contract::NestedStruct")]
struct FacetNestedStruct {
    title: String,
    items: Option<Vec<FacetRoundtripStruct>>,
    metadata: HashMap<String, Option<i32>>,
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

struct FacetI64ObjectAdapter;

impl facet_runtime::adapters::FieldCodec<i64> for FacetI64ObjectAdapter {
    fn type_ir() -> TypeIR {
        TypeIR::class("AdapterI64Wrapper")
    }

    fn register(ctx: facet_runtime::adapters::FieldCodecRegisterContext<'_>) {
        use facet_runtime::internal_baml_jinja::types::{Class, Name};
        let reg = ctx.registry;

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

#[derive(Debug, Clone, PartialEq)]
#[bamltype::BamlType]
#[baml(internal_name = "contract::WithAdapter")]
struct FacetWithAdapter {
    #[baml(with = "FacetI64ObjectAdapter")]
    id: i64,
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

#[test]
fn contract_render_schema_default_fixture() {
    let rendered =
        facet_runtime::render_schema::<FacetUser>(facet_runtime::RenderOptions::default())
            .expect("render")
            .unwrap_or_default();
    expect![[r#"
        Answer in JSON using this schema:
        {
          fullName: string,
          age: int,
        }"#]]
    .assert_eq(&rendered);
}

#[test]
fn contract_render_schema_hoisted_fixture() {
    let rendered = facet_runtime::render_schema::<FacetShape>(
        facet_runtime::RenderOptions::hoist_classes(facet_runtime::HoistClasses::All),
    )
    .expect("render")
    .unwrap_or_default();
    expect![[r#"
        ContractShape_Circle {
          kind: "Circle",
          radius: float,
        }

        ContractShape_Square {
          kind: "Square",
          side: float,
        }

        Answer in JSON using any of these schemas:
        ContractShape_Circle or ContractShape_Square"#]]
    .assert_eq(&rendered);
}

#[test]
fn contract_render_schema_ordering_fixture() {
    let rendered = facet_runtime::render_schema::<FacetOrderRoot>(
        facet_runtime::RenderOptions::hoist_classes(facet_runtime::HoistClasses::All),
    )
    .expect("render")
    .unwrap_or_default();
    expect![[r#"
        OrderA {
          value: int,
        }

        OrderB {
          value: int,
        }

        OrderRoot {
          b: OrderB,
          a: OrderA,
        }

        Answer in JSON using this schema: OrderRoot"#]]
    .assert_eq(&rendered);
}

#[test]
fn contract_parse_envelope_fixture() {
    let raw = r#"{ "kind": "Circle", "radius": 2.5 }"#;
    let parsed = facet_runtime::parse_llm_output::<FacetShape>(raw, true).expect("parse");
    expect![[r#"
        {
          "baml_value": {
            "kind": "class",
            "name": "contract::Shape__Circle",
            "fields": {
              "kind": {
                "kind": "string",
                "value": "Circle"
              },
              "radius": {
                "kind": "float",
                "value": 2.5
              }
            }
          },
          "flags": "[UnionMatch(0, []), FirstMatch(0, [])]",
          "checks": "[]",
          "explanations": []
        }"#]]
    .assert_eq(&parsed_fixture(&parsed));
}

#[test]
fn contract_parse_streaming_mode_fixture() {
    let raw = r#"{ "name": "Ada", "age": 36 }"#;
    let parsed = facet_runtime::parse_llm_output::<FacetFuzzUser>(raw, false).expect("parse");
    expect![[r#"
        {
          "baml_value": {
            "kind": "class",
            "name": "contract::FuzzUser",
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
          "flags": "[FirstMatch(0, []), OptionalDefaultFromNoValue, Pending]",
          "checks": "[]",
          "explanations": []
        }"#]]
    .assert_eq(&parsed_fixture(&parsed));
}

#[test]
fn contract_int_repr_string_fixture() {
    let raw = r#"{ "id": "18446744073709551615" }"#;
    let parsed = facet_runtime::parse_llm_output::<FacetBigIntString>(raw, true).expect("parse");
    expect![[r#"
        {
          "baml_value": {
            "kind": "class",
            "name": "contract::BigIntString",
            "fields": {
              "id": {
                "kind": "string",
                "value": "18446744073709551615"
              }
            }
          },
          "flags": "[FirstMatch(0, [])]",
          "checks": "[]",
          "explanations": []
        }"#]]
    .assert_eq(&parsed_fixture(&parsed));
}

#[test]
fn contract_map_key_pairs_parse_and_registration_fixture() {
    let raw = r#"{ "values": [ { "key": 1, "value": "a" }, { "key": 2, "value": "b" } ] }"#;
    let parsed = facet_runtime::parse_llm_output::<FacetMapPairs>(raw, true).expect("parse");

    let entry_name = format!(
        "{}::values__Entry",
        <FacetMapPairs as facet_runtime::BamlType>::baml_internal_name()
    );
    let class = <FacetMapPairs as facet_runtime::BamlType>::baml_output_format()
        .classes
        .get(&(entry_name, StreamingMode::NonStreaming))
        .expect("entry class");

    let snapshot = serde_json::to_string_pretty(&json!({
        "parsed": serde_json::from_str::<serde_json::Value>(&parsed_fixture(&parsed)).expect("json parsed fixture"),
        "entry_rendered_name": class.name.rendered_name(),
    }))
    .expect("serialize");

    expect![[r#"
        {
          "parsed": {
            "baml_value": {
              "kind": "class",
              "name": "contract::MapPairs",
              "fields": {
                "values": {
                  "kind": "list",
                  "value": [
                    {
                      "kind": "class",
                      "name": "contract::MapPairs::values__Entry",
                      "fields": {
                        "key": {
                          "kind": "int",
                          "value": 1
                        },
                        "value": {
                          "kind": "string",
                          "value": "a"
                        }
                      }
                    },
                    {
                      "kind": "class",
                      "name": "contract::MapPairs::values__Entry",
                      "fields": {
                        "key": {
                          "kind": "int",
                          "value": 2
                        },
                        "value": {
                          "kind": "string",
                          "value": "b"
                        }
                      }
                    }
                  ]
                }
              }
            },
            "flags": "[FirstMatch(0, [])]",
            "checks": "[]",
            "explanations": []
          },
          "entry_rendered_name": "valuesEntry"
        }"#]]
    .assert_eq(&snapshot);
}

#[test]
fn contract_check_results_fixture() {
    let raw = r#"{ "value": 3 }"#;
    let parsed = facet_runtime::parse_llm_output::<FacetChecked>(raw, true).expect("parse");
    expect![[r#"
        {
          "baml_value": {
            "kind": "class",
            "name": "contract::Checked",
            "fields": {
              "value": {
                "kind": "int",
                "value": 3
              }
            }
          },
          "flags": "[FirstMatch(0, []), ConstraintResults([(\"positive\", JinjaExpression(\"this > 0\"), true)])]",
          "checks": "[ResponseCheck { name: \"positive\", expression: \"this > 0\", status: \"succeeded\" }]",
          "explanations": []
        }"#]].assert_eq(&parsed_fixture(&parsed));
}

#[test]
fn contract_assert_error_shape_fixture() {
    let raw = r#"{ "value": -1 }"#;
    let err = facet_runtime::parse_llm_output::<FacetAsserted>(raw, true).expect_err("err");

    let facet_runtime::BamlParseError::ConstraintAssertsFailed { failed } = err else {
        panic!("expected assert failure error");
    };

    let snapshot = serde_json::to_string_pretty(&json!({
        "failed": format!("{:?}", failed),
    }))
    .expect("serialize");

    expect![[r#"
        {
          "failed": "[ResponseCheck { name: \"positive\", expression: \"this > 0\", status: \"failed\" }]"
        }"#]].assert_eq(&snapshot);
}

#[test]
fn contract_markdown_flags_and_explanations_fixture() {
    let markdown_raw = "```json\n{ \"name\": \"Ada\", \"age\": 36 }\n```";
    let markdown =
        facet_runtime::parse_llm_output::<FacetFuzzUser>(markdown_raw, true).expect("parse");

    let explain_raw = r#"{ "values": { "ok": 1, "bad": "oops" } }"#;
    let explain =
        facet_runtime::parse_llm_output::<FacetFuzzExplain>(explain_raw, true).expect("parse");

    let snapshot = serde_json::to_string_pretty(&json!({
        "markdown": serde_json::from_str::<serde_json::Value>(&parsed_fixture(&markdown)).expect("json markdown fixture"),
        "explain": serde_json::from_str::<serde_json::Value>(&parsed_fixture(&explain)).expect("json explain fixture"),
    }))
    .expect("serialize");

    expect![[r#"
        {
          "markdown": {
            "baml_value": {
              "kind": "class",
              "name": "contract::FuzzUser",
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
            "flags": "[FirstMatch(0, []), ObjectFromMarkdown(0), Incomplete, FirstMatch(0, []), Incomplete, OptionalDefaultFromNoValue, Pending]",
            "checks": "[]",
            "explanations": []
          },
          "explain": {
            "baml_value": {
              "kind": "class",
              "name": "contract::FuzzExplain",
              "fields": {
                "values": {
                  "kind": "map",
                  "value": {
                    "ok": {
                      "kind": "int",
                      "value": 1
                    }
                  }
                }
              }
            },
            "flags": "[FirstMatch(0, []), ObjectToMap(Object([(\"ok\", Number(Number(1), Complete)), (\"bad\", String(\"oops\", Complete))], Complete)), MapValueParseError(\"bad\", ParsingError { scope: [\"values\", \"bad\"], reason: \"Expected int, got String(\\\"oops\\\", Complete).\", causes: [] })]",
            "checks": "[]",
            "explanations": [
              "<root>.values: error while parsing map\n  - values.bad: Expected int, got String(\"oops\", Complete)."
            ]
          }
        }"#]].assert_eq(&snapshot);
}

#[test]
fn contract_unit_enum_alias_parse_fixture() {
    let raw = r#""green""#;
    let parsed = facet_runtime::parse_llm_output::<FacetColorAlias>(raw, true).expect("parse");
    expect![[r#"
        {
          "baml_value": {
            "kind": "enum",
            "name": "contract::ColorAlias",
            "variant": "Green"
          },
          "flags": "[SubstringMatch(\"\\\"green\\\"\")]",
          "checks": "[]",
          "explanations": []
        }"#]]
    .assert_eq(&parsed_fixture(&parsed));
}

#[test]
fn contract_as_union_type_ir_and_parse_fixture() {
    let literals = sorted_union_string_literals(
        <FacetAsUnionColor as facet_runtime::BamlType>::baml_type_ir(),
    );
    let raw = r#""green""#;
    let parsed = facet_runtime::parse_llm_output::<FacetAsUnionColor>(raw, true).expect("parse");

    let snapshot = serde_json::to_string_pretty(&json!({
        "literals": literals,
        "parsed": serde_json::from_str::<serde_json::Value>(&parsed_fixture(&parsed)).expect("json parsed fixture"),
    }))
    .expect("serialize");

    expect![[r#"
        {
          "literals": [
            "Red",
            "green"
          ],
          "parsed": {
            "baml_value": {
              "kind": "string",
              "value": "green"
            },
            "flags": "[UnionMatch(1, []), FirstMatch(0, [])]",
            "checks": "[]",
            "explanations": []
          }
        }"#]]
    .assert_eq(&snapshot);
}

#[test]
fn contract_data_enum_docs_render_fixture() {
    let rendered =
        facet_runtime::render_schema::<FacetDocShape>(facet_runtime::RenderOptions::default())
            .expect("render")
            .unwrap_or_default();
    expect![[r#"
        Answer in JSON using any of these schemas:
        {
          // A circle, defined by its radius.

          type: "Circle",
          // Radius in meters.
          radius: float,
        } or {
          type: "Rectangle",
          width: float,
          height: float,
        }"#]]
    .assert_eq(&rendered);
}

#[test]
fn contract_int_repr_option_fixture() {
    let parsed_num = facet_runtime::parse_llm_output::<FacetBigIntOption>(r#"{ "id": 42 }"#, true)
        .expect("parse num");
    let parsed_null =
        facet_runtime::parse_llm_output::<FacetBigIntOption>(r#"{ "id": null }"#, true)
            .expect("parse null");

    let snapshot = serde_json::to_string_pretty(&json!({
        "num": serde_json::from_str::<serde_json::Value>(&parsed_fixture(&parsed_num)).expect("json parsed fixture"),
        "null": serde_json::from_str::<serde_json::Value>(&parsed_fixture(&parsed_null)).expect("json parsed fixture"),
    }))
    .expect("serialize");

    expect![[r#"
        {
          "num": {
            "baml_value": {
              "kind": "class",
              "name": "contract::BigIntOption",
              "fields": {
                "id": {
                  "kind": "int",
                  "value": 42
                }
              }
            },
            "flags": "[FirstMatch(0, []), UnionMatch(0, [])]",
            "checks": "[]",
            "explanations": []
          },
          "null": {
            "baml_value": {
              "kind": "class",
              "name": "contract::BigIntOption",
              "fields": {
                "id": {
                  "kind": "null"
                }
              }
            },
            "flags": "[FirstMatch(0, [])]",
            "checks": "[]",
            "explanations": []
          }
        }"#]]
    .assert_eq(&snapshot);
}

#[test]
fn contract_map_key_repr_string_and_option_fixture() {
    let parsed_keys = facet_runtime::parse_llm_output::<FacetMapKeys>(
        r#"{ "values": { "1": "a", "2": "b" } }"#,
        true,
    )
    .expect("parse keys");
    let parsed_option = facet_runtime::parse_llm_output::<FacetMapKeysOption>(
        r#"{ "values": { "10": "x" } }"#,
        true,
    )
    .expect("parse option");

    let snapshot = serde_json::to_string_pretty(&json!({
        "keys": serde_json::from_str::<serde_json::Value>(&parsed_fixture(&parsed_keys)).expect("json parsed fixture"),
        "option": serde_json::from_str::<serde_json::Value>(&parsed_fixture(&parsed_option)).expect("json parsed fixture"),
    }))
    .expect("serialize");

    expect![[r#"
        {
          "keys": {
            "baml_value": {
              "kind": "class",
              "name": "contract::MapKeys",
              "fields": {
                "values": {
                  "kind": "map",
                  "value": {
                    "1": {
                      "kind": "string",
                      "value": "a"
                    },
                    "2": {
                      "kind": "string",
                      "value": "b"
                    }
                  }
                }
              }
            },
            "flags": "[FirstMatch(0, []), ObjectToMap(Object([(\"1\", String(\"a\", Complete)), (\"2\", String(\"b\", Complete))], Complete))]",
            "checks": "[]",
            "explanations": []
          },
          "option": {
            "baml_value": {
              "kind": "class",
              "name": "contract::MapKeysOption",
              "fields": {
                "values": {
                  "kind": "map",
                  "value": {
                    "10": {
                      "kind": "string",
                      "value": "x"
                    }
                  }
                }
              }
            },
            "flags": "[FirstMatch(0, []), ObjectToMap(Object([(\"10\", String(\"x\", Complete))], Complete)), UnionMatch(0, [])]",
            "checks": "[]",
            "explanations": []
          }
        }"#]].assert_eq(&snapshot);
}

#[test]
fn contract_rename_all_fixture() {
    let raw = r#"{ "fullName": "Ada" }"#;
    let parsed = facet_runtime::parse_llm_output::<FacetRenameAllUser>(raw, true).expect("parse");
    expect![[r#"
        {
          "baml_value": {
            "kind": "class",
            "name": "contract::RenameAllUser",
            "fields": {
              "full_name": {
                "kind": "string",
                "value": "Ada"
              }
            }
          },
          "flags": "[FirstMatch(0, [])]",
          "checks": "[]",
          "explanations": []
        }"#]]
    .assert_eq(&parsed_fixture(&parsed));
}

#[test]
fn contract_recursion_class_detection_fixture() {
    let output = <FacetNode as facet_runtime::BamlType>::baml_output_format();
    let mut recursive = output.recursive_classes.iter().cloned().collect::<Vec<_>>();
    recursive.sort();

    let snapshot = serde_json::to_string_pretty(&json!({
        "recursive_classes": recursive,
    }))
    .expect("serialize");

    expect![[r#"
        {
          "recursive_classes": [
            "contract::Node"
          ]
        }"#]]
    .assert_eq(&snapshot);
}

#[test]
fn contract_internal_name_collision_behavior_fixture() {
    let a_name = <facet_collision_a::User as facet_runtime::BamlType>::baml_internal_name();
    let b_name = <facet_collision_b::User as facet_runtime::BamlType>::baml_internal_name();

    let a_class = <facet_collision_a::User as facet_runtime::BamlType>::baml_output_format()
        .classes
        .get(&(a_name.to_string(), StreamingMode::NonStreaming))
        .expect("class a");
    let b_class = <facet_collision_b::User as facet_runtime::BamlType>::baml_output_format()
        .classes
        .get(&(b_name.to_string(), StreamingMode::NonStreaming))
        .expect("class b");

    let snapshot = serde_json::to_string_pretty(&json!({
        "a_name": a_name,
        "b_name": b_name,
        "a_rendered": a_class.name.rendered_name(),
        "b_rendered": b_class.name.rendered_name(),
    }))
    .expect("serialize");

    expect![[r#"
        {
          "a_name": "contract_frozen_oracle::facet_collision_a::User",
          "b_name": "contract_frozen_oracle::facet_collision_b::User",
          "a_rendered": "User",
          "b_rendered": "User"
        }"#]]
    .assert_eq(&snapshot);
}

#[test]
fn contract_with_adapter_schema_and_parse_fixture() {
    let output = <FacetWithAdapter as facet_runtime::BamlType>::baml_output_format();
    let adapter = output
        .classes
        .get(&("AdapterI64Wrapper".to_string(), StreamingMode::NonStreaming))
        .expect("adapter class");
    let adapter_fields = adapter
        .fields
        .iter()
        .map(|(name, type_ir, _, _)| format!("{}: {type_ir:?}", name.real_name()))
        .collect::<Vec<_>>();

    let raw = r#"{ "id": { "value": "9223372036854775807" } }"#;
    let parsed = facet_runtime::parse_llm_output::<FacetWithAdapter>(raw, true).expect("parse");

    let snapshot = serde_json::to_string_pretty(&json!({
        "adapter_real_name": adapter.name.real_name(),
        "adapter_fields": adapter_fields,
        "parsed": serde_json::from_str::<serde_json::Value>(&parsed_fixture(&parsed)).expect("json parsed fixture"),
    }))
    .expect("serialize");

    expect![[r#"
        {
          "adapter_real_name": "AdapterI64Wrapper",
          "adapter_fields": [
            "value: Primitive(String, TypeMeta { constraints: [], streaming_behavior: StreamingBehavior { needed: false, done: false, state: false } })"
          ],
          "parsed": {
            "baml_value": {
              "kind": "class",
              "name": "contract::WithAdapter",
              "fields": {
                "id": {
                  "kind": "class",
                  "name": "AdapterI64Wrapper",
                  "fields": {
                    "value": {
                      "kind": "string",
                      "value": "9223372036854775807"
                    }
                  }
                }
              }
            },
            "flags": "[FirstMatch(0, [])]",
            "checks": "[]",
            "explanations": []
          }
        }"#]].assert_eq(&snapshot);
}

fn assert_default_adapter_roundtrip<T>(value: T)
where
    T: Clone + std::fmt::Debug + PartialEq + facet_runtime::facet::Facet<'static>,
{
    let baml = facet_runtime::to_baml_value_lossy(&value);
    let back = facet_runtime::try_from_baml_value::<T>(baml).expect("roundtrip");
    assert_eq!(back, value);
}

#[test]
fn contract_default_adapter_roundtrips() {
    assert_default_adapter_roundtrip("hello".to_string());
    assert_default_adapter_roundtrip(true);
    assert_default_adapter_roundtrip(123i32);
    assert_default_adapter_roundtrip(-99i64);
    assert_default_adapter_roundtrip(3.5f32);
    assert_default_adapter_roundtrip(9.75f64);

    assert_default_adapter_roundtrip(Some(42i32));
    assert_default_adapter_roundtrip(None::<i32>);
    assert_default_adapter_roundtrip(vec!["a".to_string(), "b".to_string()]);
    assert_default_adapter_roundtrip(Box::new("boxed".to_string()));
    assert_default_adapter_roundtrip(Arc::new("arc".to_string()));
    assert_default_adapter_roundtrip(Rc::new("rc".to_string()));

    let mut hm = HashMap::new();
    hm.insert("answer".to_string(), 42i32);
    assert_default_adapter_roundtrip(hm);

    let mut bt = BTreeMap::new();
    bt.insert("left".to_string(), 1i64);
    bt.insert("right".to_string(), 2i64);
    assert_default_adapter_roundtrip(bt);

    let nested = FacetNestedStruct {
        title: "nested".to_string(),
        items: Some(vec![FacetRoundtripStruct {
            name: "child".to_string(),
            count: 2,
            tags: vec!["x".to_string(), "y".to_string()],
            meta: None,
        }]),
        metadata: HashMap::from([("alpha".to_string(), Some(1)), ("beta".to_string(), None)]),
    };
    assert_default_adapter_roundtrip(nested);
    assert_default_adapter_roundtrip(FacetRoundtripUnitEnum::Beta);
    assert_default_adapter_roundtrip(FacetRoundtripDataEnum::Message {
        body: "hello".to_string(),
        count: 3,
    });
}

#[test]
fn contract_default_adapter_error_shape_fixture() {
    let invalid = BamlValue::Class(
        "contract::Unsigned32".to_string(),
        [("value".to_string(), BamlValue::Int(-1))]
            .into_iter()
            .collect(),
    );

    let err =
        facet_runtime::try_from_baml_value::<FacetUnsigned32>(invalid).expect_err("should fail");

    let snapshot = serde_json::to_string_pretty(&json!({
        "expected": err.expected,
        "path": err.path,
        "display": err.to_string(),
    }))
    .expect("serialize");

    expect![[r#"
        {
          "expected": "u32",
          "path": [
            "value"
          ],
          "display": "integer out of range (expected u32, got -1) at value"
        }"#]]
    .assert_eq(&snapshot);
}

#[test]
fn contract_direct_integer_string_conversion_error_fixture() {
    let err = facet_runtime::try_from_baml_value::<i64>(BamlValue::String("123".into()))
        .expect_err("should reject");

    let snapshot = serde_json::to_string_pretty(&json!({
        "expected": err.expected,
        "path": err.path,
        "display": err.to_string(),
    }))
    .expect("serialize");

    expect![[r#"
        {
          "expected": "int",
          "path": [],
          "display": "expected a int (expected int, got String(\"123\")) at <root>"
        }"#]]
    .assert_eq(&snapshot);
}

#[test]
fn contract_direct_map_pairs_conversion_error_fixture() {
    let pair_entry = BamlValue::Map(
        [
            ("key".to_string(), BamlValue::String("k".to_string())),
            ("value".to_string(), BamlValue::Int(1)),
        ]
        .into_iter()
        .collect(),
    );
    let raw = BamlValue::List(vec![pair_entry]);

    let err =
        facet_runtime::try_from_baml_value::<HashMap<String, i64>>(raw).expect_err("should reject");

    let snapshot = serde_json::to_string_pretty(&json!({
        "expected": err.expected,
        "path": err.path,
        "display": err.to_string(),
    }))
    .expect("serialize");

    expect![[r#"
        {
          "expected": "map",
          "path": [],
          "display": "expected a map (expected map, got List([Map({\"key\": String(\"k\"), \"value\": Int(1)})])) at <root>"
        }"#]].assert_eq(&snapshot);
}

#[test]
fn contract_schema_fingerprint_fixture() {
    let output = <FacetUser as facet_runtime::BamlType>::baml_output_format();
    let fp = facet_runtime::schema_fingerprint(output, facet_runtime::RenderOptions::default())
        .expect("fingerprint");

    expect!["bd09ecdbcb1ddf0746150789905fcce576fb67b74499a9d4738ca28018ae859c"].assert_eq(&fp);
}
