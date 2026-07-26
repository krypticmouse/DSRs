use std::collections::HashMap;
use std::ffi::CString;

use proptest::prelude::*;
use proptest::string::string_regex;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::{Map, Value, json};

use super::{
    AdapterField, AdapterSpec, CompiledSpec, TypeIR, compile_spec, parse_response_core,
    render_field_structure_core,
};

#[derive(Clone, Debug)]
enum FuzzType {
    Str,
    Int,
    Float,
    Bool,
    LiteralStr(String),
    LiteralInt(i64),
    LiteralBool(bool),
    Enum(Vec<String>),
    List(Box<FuzzType>),
    Dict(Box<FuzzType>),
    Optional(Box<FuzzType>),
    Union(Vec<FuzzType>),
    Object(Vec<FuzzObjectField>),
}

#[derive(Clone, Debug)]
struct FuzzObjectField {
    name: String,
    description: String,
    required: bool,
    ty: FuzzType,
}

#[derive(Clone, Debug)]
struct FuzzOutputCase {
    name: String,
    description: String,
    ty: FuzzType,
    use_null_if_optional: bool,
    loose_jsonish: bool,
}

fn description_strategy() -> BoxedStrategy<String> {
    string_regex("[A-Za-z0-9 _-]{3,24}")
        .expect("valid description regex")
        .boxed()
}

fn dedupe_non_empty_strings(mut values: Vec<String>) -> Vec<String> {
    values.sort();
    values.dedup();
    if values.is_empty() {
        return vec!["alpha".to_string(), "beta".to_string()];
    }
    if values.len() == 1 {
        values.push(format!("{}_alt", values[0]));
    }
    values
}

fn leaf_type_strategy() -> BoxedStrategy<FuzzType> {
    prop_oneof![
        Just(FuzzType::Str),
        Just(FuzzType::Int),
        Just(FuzzType::Float),
        Just(FuzzType::Bool),
        string_regex("[a-zA-Z0-9_-]{1,20}")
            .expect("valid literal string regex")
            .prop_map(FuzzType::LiteralStr),
        any::<i16>().prop_map(|n| FuzzType::LiteralInt(n as i64)),
        any::<bool>().prop_map(FuzzType::LiteralBool),
        prop::collection::vec(
            string_regex("[a-z]{1,10}").expect("valid enum value regex"),
            2..6,
        )
        .prop_map(|values| FuzzType::Enum(dedupe_non_empty_strings(values))),
    ]
    .boxed()
}

fn fuzz_type_strategy() -> BoxedStrategy<FuzzType> {
    leaf_type_strategy()
        .prop_recursive(4, 128, 8, |inner| {
            prop_oneof![
                inner.clone().prop_map(|ty| FuzzType::List(Box::new(ty))),
                inner.clone().prop_map(|ty| FuzzType::Dict(Box::new(ty))),
                inner
                    .clone()
                    .prop_map(|ty| FuzzType::Optional(Box::new(ty))),
                prop::collection::vec(inner.clone(), 2..5).prop_map(FuzzType::Union),
                prop::collection::vec(
                    (
                        inner.clone(),
                        any::<bool>(),
                        prop::option::of(description_strategy()),
                    ),
                    1..5,
                )
                .prop_map(|items| {
                    let fields = items
                        .into_iter()
                        .enumerate()
                        .map(|(idx, (ty, required, description))| FuzzObjectField {
                            name: format!("field_{}", idx + 1),
                            description: description.unwrap_or_default(),
                            required,
                            ty,
                        })
                        .collect();
                    FuzzType::Object(fields)
                }),
            ]
            .boxed()
        })
        .boxed()
}

fn output_cases_strategy() -> BoxedStrategy<Vec<FuzzOutputCase>> {
    prop::collection::vec(
        (
            fuzz_type_strategy(),
            any::<bool>(),
            any::<bool>(),
            prop::option::of(description_strategy()),
        ),
        1..5,
    )
    .prop_map(|entries| {
        entries
            .into_iter()
            .enumerate()
            .map(
                |(idx, (ty, use_null_if_optional, loose_jsonish, description))| FuzzOutputCase {
                    name: format!("output_{}", idx + 1),
                    description: description.unwrap_or_default(),
                    ty,
                    use_null_if_optional,
                    loose_jsonish,
                },
            )
            .collect()
    })
    .boxed()
}

fn schema_from_type(ty: &FuzzType, hint: &str) -> Value {
    match ty {
        FuzzType::Str => json!({"type": "string"}),
        FuzzType::Int => json!({"type": "integer"}),
        FuzzType::Float => json!({"type": "number"}),
        FuzzType::Bool => json!({"type": "boolean"}),
        FuzzType::LiteralStr(value) => json!({"const": value}),
        FuzzType::LiteralInt(value) => json!({"const": value}),
        FuzzType::LiteralBool(value) => json!({"const": value}),
        FuzzType::Enum(values) => json!({
            "title": format!("{hint}Enum"),
            "type": "string",
            "enum": values,
        }),
        FuzzType::List(inner) => json!({
            "type": "array",
            "items": schema_from_type(inner, &format!("{hint}Item")),
        }),
        FuzzType::Dict(value_type) => json!({
            "type": "object",
            "additionalProperties": schema_from_type(value_type, &format!("{hint}Value")),
        }),
        FuzzType::Optional(inner) => json!({
            "anyOf": [
                schema_from_type(inner, &format!("{hint}Some")),
                {"type": "null"},
            ]
        }),
        FuzzType::Union(choices) => json!({
            "anyOf": choices
                .iter()
                .enumerate()
                .map(|(idx, choice)| schema_from_type(choice, &format!("{hint}Variant{}", idx + 1)))
                .collect::<Vec<_>>()
        }),
        FuzzType::Object(fields) => {
            let mut properties = Map::new();
            let mut required = Vec::new();

            for field in fields {
                let mut field_schema =
                    schema_from_type(&field.ty, &format!("{hint}{}", field.name.replace('_', "")));

                if !field.description.is_empty()
                    && let Value::Object(map) = &mut field_schema
                {
                    map.insert(
                        "description".to_string(),
                        Value::String(field.description.clone()),
                    );
                }

                if field.required {
                    required.push(field.name.clone());
                }
                properties.insert(field.name.clone(), field_schema);
            }

            json!({
                "title": format!("{hint}Object"),
                "description": format!("{hint} object"),
                "type": "object",
                "properties": properties,
                "required": required,
            })
        }
    }
}

fn value_from_type(ty: &FuzzType, use_null_if_optional: bool) -> Value {
    match ty {
        FuzzType::Str => Value::String("plain text value".to_string()),
        FuzzType::Int => json!(42),
        FuzzType::Float => json!(3.25),
        FuzzType::Bool => json!(true),
        FuzzType::LiteralStr(value) => Value::String(value.clone()),
        FuzzType::LiteralInt(value) => json!(value),
        FuzzType::LiteralBool(value) => json!(value),
        FuzzType::Enum(values) => Value::String(values[0].clone()),
        FuzzType::List(inner) => Value::Array(vec![
            value_from_type(inner, false),
            value_from_type(inner, false),
        ]),
        FuzzType::Dict(inner) => {
            let mut map = Map::new();
            map.insert("k1".to_string(), value_from_type(inner, false));
            map.insert("k2".to_string(), value_from_type(inner, false));
            Value::Object(map)
        }
        FuzzType::Optional(inner) => {
            if use_null_if_optional {
                Value::Null
            } else {
                value_from_type(inner, false)
            }
        }
        FuzzType::Union(choices) => value_from_type(&choices[0], use_null_if_optional),
        FuzzType::Object(fields) => {
            let mut map = Map::new();
            for field in fields {
                map.insert(
                    field.name.clone(),
                    value_from_type(&field.ty, use_null_if_optional),
                );
            }
            Value::Object(map)
        }
    }
}

fn loosen_top_level_jsonish(text: String, value: &Value) -> String {
    if value.is_object() {
        return text.replacen("\n}", ",\n}", 1);
    }
    if value.is_array() {
        return text.replacen("\n]", ",\n]", 1);
    }
    text
}

fn completion_text_for_value(value: &Value, loose_jsonish: bool) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Number(number) => number.to_string(),
        Value::Bool(flag) => flag.to_string(),
        Value::Null => "null".to_string(),
        _ => {
            let text = serde_json::to_string_pretty(value).expect("valid JSON value serialization");
            if loose_jsonish {
                loosen_top_level_jsonish(text, value)
            } else {
                text
            }
        }
    }
}

fn build_spec_completion_and_expected(
    cases: &[FuzzOutputCase],
) -> (AdapterSpec, String, Map<String, Value>) {
    let output_fields = cases
        .iter()
        .map(|case| AdapterField {
            name: case.name.clone(),
            description: case.description.clone(),
            format: None,
            schema: schema_from_type(&case.ty, &case.name),
        })
        .collect::<Vec<_>>();

    let mut expected = Map::new();
    let mut sections = Vec::new();

    for case in cases {
        let value = value_from_type(&case.ty, case.use_null_if_optional);
        expected.insert(case.name.clone(), value.clone());

        sections.push(format!(
            "[[ ## {} ## ]]\n{}",
            case.name,
            completion_text_for_value(&value, case.loose_jsonish)
        ));
    }

    let completion = format!("{}\n\n[[ ## completed ## ]]\n", sections.join("\n\n"));

    (
        AdapterSpec {
            input_fields: Vec::new(),
            output_fields,
            instruction: "property test instruction".to_string(),
        },
        completion,
        expected,
    )
}

fn target_output_class(
    compiled: &CompiledSpec,
) -> Option<&bamltype::internal_baml_jinja::types::Class> {
    let TypeIR::Class { name, mode, .. } = &compiled.output_format.target else {
        return None;
    };

    compiled.output_format.classes.get(&(name.clone(), *mode))
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 256, .. ProptestConfig::default() })]

    #[test]
    fn prop_generated_schemas_compile_render_parse_and_preserve_descriptions(
        cases in output_cases_strategy()
    ) {
        let (spec, completion, expected) = build_spec_completion_and_expected(&cases);

        let compiled = compile_spec(&spec)
            .unwrap_or_else(|err| panic!("compile_spec failed for generated case: {err}\nSpec: {:?}", spec));

        let rendered = render_field_structure_core(&spec)
            .unwrap_or_else(|err| panic!("render_field_structure_core failed: {err}\nSpec: {:?}", spec));

        for case in &cases {
            prop_assert!(
                rendered.contains(&format!("[[ ## {} ## ]]", case.name)),
                "rendered field structure missing output marker for {}\nRendered:\n{}",
                case.name,
                rendered,
            );
        }

        let parsed = parse_response_core(&spec, &completion, true)
            .unwrap_or_else(|err| panic!("parse_response_core failed: {err}\nCompletion:\n{completion}\nSpec: {:?}", spec));

        prop_assert_eq!(parsed, expected, "parsed output mismatch for generated case");

        let class = target_output_class(&compiled)
            .unwrap_or_else(|| panic!("expected class target in compiled output format"));

        let mut class_field_descs: HashMap<String, String> = HashMap::new();
        for (name, _, description, _) in &class.fields {
            class_field_descs.insert(
                name.real_name().to_string(),
                description.clone().unwrap_or_default(),
            );
        }

        for case in &cases {
            let got = class_field_descs
                .get(&case.name)
                .cloned()
                .unwrap_or_default();
            prop_assert_eq!(
                got,
                case.description.clone(),
                "output field description should be preserved for {}",
                case.name
            );
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 128, .. ProptestConfig::default() })]

    #[test]
    fn prop_defs_and_refs_parse_correctly_with_jsonish_noise(
        person_name in string_regex("[A-Za-z]{3,20}").expect("valid person name regex"),
        street in string_regex("[A-Za-z0-9 ]{3,24}").expect("valid street regex"),
        city in string_regex("[A-Za-z]{3,20}").expect("valid city regex"),
        postal in 10000u32..99999u32,
        age in 1u8..110u8,
        loose_jsonish in any::<bool>(),
    ) {
        let schema = json!({
            "$defs": {
                "Address": {
                    "type": "object",
                    "title": "Address",
                    "description": "Postal address",
                    "properties": {
                        "street": {"type": "string", "description": "Street line"},
                        "city": {"type": "string", "description": "City"},
                        "postal": {"type": "integer", "description": "Postal code"}
                    },
                    "required": ["street", "city", "postal"]
                },
                "Profile": {
                    "type": "object",
                    "title": "Profile",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                        "address": {"$ref": "#/$defs/Address"}
                    },
                    "required": ["name", "age", "address"]
                }
            },
            "$ref": "#/$defs/Profile"
        });

        let value = json!({
            "name": person_name,
            "age": age,
            "address": {
                "street": street,
                "city": city,
                "postal": postal,
            }
        });

        let spec = AdapterSpec {
            input_fields: Vec::new(),
            output_fields: vec![AdapterField {
                name: "profile".to_string(),
                description: "Nested profile payload".to_string(),
                format: None,
                schema,
            }],
            instruction: "Extract profile data".to_string(),
        };

        let rendered = render_field_structure_core(&spec)
            .unwrap_or_else(|err| panic!("render failed: {err}"));
        prop_assert!(rendered.contains("[[ ## profile ## ]]"));

        let completion = format!(
            "[[ ## profile ## ]]\n{}\n\n[[ ## completed ## ]]\n",
            completion_text_for_value(&value, loose_jsonish)
        );

        let parsed = parse_response_core(&spec, &completion, true)
            .unwrap_or_else(|err| panic!("parse failed: {err}\nCompletion:\n{completion}"));

        prop_assert_eq!(parsed.get("profile"), Some(&value));
    }
}

fn load_pydantic_typing_cases_from_python() -> Option<Vec<(String, Value, Value, String)>> {
    Python::initialize();

    Python::attach(|py| {
        if py.import("pydantic").is_err() {
            return None;
        }

        let code = CString::new(
            r#"
import json
from datetime import datetime, date, time
from typing import Annotated, Literal, Optional, Union
from pydantic import BaseModel, Field, TypeAdapter

class Address(BaseModel):
    street: Annotated[str, Field(description='street line', min_length=1)]
    zipcode: Annotated[int, Field(ge=1, le=99999)]

class Contact(BaseModel):
    email: str
    phone: str | None = None

class User(BaseModel):
    name: Annotated[str, Field(description='full legal name')]
    age: Annotated[int, Field(ge=0)]
    active: bool
    tags: list[str]
    scores: dict[str, float]
    address: Address
    contact: Contact | None = None

cases = [
    ('str', str, 'hello world', 'plain string'),
    ('int', int, 42, 'plain int'),
    ('float', float, 3.5, 'plain float'),
    ('bool', bool, True, 'plain bool'),
    ('annotated_str', Annotated[str, Field(description='annotated string', min_length=1)], 'annotated', 'annotated str'),
    ('literal', Literal['alpha', 'beta'], 'alpha', 'literal'),
    ('optional', Optional[int], None, 'optional int'),
    ('union_int_str', Union[int, str], 7, 'union'),
    ('list_annotated_int', list[Annotated[int, Field(ge=0, le=10)]], [1, 2, 3], 'list annotated int'),
    ('dict_list_int', dict[str, list[int]], {'a': [1, 2], 'b': [3]}, 'dict of list[int]'),
    ('tuple_int_str', tuple[int, str], [9, 'x'], 'tuple[int, str]'),
    ('set_str', set[str], ['one', 'two'], 'set[str]'),
    ('datetime', datetime, '2025-01-02T03:04:05', 'datetime string format'),
    ('date', date, '2025-01-02', 'date string format'),
    ('time', time, '03:04:05', 'time string format'),
    ('address_model', Address, {'street': 'main', 'zipcode': 94105}, 'base model'),
    (
        'user_model',
        User,
        {
            'name': 'Ada',
            'age': 33,
            'active': True,
            'tags': ['ml'],
            'scores': {'q': 0.9},
            'address': {'street': 'main', 'zipcode': 94105},
            'contact': {'email': 'a@b.com', 'phone': None}
        },
        'nested base model',
    ),
]

rows = []
for name, annotation, sample, description in cases:
    schema = TypeAdapter(annotation).json_schema()
    rows.append((name, json.dumps(schema), json.dumps(sample), description))
"#,
        )
        .ok()?;

        let locals = PyDict::new(py);
        py.run(code.as_c_str(), None, Some(&locals)).ok()?;
        let rows_obj = locals.get_item("rows").ok().flatten()?;

        let rows: Vec<(String, String, String, String)> = rows_obj.extract().ok()?;
        let mut parsed_rows = Vec::with_capacity(rows.len());
        for (name, schema_json, sample_json, description) in rows {
            let schema = serde_json::from_str::<Value>(&schema_json).ok()?;
            let sample = serde_json::from_str::<Value>(&sample_json).ok()?;
            parsed_rows.push((name, schema, sample, description));
        }

        Some(parsed_rows)
    })
}

#[test]
fn pydantic_typing_matrix_from_python_typeadapter_roundtrips() {
    let Some(cases) = load_pydantic_typing_cases_from_python() else {
        eprintln!(
            "Skipping Python-driven typing matrix test: `pydantic` not available in interpreter"
        );
        return;
    };

    assert!(
        !cases.is_empty(),
        "Python-driven typing matrix should produce at least one case"
    );

    for (case_name, schema, sample, description) in cases {
        let spec = AdapterSpec {
            input_fields: Vec::new(),
            output_fields: vec![AdapterField {
                name: "value".to_string(),
                description: description.clone(),
                format: None,
                schema,
            }],
            instruction: format!("typing case: {case_name}"),
        };

        let rendered = render_field_structure_core(&spec)
            .unwrap_or_else(|err| panic!("render failed for `{case_name}`: {err}"));
        assert!(
            rendered.contains("[[ ## value ## ]]"),
            "rendered structure missing field marker for `{case_name}`"
        );

        let completion = format!(
            "[[ ## value ## ]]\n{}\n\n[[ ## completed ## ]]\n",
            completion_text_for_value(&sample, true)
        );

        let parsed = parse_response_core(&spec, &completion, true).unwrap_or_else(|err| {
            panic!(
                "parse failed for `{case_name}`: {err}\nCompletion:\n{completion}\nRendered:\n{rendered}"
            )
        });

        assert_eq!(
            parsed.get("value"),
            Some(&sample),
            "parsed value mismatch for `{case_name}`"
        );

        let compiled = compile_spec(&spec)
            .unwrap_or_else(|err| panic!("compile_spec failed for `{case_name}`: {err}"));
        let class = target_output_class(&compiled)
            .unwrap_or_else(|| panic!("missing target output class for `{case_name}`"));
        let field_desc = class
            .fields
            .iter()
            .find(|(name, _, _, _)| name.real_name() == "value")
            .and_then(|(_, _, desc, _)| desc.clone())
            .unwrap_or_default();
        assert_eq!(
            field_desc, description,
            "field description mismatch for `{case_name}`"
        );
    }
}
