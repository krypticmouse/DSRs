use std::collections::HashMap;

use baml_bridge::{parse_llm_output, BamlType};
use baml_bridge::jsonish::deserializer::deserialize_flags::Flag;

#[derive(Debug, Clone, PartialEq, BamlType)]
struct FuzzUser {
    name: String,
    age: u32,
    nickname: Option<String>,
}

#[derive(Debug, Clone, PartialEq, BamlType)]
struct FuzzExplain {
    values: HashMap<String, i64>,
}

#[test]
fn markdown_fence_parses_and_sets_flag() {
    let raw = "```json\n{ \"name\": \"Ada\", \"age\": 36 }\n```";
    let parsed = parse_llm_output::<FuzzUser>(raw, true).expect("parse");

    assert_eq!(parsed.value.name, "Ada");
    assert!(parsed
        .flags
        .iter()
        .any(|flag| matches!(flag, Flag::ObjectFromMarkdown(_))));
}

#[test]
fn trailing_comma_parses() {
    let raw = r#"{ "name": "Ada", "age": 36, }"#;
    let parsed = parse_llm_output::<FuzzUser>(raw, true).expect("parse");
    assert_eq!(parsed.value.age, 36);
}

#[test]
fn extra_keys_are_ignored() {
    let raw = r#"{ "name": "Ada", "age": 36, "extra": "ignored" }"#;
    let parsed = parse_llm_output::<FuzzUser>(raw, true).expect("parse");
    assert_eq!(parsed.value.name, "Ada");
    assert_eq!(parsed.value.nickname, None);
}

#[test]
fn explanations_surface_on_map_parse_error() {
    let raw = r#"{ "values": { "ok": 1, "bad": "oops" } }"#;
    let parsed = parse_llm_output::<FuzzExplain>(raw, true).expect("parse");
    assert_eq!(parsed.value.values.get("ok"), Some(&1));
    assert!(!parsed.explanations.is_empty());
}
