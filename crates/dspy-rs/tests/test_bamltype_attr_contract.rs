use dspy_rs::{BamlType, BamlTypeTrait, RenderOptions};

#[derive(Debug, Clone, PartialEq)]
#[BamlType]
#[baml(internal_name = "contract::DsrsUser")]
struct DsrsUser {
    #[baml(alias = "fullName")]
    name: String,
    age: u32,
}

#[test]
fn bamltype_attribute_macro_works_from_dspy_rs() {
    let schema = <DsrsUser as BamlTypeTrait>::baml_output_format()
        .render(RenderOptions::default())
        .expect("render schema")
        .unwrap_or_default();
    assert!(schema.contains("fullName"));

    let raw = r#"{ "fullName": "Ada", "age": 36 }"#;
    let parsed = dspy_rs::bamltype::parse_llm_output::<DsrsUser>(raw, true).expect("parse");
    assert_eq!(parsed.value.name, "Ada");
    assert_eq!(parsed.value.age, 36);
}
