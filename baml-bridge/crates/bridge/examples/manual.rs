use baml_bridge::{parse_llm_output, render_schema, BamlType};
use baml_bridge::internal_baml_jinja::types::RenderOptions;

/// A user record returned by the model.
#[derive(Debug, Clone, PartialEq, BamlType)]
struct User {
    /// Full name for display.
    #[baml(alias = "fullName")]
    name: String,
    /// Age in years.
    age: i64,
}

fn main() {
    let schema = render_schema::<User>(RenderOptions::default())
        .expect("render failed")
        .unwrap_or_default();
    println!("Schema:\n{}", schema);

    let raw = r#"{ "name": "Ada Lovelace", "age": 36 }"#;
    let parsed = parse_llm_output::<User>(raw, true).expect("parse failed");
    println!("Parsed: {:?}", parsed.value);
}
