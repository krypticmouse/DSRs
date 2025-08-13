pub mod adapter;
pub mod core;
pub mod data;
pub mod evaluate;
pub mod field;
pub mod lm;
pub mod programs;
pub mod providers;

pub mod internal;

pub use dsrs_macros::*;

#[macro_export]
macro_rules! sign {
    // Example Usage: signature! {
    //     question: String, random: bool -> answer: String
    // }
    //
    // Example Output:
    //
    // #[derive(Signature)]
    // struct InlineSignature {
    //     question: In<String>,
    //     random: In<bool>,
    //     answer: Out<String>,
    // }
    //
    // InlineSignature::new()

    // Pattern: input fields -> output fields
    { ($($input_name:ident : $input_type:ty),* $(,)?) -> $($output_name:ident : $output_type:ty),* $(,)? } => {{
        use dspy_rs::internal::{MetaField, MetaSignature};
        use indexmap::IndexMap;

        let mut input_fields = IndexMap::new();
        let mut output_fields = IndexMap::new();

        // Add input fields
        $(
            let json_value = serde_json::to_value(&schemars::schema_for!($input_type)).unwrap();

            let schema = if let Some(properties) = json_value.as_object()
                .and_then(|obj| obj.get("properties"))
                .and_then(|props| props.as_object()) {
                serde_json::to_string(&properties).unwrap_or_else(|_| "".to_string())
            } else {
                "".to_string()
            };

            input_fields.insert(
                stringify!($input_name).to_string(),
                MetaField {
                    desc: String::new(),
                    schema: schema,
                    data_type: stringify!($input_type).to_string(),
                    __dsrs_field_type: "Input".to_string(),
                }
            );
        )*

        // Add output fields
        $(
            let json_value = serde_json::to_value(&schemars::schema_for!($output_type)).unwrap();

            let schema = if let Some(properties) = json_value.as_object()
                .and_then(|obj| obj.get("properties"))
                .and_then(|props| props.as_object()) {
                serde_json::to_string(&properties).unwrap_or_else(|_| "".to_string())
            } else {
                "".to_string()
            };

            output_fields.insert(
                stringify!($output_name).to_string(),
                MetaField {
                    desc: String::new(),
                    schema: schema,
                    data_type: stringify!($output_type).to_string(),
                    __dsrs_field_type: "Output".to_string(),
                }
            );
        )*

        // Generate instruction string
        let input_names: Vec<&str> = vec![$(stringify!($input_name)),*];
        let output_names: Vec<&str> = vec![$(stringify!($output_name)),*];

        let instruction = "".to_string();

        MetaSignature {
            name: "InlineSignature".to_string(),
            instruction,
            input_fields,
            output_fields,
        }
    }};
}

#[macro_export]
macro_rules! example {
    // Pattern: { "key": "value", ... }
    { $($key:literal : $value:expr),* $(,)? } => {{
        use std::collections::HashMap;
        use dspy_rs::data::example::Example;

        let mut fields = HashMap::new();
        $(
            fields.insert($key.to_string(), $value.to_string());
        )*

        Example::new(
            fields,
            vec![],
            vec![],
        )
    }};
}
