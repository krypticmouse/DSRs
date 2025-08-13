use crate::core::Adapter;
use crate::core::Signature;
use crate::data::{
    example::Example,
    prediction::{LmUsage, Prediction},
};
use crate::internal::{MetaField, MetaSignature};
use serde_json::json;
use std::collections::HashMap;

use async_openai::types::CreateChatCompletionResponse;
use indexmap::IndexMap;

#[derive(Default, Clone)]
pub struct ChatAdapter;

fn get_type_hint(field: &MetaField) -> String {
    if field.schema.is_empty() && field.data_type == "String" {
        String::new()
    } else {
        format!(" (must be formatted as valid Rust {})", field.data_type)
    }
}

impl ChatAdapter {
    fn get_field_attribute_list(&self, field_iter: &IndexMap<String, MetaField>) -> String {
        let mut field_attributes = String::new();
        for (i, (field_name, field)) in field_iter.iter().enumerate() {
            field_attributes
                .push_str(format!("{}. `{field_name}` ({})", i + 1, field.data_type).as_str());
            if !field.desc.is_empty() {
                field_attributes.push_str(format!(": {}", field.desc).as_str());
            }
            field_attributes.push('\n');
        }
        field_attributes
    }

    fn get_field_structure(&self, field_iter: &IndexMap<String, MetaField>) -> String {
        let mut field_structure = String::new();
        for (field_name, field) in field_iter {
            let schema_str = field.schema.clone();
            let schema_prompt = if schema_str.is_empty() && field.data_type == "String" {
                "".to_string()
            } else if !schema_str.is_empty() {
                format!(
                    "\t# note: the value you produce must adhere to the JSON schema: {schema_str}"
                )
            } else {
                format!(
                    "\t# note: the value you produce must be a single {} value",
                    field.data_type
                )
            };

            field_structure.push_str(
                format!("[[ ## {field_name} ## ]]\n{field_name}{schema_prompt}\n\n").as_str(),
            );
        }
        field_structure
    }

    fn parse_response(
        &self,
        signature: &MetaSignature,
        response: CreateChatCompletionResponse,
    ) -> Prediction {
        let mut output = HashMap::new();

        let response_content = response.choices[0].message.content.as_ref().unwrap();

        for (field_name, field) in signature.output_fields.iter() {
            let field_value = response_content
                .split(format!("[[ ## {field_name} ## ]]\n").as_str())
                .nth(1)
                .unwrap();

            let extracted_field = field_value.split("[[ ## ").nth(0).unwrap().trim();

            if field.schema.is_empty() && field.data_type == "String" {
                output.insert(field_name.clone(), json!(extracted_field));
            } else {
                output.insert(
                    field_name.clone(),
                    serde_json::from_str(extracted_field).unwrap(),
                );
            }
        }

        Prediction {
            data: output,
            lm_usage: LmUsage::default(),
        }
    }
}

impl Adapter for ChatAdapter {
    fn format_field_description(&self, signature: &impl Signature) -> String {
        let input_field_description = self.get_field_attribute_list(&signature.input_fields);
        let output_field_description = self.get_field_attribute_list(&signature.output_fields);

        format!(
            "Your input fields are:\n{input_field_description}\nYour output fields are:\n{output_field_description}"
        )
    }

    fn format_field_structure(&self, signature: &impl Signature) -> String {
        let input_field_structure = self.get_field_structure(&signature.input_fields);
        let output_field_structure = self.get_field_structure(&signature.output_fields);

        format!(
            "All interactions will be structured in the following way, with the appropriate values filled in.\n\n{input_field_structure}{output_field_structure}[[ ## completed ## ]]\n"
        )
    }

    fn format_task_description(&self, signature: &impl Signature) -> String {
        let instruction = if signature.instruction.is_empty() {
            format!(
                "Given the fields {}, produce the fields {}.",
                signature
                    .input_fields
                    .keys()
                    .map(|k| format!("`{k}`"))
                    .collect::<Vec<String>>()
                    .join(", "),
                signature
                    .output_fields
                    .keys()
                    .map(|k| format!("`{k}`"))
                    .collect::<Vec<String>>()
                    .join(", ")
            )
        } else {
            signature.instruction.clone()
        };

        format!("In adhering to this structure, your objective is:\n\t{instruction}")
    }

    fn format_user_message(&self, signature: &impl Signature, inputs: Example) -> String {
        let mut input_str = String::new();
        for (field_name, _) in signature.input_fields.iter() {
            input_str.push_str(
                format!(
                    "[[ ## {field_name} ## ]]\n{field_value}\n\n",
                    field_name = field_name,
                    field_value = inputs.get(field_name, None)
                )
                .as_str(),
            );
        }

        let first_output_field = &signature.output_fields.keys()[0];
        let first_output_field_value = signature.output_fields.get(first_output_field).unwrap();

        let type_hint = get_type_hint(first_output_field_value);

        let mut user_message = format!(
            "Respond with the corresponding output fields, starting with the field `{first_output_field}`{type_hint},"
        );
        for (field_name, field) in signature.output_fields.iter().skip(1) {
            user_message
                .push_str(format!(" then `{field_name}`{},", get_type_hint(field)).as_str());
        }
        user_message.push_str(" and then ending with the marker for `completed`.");

        format!("{input_str}{user_message}")
    }
}
