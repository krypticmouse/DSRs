use anyhow::Result;
use serde_json::{Value, json};
use std::collections::HashMap;

use crate::core::{Chat, LM, Message, MetaSignature};
use crate::data::{example::Example, prediction::Prediction};
use crate::utils::get_iter_from_value;

#[derive(Default, Clone)]
pub struct ChatAdapter;

fn get_type_hint(field: &Value) -> String {
    if field["schema"].as_str().unwrap().is_empty()
        && field["data_type"].as_str().unwrap() == "String"
    {
        String::new()
    } else {
        format!(
            " (must be formatted as valid Rust {})",
            field["data_type"].as_str().unwrap()
        )
    }
}

impl ChatAdapter {
    fn get_field_attribute_list(
        &self,
        field_iter: impl Iterator<Item = (String, Value)>,
    ) -> String {
        let mut field_attributes = String::new();
        for (i, (field_name, field)) in field_iter.enumerate() {
            let data_type = field["data_type"].as_str().unwrap();
            let desc = field["desc"].as_str().unwrap();

            field_attributes.push_str(format!("{}. `{field_name}` ({data_type})", i + 1).as_str());
            if !desc.is_empty() {
                field_attributes.push_str(format!(": {desc}").as_str());
            }
            field_attributes.push('\n');
        }
        field_attributes
    }

    fn get_field_structure(&self, field_iter: impl Iterator<Item = (String, Value)>) -> String {
        let mut field_structure = String::new();
        for (field_name, field) in field_iter {
            let schema_str = field["schema"].as_str().unwrap();
            let data_type = field["data_type"].as_str().unwrap();

            let schema_prompt = if schema_str.is_empty() && data_type == "String" {
                "".to_string()
            } else if !schema_str.is_empty() {
                format!(
                    "\t# note: the value you produce must adhere to the JSON schema: {schema_str}"
                )
            } else {
                format!("\t# note: the value you produce must be a single {data_type} value",)
            };

            field_structure.push_str(
                format!("[[ ## {field_name} ## ]]\n{field_name}{schema_prompt}\n\n").as_str(),
            );
        }
        field_structure
    }

    fn format_system_message(&self, signature: &dyn MetaSignature) -> String {
        let field_description = self.format_field_description(signature);
        let field_structure = self.format_field_structure(signature);
        let task_description = self.format_task_description(signature);

        format!("{field_description}\n{field_structure}\n{task_description}")
    }

    fn format_field_description(&self, signature: &dyn MetaSignature) -> String {
        let input_field_description =
            self.get_field_attribute_list(get_iter_from_value(&signature.input_fields()));
        let output_field_description =
            self.get_field_attribute_list(get_iter_from_value(&signature.output_fields()));

        format!(
            "Your input fields are:\n{input_field_description}\nYour output fields are:\n{output_field_description}"
        )
    }

    fn format_field_structure(&self, signature: &dyn MetaSignature) -> String {
        let input_field_structure =
            self.get_field_structure(get_iter_from_value(&signature.input_fields()));
        let output_field_structure =
            self.get_field_structure(get_iter_from_value(&signature.output_fields()));

        format!(
            "All interactions will be structured in the following way, with the appropriate values filled in.\n\n{input_field_structure}{output_field_structure}[[ ## completed ## ]]\n"
        )
    }

    fn format_task_description(&self, signature: &dyn MetaSignature) -> String {
        let instruction = if signature.instruction().is_empty() {
            format!(
                "Given the fields {}, produce the fields {}.",
                signature
                    .input_fields()
                    .as_object()
                    .unwrap()
                    .keys()
                    .map(|k| format!("`{k}`"))
                    .collect::<Vec<String>>()
                    .join(", "),
                signature
                    .output_fields()
                    .as_object()
                    .unwrap()
                    .keys()
                    .map(|k| format!("`{k}`"))
                    .collect::<Vec<String>>()
                    .join(", ")
            )
        } else {
            signature.instruction().clone()
        };

        format!("In adhering to this structure, your objective is:\n\t{instruction}")
    }

    fn format_user_message(&self, signature: &dyn MetaSignature, inputs: Example) -> String {
        let mut input_str = String::new();
        for (field_name, _) in get_iter_from_value(&signature.input_fields()) {
            input_str.push_str(
                format!(
                    "[[ ## {field_name} ## ]]\n{field_value}\n\n",
                    field_name = field_name,
                    field_value = inputs.get(field_name.as_str(), None)
                )
                .as_str(),
            );
        }

        let first_output_field = signature
            .output_fields()
            .as_object()
            .unwrap()
            .keys()
            .next()
            .unwrap()
            .clone();
        let first_output_field_value = signature
            .output_fields()
            .as_object()
            .unwrap()
            .get(&first_output_field)
            .unwrap()
            .clone();

        let type_hint = get_type_hint(&first_output_field_value);

        let mut user_message = format!(
            "Respond with the corresponding output fields, starting with the field `{first_output_field}`{type_hint},"
        );
        for (field_name, field) in get_iter_from_value(&signature.output_fields()).skip(1) {
            user_message
                .push_str(format!(" then `{field_name}`{},", get_type_hint(&field)).as_str());
        }
        user_message.push_str(" and then ending with the marker for `completed`.");

        format!("{input_str}{user_message}")
    }

    pub fn format(&self, signature: &dyn MetaSignature, inputs: Example) -> Chat {
        let system_message = self.format_system_message(signature);
        let user_message = self.format_user_message(signature, inputs);

        let mut chat = Chat::new(vec![]);
        chat.push("system", &system_message);
        chat.push("user", &user_message);

        chat
    }

    pub fn parse_response(
        &self,
        signature: &dyn MetaSignature,
        response: Message,
    ) -> HashMap<String, Value> {
        let mut output = HashMap::new();

        let response_content = response.content();

        for (field_name, field) in get_iter_from_value(&signature.output_fields()) {
            let field_value = response_content
                .split(format!("[[ ## {field_name} ## ]]\n").as_str())
                .nth(1)
                .unwrap();

            let extracted_field = field_value.split("[[ ## ").nth(0).unwrap().trim();
            let data_type = field["data_type"].as_str().unwrap();
            let schema = field["schema"].as_str().unwrap();

            if schema.is_empty() && data_type == "String" {
                output.insert(field_name.clone(), json!(extracted_field));
            } else {
                output.insert(
                    field_name.clone(),
                    serde_json::from_str(extracted_field).unwrap(),
                );
            }
        }

        output
    }

    pub async fn call(
        &self,
        lm: &mut LM,
        signature: &dyn MetaSignature,
        inputs: Example,
    ) -> Result<Prediction> {
        let messages = self.format(signature, inputs);
        let (response, usage) = lm.call(messages, "predict").await?;
        let output = self.parse_response(signature, response);

        Ok(Prediction {
            data: output,
            lm_usage: usage,
        })
    }
}
