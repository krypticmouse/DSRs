use crate::core::{Adapter, Chat, Message, Signature};
use anyhow::Result;
use serde_json::{Value, json};

#[derive(Default, Clone)]
pub struct ChatAdapter;

fn get_type_hint(field_schema: &Value) -> String {
    if let Some(field_type) = field_schema.get("type").and_then(|t| t.as_str()) {
        if field_type == "string" {
            String::new()
        } else {
            format!(" (must be formatted as valid JSON {})", field_type)
        }
    } else {
        String::new()
    }
}

impl ChatAdapter {
    fn get_field_attribute_list(&self, fields: &[(String, String)]) -> String {
        let mut field_attributes = String::new();
        for (i, (field_name, description)) in fields.iter().enumerate() {
            field_attributes.push_str(format!("{}. `{field_name}`", i + 1).as_str());
            if !description.is_empty() {
                field_attributes.push_str(format!(": {}", description).as_str());
            }
            field_attributes.push('\n');
        }
        field_attributes
    }

    fn get_field_structure(&self, schema: &Value) -> String {
        let mut field_structure = String::new();
        if let Some(properties) = schema.get("properties").and_then(|p| p.as_object()) {
            for (field_name, field_schema) in properties {
                let schema_prompt =
                    if let Some(field_type) = field_schema.get("type").and_then(|t| t.as_str()) {
                        if field_type == "string" {
                            "".to_string()
                        } else {
                            format!(
                                "\t# note: the value you produce must be valid JSON of type {}",
                                field_type
                            )
                        }
                    } else {
                        format!(
                            "\t# note: the value you produce must adhere to the schema: {}",
                            serde_json::to_string(field_schema).unwrap_or_default()
                        )
                    };

                field_structure.push_str(
                    format!("[[ ## {field_name} ## ]]\n{field_name}{schema_prompt}\n\n").as_str(),
                );
            }
        }
        field_structure
    }

    fn format_field_description(&self, signature: &impl Signature) -> String {
        let input_fields = signature.metadata().input_fields();
        let output_fields = signature.metadata().output_fields();
        let input_field_description = self.get_field_attribute_list(&input_fields);
        let output_field_description = self.get_field_attribute_list(&output_fields);

        format!(
            "Your input fields are:\n{input_field_description}\nYour output fields are:\n{output_field_description}"
        )
    }

    fn format_field_structure(&self, signature: &impl Signature) -> String {
        let input_field_structure = self.get_field_structure(&signature.metadata().input_schema);
        let output_field_structure = self.get_field_structure(&signature.metadata().output_schema);

        format!(
            "All interactions will be structured in the following way, with the appropriate values filled in.\n\n{input_field_structure}{output_field_structure}[[ ## completed ## ]]\n"
        )
    }

    fn format_task_description(&self, signature: &impl Signature) -> String {
        let metadata = signature.metadata();
        let instruction = if metadata.instructions.is_empty() {
            let input_fields = metadata.input_fields();
            let output_fields = metadata.output_fields();
            format!(
                "Given the fields {}, produce the fields {}.",
                input_fields
                    .iter()
                    .map(|(name, _)| format!("`{name}`"))
                    .collect::<Vec<String>>()
                    .join(", "),
                output_fields
                    .iter()
                    .map(|(name, _)| format!("`{name}`"))
                    .collect::<Vec<String>>()
                    .join(", ")
            )
        } else {
            metadata.instructions.clone()
        };

        format!("In adhering to this structure, your objective is:\n\t{instruction}")
    }

    fn format_system_message(&self, signature: &impl Signature) -> String {
        let field_description = self.format_field_description(signature);
        let field_structure = self.format_field_structure(signature);
        let task_description = self.format_task_description(signature);

        format!("{field_description}\n\n{task_description}\n\n{field_structure}")
    }

    fn format_user_message<S: Signature>(&self, signature: &S, inputs: &S::Inputs) -> String {
        let mut input_str = String::new();

        // Extract field values using the signature's extract_fields method
        let field_values = signature.extract_fields(inputs);
        let input_fields = signature.metadata().input_fields();

        for ((field_name, _), field_value) in input_fields.iter().zip(field_values.into_iter()) {
            let field_value_str: String = field_value.into();
            input_str.push_str(
                format!(
                    "[[ ## {field_name} ## ]]\n{field_value_str}\n\n",
                    field_name = field_name,
                    field_value_str = field_value_str
                )
                .as_str(),
            );
        }

        let output_fields = signature.metadata().output_fields();
        if let Some((first_field_name, _)) = output_fields.first() {
            let first_field_schema = signature
                .metadata()
                .output_schema
                .get("properties")
                .and_then(|p| p.get(first_field_name))
                .unwrap_or(&Value::Null);

            let type_hint = get_type_hint(first_field_schema);

            let mut user_message = format!(
                "Respond with the corresponding output fields, starting with the field `{first_field_name}`{type_hint},"
            );

            for (field_name, _) in output_fields.iter().skip(1) {
                let field_schema = signature
                    .metadata()
                    .output_schema
                    .get("properties")
                    .and_then(|p| p.get(field_name))
                    .unwrap_or(&Value::Null);
                user_message.push_str(
                    format!(" then `{field_name}`{},", get_type_hint(field_schema)).as_str(),
                );
            }
            user_message.push_str(" and then ending with the marker for `completed`.");

            format!("{input_str}{user_message}")
        } else {
            format!("{input_str}Respond with the completed marker.")
        }
    }

    fn parse_response<S: Signature>(&self, signature: &S, response: Message) -> Result<S::Outputs> {
        let response_content = response.content();

        let mut output_json = serde_json::Map::new();
        let output_fields = signature.metadata().output_fields();

        for (field_name, _) in output_fields.iter() {
            let field_marker = format!("[[ ## {field_name} ## ]]\n");
            let field_value = response_content
                .split(&field_marker)
                .nth(1)
                .ok_or_else(|| anyhow::anyhow!("Field '{}' not found in response", field_name))?;

            let extracted_field = field_value
                .split("[[ ## ")
                .next()
                .unwrap_or(field_value)
                .trim();

            // Get field schema to determine how to parse the value
            let field_schema = signature
                .metadata()
                .output_schema
                .get("properties")
                .and_then(|p| p.get(field_name));

            let parsed_value = if let Some(schema) = field_schema {
                if let Some(field_type) = schema.get("type").and_then(|t| t.as_str()) {
                    match field_type {
                        "string" => json!(extracted_field),
                        _ => serde_json::from_str(extracted_field)
                            .unwrap_or_else(|_| json!(extracted_field)),
                    }
                } else {
                    serde_json::from_str(extracted_field).unwrap_or_else(|_| json!(extracted_field))
                }
            } else {
                json!(extracted_field)
            };

            output_json.insert(field_name.clone(), parsed_value);
        }

        let output_value = Value::Object(output_json);
        serde_json::from_value(output_value)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize output: {}", e))
    }
}

impl Adapter for ChatAdapter {
    fn format<S: Signature>(&self, signature: &S, inputs: &S::Inputs) -> Chat {
        let system_message = self.format_system_message(signature);
        let user_message = self.format_user_message(signature, inputs);

        let mut chat = Chat::new(vec![]);
        chat.push("system", system_message.as_str());
        chat.push("user", user_message.as_str());

        chat
    }

    fn parse<S: Signature>(&self, signature: &S, response: Message) -> Result<S::Outputs> {
        self.parse_response(signature, response)
    }
}
