use anyhow::Result;
use indexmap::IndexMap;
use rig::tool::ToolDyn;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Arc;

use super::Adapter;
use crate::baml_bridge::BamlValueConvert;
use crate::baml_bridge::ToBamlValue;
use crate::baml_bridge::jsonish;
use crate::baml_bridge::jsonish::BamlValueWithFlags;
use crate::baml_bridge::jsonish::deserializer::coercer::run_user_checks;
use crate::baml_bridge::jsonish::deserializer::deserialize_flags::DeserializerConditions;
use crate::serde_utils::get_iter_from_value;
use crate::{
    BamlValue, Cache, Chat, ConstraintLevel, ConstraintResult, Example, FieldMeta, Flag,
    JsonishError, LM, Message, MetaSignature, ParseError, Prediction, RenderOptions, Signature,
};
use crate::utils::cache::CallResult as CacheCallResult;

#[derive(Default, Clone)]
pub struct ChatAdapter;

fn get_type_hint(field: &Value) -> String {
    let schema = &field["schema"];
    let type_str = field["type"].as_str().unwrap_or("String");

    // Check if schema exists and is not empty (either as string or object)
    let has_schema = if let Some(s) = schema.as_str() {
        !s.is_empty()
    } else {
        schema.is_object()
    };

    if !has_schema && type_str == "String" {
        String::new()
    } else {
        format!(" (must be formatted as valid Rust {type_str})")
    }
}

impl ChatAdapter {
    fn get_field_attribute_list(
        &self,
        field_iter: impl Iterator<Item = (String, Value)>,
    ) -> String {
        let mut field_attributes = String::new();
        for (i, (field_name, field)) in field_iter.enumerate() {
            let data_type = field["type"].as_str().unwrap_or("String");
            let desc = field["desc"].as_str().unwrap_or("");

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
            let schema = &field["schema"];
            let data_type = field["type"].as_str().unwrap_or("String");

            // Handle schema as either string or JSON object
            let schema_prompt = if let Some(s) = schema.as_str() {
                if s.is_empty() && data_type == "String" {
                    "".to_string()
                } else if !s.is_empty() {
                    format!("\t# note: the value you produce must adhere to the JSON schema: {s}")
                } else {
                    format!("\t# note: the value you produce must be a single {data_type} value")
                }
            } else if schema.is_object() || schema.is_array() {
                // Convert JSON object/array to string for display
                let schema_str = schema.to_string();
                format!(
                    "\t# note: the value you produce must adhere to the JSON schema: {schema_str}"
                )
            } else if data_type == "String" {
                "".to_string()
            } else {
                format!("\t# note: the value you produce must be a single {data_type} value")
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

    fn format_user_message(&self, signature: &dyn MetaSignature, inputs: &Example) -> String {
        let mut input_str = String::new();
        for (field_name, _) in get_iter_from_value(&signature.input_fields()) {
            let field_value = inputs.get(field_name.as_str(), None);
            // Extract the actual string value if it's a JSON string, otherwise use as is
            let field_value_str = if let Some(s) = field_value.as_str() {
                s.to_string()
            } else {
                field_value.to_string()
            };

            input_str
                .push_str(format!("[[ ## {field_name} ## ]]\n{field_value_str}\n\n",).as_str());
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

    fn format_assistant_message(&self, signature: &dyn MetaSignature, outputs: &Example) -> String {
        let mut assistant_message = String::new();
        for (field_name, _) in get_iter_from_value(&signature.output_fields()) {
            let field_value = outputs.get(field_name.as_str(), None);
            // Extract the actual string value if it's a JSON string, otherwise use as is
            let field_value_str = if let Some(s) = field_value.as_str() {
                s.to_string()
            } else {
                field_value.to_string()
            };

            assistant_message
                .push_str(format!("[[ ## {field_name} ## ]]\n{field_value_str}\n\n",).as_str());
        }
        assistant_message.push_str("[[ ## completed ## ]]\n");
        assistant_message
    }

    fn format_demos(&self, signature: &dyn MetaSignature, demos: &Vec<Example>) -> Chat {
        let mut chat = Chat::new(vec![]);

        for demo in demos {
            let user_message = self.format_user_message(signature, demo);
            let assistant_message = self.format_assistant_message(signature, demo);
            chat.push("user", &user_message);
            chat.push("assistant", &assistant_message);
        }

        chat
    }

    pub fn format_system_message_typed<S: Signature>(&self) -> Result<String> {
        self.format_system_message_typed_with_instruction::<S>(None)
    }

    pub fn format_system_message_typed_with_instruction<S: Signature>(
        &self,
        instruction_override: Option<&str>,
    ) -> Result<String> {
        let mut parts = Vec::new();
        parts.push(self.format_field_descriptions_typed::<S>());
        parts.push(self.format_field_structure_typed::<S>());

        let schema = S::output_format_content()
            .render(RenderOptions::default())?
            .unwrap_or_default();
        if !schema.is_empty() {
            parts.push(format!("Answer in this schema:\n{schema}"));
        }

        let instruction = instruction_override.unwrap_or(S::instruction());
        if !instruction.is_empty() {
            parts.push(instruction.to_string());
        }

        Ok(parts.join("\n\n"))
    }

    fn format_field_descriptions_typed<S: Signature>(&self) -> String {
        let mut lines = Vec::new();
        lines.push("Your input fields are:".to_string());
        for (i, field) in S::input_fields().iter().enumerate() {
            let type_name = (field.type_ir)().diagnostic_repr().to_string();
            let mut line = format!("{}. `{}` ({type_name})", i + 1, field.name);
            if !field.description.is_empty() {
                line.push_str(": ");
                line.push_str(field.description);
            }
            lines.push(line);
        }

        lines.push(String::new());
        lines.push("Your output fields are:".to_string());
        for (i, field) in S::output_fields().iter().enumerate() {
            let type_name = (field.type_ir)().diagnostic_repr().to_string();
            let mut line = format!("{}. `{}` ({type_name})", i + 1, field.name);
            if !field.description.is_empty() {
                line.push_str(": ");
                line.push_str(field.description);
            }
            lines.push(line);
        }

        lines.join("\n")
    }

    fn format_field_structure_typed<S: Signature>(&self) -> String {
        let mut lines = vec![
            "All interactions will be structured in the following way, with the appropriate values filled in.".to_string(),
            String::new(),
        ];

        for field in S::input_fields() {
            lines.push(format!("[[ ## {} ## ]]", field.name));
            lines.push(field.name.to_string());
            lines.push(String::new());
        }

        for field in S::output_fields() {
            lines.push(format!("[[ ## {} ## ]]", field.name));
            lines.push(field.name.to_string());
            lines.push(String::new());
        }

        lines.push("[[ ## completed ## ]]".to_string());

        lines.join("\n")
    }

    pub fn format_user_message_typed<S: Signature>(&self, input: &S::Input) -> String
    where
        S::Input: ToBamlValue,
    {
        let baml_value = input.to_baml_value();
        let Some(fields) = baml_value_fields(&baml_value) else {
            return String::new();
        };

        let mut result = String::new();
        for field_spec in S::input_fields() {
            if let Some(value) = fields.get(field_spec.rust_name) {
                result.push_str(&format!("[[ ## {} ## ]]\n", field_spec.name));
                result.push_str(&format_baml_value_for_prompt(value));
                result.push_str("\n\n");
            }
        }

        result
    }

    pub fn format_assistant_message_typed<S: Signature>(&self, output: &S::Output) -> String
    where
        S::Output: ToBamlValue,
    {
        let baml_value = output.to_baml_value();
        let Some(fields) = baml_value_fields(&baml_value) else {
            return String::new();
        };

        let mut result = String::new();
        for field_spec in S::output_fields() {
            if let Some(value) = fields.get(field_spec.rust_name) {
                result.push_str(&format!("[[ ## {} ## ]]\n", field_spec.name));
                result.push_str(&format_baml_value_for_prompt(value));
                result.push_str("\n\n");
            }
        }
        result.push_str("[[ ## completed ## ]]\n");

        result
    }

    pub fn format_demo_typed<S: Signature>(&self, demo: S) -> (String, String)
    where
        S::Input: ToBamlValue,
        S::Output: ToBamlValue,
    {
        let (input, output) = demo.into_parts();
        let user_msg = self.format_user_message_typed::<S>(&input);
        let assistant_msg = self.format_assistant_message_typed::<S>(&output);
        (user_msg, assistant_msg)
    }

    #[allow(clippy::result_large_err)]
    pub fn parse_response_typed<S: Signature>(
        &self,
        response: &Message,
    ) -> std::result::Result<(S::Output, IndexMap<String, FieldMeta>), ParseError> {
        let content = response.content();
        let output_format = S::output_format_content();

        let mut metas = IndexMap::new();
        let mut errors = Vec::new();
        let mut output_map = crate::baml_bridge::baml_types::BamlMap::new();

        for field in S::output_fields() {
            let rust_name = field.rust_name.to_string();
            let type_ir = (field.type_ir)();

            let raw_text = match extract_field(&content, field.name) {
                Ok(text) => text,
                Err(_) => {
                    errors.push(ParseError::MissingField {
                        field: rust_name.clone(),
                        raw_response: content.to_string(),
                    });
                    continue;
                }
            };

            let parsed: BamlValueWithFlags =
                match jsonish::from_str(output_format, &type_ir, &raw_text, true) {
                    Ok(value) => value,
                    Err(err) => {
                        errors.push(ParseError::CoercionFailed {
                            field: rust_name.clone(),
                            expected_type: type_ir.diagnostic_repr().to_string(),
                            raw_text: raw_text.clone(),
                            source: JsonishError::from(err),
                        });
                        continue;
                    }
                };

            let baml_value: BamlValue = parsed.clone().into();

            let mut flags = Vec::new();
            collect_flags(&parsed, &mut flags);

            let mut checks = Vec::new();
            match run_user_checks(&baml_value, &type_ir) {
                Ok(results) => {
                    for (constraint, passed) in results {
                        let label = constraint.label.as_deref().unwrap_or_else(|| {
                            if constraint.level == ConstraintLevel::Assert {
                                "assert"
                            } else {
                                "check"
                            }
                        });
                        let expression = constraint.expression.to_string();
                        if constraint.level == ConstraintLevel::Assert && !passed {
                            errors.push(ParseError::AssertFailed {
                                field: rust_name.clone(),
                                label: label.to_string(),
                                expression: expression.clone(),
                                value: baml_value.clone(),
                            });
                        }
                        if constraint.level == ConstraintLevel::Check {
                            checks.push(ConstraintResult {
                                label: label.to_string(),
                                expression,
                                passed,
                            });
                        }
                    }
                }
                Err(err) => {
                    errors.push(ParseError::ExtractionFailed {
                        field: rust_name.clone(),
                        raw_response: content.to_string(),
                        reason: err.to_string(),
                    });
                    continue;
                }
            }

            metas.insert(
                rust_name.clone(),
                FieldMeta {
                    raw_text,
                    flags,
                    checks,
                },
            );

            output_map.insert(rust_name, baml_value);
        }

        if !errors.is_empty() {
            let partial = if output_map.is_empty() {
                None
            } else {
                Some(BamlValue::Map(output_map))
            };
            return Err(ParseError::Multiple { errors, partial });
        }

        let typed_output = <S::Output as BamlValueConvert>::try_from_baml_value(
            BamlValue::Map(output_map),
            Vec::new(),
        )
        .map_err(|err| ParseError::ExtractionFailed {
            field: "<all>".to_string(),
            raw_response: content.to_string(),
            reason: err.to_string(),
        })?;

        Ok((typed_output, metas))
    }
}

fn extract_field(content: &str, field_name: &str) -> std::result::Result<String, String> {
    let start_marker = format!("[[ ## {} ## ]]", field_name);
    let start_pos = content
        .find(&start_marker)
        .ok_or_else(|| format!("marker not found: {start_marker}"))?;
    let after_marker = start_pos + start_marker.len();
    let remaining = &content[after_marker..];
    let end_pos = remaining.find("[[ ##").unwrap_or(remaining.len());
    let extracted = remaining[..end_pos].trim();
    Ok(extracted.to_string())
}

fn baml_value_fields(
    value: &BamlValue,
) -> Option<&crate::baml_bridge::baml_types::BamlMap<String, BamlValue>> {
    match value {
        BamlValue::Class(_, fields) => Some(fields),
        BamlValue::Map(fields) => Some(fields),
        _ => None,
    }
}

fn format_baml_value_for_prompt(value: &BamlValue) -> String {
    match value {
        BamlValue::String(s) => s.clone(),
        BamlValue::Null => "null".to_string(),
        other => serde_json::to_string(other).unwrap_or_else(|_| "<error>".to_string()),
    }
}

fn collect_flags(value: &BamlValueWithFlags, flags: &mut Vec<Flag>) {
    collect_flags_recursive(value, flags);
}

fn collect_flags_recursive(value: &BamlValueWithFlags, flags: &mut Vec<Flag>) {
    match value {
        BamlValueWithFlags::String(v) => {
            collect_from_conditions(&v.flags, flags);
        }
        BamlValueWithFlags::Int(v) => {
            collect_from_conditions(&v.flags, flags);
        }
        BamlValueWithFlags::Float(v) => {
            collect_from_conditions(&v.flags, flags);
        }
        BamlValueWithFlags::Bool(v) => {
            collect_from_conditions(&v.flags, flags);
        }
        BamlValueWithFlags::Enum(_, _, v) => {
            collect_from_conditions(&v.flags, flags);
        }
        BamlValueWithFlags::Media(_, v) => {
            collect_from_conditions(&v.flags, flags);
        }
        BamlValueWithFlags::List(conds, _, items) => {
            collect_from_conditions(conds, flags);
            for item in items {
                collect_flags_recursive(item, flags);
            }
        }
        BamlValueWithFlags::Map(conds, _, items) => {
            collect_from_conditions(conds, flags);
            for (_, (entry_flags, entry_value)) in items {
                collect_from_conditions(entry_flags, flags);
                collect_flags_recursive(entry_value, flags);
            }
        }
        BamlValueWithFlags::Class(_, conds, _, fields) => {
            collect_from_conditions(conds, flags);
            for (_, field_value) in fields {
                collect_flags_recursive(field_value, flags);
            }
        }
        BamlValueWithFlags::Null(_, conds) => {
            collect_from_conditions(conds, flags);
        }
    }
}

fn collect_from_conditions(conditions: &DeserializerConditions, flags: &mut Vec<Flag>) {
    flags.extend(conditions.flags.iter().cloned());
}

#[async_trait::async_trait]
impl Adapter for ChatAdapter {
    fn format(&self, signature: &dyn MetaSignature, inputs: Example) -> Chat {
        let system_message = self.format_system_message(signature);
        let user_message = self.format_user_message(signature, &inputs);

        let demos = signature.demos();
        let demos = self.format_demos(signature, &demos);

        let mut chat = Chat::new(vec![]);
        chat.push("system", &system_message);
        chat.push_all(&demos);
        chat.push("user", &user_message);

        chat
    }

    fn parse_response(
        &self,
        signature: &dyn MetaSignature,
        response: Message,
    ) -> HashMap<String, Value> {
        let mut output = HashMap::new();

        let response_content = response.content();

        for (field_name, field) in get_iter_from_value(&signature.output_fields()) {
            let field_value = response_content
                .split(format!("[[ ## {field_name} ## ]]\n").as_str())
                .nth(1);

            if field_value.is_none() {
                continue; // Skip field if not found in response
            }
            let field_value = field_value.unwrap();

            let extracted_field = field_value.split("[[ ## ").nth(0).unwrap().trim();
            let data_type = field["type"].as_str().unwrap();
            let schema = &field["schema"];

            // Check if schema exists (as string or object)
            let has_schema = if let Some(s) = schema.as_str() {
                !s.is_empty()
            } else {
                schema.is_object() || schema.is_array()
            };

            if !has_schema && data_type == "String" {
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

    async fn call(
        &self,
        lm: Arc<LM>,
        signature: &dyn MetaSignature,
        inputs: Example,
        tools: Vec<Arc<dyn ToolDyn>>,
    ) -> Result<Prediction> {
        // Check cache first (release lock immediately after checking)
        if lm.cache
            && let Some(cache) = lm.cache_handler.as_ref()
        {
            let cache_key = inputs.clone();
            if let Some(cached) = cache.lock().await.get(cache_key).await? {
                return Ok(cached);
            }
        }

        let messages = self.format(signature, inputs.clone());
        let response = lm.call(messages, tools).await?;
        let prompt_str = response.chat.to_json().to_string();

        let mut output = self.parse_response(signature, response.output);
        if !response.tool_calls.is_empty() {
            output.insert(
                "tool_calls".to_string(),
                response
                    .tool_calls
                    .into_iter()
                    .map(|call| json!(call))
                    .collect::<Value>(),
            );
            output.insert(
                "tool_executions".to_string(),
                response
                    .tool_executions
                    .into_iter()
                    .map(|execution| json!(execution))
                    .collect::<Value>(),
            );
        }

        let prediction = Prediction {
            data: output,
            lm_usage: response.usage,
            node_id: None,
        };

        // Store in cache if enabled
        if lm.cache
            && let Some(cache) = lm.cache_handler.as_ref()
        {
            let (tx, rx) = tokio::sync::mpsc::channel(1);
            let cache_clone = cache.clone();
            let inputs_clone = inputs.clone();

            // Spawn the cache insert operation to avoid deadlock
            tokio::spawn(async move {
                let _ = cache_clone.lock().await.insert(inputs_clone, rx).await;
            });

            // Send the result to the cache
            tx.send(CacheCallResult {
                prompt: prompt_str,
                prediction: prediction.clone(),
            })
            .await
            .map_err(|_| anyhow::anyhow!("Failed to send to cache"))?;
        }

        Ok(prediction)
    }
}
