use anyhow::Result;
use bamltype::jsonish;
use bamltype::jsonish::BamlValueWithFlags;
use bamltype::jsonish::deserializer::coercer::run_user_checks;
use bamltype::jsonish::deserializer::deserialize_flags::DeserializerConditions;
use indexmap::IndexMap;
use regex::Regex;
use rig::tool::ToolDyn;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::{Arc, LazyLock};
use tracing::{Instrument, debug, trace};

use super::Adapter;
use crate::serde_utils::get_iter_from_value;
use crate::utils::cache::CacheEntry;
use crate::{
    BamlType, BamlValue, Cache, Chat, ConstraintLevel, ConstraintResult, Example, FieldMeta, Flag,
    JsonishError, LM, Message, MetaSignature, OutputFormatContent, ParseError, Prediction,
    RenderOptions, Signature, TypeIR,
};

#[derive(Default, Clone)]
pub struct ChatAdapter;

static FIELD_HEADER_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\[\[ ## (\w+) ## \]\]").unwrap());

fn render_field_type_schema(
    parent_format: &OutputFormatContent,
    type_ir: &TypeIR,
) -> Result<String> {
    let field_format = OutputFormatContent {
        enums: parent_format.enums.clone(),
        classes: parent_format.classes.clone(),
        recursive_classes: parent_format.recursive_classes.clone(),
        structural_recursive_aliases: parent_format.structural_recursive_aliases.clone(),
        target: type_ir.clone(),
    };

    let schema = field_format
        .render(RenderOptions::default().with_prefix(None))?
        .unwrap_or_else(|| type_ir.diagnostic_repr().to_string());

    Ok(schema)
}

fn resolve_rendered_type_token(token: &str, output_format: Option<&OutputFormatContent>) -> String {
    if let Some(output_format) = output_format {
        if let Some(class) = output_format
            .classes
            .iter()
            .find_map(|((name, _), class)| (name == token).then_some(class))
        {
            return class.name.rendered_name().to_string();
        }

        if let Some(enm) = output_format.enums.get(token) {
            return enm.name.rendered_name().to_string();
        }
    }

    token.rsplit("::").next().unwrap_or(token).to_string()
}

fn simplify_type_name(raw: &str, output_format: Option<&OutputFormatContent>) -> String {
    let mut result = String::with_capacity(raw.len());
    let mut chars = raw.chars();
    while let Some(ch) = chars.next() {
        if ch == '`' {
            let mut token = String::new();
            for next in chars.by_ref() {
                if next == '`' {
                    break;
                }
                token.push(next);
            }
            let rendered = resolve_rendered_type_token(&token, output_format);
            result.push_str(&rendered);
        } else {
            result.push(ch);
        }
    }
    result
}

fn render_type_name_for_prompt(
    type_ir: &TypeIR,
    output_format: Option<&OutputFormatContent>,
) -> String {
    let raw = type_ir.diagnostic_repr().to_string();
    let simplified = simplify_type_name(&raw, output_format);
    simplified
        .replace("class ", "")
        .replace("enum ", "")
        .replace(" | ", " or ")
        .trim()
        .to_string()
}

fn split_schema_definitions(schema: &str) -> Option<(String, String)> {
    let lines: Vec<&str> = schema.lines().collect();
    let mut index = 0;
    let mut definitions = Vec::new();
    let mut parsed_any = false;

    while index < lines.len() {
        let start_index = index;

        while index < lines.len() && lines[index].trim().is_empty() {
            index += 1;
        }

        while index < lines.len() && lines[index].trim_start().starts_with("//") {
            index += 1;
        }

        while index < lines.len() && lines[index].trim().is_empty() {
            index += 1;
        }

        if index >= lines.len() {
            break;
        }

        let name_line = lines[index].trim();
        if name_line.is_empty() {
            break;
        }
        index += 1;

        if index >= lines.len() || lines[index].trim() != "----" {
            index = start_index;
            break;
        }
        index += 1;

        let mut values_found = 0;
        while index < lines.len() {
            let trimmed = lines[index].trim_start();
            if trimmed.is_empty() {
                break;
            }
            if trimmed.starts_with('-') {
                values_found += 1;
                index += 1;
                continue;
            }
            break;
        }

        if values_found == 0 {
            index = start_index;
            break;
        }

        let mut block_end = index;
        if index < lines.len() && lines[index].trim().is_empty() {
            index += 1;
            block_end = index;
        }

        definitions.extend_from_slice(&lines[start_index..block_end]);
        parsed_any = true;
    }

    if !parsed_any {
        return None;
    }

    let mut main_lines = Vec::new();
    if index < lines.len() {
        main_lines.extend_from_slice(&lines[index..]);
    }

    let defs = definitions.join("\n").trim_end().to_string();
    let main = main_lines.join("\n").trim_start().to_string();
    if defs.is_empty() || main.is_empty() {
        None
    } else {
        Some((defs, main))
    }
}

fn format_schema_for_prompt(schema: &str) -> String {
    let Some((definitions, main)) = split_schema_definitions(schema) else {
        return schema.to_string();
    };

    format!("Definitions (used below):\n\n{definitions}\n\n{main}")
}

impl ChatAdapter {
    fn format_task_description_typed<S: Signature>(
        &self,
        instruction_override: Option<&str>,
    ) -> String {
        let instruction = instruction_override.unwrap_or(S::instruction());
        let instruction = if instruction.is_empty() {
            let input_fields = S::input_fields()
                .iter()
                .map(|field| format!("`{}`", field.name))
                .collect::<Vec<_>>()
                .join(", ");
            let output_fields = S::output_fields()
                .iter()
                .map(|field| format!("`{}`", field.name))
                .collect::<Vec<_>>()
                .join(", ");
            format!("Given the fields {input_fields}, produce the fields {output_fields}.")
        } else {
            instruction.to_string()
        };

        let mut indented = String::new();
        for line in instruction.lines() {
            indented.push('\n');
            indented.push_str("        ");
            indented.push_str(line);
        }

        format!("In adhering to this structure, your objective is: {indented}")
    }

    fn format_response_instructions_typed<S: Signature>(&self) -> String {
        let mut output_fields = S::output_fields().iter();
        let Some(first_field) = output_fields.next() else {
            return "Respond with the marker for `[[ ## completed ## ]]`.".to_string();
        };

        let mut message = format!(
            "Respond with the corresponding output fields, starting with the field `[[ ## {} ## ]]`,",
            first_field.name
        );
        for field in output_fields {
            message.push_str(&format!(" then `[[ ## {} ## ]]`,", field.name));
        }
        message.push_str(" and then ending with the marker for `[[ ## completed ## ]]`.");

        message
    }

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

        let mut indented = String::new();
        for line in instruction.lines() {
            indented.push('\n');
            indented.push_str("        ");
            indented.push_str(line);
        }

        format!("In adhering to this structure, your objective is: {indented}")
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
        let mut user_message = format!(
            "Respond with the corresponding output fields, starting with the field `[[ ## {first_output_field} ## ]]`,"
        );
        for (field_name, _) in get_iter_from_value(&signature.output_fields()).skip(1) {
            user_message.push_str(format!(" then `[[ ## {field_name} ## ]]`,").as_str());
        }
        user_message.push_str(" and then ending with the marker for `[[ ## completed ## ]]`.");

        format!("{input_str}{user_message}")
    }

    fn format_assistant_message(&self, signature: &dyn MetaSignature, outputs: &Example) -> String {
        let mut sections = Vec::new();
        for (field_name, _) in get_iter_from_value(&signature.output_fields()) {
            let field_value = outputs.get(field_name.as_str(), None);
            // Extract the actual string value if it's a JSON string, otherwise use as is
            let field_value_str = if let Some(s) = field_value.as_str() {
                s.to_string()
            } else {
                field_value.to_string()
            };

            sections.push(format!("[[ ## {field_name} ## ]]\n{field_value_str}"));
        }
        let mut assistant_message = sections.join("\n\n");
        assistant_message.push_str("\n\n[[ ## completed ## ]]\n");
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

    #[tracing::instrument(
        name = "dsrs.adapter.chat.format_system_typed",
        level = "trace",
        skip(self),
        fields(
            signature = std::any::type_name::<S>(),
            instruction_override = instruction_override.is_some()
        )
    )]
    pub fn format_system_message_typed_with_instruction<S: Signature>(
        &self,
        instruction_override: Option<&str>,
    ) -> Result<String> {
        let parts = [
            self.format_field_descriptions_typed::<S>(),
            self.format_field_structure_typed::<S>()?,
            self.format_response_instructions_typed::<S>(),
            self.format_task_description_typed::<S>(instruction_override),
        ];

        let system = parts.join("\n\n");
        trace!(system_len = system.len(), "formatted typed system prompt");
        Ok(system)
    }

    fn format_field_descriptions_typed<S: Signature>(&self) -> String {
        let input_format = <S::Input as BamlType>::baml_output_format();
        let output_format = S::output_format_content();

        let mut lines = Vec::new();
        lines.push("Your input fields are:".to_string());
        for (i, field) in S::input_fields().iter().enumerate() {
            let type_name = render_type_name_for_prompt(&(field.type_ir)(), Some(input_format));
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
            let type_name = render_type_name_for_prompt(&(field.type_ir)(), Some(output_format));
            let mut line = format!("{}. `{}` ({type_name})", i + 1, field.name);
            if !field.description.is_empty() {
                line.push_str(": ");
                line.push_str(field.description);
            }
            lines.push(line);
        }

        lines.join("\n")
    }

    fn format_field_structure_typed<S: Signature>(&self) -> Result<String> {
        let mut lines = vec![
            "All interactions will be structured in the following way, with the appropriate values filled in.".to_string(),
            String::new(),
        ];

        for field in S::input_fields() {
            lines.push(format!("[[ ## {} ## ]]", field.name));
            lines.push(field.name.to_string());
            lines.push(String::new());
        }

        let parent_format = S::output_format_content();
        for field in S::output_fields() {
            let type_ir = (field.type_ir)();
            let type_name = render_type_name_for_prompt(&type_ir, Some(parent_format));
            let schema = render_field_type_schema(parent_format, &type_ir)?;
            lines.push(format!("[[ ## {} ## ]]", field.name));
            lines.push(format!(
                "Output field `{}` should be of type: {type_name}",
                field.name
            ));
            if !schema.is_empty() && schema != type_name {
                lines.push(String::new());
                lines.push(format_schema_for_prompt(&schema));
            }
            lines.push(String::new());
        }

        lines.push("[[ ## completed ## ]]".to_string());

        Ok(lines.join("\n"))
    }

    pub fn format_user_message_typed<S: Signature>(&self, input: &S::Input) -> String
    where
        S::Input: BamlType,
    {
        let baml_value = input.to_baml_value();
        let Some(fields) = baml_value_fields(&baml_value) else {
            return String::new();
        };
        let input_output_format = <S::Input as BamlType>::baml_output_format();

        let mut result = String::new();
        for field_spec in S::input_fields() {
            if let Some(value) = fields.get(field_spec.rust_name) {
                result.push_str(&format!("[[ ## {} ## ]]\n", field_spec.name));
                result.push_str(&format_baml_value_for_prompt_typed(
                    value,
                    input_output_format,
                    field_spec.format,
                ));
                result.push_str("\n\n");
            }
        }

        result
    }

    pub fn format_assistant_message_typed<S: Signature>(&self, output: &S::Output) -> String
    where
        S::Output: BamlType,
    {
        let baml_value = output.to_baml_value();
        let Some(fields) = baml_value_fields(&baml_value) else {
            return String::new();
        };

        let mut sections = Vec::new();
        for field_spec in S::output_fields() {
            if let Some(value) = fields.get(field_spec.rust_name) {
                sections.push(format!(
                    "[[ ## {} ## ]]\n{}",
                    field_spec.name,
                    format_baml_value_for_prompt(value)
                ));
            }
        }
        let mut result = sections.join("\n\n");
        result.push_str("\n\n[[ ## completed ## ]]\n");

        result
    }

    pub fn format_demo_typed<S: Signature>(&self, demo: S) -> (String, String)
    where
        S::Input: BamlType,
        S::Output: BamlType,
    {
        let (input, output) = demo.into_parts();
        let user_msg = self.format_user_message_typed::<S>(&input);
        let assistant_msg = self.format_assistant_message_typed::<S>(&output);
        (user_msg, assistant_msg)
    }

    #[allow(clippy::result_large_err)]
    #[tracing::instrument(
        name = "dsrs.adapter.chat.parse_typed",
        level = "debug",
        skip(self, response),
        fields(
            signature = std::any::type_name::<S>(),
            output_field_count = S::output_fields().len()
        )
    )]
    pub fn parse_response_typed<S: Signature>(
        &self,
        response: &Message,
    ) -> std::result::Result<(S::Output, IndexMap<String, FieldMeta>), ParseError> {
        let content = response.content();
        let output_format = S::output_format_content();
        let sections = parse_sections(&content);

        let mut metas = IndexMap::new();
        let mut errors = Vec::new();
        let mut output_map = bamltype::baml_types::BamlMap::new();
        let mut checks_total = 0usize;
        let mut checks_failed = 0usize;
        let mut asserts_failed = 0usize;

        for field in S::output_fields() {
            let rust_name = field.rust_name.to_string();
            let type_ir = (field.type_ir)();

            let raw_text = match sections.get(field.name) {
                Some(text) => text.clone(),
                None => {
                    debug!(field = %rust_name, "missing output field in response");
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
                        debug!(
                            field = %rust_name,
                            expected_type = %type_ir.diagnostic_repr(),
                            raw_text_len = raw_text.len(),
                            "typed coercion failed"
                        );
                        trace!(
                            field = %rust_name,
                            raw_preview = %crate::truncate(&raw_text, 160),
                            "typed coercion failed preview"
                        );
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
            collect_flags_recursive(&parsed, &mut flags);

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
                            asserts_failed += 1;
                            debug!(
                                field = %rust_name,
                                label,
                                "typed assert constraint failed"
                            );
                            errors.push(ParseError::AssertFailed {
                                field: rust_name.clone(),
                                label: label.to_string(),
                                expression: expression.clone(),
                                value: baml_value.clone(),
                            });
                        }
                        if constraint.level == ConstraintLevel::Check {
                            checks_total += 1;
                            if !passed {
                                checks_failed += 1;
                                trace!(
                                    field = %rust_name,
                                    label,
                                    "typed check constraint failed"
                                );
                            }
                            checks.push(ConstraintResult {
                                label: label.to_string(),
                                expression,
                                passed,
                            });
                        }
                    }
                }
                Err(err) => {
                    debug!(
                        field = %rust_name,
                        reason = %err,
                        "typed extraction failed while running checks"
                    );
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
            debug!(
                errors = errors.len(),
                checks_total, checks_failed, asserts_failed, "typed parse returned errors"
            );
            let partial = if output_map.is_empty() {
                None
            } else {
                Some(BamlValue::Class(
                    <S::Output as BamlType>::baml_internal_name().to_string(),
                    output_map,
                ))
            };
            return Err(ParseError::Multiple { errors, partial });
        }

        let typed_output = <S::Output as BamlType>::try_from_baml_value(BamlValue::Class(
            <S::Output as BamlType>::baml_internal_name().to_string(),
            output_map,
        ))
        .map_err(|err| ParseError::ExtractionFailed {
            field: "<all>".to_string(),
            raw_response: content.to_string(),
            reason: err.to_string(),
        })?;
        debug!(
            parsed_fields = metas.len(),
            checks_total, checks_failed, asserts_failed, "typed parse completed"
        );

        Ok((typed_output, metas))
    }

    #[tracing::instrument(
        name = "dsrs.adapter.chat.parse",
        level = "debug",
        skip(self, signature, response),
        fields(
            output_field_count = signature
                .output_fields()
                .as_object()
                .map(|fields| fields.len())
                .unwrap_or_default()
        )
    )]
    fn parse_response_strict(
        &self,
        signature: &dyn MetaSignature,
        response: Message,
    ) -> Result<HashMap<String, Value>> {
        let mut output = HashMap::new();

        let response_content = response.content();
        let sections = parse_sections(&response_content);

        for (field_name, field) in get_iter_from_value(&signature.output_fields()) {
            let Some(field_value) = sections.get(&field_name) else {
                debug!(
                    field = %field_name,
                    "legacy parse missing required output field"
                );
                return Err(anyhow::anyhow!(
                    "missing required field `{}` in model output",
                    field_name
                ));
            };
            let extracted_field = field_value.as_str();
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
                let value = serde_json::from_str(extracted_field).map_err(|err| {
                    debug!(
                        field = %field_name,
                        data_type,
                        raw_text_len = extracted_field.len(),
                        error = %err,
                        "legacy parse json coercion failed"
                    );
                    anyhow::anyhow!(
                        "failed to parse field `{}` as {} from model output: {}",
                        field_name,
                        data_type,
                        err
                    )
                })?;
                output.insert(field_name.clone(), value);
            }
        }

        debug!(parsed_fields = output.len(), "legacy parse completed");
        Ok(output)
    }
}

fn parse_sections(content: &str) -> IndexMap<String, String> {
    let mut sections: Vec<(Option<String>, Vec<String>)> = vec![(None, Vec::new())];

    for line in content.lines() {
        let trimmed = line.trim();
        if let Some(caps) = FIELD_HEADER_PATTERN.captures(trimmed) {
            let header = caps.get(1).unwrap().as_str().to_string();
            let marker = caps.get(0).unwrap();
            let remaining = trimmed[marker.end()..].trim();

            let mut lines = Vec::new();
            if !remaining.is_empty() {
                lines.push(remaining.to_string());
            }
            sections.push((Some(header), lines));
        } else if let Some((_, lines)) = sections.last_mut() {
            lines.push(line.to_string());
        }
    }

    let mut parsed = IndexMap::new();
    for (header, lines) in sections {
        let Some(name) = header else {
            continue;
        };
        if parsed.contains_key(&name) {
            continue;
        }
        parsed.insert(name, lines.join("\n").trim().to_string());
    }

    parsed
}

fn baml_value_fields(
    value: &BamlValue,
) -> Option<&bamltype::baml_types::BamlMap<String, BamlValue>> {
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

fn format_baml_value_for_prompt_typed(
    value: &BamlValue,
    output_format: &OutputFormatContent,
    format: Option<&str>,
) -> String {
    let format = match format {
        Some(format) => format,
        None => {
            if let BamlValue::String(s) = value {
                return s.clone();
            }
            "json"
        }
    };

    bamltype::internal_baml_jinja::format_baml_value(value, output_format, format)
        .unwrap_or_else(|_| "<error>".to_string())
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
    #[tracing::instrument(
        name = "dsrs.adapter.chat.format",
        level = "trace",
        skip(self, signature, inputs),
        fields(
            input_fields = inputs.input_keys.len(),
            output_fields = inputs.output_keys.len()
        )
    )]
    fn format(&self, signature: &dyn MetaSignature, inputs: Example) -> Chat {
        let system_message = self.format_system_message(signature);
        let user_message = self.format_user_message(signature, &inputs);

        let demo_examples = signature.demos();
        let demos = self.format_demos(signature, &demo_examples);

        let mut chat = Chat::new(vec![]);
        chat.push("system", &system_message);
        chat.push_all(&demos);
        chat.push("user", &user_message);

        trace!(
            demo_count = demo_examples.len(),
            system_len = system_message.len(),
            user_len = user_message.len(),
            message_count = chat.len(),
            "legacy prompt formatted"
        );

        chat
    }

    fn parse_response(
        &self,
        signature: &dyn MetaSignature,
        response: Message,
    ) -> HashMap<String, Value> {
        self.parse_response_strict(signature, response)
            .unwrap_or_else(|err| panic!("legacy parse failed: {err}"))
    }

    #[tracing::instrument(
        name = "dsrs.adapter.chat.call",
        level = "debug",
        skip(self, lm, signature, inputs, tools),
        fields(
            cache_enabled = lm.cache,
            tool_count = tools.len(),
            input_field_count = inputs.data.len()
        )
    )]
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
                debug!(
                    cache_hit = true,
                    output_fields = cached.data.len(),
                    "adapter cache hit"
                );
                return Ok(cached);
            }
            debug!(cache_hit = false, "adapter cache miss");
        }
        let messages = self.format(signature, inputs.clone());
        trace!(message_count = messages.len(), "adapter formatted chat");
        let response = lm.call(messages, tools).await?;
        debug!(
            prompt_tokens = response.usage.prompt_tokens,
            completion_tokens = response.usage.completion_tokens,
            total_tokens = response.usage.total_tokens,
            tool_calls = response.tool_calls.len(),
            "adapter lm call complete"
        );
        let prompt_str = response.chat.to_json().to_string();

        let mut output = self.parse_response_strict(signature, response.output)?;
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
        debug!(output_fields = output.len(), "adapter parsed output");

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
            tokio::spawn(
                async move {
                    let _ = cache_clone.lock().await.insert(inputs_clone, rx).await;
                }
                .instrument(tracing::Span::current()),
            );
            trace!("spawned async cache insert");

            // Send the result to the cache
            tx.send(CacheEntry {
                prompt: prompt_str,
                prediction: prediction.clone(),
            })
            .await
            .map_err(|_| anyhow::anyhow!("Failed to send to cache"))?;
            trace!("sent prediction to cache insert task");
        }

        Ok(prediction)
    }
}
