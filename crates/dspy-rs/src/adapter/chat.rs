use anyhow::Result;
use bamltype::jsonish;
use bamltype::jsonish::BamlValueWithFlags;
use bamltype::jsonish::deserializer::coercer::run_user_checks;
use bamltype::jsonish::deserializer::deserialize_flags::DeserializerConditions;
use indexmap::IndexMap;
use regex::Regex;
use std::sync::LazyLock;
use tracing::{debug, trace};

use super::Adapter;
use crate::CallMetadata;
use crate::{
    BamlType, BamlValue, ConstraintLevel, ConstraintResult, FieldMeta, Flag, JsonishError,
    Message, OutputFormatContent, ParseError, PredictError, Predicted, RenderOptions, Signature,
    TypeIR,
};

#[derive(Default, Clone)]
pub struct ChatAdapter;

static FIELD_HEADER_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\[\[ ## ([^#]+?) ## \]\]").unwrap());

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
    fn format_task_description_schema(
        &self,
        schema: &crate::SignatureSchema,
        instruction_override: Option<&str>,
    ) -> String {
        let instruction = instruction_override.unwrap_or(schema.instruction());
        let instruction = if instruction.is_empty() {
            let input_fields = schema
                .input_fields()
                .iter()
                .map(|field| format!("`{}`", field.lm_name))
                .collect::<Vec<_>>()
                .join(", ");
            let output_fields = schema
                .output_fields()
                .iter()
                .map(|field| format!("`{}`", field.lm_name))
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

    fn format_task_description_typed<S: Signature>(
        &self,
        instruction_override: Option<&str>,
    ) -> String {
        self.format_task_description_schema(S::schema(), instruction_override)
    }

    fn format_response_instructions_schema(&self, schema: &crate::SignatureSchema) -> String {
        let mut output_fields = schema.output_fields().iter();
        let Some(first_field) = output_fields.next() else {
            return "Respond with the marker for `[[ ## completed ## ]]`.".to_string();
        };

        let mut message = format!(
            "Respond with the corresponding output fields, starting with the field `[[ ## {} ## ]]`,",
            first_field.lm_name
        );
        for field in output_fields {
            message.push_str(&format!(" then `[[ ## {} ## ]]`,", field.lm_name));
        }
        message.push_str(" and then ending with the marker for `[[ ## completed ## ]]`.");

        message
    }

    fn format_response_instructions_typed<S: Signature>(&self) -> String {
        self.format_response_instructions_schema(S::schema())
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
        self.build_system(S::schema(), instruction_override)
    }

    pub fn build_system(
        &self,
        schema: &crate::SignatureSchema,
        instruction_override: Option<&str>,
    ) -> Result<String> {
        let parts = [
            self.format_field_descriptions_schema(schema),
            self.format_field_structure_schema(schema)?,
            self.format_response_instructions_schema(schema),
            self.format_task_description_schema(schema, instruction_override),
        ];

        let system = parts.join("\n\n");
        trace!(system_len = system.len(), "formatted schema system prompt");
        Ok(system)
    }

    fn format_field_descriptions_schema(&self, schema: &crate::SignatureSchema) -> String {
        let output_format = schema.output_format();

        let mut lines = Vec::new();
        lines.push("Your input fields are:".to_string());
        for (i, field) in schema.input_fields().iter().enumerate() {
            let type_name = render_type_name_for_prompt(&field.type_ir, None);
            let mut line = format!("{}. `{}` ({type_name})", i + 1, field.lm_name);
            if !field.docs.is_empty() {
                line.push_str(": ");
                line.push_str(&field.docs);
            }
            lines.push(line);
        }

        lines.push(String::new());
        lines.push("Your output fields are:".to_string());
        for (i, field) in schema.output_fields().iter().enumerate() {
            let type_name = render_type_name_for_prompt(&field.type_ir, Some(output_format));
            let mut line = format!("{}. `{}` ({type_name})", i + 1, field.lm_name);
            if !field.docs.is_empty() {
                line.push_str(": ");
                line.push_str(&field.docs);
            }
            lines.push(line);
        }

        lines.join("\n")
    }

    fn format_field_descriptions_typed<S: Signature>(&self) -> String {
        self.format_field_descriptions_schema(S::schema())
    }

    fn format_field_structure_schema(&self, schema: &crate::SignatureSchema) -> Result<String> {
        let mut lines = vec![
            "All interactions will be structured in the following way, with the appropriate values filled in.".to_string(),
            String::new(),
        ];

        for field in schema.input_fields() {
            lines.push(format!("[[ ## {} ## ]]", field.lm_name));
            lines.push(field.lm_name.to_string());
            lines.push(String::new());
        }

        let parent_format = schema.output_format();
        for field in schema.output_fields() {
            let type_name = render_type_name_for_prompt(&field.type_ir, Some(parent_format));
            let rendered_schema = render_field_type_schema(parent_format, &field.type_ir)?;
            lines.push(format!("[[ ## {} ## ]]", field.lm_name));
            lines.push(format!(
                "Output field `{}` should be of type: {type_name}",
                field.lm_name
            ));
            if !rendered_schema.is_empty() && rendered_schema != type_name {
                lines.push(String::new());
                lines.push(format_schema_for_prompt(&rendered_schema));
            }
            lines.push(String::new());
        }

        lines.push("[[ ## completed ## ]]".to_string());
        Ok(lines.join("\n"))
    }

    fn format_field_structure_typed<S: Signature>(&self) -> Result<String> {
        self.format_field_structure_schema(S::schema())
    }

    pub fn format_user_message_typed<S: Signature>(&self, input: &S::Input) -> String
    where
        S::Input: BamlType,
    {
        self.format_input(S::schema(), input)
    }

    pub fn format_input<I>(&self, schema: &crate::SignatureSchema, input: &I) -> String
    where
        I: BamlType + for<'a> facet::Facet<'a>,
    {
        let baml_value = input.to_baml_value();
        let input_output_format = <I as BamlType>::baml_output_format();

        let mut result = String::new();
        for field_spec in schema.input_fields() {
            if let Some(value) = value_for_path_relaxed(&baml_value, field_spec.path()) {
                result.push_str(&format!("[[ ## {} ## ]]\n", field_spec.lm_name));
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

    pub fn format_input_baml(&self, schema: &crate::SignatureSchema, input: &BamlValue) -> String {
        let mut result = String::new();
        for field_spec in schema.input_fields() {
            if let Some(value) = value_for_path_relaxed(input, field_spec.path()) {
                result.push_str(&format!("[[ ## {} ## ]]\n", field_spec.lm_name));
                result.push_str(&format_baml_value_for_prompt_typed(
                    value,
                    schema.output_format(),
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
        self.format_output(S::schema(), output)
    }

    pub fn format_output<O>(&self, schema: &crate::SignatureSchema, output: &O) -> String
    where
        O: BamlType + for<'a> facet::Facet<'a>,
    {
        let baml_value = output.to_baml_value();

        let mut sections = Vec::new();
        for field_spec in schema.output_fields() {
            if let Some(value) = value_for_path_relaxed(&baml_value, field_spec.path()) {
                sections.push(format!(
                    "[[ ## {} ## ]]\n{}",
                    field_spec.lm_name,
                    format_baml_value_for_prompt(value)
                ));
            }
        }
        let mut result = sections.join("\n\n");
        result.push_str("\n\n[[ ## completed ## ]]\n");

        result
    }

    pub fn format_output_baml(
        &self,
        schema: &crate::SignatureSchema,
        output: &BamlValue,
    ) -> String {
        let mut sections = Vec::new();
        for field_spec in schema.output_fields() {
            if let Some(value) = value_for_path_relaxed(output, field_spec.path()) {
                sections.push(format!(
                    "[[ ## {} ## ]]\n{}",
                    field_spec.lm_name,
                    format_baml_value_for_prompt(value)
                ));
            }
        }
        let mut result = sections.join("\n\n");
        result.push_str("\n\n[[ ## completed ## ]]\n");
        result
    }

    pub fn format_demo_typed<S: Signature>(
        &self,
        demo: &crate::predictors::Demo<S>,
    ) -> (String, String)
    where
        S::Input: BamlType,
        S::Output: BamlType,
    {
        let user_msg = self.format_user_message_typed::<S>(&demo.input);
        let assistant_msg = self.format_assistant_message_typed::<S>(&demo.output);
        (user_msg, assistant_msg)
    }

    #[allow(clippy::result_large_err)]
    #[tracing::instrument(
        name = "dsrs.adapter.chat.parse_typed",
        level = "debug",
        skip(self, response),
        fields(
            signature = std::any::type_name::<S>(),
            output_field_count = S::schema().output_fields().len()
        )
    )]
    pub fn parse_response_typed<S: Signature>(
        &self,
        response: &Message,
    ) -> std::result::Result<(S::Output, IndexMap<String, FieldMeta>), ParseError> {
        self.parse_output_with_meta::<S::Output>(S::schema(), response)
    }

    #[allow(clippy::result_large_err)]
    pub fn parse_output_with_meta<O>(
        &self,
        schema: &crate::SignatureSchema,
        response: &Message,
    ) -> std::result::Result<(O, IndexMap<String, FieldMeta>), ParseError>
    where
        O: BamlType + for<'a> facet::Facet<'a>,
    {
        let content = response.content();
        let output_format = schema.output_format();
        let sections = parse_sections(&content);

        let mut metas = IndexMap::new();
        let mut errors = Vec::new();
        let mut output_map = bamltype::baml_types::BamlMap::new();
        let mut checks_total = 0usize;
        let mut checks_failed = 0usize;
        let mut asserts_failed = 0usize;

        for field in schema.output_fields() {
            let rust_name = field.rust_name.clone();
            let type_ir = field.type_ir.clone();

            let raw_text = match sections.get(field.lm_name) {
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

            insert_baml_at_path(&mut output_map, field.path(), baml_value);
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
                    <O as BamlType>::baml_internal_name().to_string(),
                    output_map,
                ))
            };
            return Err(ParseError::Multiple { errors, partial });
        }

        let typed_output = <O as BamlType>::try_from_baml_value(BamlValue::Class(
            <O as BamlType>::baml_internal_name().to_string(),
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

    #[allow(clippy::result_large_err)]
    pub fn parse_output<O>(
        &self,
        schema: &crate::SignatureSchema,
        response: &Message,
    ) -> std::result::Result<O, ParseError>
    where
        O: BamlType + for<'a> facet::Facet<'a>,
    {
        let (output, _) = self.parse_output_with_meta::<O>(schema, response)?;
        Ok(output)
    }

    #[allow(clippy::result_large_err)]
    pub fn parse_output_baml_with_meta(
        &self,
        schema: &crate::SignatureSchema,
        response: &Message,
    ) -> std::result::Result<(BamlValue, IndexMap<String, FieldMeta>), ParseError> {
        let content = response.content();
        let output_format = schema.output_format();
        let sections = parse_sections(&content);

        let mut metas = IndexMap::new();
        let mut errors = Vec::new();
        let mut output_map = bamltype::baml_types::BamlMap::new();

        for field in schema.output_fields() {
            let rust_name = field.rust_name.clone();
            let type_ir = field.type_ir.clone();

            let raw_text = match sections.get(field.lm_name) {
                Some(text) => text.clone(),
                None => {
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
            insert_baml_at_path(&mut output_map, field.path(), baml_value);
        }

        if !errors.is_empty() {
            let partial = if output_map.is_empty() {
                None
            } else {
                Some(BamlValue::Class("DynamicOutput".to_string(), output_map))
            };
            return Err(ParseError::Multiple { errors, partial });
        }

        Ok((
            BamlValue::Class("DynamicOutput".to_string(), output_map),
            metas,
        ))
    }

    #[allow(clippy::result_large_err)]
    pub fn parse_output_baml(
        &self,
        schema: &crate::SignatureSchema,
        response: &Message,
    ) -> std::result::Result<BamlValue, ParseError> {
        let (output, _) = self.parse_output_baml_with_meta(schema, response)?;
        Ok(output)
    }

    pub fn parse_sections(content: &str) -> IndexMap<String, String> {
        crate::adapter::chat::parse_sections(content)
    }

    pub fn parse_response_with_schema<S: Signature>(
        &self,
        response: Message,
    ) -> std::result::Result<Predicted<S::Output>, PredictError> {
        let raw_response = response.content();
        let (output, field_meta) = self
            .parse_response_typed::<S>(&response)
            .map_err(|source| PredictError::Parse {
                source,
                raw_response: raw_response.clone(),
                lm_usage: crate::LmUsage::default(),
            })?;
        let metadata = CallMetadata::new(
            raw_response,
            crate::LmUsage::default(),
            Vec::new(),
            Vec::new(),
            None,
            field_meta,
        );
        Ok(Predicted::new(output, metadata))
    }

}

fn parse_sections(content: &str) -> IndexMap<String, String> {
    let mut sections: Vec<(Option<String>, Vec<String>)> = vec![(None, Vec::new())];

    for line in content.lines() {
        let trimmed = line.trim();
        if let Some(caps) = FIELD_HEADER_PATTERN.captures(trimmed) {
            let header = caps.get(1).unwrap().as_str().trim().to_string();
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

fn value_for_path_relaxed<'a>(
    value: &'a BamlValue,
    path: &crate::FieldPath,
) -> Option<&'a BamlValue> {
    let mut current = value;
    let parts: Vec<_> = path.iter().collect();
    let mut idx = 0usize;
    while idx < parts.len() {
        match current {
            BamlValue::Class(_, fields) | BamlValue::Map(fields) => {
                if let Some(next) = fields.get(parts[idx]) {
                    current = next;
                    idx += 1;
                    continue;
                }
                if idx + 1 < parts.len() {
                    if let Some(next) = fields.get(parts[idx + 1]) {
                        current = next;
                        idx += 2;
                        continue;
                    }
                }
                return None;
            }
            _ => return None,
        }
    }
    Some(current)
}

fn insert_baml_at_path(
    root: &mut bamltype::baml_types::BamlMap<String, BamlValue>,
    path: &crate::FieldPath,
    value: BamlValue,
) {
    let parts: Vec<_> = path.iter().collect();
    if parts.is_empty() {
        return;
    }
    insert_baml_at_parts(root, &parts, value);
}

fn insert_baml_at_parts(
    root: &mut bamltype::baml_types::BamlMap<String, BamlValue>,
    parts: &[&'static str],
    value: BamlValue,
) {
    if parts.len() == 1 {
        root.insert(parts[0].to_string(), value);
        return;
    }

    let key = parts[0].to_string();
    let entry = root
        .entry(key)
        .or_insert_with(|| BamlValue::Map(bamltype::baml_types::BamlMap::new()));

    if !matches!(entry, BamlValue::Map(_) | BamlValue::Class(_, _)) {
        *entry = BamlValue::Map(bamltype::baml_types::BamlMap::new());
    }

    let child = match entry {
        BamlValue::Map(map) | BamlValue::Class(_, map) => map,
        _ => unreachable!(),
    };

    insert_baml_at_parts(child, &parts[1..], value);
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

impl Adapter for ChatAdapter {}
