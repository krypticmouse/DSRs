use anyhow::Result;
use bamltype::jsonish;
use bamltype::jsonish::BamlValueWithFlags;
use bamltype::jsonish::deserializer::coercer::run_user_checks;
use bamltype::jsonish::deserializer::deserialize_flags::DeserializerConditions;
use indexmap::IndexMap;
use minijinja::UndefinedBehavior;
use minijinja::value::{Kwargs, Value as MiniJinjaValue};
use regex::Regex;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};
use tracing::{debug, trace};

use super::Adapter;
use crate::CallMetadata;
use crate::{
    BamlType, BamlValue, ConstraintLevel, ConstraintResult, FieldMeta, Flag, InputRenderSpec,
    JsonishError, Message, OutputFormatContent, ParseError, PredictError, Predicted, RenderOptions,
    Signature, TypeIR,
};

/// Builds prompts and parses responses using the `[[ ## field ## ]]` delimiter protocol.
///
/// The adapter is stateless — all state comes from the [`SignatureSchema`](crate::SignatureSchema)
/// passed to each method. Two usage patterns:
///
/// - **High-level** (what [`Predict`](crate::Predict) uses): `format_system_message_typed`,
///   `format_user_message_typed`, `parse_response_typed` — all parameterized by `S: Signature`.
/// - **Building blocks** (for module authors): `build_system`, `format_input`, `format_output`,
///   `parse_output`, `parse_sections` — parameterized by `&SignatureSchema`, not a Signature type.
///
/// The building blocks exist so module authors can compose custom prompt flows (e.g.
/// ReAct's action/extract loop) without reimplementing the delimiter protocol.
#[derive(Default, Clone)]
pub struct ChatAdapter;

static FIELD_HEADER_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\[\[ ## ([^#]+?) ## \]\]").unwrap());

const INPUT_RENDER_TEMPLATE_NAME: &str = "__input_field__";

#[derive(Clone)]
struct CachedInputRenderTemplate {
    env: minijinja::Environment<'static>,
}

static INPUT_RENDER_TEMPLATE_CACHE: LazyLock<
    Mutex<HashMap<&'static str, CachedInputRenderTemplate>>,
> = LazyLock::new(|| Mutex::new(HashMap::new()));

fn regex_match(value: String, regex: String) -> bool {
    match Regex::new(&regex) {
        Ok(re) => re.is_match(&value),
        Err(_) => false,
    }
}

fn sum_filter(value: Vec<MiniJinjaValue>) -> MiniJinjaValue {
    let int_sum: Option<i64> = value
        .iter()
        .map(|value| <i64>::try_from(value.clone()).ok())
        .collect::<Option<Vec<_>>>()
        .map(|ints| ints.into_iter().sum());
    let float_sum: Option<f64> = value
        .into_iter()
        .map(|value| <f64>::try_from(value).ok())
        .collect::<Option<Vec<_>>>()
        .map(|floats| floats.into_iter().sum());
    int_sum.map_or(
        float_sum.map_or(MiniJinjaValue::from(0), MiniJinjaValue::from),
        MiniJinjaValue::from,
    )
}

fn truncate_filter(
    value: String,
    positional_length: Option<usize>,
    kwargs: Kwargs,
) -> Result<String, minijinja::Error> {
    let kwarg_length: Option<usize> = kwargs.get("length")?;
    let length = kwarg_length.or(positional_length).unwrap_or(255);
    let killwords: Option<bool> = kwargs.get("killwords")?;
    let leeway: Option<usize> = kwargs.get("leeway")?;
    let end: Option<String> = kwargs.get("end")?;
    kwargs.assert_all_used()?;

    let killwords = killwords.unwrap_or(false);
    let leeway = leeway.unwrap_or(5);
    let end = end.unwrap_or_else(|| "...".to_string());
    let value_len = value.chars().count();

    if value_len <= length.saturating_add(leeway) {
        return Ok(value);
    }

    let trim_to = length.saturating_sub(end.chars().count());
    if trim_to == 0 {
        return Ok(end.chars().take(length).collect());
    }

    let mut truncated: String = value.chars().take(trim_to).collect();
    if !killwords {
        if let Some(index) = truncated.rfind(char::is_whitespace) {
            if index > 0 {
                truncated.truncate(index);
            }
        }
        truncated = truncated.trim_end().to_string();
    }

    Ok(format!("{truncated}{end}"))
}

fn build_input_render_environment() -> minijinja::Environment<'static> {
    // Keep this setup aligned with BAML's jinja env defaults, then add contrib filters.
    let mut env = minijinja::Environment::new();
    env.set_formatter(|output, state, value| {
        let value = if value.is_none() {
            &MiniJinjaValue::from("null")
        } else {
            value
        };
        minijinja::escape_formatter(output, state, value)
    });
    env.set_debug(true);
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    env.set_undefined_behavior(UndefinedBehavior::Strict);
    env.add_filter("regex_match", regex_match);
    env.add_filter("sum", sum_filter);
    env.add_filter("truncate", truncate_filter);
    env
}

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
    // TODO(post-hardening): This parser is intentionally heuristic. Keep this
    // behavior covered by tests when schema rendering changes.
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

    /// Builds the system message for a signature using its default instruction.
    ///
    /// Shorthand for `format_system_message_typed_with_instruction::<S>(None)`.
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
    /// Builds the system message for a signature with an optional instruction override.
    ///
    /// The system message includes:
    /// 1. Field descriptions (names, types, doc comments)
    /// 2. Field structure template (the `[[ ## field ## ]]` layout the LM should follow)
    /// 3. Response instructions (which fields to produce, in what order)
    /// 4. Task description (the signature's instruction or the override)
    pub fn format_system_message_typed_with_instruction<S: Signature>(
        &self,
        instruction_override: Option<&str>,
    ) -> Result<String> {
        self.build_system(S::schema(), instruction_override)
    }

    /// Builds a system message from a [`SignatureSchema`](crate::SignatureSchema) directly.
    ///
    /// The schema-based equivalent of [`format_system_message_typed_with_instruction`](ChatAdapter::format_system_message_typed_with_instruction).
    /// Use this when you have a schema but not a concrete `S: Signature` type (e.g.
    /// in dynamic or schema-transformed contexts).
    ///
    /// # Errors
    ///
    /// Returns an error if the output format rendering fails (malformed type IR).
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

    /// Formats a typed input value as a user message with `[[ ## field ## ]]` delimiters.
    ///
    /// Each input field is serialized via `BamlType::to_baml_value()` and formatted
    /// according to its field path (handling flattened fields). Appends the response
    /// instructions telling the LM which output fields to produce.
    pub fn format_user_message_typed<S: Signature>(&self, input: &S::Input) -> String
    where
        S::Input: BamlType,
    {
        self.format_input(S::schema(), input)
    }

    /// Formats an input value using a schema — the building-block version of
    /// [`format_user_message_typed`](ChatAdapter::format_user_message_typed).
    ///
    /// Navigates the `BamlValue` using each field's [`FieldPath`](crate::FieldPath) to
    /// handle flattened structs correctly. A field with path `["inner", "question"]` is
    /// extracted from the nested structure but rendered as a flat `[[ ## question ## ]]`
    /// section in the prompt. Appends response instructions so the LM sees
    /// output-field ordering guidance in the latest user turn.
    pub fn format_input<I>(&self, schema: &crate::SignatureSchema, input: &I) -> String
    where
        I: BamlType + for<'a> facet::Facet<'a>,
    {
        let baml_value = input.to_baml_value();
        let input_output_format = <I as BamlType>::baml_output_format();
        let input_json = build_input_context_value(schema, &baml_value);
        let vars = Value::Object(serde_json::Map::new());

        let mut result = String::new();
        for field_spec in schema.input_fields() {
            if let Some(value) = value_for_path_relaxed(&baml_value, field_spec.path()) {
                result.push_str(&format!("[[ ## {} ## ]]\n", field_spec.lm_name));
                result.push_str(&render_input_field(
                    field_spec,
                    value,
                    &input_json,
                    input_output_format,
                    &vars,
                ));
                result.push_str("\n\n");
            }
        }

        result.push_str(&self.format_response_instructions_schema(schema));
        result
    }

    /// Formats a typed output value as an assistant message for few-shot demos.
    ///
    /// Each output field is serialized and delimited with `[[ ## field ## ]]` markers,
    /// ending with `[[ ## completed ## ]]`. Used internally by [`Predict`](crate::Predict)
    /// to format demo assistant messages.
    pub fn format_assistant_message_typed<S: Signature>(&self, output: &S::Output) -> String
    where
        S::Output: BamlType,
    {
        self.format_output(S::schema(), output)
    }

    /// Formats an output value using a schema — the building-block version of
    /// [`format_assistant_message_typed`](ChatAdapter::format_assistant_message_typed).
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

    /// Formats a demo example as a (user_message, assistant_message) pair.
    ///
    /// Convenience method that calls [`format_user_message_typed`](ChatAdapter::format_user_message_typed)
    /// and [`format_assistant_message_typed`](ChatAdapter::format_assistant_message_typed).
    pub fn format_demo_typed<S: Signature>(
        &self,
        demo: &crate::predictors::Example<S>,
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
    /// Parses an LM response into a typed output with per-field metadata.
    ///
    /// The full parsing pipeline:
    /// 1. Split the response into `[[ ## field ## ]]` sections
    /// 2. For each output field in the schema, find its section by LM name
    /// 3. Coerce the raw text to the field's type via jsonish
    /// 4. Run `#[check]` and `#[assert]` constraints
    /// 5. Assemble the flat fields into the nested typed output via field paths
    ///
    /// Returns the typed output and a map of [`FieldMeta`] with
    /// per-field raw text, parse flags, and constraint results.
    ///
    /// # Errors
    ///
    /// Returns [`ParseError`] variants:
    /// - `MissingField` — an output field's `[[ ## field ## ]]` section wasn't found
    /// - `CoercionFailed` — jsonish couldn't parse the raw text into the expected type
    /// - `AssertFailed` — a `#[assert(...)]` constraint failed
    /// - `ExtractionFailed` — the assembled BamlValue couldn't convert to the typed output
    /// - `Multiple` — several of the above; includes a partial BamlValue if some fields parsed
    pub fn parse_response_typed<S: Signature>(
        &self,
        response: &Message,
    ) -> std::result::Result<(S::Output, IndexMap<String, FieldMeta>), ParseError> {
        self.parse_output_with_meta::<S::Output>(S::schema(), response)
    }

    #[allow(clippy::result_large_err)]
    /// Parses an LM response against a schema, returning typed output and field metadata.
    ///
    /// Schema-based equivalent of [`parse_response_typed`](ChatAdapter::parse_response_typed).
    /// Use when you have a schema but not a `S: Signature` type.
    ///
    /// # Errors
    ///
    /// Same as [`parse_response_typed`](ChatAdapter::parse_response_typed).
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
    /// Parses an LM response into a typed output, discarding field metadata.
    ///
    /// Convenience wrapper around [`parse_output_with_meta`](ChatAdapter::parse_output_with_meta).
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

    /// Splits raw LM response text into named sections by `[[ ## field ## ]]` delimiters.
    ///
    /// Returns an ordered map of field_name → section_content. The `completed` marker
    /// is included as a section (usually empty). Duplicate section names keep the first
    /// occurrence. Content before the first delimiter is discarded.
    pub fn parse_sections(content: &str) -> IndexMap<String, String> {
        crate::adapter::chat::parse_sections(content)
    }

    /// Parses a raw [`Message`] into a [`Predicted<S::Output>`](crate::Predicted).
    ///
    /// Convenience wrapper that calls [`parse_response_typed`](ChatAdapter::parse_response_typed)
    /// and wraps the result in [`Predicted`] with default metadata
    /// (zero usage, no tool calls). Useful for testing or replaying saved responses.
    ///
    /// # Errors
    ///
    /// Parse failures are wrapped as [`PredictError::Parse`].
    #[expect(
        clippy::result_large_err,
        reason = "Public API returns PredictError directly for downstream matching."
    )]
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
            // TODO(post-hardening): We currently keep the first occurrence to avoid
            // late duplicate markers silently overwriting earlier parsed fields.
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
                // Flattened wrappers may remove one or more intermediate path
                // segments (`outer.inner.answer` serialized as `answer`), so
                // probe ahead for the next segment visible at this level.
                let mut matched = None;
                for (look_ahead, part) in parts.iter().enumerate().skip(idx + 1) {
                    if let Some(next) = fields.get(*part) {
                        matched = Some((look_ahead, next));
                        break;
                    }
                }
                if let Some((look_ahead, next)) = matched {
                    current = next;
                    idx = look_ahead + 1;
                    continue;
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

fn render_input_field(
    field_spec: &crate::FieldSchema,
    value: &BamlValue,
    input: &Value,
    output_format: &OutputFormatContent,
    vars: &Value,
) -> String {
    match field_spec.input_render {
        InputRenderSpec::Default => match value {
            BamlValue::String(s) => s.clone(),
            _ => bamltype::internal_baml_jinja::format_baml_value(value, output_format, "json")
                .unwrap_or_else(|_| "<error>".to_string()),
        },
        InputRenderSpec::Format(format) => {
            bamltype::internal_baml_jinja::format_baml_value(value, output_format, format)
                .unwrap_or_else(|_| "<error>".to_string())
        }
        InputRenderSpec::Jinja(template) => {
            render_input_field_jinja(template, field_spec, value, input, output_format, vars)
        }
    }
}

fn build_input_context_value(schema: &crate::SignatureSchema, root: &BamlValue) -> Value {
    let mut input_json = baml_value_to_render_json(root);
    let Some(root_map) = input_json.as_object_mut() else {
        return input_json;
    };

    // Provide alias lookups for top-level fields so templates can use either
    // Rust field names (`input.question`) or prompt aliases (`input.query`).
    for field_spec in schema.input_fields() {
        if field_spec.rust_name.contains('.') || field_spec.lm_name == field_spec.rust_name {
            continue;
        }
        if field_spec.path().iter().nth(1).is_some() {
            continue;
        }
        if let Some(value) = root_map.get(field_spec.rust_name.as_str()).cloned() {
            root_map
                .entry(field_spec.lm_name.to_string())
                .or_insert(value);
        }
    }

    input_json
}

fn baml_value_to_render_json(value: &BamlValue) -> Value {
    serde_json::to_value(value).unwrap_or(Value::Null)
}

fn render_input_field_jinja(
    template: &'static str,
    field_spec: &crate::FieldSchema,
    value: &BamlValue,
    input: &Value,
    _output_format: &OutputFormatContent,
    vars: &Value,
) -> String {
    let env = {
        let mut cache = INPUT_RENDER_TEMPLATE_CACHE
            .lock()
            .expect("input render template cache lock poisoned");
        cache
            .entry(template)
            .or_insert_with(|| {
                let mut env = build_input_render_environment();
                env.add_template(INPUT_RENDER_TEMPLATE_NAME, template)
                    .unwrap_or_else(|err| {
                        panic!(
                            "failed to compile cached input render template for `{}` ({}): {err}",
                            field_spec.lm_name, field_spec.rust_name
                        )
                    });
                CachedInputRenderTemplate { env }
            })
            .env
            .clone()
    };

    let compiled = env
        .get_template(INPUT_RENDER_TEMPLATE_NAME)
        .unwrap_or_else(|err| {
            panic!(
                "failed to fetch cached input render template for `{}` ({}): {err}",
                field_spec.lm_name, field_spec.rust_name
            )
        });

    let this = baml_value_to_render_json(value);
    let field = json!({
        "name": field_spec.lm_name,
        "rust_name": field_spec.rust_name,
        "type": field_spec.type_ir.diagnostic_repr().to_string(),
    });
    let context = json!({
        "this": this,
        "input": input,
        "field": field,
        "vars": vars,
    });

    compiled
        .render(minijinja::Value::from_serialize(context))
        .unwrap_or_else(|err| {
            panic!(
                "failed to render input field `{}` (rust `{}`) with #[render(jinja = ...)] template `{}`: {err}",
                field_spec.lm_name, field_spec.rust_name, template
            )
        })
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
