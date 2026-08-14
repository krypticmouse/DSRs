use anyhow::Result;
use indexmap::IndexMap;
use minijinja::UndefinedBehavior;
use minijinja::value::{Kwargs, Value as MiniJinjaValue};
use regex::Regex;
use serde_json::{Map, Value, json};
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, RwLock};
use tracing::{debug, trace};

use crate::CallMetadata;
use crate::typesys::coerce::coerce;
use crate::typesys::constraint::evaluate_constraint_expression;
use crate::typesys::render::{schema_block, type_name};
use crate::{
    ConstraintKind, ConstraintResult, FieldMeta, FieldSchema, InputRenderSpec,
    JsonishError, Message, ParseError, PredictError, Predicted, Schema, Signature,
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

struct CachedInputRenderTemplate {
    env: minijinja::Environment<'static>,
}

static INPUT_RENDER_TEMPLATE_CACHE: LazyLock<
    RwLock<HashMap<&'static str, Arc<CachedInputRenderTemplate>>>,
> = LazyLock::new(|| RwLock::new(HashMap::new()));

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
        if let Some(index) = truncated.rfind(char::is_whitespace)
            && index > 0
        {
            truncated.truncate(index);
        }
        truncated = truncated.trim_end().to_string();
    }

    Ok(format!("{truncated}{end}"))
}

fn build_input_render_environment() -> minijinja::Environment<'static> {
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
        let output_schema = schema.output_schema();

        let mut lines = Vec::new();
        lines.push("Your input fields are:".to_string());
        for (i, field) in schema.input_fields().iter().enumerate() {
            let type_name = type_name(&field.type_ir, None);
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
            let type_name = type_name(&field.type_ir, Some(output_schema));
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

        let output_schema = schema.output_schema();
        for field in schema.output_fields() {
            let type_name = type_name(&field.type_ir, Some(output_schema));
            let rendered_schema = schema_block(&field.type_ir, output_schema);
            lines.push(format!("[[ ## {} ## ]]", field.lm_name));
            lines.push(format!(
                "Output field `{}` should be of type: {type_name}",
                field.lm_name
            ));
            if !rendered_schema.is_empty() && rendered_schema != type_name {
                lines.push(String::new());
                lines.push(rendered_schema);
            }
            lines.push(String::new());
        }

        lines.push("[[ ## completed ## ]]".to_string());
        Ok(lines.join("\n"))
    }

    /// Formats a typed input value as a user message with `[[ ## field ## ]]` delimiters.
    ///
    /// Each input field is serialized via serde and formatted according to its field path
    /// (handling flattened fields). Appends the response instructions telling the LM which
    /// output fields to produce.
    pub fn format_user_message_typed<S: Signature>(&self, input: &S::Input) -> String
    where
        S::Input: Schema,
    {
        self.format_input(S::schema(), input)
    }

    /// Formats an input value using a schema — the building-block version of
    /// [`format_user_message_typed`](ChatAdapter::format_user_message_typed).
    ///
    /// Navigates the serialized JSON using each field's [`FieldPath`](crate::FieldPath) to
    /// handle flattened structs correctly. A field with path `["inner", "question"]` is
    /// extracted from the flattened structure but rendered as a flat `[[ ## question ## ]]`
    /// section in the prompt. Appends response instructions so the LM sees output-field
    /// ordering guidance in the latest user turn.
    pub fn format_input<I>(&self, schema: &crate::SignatureSchema, input: &I) -> String
    where
        I: Schema + for<'a> facet::Facet<'a>,
    {
        let json = serde_json::to_value(input).unwrap_or(Value::Null);
        // The aliased input-context tree is only read by `#[render(jinja = ...)]`
        // fields — skip the full clone it requires when no field uses Jinja.
        let has_jinja_field = schema
            .input_fields()
            .iter()
            .any(|field| matches!(field.input_render, InputRenderSpec::Jinja(_)));
        let input_json = if has_jinja_field {
            build_input_context_value(schema, &json)
        } else {
            Value::Null
        };
        let vars = Value::Object(Map::new());

        let mut result = String::new();
        for field_spec in schema.input_fields() {
            if let Some(value) = value_for_path_relaxed(&json, field_spec.path()) {
                result.push_str(&format!("[[ ## {} ## ]]\n", field_spec.lm_name));
                result.push_str(&render_input_field(field_spec, value, &input_json, &vars));
                result.push_str("\n\n");
            }
        }

        result.push_str(
            schema.response_instructions_cached(|| self.format_response_instructions_schema(schema)),
        );
        result
    }

    /// Formats a typed output value as an assistant message for few-shot demos.
    ///
    /// Each output field is serialized and delimited with `[[ ## field ## ]]` markers,
    /// ending with `[[ ## completed ## ]]`. Used internally by [`Predict`](crate::Predict)
    /// to format demo assistant messages.
    pub fn format_assistant_message_typed<S: Signature>(&self, output: &S::Output) -> String
    where
        S::Output: Schema,
    {
        self.format_output(S::schema(), output)
    }

    /// Formats an output value using a schema — the building-block version of
    /// [`format_assistant_message_typed`](ChatAdapter::format_assistant_message_typed).
    pub fn format_output<O>(&self, schema: &crate::SignatureSchema, output: &O) -> String
    where
        O: Schema + for<'a> facet::Facet<'a>,
    {
        let json = serde_json::to_value(output).unwrap_or(Value::Null);

        let mut sections = Vec::new();
        for field_spec in schema.output_fields() {
            if let Some(value) = value_for_path_relaxed(&json, field_spec.path()) {
                sections.push(format!(
                    "[[ ## {} ## ]]\n{}",
                    field_spec.lm_name,
                    format_json_value_for_prompt(value)
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
        S::Input: Schema,
        S::Output: Schema,
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
    /// 3. Coerce the raw text to the field's type via the in-house coercer
    /// 4. Run `#[check]` and `#[assert]` constraints
    /// 5. Assemble the flat fields into the typed output via serde
    ///
    /// Returns the typed output and a map of [`FieldMeta`] with per-field raw text, parse
    /// flags, and constraint results.
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
    pub fn parse_output_with_meta<O>(
        &self,
        schema: &crate::SignatureSchema,
        response: &Message,
    ) -> std::result::Result<(O, IndexMap<String, FieldMeta>), ParseError>
    where
        O: Schema + for<'a> facet::Facet<'a>,
    {
        let content = response.text_content_cow();
        let output_schema = schema.output_schema();
        let sections = parse_sections_cow(&content);

        let mut metas = IndexMap::new();
        let mut errors = Vec::new();
        // Coerced fields keyed by their leaf name, fed straight into serde's
        // MapDeserializer below — no intermediate `Value::Object` tree.
        let mut output_fields: Vec<(&'static str, Value)> =
            Vec::with_capacity(schema.output_fields().len());
        let mut checks_total = 0usize;
        let mut checks_failed = 0usize;
        let mut asserts_failed = 0usize;

        for field in schema.output_fields() {
            let rust_name = field.rust_name.as_str();
            let field_type = &field.type_ir;

            let raw_text: &str = match sections.get(field.lm_name) {
                Some(text) => text.as_ref(),
                None => {
                    debug!(field = %rust_name, "missing output field in response");
                    errors.push(ParseError::MissingField {
                        field: rust_name.to_string(),
                        raw_response: content.to_string(),
                    });
                    continue;
                }
            };

            let coerced = match coerce(raw_text, field_type, output_schema) {
                Ok(value) => value,
                Err(err) => {
                    let expected_type = type_name(field_type, Some(output_schema));
                    debug!(
                        field = %rust_name,
                        expected_type = %expected_type,
                        raw_text_len = raw_text.len(),
                        "typed coercion failed"
                    );
                    trace!(
                        field = %rust_name,
                        raw_preview = %crate::truncate(raw_text, 160),
                        "typed coercion failed preview"
                    );
                    errors.push(ParseError::CoercionFailed {
                        field: rust_name.to_string(),
                        expected_type,
                        raw_text: raw_text.to_string(),
                        source: JsonishError::from(err),
                    });
                    continue;
                }
            };

            // Constraints are evaluated straight off the `'static` specs — the
            // compiled expressions are cached process-wide, so no per-call
            // Environment build, recompile, or `Vec<Constraint>` allocation.
            let mut checks = Vec::new();
            for spec in field.constraints {
                let passed = evaluate_constraint_expression(spec.expression, &coerced.value);
                match spec.kind {
                    ConstraintKind::Assert => {
                        if !passed {
                            asserts_failed += 1;
                            debug!(field = %rust_name, label = %spec.label, "typed assert constraint failed");
                            errors.push(ParseError::AssertFailed {
                                field: rust_name.to_string(),
                                label: spec.label.to_string(),
                                expression: spec.expression.to_string(),
                                value: coerced.value.clone(),
                            });
                        }
                    }
                    ConstraintKind::Check => {
                        checks_total += 1;
                        if !passed {
                            checks_failed += 1;
                            trace!(field = %rust_name, label = %spec.label, "typed check constraint failed");
                        }
                        checks.push(ConstraintResult {
                            label: spec.label.to_string(),
                            expression: spec.expression.to_string(),
                            passed,
                        });
                    }
                }
            }

            metas.insert(
                field.rust_name.clone(),
                FieldMeta {
                    raw_text: raw_text.to_string(),
                    flags: coerced.flags,
                    checks,
                },
            );

            // `#[serde(flatten)]` wrappers serialize flat, so every field keys at
            // the top level by its final path segment. Duplicate leaves keep the
            // last value, matching the previous `Map::insert` behavior.
            if let Some(leaf) = field.path().iter().last() {
                if let Some(existing) = output_fields.iter_mut().find(|(name, _)| *name == leaf) {
                    existing.1 = coerced.value;
                } else {
                    output_fields.push((leaf, coerced.value));
                }
            }
        }

        if !errors.is_empty() {
            debug!(
                errors = errors.len(),
                checks_total, checks_failed, asserts_failed, "typed parse returned errors"
            );
            let partial = if output_fields.is_empty() {
                None
            } else {
                Some(Value::Object(
                    output_fields
                        .into_iter()
                        .map(|(name, value)| (name.to_string(), value))
                        .collect(),
                ))
            };
            return Err(ParseError::Multiple { errors, partial });
        }

        // Deserialize straight from the coerced field pairs — the historical
        // `Value::Object` assembly + `from_value` re-walk is skipped entirely.
        let typed_output = O::deserialize(serde::de::value::MapDeserializer::<
            _,
            serde_json::Error,
        >::new(output_fields.into_iter()))
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
        O: Schema + for<'a> facet::Facet<'a>,
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
    parse_sections_cow(content)
        .into_iter()
        .map(|(name, text)| (name.to_string(), text.into_owned()))
        .collect()
}

/// Zero-copy section splitter: each section's content is a byte range of `content`,
/// so no per-line Strings are allocated. Only sections containing `\r` fall back to
/// an owned, line-normalized copy (matching the historical `lines().join("\n")`
/// behavior). Duplicate section names keep the first occurrence.
fn parse_sections_cow(content: &str) -> IndexMap<&str, std::borrow::Cow<'_, str>> {
    let base = content.as_ptr() as usize;
    let mut ranges: IndexMap<&str, (usize, usize)> = IndexMap::new();
    // The currently open section: (name, content start offset).
    let mut open: Option<(&str, usize)> = None;

    for line in content.lines() {
        let trimmed = line.trim();
        let Some(caps) = FIELD_HEADER_PATTERN.captures(trimmed) else {
            continue;
        };
        let line_start = line.as_ptr() as usize - base;
        if let Some((name, start)) = open.take()
            && !ranges.contains_key(name)
        {
            ranges.insert(name, (start, line_start));
        }
        let header = caps.get(1).unwrap().as_str().trim();
        let marker_end = caps.get(0).unwrap().end();
        let content_start = (trimmed.as_ptr() as usize - base) + marker_end;
        open = Some((header, content_start));
    }
    if let Some((name, start)) = open
        && !ranges.contains_key(name)
    {
        ranges.insert(name, (start, content.len()));
    }

    ranges
        .into_iter()
        .map(|(name, (start, end))| {
            let slice = content[start..end].trim();
            let text = if slice.contains('\r') {
                std::borrow::Cow::Owned(
                    slice.lines().collect::<Vec<_>>().join("\n").trim().to_string(),
                )
            } else {
                std::borrow::Cow::Borrowed(slice)
            };
            (name, text)
        })
        .collect()
}

/// Navigates `value` by `path`, tolerating flattened wrappers whose intermediate segments
/// are absent from the serialized (flat) structure.
fn value_for_path_relaxed<'a>(value: &'a Value, path: &crate::FieldPath) -> Option<&'a Value> {
    let mut current = value;
    let parts: Vec<_> = path.iter().collect();
    let mut idx = 0usize;
    while idx < parts.len() {
        match current {
            Value::Object(fields) => {
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

fn format_json_value_for_prompt(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Null => "null".to_string(),
        other => serde_json::to_string(other).unwrap_or_else(|_| "<error>".to_string()),
    }
}

fn render_input_field(
    field_spec: &FieldSchema,
    value: &Value,
    input: &Value,
    vars: &Value,
) -> String {
    match field_spec.input_render {
        InputRenderSpec::Default => match value {
            Value::String(s) => s.clone(),
            _ => serde_json::to_string(value).unwrap_or_else(|_| "<error>".to_string()),
        },
        InputRenderSpec::Format(format) => crate::typesys::format_value(value, format),
        InputRenderSpec::Jinja(template) => {
            render_input_field_jinja(template, field_spec, value, input, vars)
        }
    }
}

fn build_input_context_value(schema: &crate::SignatureSchema, root: &Value) -> Value {
    let mut input_json = root.clone();
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

fn render_input_field_jinja(
    template: &'static str,
    field_spec: &FieldSchema,
    value: &Value,
    input: &Value,
    vars: &Value,
) -> String {
    let cached = {
        let cache = INPUT_RENDER_TEMPLATE_CACHE
            .read()
            .expect("input render template cache lock poisoned");
        cache.get(template).cloned()
    };
    let cached = match cached {
        Some(cached) => cached,
        None => {
            let mut env = build_input_render_environment();
            env.add_template(INPUT_RENDER_TEMPLATE_NAME, template)
                .unwrap_or_else(|err| {
                    panic!(
                        "failed to compile cached input render template for `{}` ({}): {err}",
                        field_spec.lm_name, field_spec.rust_name
                    )
                });
            let entry = Arc::new(CachedInputRenderTemplate { env });
            let mut cache = INPUT_RENDER_TEMPLATE_CACHE
                .write()
                .expect("input render template cache lock poisoned");
            cache.entry(template).or_insert(entry).clone()
        }
    };

    let compiled = cached
        .env
        .get_template(INPUT_RENDER_TEMPLATE_NAME)
        .unwrap_or_else(|err| {
            panic!(
                "failed to fetch cached input render template for `{}` ({}): {err}",
                field_spec.lm_name, field_spec.rust_name
            )
        });

    let this = value.clone();
    let field = json!({
        "name": field_spec.lm_name,
        "rust_name": field_spec.rust_name,
        "type": type_name(&field_spec.type_ir, None),
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

