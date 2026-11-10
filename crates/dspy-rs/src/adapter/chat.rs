use anyhow::{Context, Result};
use indexmap::IndexMap;
use minijinja::UndefinedBehavior;
use minijinja::value::{Kwargs, Value as MiniJinjaValue};
use regex::Regex;
use serde_json::{Map, Value, json};
use std::sync::LazyLock;
use tracing::{debug, trace};

use crate::ir::{RenderSpec, SignatureDef};
use crate::trace::JsonMap;
use crate::typesys::coerce::{coerce_value_with_checks, coerce_with_checks};
use crate::typesys::constraint::evaluate_expression;
use crate::typesys::render::{schema_block, type_name};
use crate::typesys::{
    FieldType, TypeTable, alias_output_fields, canonicalize_output_aliases, json_schema_for_outputs,
};
use crate::{ConstraintKind, ConstraintResult, FieldMeta, JsonishError, Message, ParseError};

/// Builds prompts and parses responses using the `[[ ## field ## ]]` delimiter protocol.
///
/// The adapter is stateless — all state comes from the [`SignatureDef`] passed to
/// each method: `build_system_def`, `format_input_def`, `format_output_def`,
/// `parse_output_def`, plus the free `parse_sections`. This is the single prompt
/// lane; [`Predict`](crate::Predict) reaches it through the IR interpreter, and
/// module authors can call the `*_def` building blocks directly to compose custom
/// prompt flows without reimplementing the delimiter protocol.
#[derive(Default, Clone)]
pub struct ChatAdapter;

static FIELD_HEADER_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\[\[ ## ([^#]+?) ## \]\]").unwrap());

const INPUT_RENDER_TEMPLATE_NAME: &str = "__input_field__";

fn regex_match(value: String, regex: String) -> bool {
    match Regex::new(&regex) {
        Ok(re) => re.is_match(&value),
        Err(_) => false,
    }
}

fn sum_filter(value: Vec<MiniJinjaValue>) -> Result<MiniJinjaValue, minijinja::Error> {
    let ints: Option<Vec<i64>> = value
        .iter()
        .map(|value| <i64>::try_from(value.clone()).ok())
        .collect();
    if let Some(ints) = ints {
        let sum = ints
            .into_iter()
            .try_fold(0_i64, i64::checked_add)
            .ok_or_else(|| {
                minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, "sum overflow")
            })?;
        return Ok(MiniJinjaValue::from(sum));
    }
    let float_sum: Option<f64> = value
        .into_iter()
        .map(|value| <f64>::try_from(value).ok())
        .collect::<Option<Vec<_>>>()
        .map(|floats| floats.into_iter().sum());
    Ok(float_sum.map_or(MiniJinjaValue::from(0), MiniJinjaValue::from))
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

fn build_input_render_environment<'source>() -> minijinja::Environment<'source> {
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

/// A borrowed view of one signature field: everything the prompt builders
/// need, sourced from an owned [`ir::FieldDef`](crate::ir::FieldDef). The
/// historical `'static` [`FieldSchema`](crate::FieldSchema) lane rendered
/// through these same view-based functions; with `Predict` routed through the
/// interpreter, the def lane is the only prompt path.
struct FieldView<'a> {
    lm_name: &'a str,
    docs: &'a str,
    ty: &'a FieldType,
}

impl<'a> FieldView<'a> {
    fn of_def(field: &'a crate::ir::FieldDef) -> Self {
        Self {
            lm_name: &field.lm_name,
            docs: field.docs.as_deref().unwrap_or(""),
            ty: &field.ty,
        }
    }
}

fn def_views(fields: &[crate::ir::FieldDef]) -> Vec<FieldView<'_>> {
    fields.iter().map(FieldView::of_def).collect()
}

fn format_task_description_view(
    inputs: &[FieldView<'_>],
    outputs: &[FieldView<'_>],
    instruction: &str,
    instruction_override: Option<&str>,
) -> String {
    let instruction = instruction_override.unwrap_or(instruction);
    let instruction = if instruction.is_empty() {
        let input_fields = inputs
            .iter()
            .map(|field| format!("`{}`", field.lm_name))
            .collect::<Vec<_>>()
            .join(", ");
        let output_fields = outputs
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

fn format_response_instructions_view(outputs: &[FieldView<'_>]) -> String {
    let mut output_fields = outputs.iter();
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

fn format_field_descriptions_view(
    inputs: &[FieldView<'_>],
    outputs: &[FieldView<'_>],
    types: &TypeTable,
) -> String {
    let mut lines = Vec::new();
    lines.push("Your input fields are:".to_string());
    for (i, field) in inputs.iter().enumerate() {
        let type_name = type_name(field.ty, None);
        let mut line = format!("{}. `{}` ({type_name})", i + 1, field.lm_name);
        if !field.docs.is_empty() {
            line.push_str(": ");
            line.push_str(field.docs);
        }
        lines.push(line);
    }

    lines.push(String::new());
    lines.push("Your output fields are:".to_string());
    for (i, field) in outputs.iter().enumerate() {
        let type_name = type_name(field.ty, Some(types));
        let mut line = format!("{}. `{}` ({type_name})", i + 1, field.lm_name);
        if !field.docs.is_empty() {
            line.push_str(": ");
            line.push_str(field.docs);
        }
        lines.push(line);
    }

    lines.join("\n")
}

fn format_field_structure_view(
    inputs: &[FieldView<'_>],
    outputs: &[FieldView<'_>],
    types: &TypeTable,
) -> String {
    let mut lines = vec![
        "All interactions will be structured in the following way, with the appropriate values filled in.".to_string(),
        String::new(),
    ];

    for field in inputs {
        lines.push(format!("[[ ## {} ## ]]", field.lm_name));
        lines.push(field.lm_name.to_string());
        lines.push(String::new());
    }

    for field in outputs {
        let type_name = type_name(field.ty, Some(types));
        let rendered_schema = schema_block(field.ty, types);
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
    lines.join("\n")
}

fn build_system_view(
    inputs: &[FieldView<'_>],
    outputs: &[FieldView<'_>],
    types: &TypeTable,
    instruction: &str,
    instruction_override: Option<&str>,
) -> String {
    let parts = [
        format_field_descriptions_view(inputs, outputs, types),
        format_field_structure_view(inputs, outputs, types),
        format_response_instructions_view(outputs),
        format_task_description_view(inputs, outputs, instruction, instruction_override),
    ];

    let system = parts.join("\n\n");
    trace!(system_len = system.len(), "formatted schema system prompt");
    system
}

impl ChatAdapter {
    /// Builds a system message from an owned [`SignatureDef`] — no `'static`
    /// requirement anywhere. `types` resolves the def's class/enum references
    /// (for derive-bridged defs, [`SignatureDef::types_of`]).
    ///
    /// The system message includes:
    /// 1. Field descriptions (names, types, doc comments)
    /// 2. Field structure template (the `[[ ## field ## ]]` layout the LM should follow)
    /// 3. Response instructions (which fields to produce, in what order)
    /// 4. Task description (the def's instruction or the override)
    pub fn build_system_def(
        &self,
        def: &SignatureDef,
        types: &TypeTable,
        instruction_override: Option<&str>,
    ) -> String {
        build_system_view(
            &def_views(&def.inputs),
            &def_views(&def.outputs),
            types,
            &def.instruction,
            instruction_override,
        )
    }

    /// System prompt for a provider-native JSON response. Primitive native outputs
    /// use the schema supplied through the API; complex outputs retain its prompt
    /// copy so nested descriptions remain visible to grammar-only providers.
    pub fn build_system_json_def(
        &self,
        def: &SignatureDef,
        types: &TypeTable,
        instruction_override: Option<&str>,
        schema_mode: bool,
    ) -> String {
        let inputs = def_views(&def.inputs);
        let outputs = def_views(&def.outputs);
        let mut parts = vec![format_field_descriptions_view(&inputs, &outputs, types)];
        if schema_mode
            && outputs.iter().all(|field| {
                matches!(
                    field.ty,
                    FieldType::String
                        | FieldType::Int
                        | FieldType::Float
                        | FieldType::Bool
                        | FieldType::Literal(_)
                )
            })
        {
            parts.push(
                "Return compact JSON matching the provider JSON Schema. No markdown.".to_string(),
            );
            let instruction = instruction_override.unwrap_or(&def.instruction);
            if !instruction.is_empty() {
                parts.push(instruction.to_string());
            }
            return parts.join("\n\n");
        }
        parts.push("Return one JSON object containing every output field, using the field names shown above. Do not include markdown or text outside the JSON object.".to_string());
        parts.push(format!(
            "Effective output format: {}.",
            if schema_mode {
                "provider JSON Schema"
            } else {
                "provider JSON mode"
            }
        ));
        let schema =
            serde_json::to_string(&json_schema_for_outputs(def, types)).unwrap_or_default();
        parts.push(format!("The exact output schema is: {schema}"));
        parts.push(format_task_description_view(
            &inputs,
            &outputs,
            &def.instruction,
            instruction_override,
        ));
        parts.join("\n\n")
    }

    /// Formats a value-level input from an owned [`SignatureDef`] as a user
    /// message with `[[ ## field ## ]]` delimiters, with no `'static`
    /// requirement anywhere. Appends the response instructions telling the LM
    /// which output fields to produce.
    ///
    /// Fields absent from `input` are skipped (the historical relaxed path
    /// navigation for flattened structs). Jinja render templates are compiled
    /// per call — dynamic defs own their template strings, and a
    /// process-global cache keyed on them would reintroduce the leak-per-load
    /// RFC 0002 IR-1 removed.
    /// On a template error, returns an explicit marker. Use
    /// [`try_format_input_def`](Self::try_format_input_def) to handle the error.
    pub fn format_input_def(&self, def: &SignatureDef, input: &JsonMap) -> String {
        self.try_format_input_def(def, input)
            .unwrap_or_else(|_| "[[ ## input_render_error ## ]]".to_string())
    }

    /// Fallible input formatting for runtime callers that must reject bad templates.
    pub fn try_format_input_def(&self, def: &SignatureDef, input: &JsonMap) -> Result<String> {
        self.try_format_input_with_response(
            def,
            input,
            &format_response_instructions_view(&def_views(&def.outputs)),
        )
    }

    fn try_format_input_with_response(
        &self,
        def: &SignatureDef,
        input: &JsonMap,
        response_instruction: &str,
    ) -> Result<String> {
        let mut result = String::new();
        let mut input_context = None;
        for field in def.inputs.iter() {
            let Some(value) = input.get(&*field.name) else {
                continue;
            };
            result.push_str(&format!("[[ ## {} ## ]]\n", field.lm_name));
            if matches!(&field.render, RenderSpec::Jinja(_)) {
                input_context.get_or_insert_with(|| {
                    MiniJinjaValue::from_serialize(build_input_context_def(def, input))
                });
            }
            result.push_str(&render_input_field_def(
                field,
                value,
                input_context.as_ref(),
            )?);
            result.push_str("\n\n");
        }

        result.push_str(response_instruction);
        Ok(result)
    }

    /// Delimited inputs with a JSON-object response instruction.
    pub fn format_input_json_def(&self, def: &SignatureDef, input: &JsonMap) -> String {
        self.try_format_input_json_def(def, input)
            .unwrap_or_else(|_| "[[ ## input_render_error ## ]]".to_string())
    }

    /// Fallible JSON-response input formatting, sharing the alias-aware render path.
    pub fn try_format_input_json_def(&self, def: &SignatureDef, input: &JsonMap) -> Result<String> {
        self.try_format_input_with_response(
            def,
            input,
            "Return a JSON object with every output field.",
        )
    }

    pub(crate) fn build_system_scalar_def(
        &self,
        def: &SignatureDef,
        types: &TypeTable,
        instruction: &str,
    ) -> String {
        format!(
            "{}\n\nReturn only the output field's JSON value, without an object wrapper, markdown or explanation.\n\n{}",
            format_field_descriptions_view(
                &def_views(&def.inputs),
                &def_views(&def.outputs),
                types
            ),
            instruction
        )
    }

    pub(crate) fn try_format_input_scalar_def(
        &self,
        def: &SignatureDef,
        input: &JsonMap,
    ) -> Result<String> {
        self.try_format_input_with_response(def, input, &format!(
            "Compute `{}` from the inputs and return only its JSON value, without an object wrapper.",
            def.outputs[0].lm_name,
        ))
    }

    /// Formats a value-level output map as an assistant message for few-shot
    /// demos: each output field delimited with `[[ ## field ## ]]` markers,
    /// ending with `[[ ## completed ## ]]`.
    ///
    /// Fields absent from `output` are skipped (the historical relaxed path
    /// navigation for flattened structs).
    pub fn format_output_def(&self, def: &SignatureDef, output: &JsonMap) -> String {
        let mut sections = Vec::new();
        for field in def.outputs.iter() {
            if let Some(value) = output.get(&*field.name) {
                sections.push(format!(
                    "[[ ## {} ## ]]\n{}",
                    field.lm_name,
                    format_json_value_for_prompt(value)
                ));
            }
        }
        let mut result = sections.join("\n\n");
        result.push_str("\n\n[[ ## completed ## ]]\n");
        result
    }

    pub fn format_output_json_def(
        &self,
        def: &SignatureDef,
        types: &TypeTable,
        output: &JsonMap,
    ) -> String {
        let value = alias_output_fields(Value::Object(output.clone()), def, types);
        serde_json::to_string_pretty(&value).unwrap_or_else(|_| "{}".to_string())
    }

    #[allow(clippy::result_large_err)]
    /// Parses an LM response against an owned [`SignatureDef`] into a value-level
    /// output map, with no `'static` requirement anywhere.
    ///
    /// The returned [`JsonMap`] is keyed by canonical field name
    /// (`FieldDef::name`); `types` resolves class/enum references during
    /// coercion. Constraint expressions are compiled per call (owned strings —
    /// no global cache, no leak).
    #[tracing::instrument(name = "dsrs.adapter.parse", level = "debug", skip_all,
        fields(signature = %def.name, fields = def.outputs.len()))]
    pub fn parse_output_def(
        &self,
        def: &SignatureDef,
        types: &TypeTable,
        response: &Message,
    ) -> std::result::Result<(JsonMap, IndexMap<String, FieldMeta>), ParseError> {
        let content = response.text_content_cow();
        let sections = parse_sections_cow(&content);

        let mut metas = IndexMap::new();
        let mut errors = Vec::new();
        let mut output = JsonMap::new();

        for field in def.outputs.iter() {
            let raw_text: &str = match sections.get(&*field.lm_name) {
                Some(text) => text.as_ref(),
                None => {
                    debug!(field = %field.name, "missing output field in response");
                    errors.push(ParseError::MissingField {
                        field: field.name.to_string(),
                        raw_response: content.to_string(),
                    });
                    continue;
                }
            };

            let (coerced, nested_checks) = match coerce_with_checks(raw_text, &field.ty, types) {
                Ok(value) => value,
                Err(err) => {
                    let expected_type = type_name(&field.ty, Some(types));
                    debug!(
                        field = %field.name,
                        expected_type = %expected_type,
                        raw_text_len = raw_text.len(),
                        "value-level coercion failed"
                    );
                    errors.push(ParseError::CoercionFailed {
                        field: field.name.to_string(),
                        expected_type,
                        raw_text: raw_text.to_string(),
                        source: JsonishError::from(err),
                    });
                    continue;
                }
            };

            let mut checks = nested_checks
                .into_iter()
                .map(|check| ConstraintResult {
                    label: check.label,
                    expression: check.expression,
                    passed: check.passed,
                })
                .collect::<Vec<_>>();
            checks.extend(evaluate_field_constraints(
                field,
                &coerced.value,
                &mut errors,
            ));

            metas.insert(
                field.name.to_string(),
                FieldMeta {
                    raw_text: raw_text.to_string(),
                    flags: coerced.flags,
                    checks,
                },
            );
            output.insert(field.name.to_string(), coerced.value);
        }

        if !errors.is_empty() {
            debug!(errors = errors.len(), "value-level parse returned errors");
            let partial = if output.is_empty() {
                None
            } else {
                Some(Value::Object(output))
            };
            return Err(ParseError::Multiple { errors, partial });
        }

        Ok((output, metas))
    }

    /// Parses provider-native JSON output through the same coercion and constraint stage
    /// as delimiter responses. JSON object keys may use LM-facing aliases.
    pub fn parse_output_json_def(
        &self,
        def: &SignatureDef,
        types: &TypeTable,
        response: &Message,
    ) -> std::result::Result<(JsonMap, IndexMap<String, FieldMeta>), ParseError> {
        let content = response.text_content_cow();
        let parsed: Value =
            serde_json::from_str(&content).map_err(|err| ParseError::CoercionFailed {
                field: "<response>".into(),
                expected_type: "JSON object".into(),
                raw_text: content.to_string(),
                source: JsonishError::from(anyhow::anyhow!(err)),
            })?;
        self.coerce_json_output(def, types, parsed, &content)
    }

    pub(crate) fn parse_output_scalar_def(
        &self,
        def: &SignatureDef,
        types: &TypeTable,
        response: &Message,
    ) -> std::result::Result<(JsonMap, IndexMap<String, FieldMeta>), ParseError> {
        let content = response.text_content_cow();
        let parsed: Value =
            serde_json::from_str(&content).map_err(|err| ParseError::CoercionFailed {
                field: "<response>".into(),
                expected_type: "JSON scalar".into(),
                raw_text: content.to_string(),
                source: JsonishError::from(anyhow::anyhow!(err)),
            })?;
        if parsed.is_object() || parsed.is_array() {
            return Err(ParseError::CoercionFailed {
                field: "<response>".into(),
                expected_type: "JSON scalar".into(),
                raw_text: content.to_string(),
                source: JsonishError::from(anyhow::anyhow!("expected JSON scalar")),
            });
        }
        let field = &def.outputs[0];
        let values = JsonMap::from_iter([(field.lm_name.to_string(), parsed)]);
        self.coerce_json_output(def, types, Value::Object(values), &content)
    }

    fn coerce_json_output(
        &self,
        def: &SignatureDef,
        types: &TypeTable,
        parsed: Value,
        content: &str,
    ) -> std::result::Result<(JsonMap, IndexMap<String, FieldMeta>), ParseError> {
        let normalized = canonicalize_output_aliases(parsed, def, types);
        let Some(values) = normalized.as_object() else {
            return Err(ParseError::CoercionFailed {
                field: "<response>".into(),
                expected_type: "JSON object".into(),
                raw_text: content.to_string(),
                source: JsonishError::from(anyhow::anyhow!("expected JSON object")),
            });
        };
        let mut output = JsonMap::new();
        let mut metas = IndexMap::new();
        let mut errors = Vec::new();
        for field in def.outputs.iter() {
            let Some(value) = values.get(field.name.as_ref()).cloned() else {
                errors.push(ParseError::MissingField {
                    field: field.name.to_string(),
                    raw_response: content.to_string(),
                });
                continue;
            };
            let value_text = value.to_string();
            let (coerced, nested_checks) = match coerce_value_with_checks(value, &field.ty, types) {
                Ok(value) => value,
                Err(err) => {
                    errors.push(ParseError::CoercionFailed {
                        field: field.name.to_string(),
                        expected_type: type_name(&field.ty, Some(types)),
                        raw_text: value_text,
                        source: JsonishError::from(err),
                    });
                    continue;
                }
            };
            let mut checks: Vec<ConstraintResult> = nested_checks
                .into_iter()
                .map(|check| ConstraintResult {
                    label: check.label,
                    expression: check.expression,
                    passed: check.passed,
                })
                .collect();
            checks.extend(evaluate_field_constraints(
                field,
                &coerced.value,
                &mut errors,
            ));
            metas.insert(
                field.name.to_string(),
                FieldMeta {
                    raw_text: value_text,
                    flags: coerced.flags,
                    checks,
                },
            );
            output.insert(field.name.to_string(), coerced.value);
        }
        if !errors.is_empty() {
            let partial = (!output.is_empty()).then(|| Value::Object(output));
            return Err(ParseError::Multiple { errors, partial });
        }
        Ok((output, metas))
    }

    /// Splits raw LM response text into named sections by `[[ ## field ## ]]` delimiters.
    ///
    /// Returns an ordered map of field_name → section_content. The `completed` marker
    /// is included as a section (usually empty). Duplicate section names keep the first
    /// occurrence. Content before the first delimiter is discarded.
    pub fn parse_sections(content: &str) -> IndexMap<String, String> {
        crate::adapter::chat::parse_sections(content)
    }
}

fn evaluate_field_constraints(
    field: &crate::ir::FieldDef,
    value: &Value,
    errors: &mut Vec<ParseError>,
) -> Vec<ConstraintResult> {
    let mut checks = Vec::new();
    for constraint in field.constraints.iter() {
        let passed = evaluate_expression(&constraint.expr, value);
        match constraint.kind {
            ConstraintKind::Assert if !passed => {
                debug!(field = %field.name, label = %constraint.label, "value-level assert constraint failed");
                errors.push(ParseError::AssertFailed {
                    field: field.name.to_string(),
                    label: constraint.label.to_string(),
                    expression: constraint.expr.to_string(),
                    value: value.clone(),
                });
            }
            ConstraintKind::Check => checks.push(ConstraintResult {
                label: constraint.label.to_string(),
                expression: constraint.expr.to_string(),
                passed,
            }),
            ConstraintKind::Assert => {}
        }
    }
    checks
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
                    slice
                        .lines()
                        .collect::<Vec<_>>()
                        .join("\n")
                        .trim()
                        .to_string(),
                )
            } else {
                std::borrow::Cow::Borrowed(slice)
            };
            (name, text)
        })
        .collect()
}

fn format_json_value_for_prompt(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Null => "null".to_string(),
        other => serde_json::to_string(other).unwrap_or_else(|_| "<error>".to_string()),
    }
}

/// Renders one input field of a [`SignatureDef`], honoring its [`RenderSpec`].
///
/// The Jinja arm compiles the template per call: templates are owned strings on
/// a runtime [`SignatureDef`], so there is no `&'static str` key to cache on.
fn render_input_field_def(
    field: &crate::ir::FieldDef,
    value: &Value,
    input_context: Option<&MiniJinjaValue>,
) -> Result<String> {
    match &field.render {
        RenderSpec::Default => Ok(match value {
            Value::String(s) => s.clone(),
            _ => serde_json::to_string(value).unwrap_or_else(|_| "<error>".to_string()),
        }),
        RenderSpec::Format(format) => Ok(crate::typesys::format_value(value, format)),
        RenderSpec::Jinja(template) => {
            let input_context = input_context.ok_or_else(|| {
                anyhow::anyhow!("missing input render context for `{}`", field.name)
            })?;
            let mut env = build_input_render_environment();
            env.add_template(INPUT_RENDER_TEMPLATE_NAME, template)
                .with_context(|| {
                    format!(
                        "failed to compile input render template for `{}`",
                        field.name
                    )
                })?;
            let compiled = env
                .get_template(INPUT_RENDER_TEMPLATE_NAME)
                .with_context(|| {
                    format!("failed to load input render template for `{}`", field.name)
                })?;

            let field_context = json!({
                    "name": field.lm_name,
                    "rust_name": field.name,
                    "type": type_name(&field.ty, None),
            });
            let context = minijinja::context! {
                this => value,
                input => input_context,
                field => field_context,
                vars => Value::Object(Map::new()),
            };

            compiled
                .render(context)
                .with_context(|| format!("failed to render input field `{}`", field.name))
        }
    }
}

/// Alias-augmented input context for def-lane Jinja templates: templates can
/// address a field by canonical name (`input.question`) or LM alias
/// (`input.query`), mirroring [`build_input_context_value`].
fn build_input_context_def(def: &SignatureDef, input: &JsonMap) -> Value {
    let mut root = input.clone();
    for field in def.inputs.iter() {
        if field.lm_name == field.name {
            continue;
        }
        if let Some(value) = root.get(&*field.name).cloned() {
            root.entry(field.lm_name.to_string()).or_insert(value);
        }
    }
    Value::Object(root)
}
