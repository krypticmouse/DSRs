#![cfg(feature = "rlm")]

use crate::rlm_core::{RlmInputFields, RlmVariable};
use crate::{ConstraintKind, Signature, TypeIR};
use pyo3::types::{PyAnyMethods, PyDict, PyDictMethods, PySequence};
use pyo3::{Bound, Py, PyAny, Python};
use std::collections::HashMap;

use super::config::RlmConfig;
use super::history::{render_history, ReplHistoryEntry};
use super::prompt::{format_baml_shape, generate_output_schema_description, ACTION_INSTRUCTIONS_TEMPLATE};

#[derive(Debug, Clone)]
pub struct RlmAdapter {
    config: RlmConfig,
}

impl RlmAdapter {
    pub fn new(config: RlmConfig) -> Self {
        Self { config }
    }

    pub fn variable_previews<S>(&self, input: &S::Input) -> String
    where
        S: Signature,
        S::Input: RlmInputFields,
    {
        let variables = input.rlm_variables();
        if variables.is_empty() {
            return String::new();
        }

        let mut spec_map = HashMap::new();
        for field in S::input_fields() {
            spec_map.insert(field.name, *field);
        }

        Python::attach(|py| {
            let py_fields: HashMap<String, Py<PyAny>> = input
                .rlm_py_fields(py)
                .into_iter()
                .collect();
            let output_format = S::output_format_content();

            let mut blocks = Vec::new();
            for variable in variables {
                let spec = spec_map.get(variable.name.as_str()).copied();
                let shape = spec
                    .map(|spec| format_baml_shape(output_format, &(spec.type_ir)()))
                    .unwrap_or_default();
                let py_obj = py_fields
                    .get(&variable.name)
                    .map(|value| value.bind(py));

                blocks.push(format_variable_preview(&variable, spec, py_obj.as_deref(), &shape));
            }

            blocks.join("\n\n")
        })
    }

    pub fn build_prompt<S>(
        &self,
        variable_descriptions: &str,
        schema: &str,
        history: &[ReplHistoryEntry],
        iteration: usize,
    ) -> String
    where
        S: Signature,
    {
        let mut prompt = self.action_instructions::<S>();

        if !variable_descriptions.trim().is_empty() {
            prompt.push_str("\n\nvariables_info:\n");
            prompt.push_str(variable_descriptions);
        }

        if !schema.trim().is_empty() {
            prompt.push_str("\n\nOutput schema:\n");
            prompt.push_str(schema.trim());
        }

        let shapes = self.format_output_shapes::<S>();
        if !shapes.is_empty() {
            prompt.push_str("\n\nOutput shapes:\n");
            prompt.push_str(&shapes);
        }

        if !history.is_empty() {
            prompt.push_str("\n\nrepl_history:\n");
            prompt.push_str(&render_history(history, self.config.max_history_output_chars));
        }

        prompt.push_str(&format!(
            "\n\niteration: {}/{}",
            iteration, self.config.max_iterations
        ));

        prompt.push_str("\n\nReturn the next step as a ```repl``` or ```python``` code block. If you are done, call SUBMIT(field=value, ...).\n");
        prompt
    }

    pub fn build_extraction_prompt<S>(
        &self,
        variable_descriptions: &str,
        schema: &str,
        history: &[ReplHistoryEntry],
    ) -> String
    where
        S: Signature,
    {
        let mut prompt = self.action_instructions::<S>();
        prompt.push_str("\n\nYou are performing fallback extraction for a typed signature.\n");
        prompt.push_str("Use the inputs, schema, and REPL history to extract the final output.\n");

        if !variable_descriptions.trim().is_empty() {
            prompt.push_str("\nvariables_info:\n");
            prompt.push_str(variable_descriptions);
        }

        if !schema.trim().is_empty() {
            prompt.push_str("\n\nOutput schema:\n");
            prompt.push_str(schema.trim());
        } else {
            let schema = generate_output_schema_description::<S>();
            if !schema.trim().is_empty() {
                prompt.push_str("\n\nOutput schema:\n");
                prompt.push_str(schema.trim());
            }
        }

        if !history.is_empty() {
            prompt.push_str("\n\nrepl_history:\n");
            prompt.push_str(&render_history(history, self.config.max_history_output_chars));
        }

        prompt.push_str("\n\n");
        prompt.push_str(&format_output_instructions::<S>());
        prompt.push_str("\nRespond with only the structured output and no extra commentary.\n");
        prompt
    }

    fn action_instructions<S: Signature>(&self) -> String {
        let inputs = format_input_names::<S>();
        let output_fields = format_output_fields::<S>();

        ACTION_INSTRUCTIONS_TEMPLATE
            .replace("{inputs}", &inputs)
            .replace("{output_fields}", &output_fields)
            .replace("{max_llm_calls}", &self.config.max_llm_calls.to_string())
    }

    fn format_output_shapes<S: Signature>(&self) -> String {
        let output_format = S::output_format_content();
        let mut blocks = Vec::new();

        for field in S::output_fields() {
            let type_ir = (field.type_ir)();
            let shape = format_baml_shape(output_format, &type_ir);
            if shape.trim().is_empty() {
                continue;
            }

            let mut block = String::new();
            block.push_str(field.name);
            block.push_str(":\n");
            block.push_str(&indent_block(&shape, "  "));
            blocks.push(block);
        }

        blocks.join("\n\n")
    }
}

fn format_input_names<S: Signature>() -> String {
    let mut names = S::input_fields()
        .iter()
        .map(|field| format!("`{}`", field.name))
        .collect::<Vec<_>>();
    names.push("`variables_info`".to_string());
    names.push("`repl_history`".to_string());
    names.push("`iteration`".to_string());
    names.join(", ")
}

fn format_output_fields<S: Signature>() -> String {
    let mut lines = Vec::new();
    for field in S::output_fields() {
        let type_ir = (field.type_ir)();
        let type_name = format_type_name(&type_ir);
        let mut line = format!("- {}: {}", field.name, type_name);
        let mut notes = Vec::new();

        if !field.description.is_empty() {
            notes.push(field.description.to_string());
        }

        if !field.constraints.is_empty() {
            let constraint_summary = field
                .constraints
                .iter()
                .map(|constraint| {
                    let kind = match constraint.kind {
                        ConstraintKind::Check => "check",
                        ConstraintKind::Assert => "assert",
                    };
                    if constraint.label.is_empty() {
                        format!("{kind} {}", constraint.expression)
                    } else {
                        format!("{kind} {}: {}", constraint.label, constraint.expression)
                    }
                })
                .collect::<Vec<_>>()
                .join("; ");
            notes.push(constraint_summary);
        }

        if !notes.is_empty() {
            line.push_str("  # ");
            line.push_str(&notes.join("; "));
        }

        lines.push(line);
    }

    if lines.is_empty() {
        "- (no output fields)".to_string()
    } else {
        lines.join("\n")
    }
}

fn format_type_name(type_ir: &TypeIR) -> String {
    let raw = type_ir.diagnostic_repr().to_string();
    simplify_type_name(&raw)
        .replace("class ", "")
        .replace("enum ", "")
        .replace(" | ", " or ")
        .trim()
        .to_string()
}

fn simplify_type_name(raw: &str) -> String {
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
            let simplified = token.rsplit("::").next().unwrap_or(&token);
            result.push_str(simplified);
        } else {
            result.push(ch);
        }
    }
    result
}

fn format_output_instructions<S: Signature>() -> String {
    let mut fields = S::output_fields().iter();
    let Some(first) = fields.next() else {
        return "Respond with the marker for `[[ ## completed ## ]]`.".to_string();
    };

    let mut message = format!(
        "Respond with the output fields, starting with `[[ ## {} ## ]]`",
        first.name
    );
    for field in fields {
        message.push_str(&format!(", then `[[ ## {} ## ]]`", field.name));
    }
    message.push_str(", and then ending with the marker for `[[ ## completed ## ]]`.");
    message
}

fn indent_block(text: &str, prefix: &str) -> String {
    text.lines()
        .map(|line| {
            if line.is_empty() {
                line.to_string()
            } else {
                format!("{prefix}{line}")
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn format_variable_preview(
    variable: &RlmVariable,
    spec: Option<crate::FieldSpec>,
    py_obj: Option<&Bound<'_, PyAny>>,
    shape: &str,
) -> String {
    let mut output = String::new();
    output.push_str(&format!(
        "Variable: `{}` (access it in your code)\n",
        variable.name
    ));

    let type_summary = variable.type_desc.lines().next().unwrap_or("unknown");
    output.push_str(&format!("Type: {}\n", type_summary));

    let count = collection_len(py_obj);
    if let Some(count) = count {
        if is_collection(py_obj, &variable.type_desc) {
            output.push_str(&format!("Count: {}\n", format_count(count)));
        }
    }

    if let Some(spec) = spec {
        if !spec.description.is_empty() {
            output.push_str(&format!("Description: {}\n", spec.description));
        }
        if !spec.constraints.is_empty() {
            let constraints = spec
                .constraints
                .iter()
                .map(|constraint| {
                    if constraint.label.is_empty() {
                        constraint.expression.to_string()
                    } else {
                        format!("{}: {}", constraint.label, constraint.expression)
                    }
                })
                .collect::<Vec<_>>()
                .join("; ");
            if !constraints.is_empty() {
                output.push_str(&format!("Constraints: {}\n", constraints));
            }
        }
    }

    output.push_str(&format!(
        "Total length: {} characters\n",
        format_count(variable.total_length)
    ));

    let usage = format_usage_hints(variable, py_obj, count);
    if !usage.is_empty() {
        output.push_str("Usage:\n");
        for hint in usage {
            output.push_str(&format!("  - {hint}\n"));
        }
    }

    if !shape.trim().is_empty() {
        output.push_str("Shape:\n");
        output.push_str(&indent_block(shape, "  "));
        output.push('\n');
    }

    let samples = preview_samples(variable, py_obj, count);
    if !samples.is_empty() {
        output.push_str("Preview:\n");
        for sample in samples {
            output.push_str(&format!("  - {sample}\n"));
        }
    }

    output.trim_end().to_string()
}

fn format_usage_hints(
    variable: &RlmVariable,
    py_obj: Option<&Bound<'_, PyAny>>,
    count: Option<usize>,
) -> Vec<String> {
    let mut hints = Vec::new();
    if count.is_some() {
        hints.push(format!("len({})", variable.name));
    }
    if let Some(obj) = py_obj {
        if obj.hasattr("__getitem__").unwrap_or(false) {
            hints.push(format!("{}[i]", variable.name));
        }
    }
    for (name, _) in &variable.properties {
        hints.push(format!("{}.{}", variable.name, name));
    }
    hints
}

fn preview_samples(
    variable: &RlmVariable,
    py_obj: Option<&Bound<'_, PyAny>>,
    count: Option<usize>,
) -> Vec<String> {
    if let Some(obj) = py_obj {
        if let Ok(sequence) = obj.cast::<PySequence>() {
            return preview_sequence(&sequence, count.unwrap_or(0));
        }
        if let Ok(dict) = obj.cast::<PyDict>() {
            return preview_dict(&dict, count.unwrap_or(0));
        }
        if let Some(repr) = repr_string(obj) {
            return vec![truncate_preview_sample(&repr)];
        }
    }

    if !variable.preview.is_empty() {
        return vec![truncate_preview_sample(&variable.preview)];
    }

    Vec::new()
}

fn preview_sequence(sequence: &Bound<'_, PySequence>, count: usize) -> Vec<String> {
    let mut samples = Vec::new();
    let sample_count = count.min(2);
    for idx in 0..sample_count {
        if let Ok(item) = sequence.get_item(idx) {
            if let Some(repr) = repr_string(&item) {
                samples.push(truncate_preview_sample(&repr));
            }
        }
    }
    if count > sample_count {
        samples.push("...".to_string());
    }
    samples
}

fn preview_dict(dict: &Bound<'_, PyDict>, count: usize) -> Vec<String> {
    let mut samples = Vec::new();
    for (idx, (key, value)) in dict.iter().enumerate() {
        if idx >= 2 {
            break;
        }
        let key_repr = repr_string(&key).unwrap_or_else(|| "<key>".to_string());
        let value_repr = repr_string(&value).unwrap_or_else(|| "<value>".to_string());
        samples.push(truncate_preview_sample(&format!("{key_repr}: {value_repr}")));
    }
    if count > samples.len() {
        samples.push("...".to_string());
    }
    samples
}

fn repr_string(obj: &Bound<'_, PyAny>) -> Option<String> {
    obj.repr().ok().and_then(|repr| repr.extract::<String>().ok())
}

fn collection_len(py_obj: Option<&Bound<'_, PyAny>>) -> Option<usize> {
    py_obj.and_then(|obj| obj.len().ok())
}

fn is_collection(py_obj: Option<&Bound<'_, PyAny>>, type_desc: &str) -> bool {
    if let Some(obj) = py_obj {
        if obj.cast::<PySequence>().is_ok() || obj.cast::<PyDict>().is_ok() {
            return true;
        }
    }
    let lowered = type_desc.to_ascii_lowercase();
    lowered.contains("list")
        || lowered.contains("vec")
        || lowered.contains("map")
        || lowered.contains("dict")
        || lowered.contains("hashmap")
}

fn truncate_preview_sample(value: &str) -> String {
    const LIMIT: usize = 100;
    if value.chars().count() <= LIMIT {
        return value.to_string();
    }
    let mut truncated: String = value.chars().take(LIMIT).collect();
    truncated.push_str("...");
    truncated
}

fn format_count(value: usize) -> String {
    let digits = value.to_string();
    let mut formatted = String::with_capacity(digits.len() + digits.len() / 3);
    for (idx, ch) in digits.chars().rev().enumerate() {
        if idx > 0 && idx % 3 == 0 {
            formatted.push(',');
        }
        formatted.push(ch);
    }
    formatted.chars().rev().collect()
}
