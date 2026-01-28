#![cfg(feature = "rlm")]

use crate::rlm_core::RlmInputFields;
use crate::{ConstraintKind, Signature, TypeIR};

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
        input.rlm_variable_descriptions()
    }

    pub fn build_prompt<S>(
        &self,
        variable_descriptions: &str,
        schema: &str,
        history: &[ReplHistoryEntry],
    ) -> String
    where
        S: Signature,
    {
        let mut prompt = self.action_instructions::<S>();

        if !variable_descriptions.trim().is_empty() {
            prompt.push_str("\n\nInput Variables:\n");
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
            prompt.push_str("\n\nHistory:\n");
            prompt.push_str(&render_history(history, self.config.max_history_output_chars));
        }

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
        let mut prompt = String::new();
        prompt.push_str("You are performing fallback extraction for a typed signature.\n");
        prompt.push_str("Use the inputs, schema, and REPL history to extract the final output.\n");

        if !variable_descriptions.trim().is_empty() {
            prompt.push_str("\nInputs:\n");
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
            prompt.push_str("\n\nREPL history:\n");
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
    let fields = S::input_fields();
    if fields.is_empty() {
        return "(none)".to_string();
    }
    fields
        .iter()
        .map(|field| format!("`{}`", field.name))
        .collect::<Vec<_>>()
        .join(", ")
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
