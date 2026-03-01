use crate::{ConstraintKind, Signature, SignatureSchema};
use bamltype::baml_types::ir_type::{TypeGeneric, UnionTypeViewGeneric};
use bamltype::internal_baml_jinja::types::OutputFormatContent;

use super::RlmConfig;
use super::previews::{is_primitive_input_type, render_type_shape, type_label};

const PATTERNS_BLOCK: &str = r#"## Analysis Patterns

llm_query handles meaning. string operations handle structure.
combine them freely. here are common compositions:

# Semantic filter
# batch queries work best on focused, pre-filtered sets.
# narrow with .search() first, then batch-analyze the results.
relevant = [x for x, r in zip(items, llm_query_batched(
    [f"Is {x.label} about {topic}? yes/no" for x in items[:5]]
)) if 'yes' in r.lower()]

# Chain
findings = llm_query_batched([f"Key finding: {x}" for x in relevant])
answer = llm_query(f"Synthesize:\n" + "\n---\n".join(findings))

# Map-reduce
chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]
parts = llm_query_batched([f"Summarize: {c}" for c in chunks])
summary = llm_query(f"Combine:\n" + "\n".join(parts))

# Direct
quick_answer = llm_query(f"Answer directly: {question}")

# SUBMIT safely for long answers
# Build long strings via variables first to avoid unterminated triple-quote/parens errors.
direct_answer = (
    "Line 1...\n"
    "Line 2..."
)
key_findings = (
    "1. First finding...\n"
    "2. Second finding..."
)

SUBMIT(direct_answer=direct_answer, key_findings=key_findings)

## Common Mistakes
# DON'T read everything then reason:
for s in sessions:
    print(s.render())  # you just burned your entire context

# DON'T use regex when llm_query is better:
matches = [s for s in sessions if re.search(r'auth.*fail', s.render())]
# this misses "authentication broke" and "login stopped working"

# DO delegate understanding:
relevant = llm_query_batched([
    f"does this session discuss auth failures? yes/no\n{s.render()[:3000]}"
    for s in sessions
])

# DO accumulate understanding across turns:
findings = []
findings.append(f"session 3: {observation}")
findings.append(f"session 7 contradicts: {counter}")"#;

pub(super) fn render_action_instruction<S: Signature>(
    _config: &RlmConfig,
    instruction_override: Option<&str>,
    variable_schemas: &str,
) -> String {
    let schema = SignatureSchema::of::<S>();
    let task = instruction_override
        .unwrap_or_else(|| schema.instruction())
        .trim();

    let mut lines = vec![
        "## Task".to_string(),
        task.to_string(),
        String::new(),
        "## Input Variables".to_string(),
    ];

    if variable_schemas.trim().is_empty() {
        lines.push("(No complex input variables.)".to_string());
    } else {
        lines.push(variable_schemas.trim().to_string());
    }

    lines.push(String::new());
    lines.push("## Output Schema".to_string());
    lines.push("Call SUBMIT() with the following fields when you have your answer:".to_string());
    lines.push(String::new());
    lines.push("Your output fields are:".to_string());

    let output_format = schema.output_format();
    for (index, field) in schema.output_fields().iter().enumerate() {
        let type_name = type_label(&field.type_ir, output_format);
        let mut doc_lines = field.docs.lines().map(str::trim_end).collect::<Vec<_>>();
        while doc_lines.first().is_some_and(|line| line.trim().is_empty()) {
            doc_lines.remove(0);
        }
        while doc_lines.last().is_some_and(|line| line.trim().is_empty()) {
            doc_lines.pop();
        }

        if let Some(first_doc) = doc_lines.first() {
            lines.push(format!(
                "{}. `{}` ({}): {}",
                index + 1,
                field.lm_name,
                type_name,
                first_doc
            ));
            for line in doc_lines.iter().skip(1) {
                lines.push(format!("   {}", line));
            }
        } else {
            lines.push(format!(
                "{}. `{}` ({})",
                index + 1,
                field.lm_name,
                type_name
            ));
        }

        if let Some(variants) = enum_variants_line(&field.type_ir, output_format) {
            lines.push(format!("   Valid values: {variants}"));
        }

        if !is_simple_output_type(&field.type_ir) {
            lines.push("   Schema:".to_string());
            for line in render_type_shape(&field.type_ir, output_format, 5) {
                lines.push(line);
            }
        }

        lines.push(String::new());
    }

    let submit_assignments = schema
        .output_fields()
        .iter()
        .map(|field| format!("{}=...", field.lm_name))
        .collect::<Vec<_>>()
        .join(", ");
    lines.push(format!("When final, call SUBMIT({submit_assignments})."));

    lines.push(String::new());
    lines.push("## Available Tools".to_string());
    lines.push("Available in the REPL:".to_string());
    lines.push("- Input variables accessible directly by name".to_string());
    lines.push("- `llm_query(prompt)` — ask a sub-model to analyze text. processed outside your context window, so large inputs won't crowd your reasoning.".to_string());
    lines.push("- `llm_query_batched(prompts)` — batch query concurrently".to_string());
    lines.push("- `SUBMIT(field1=value1, ...)` — submit final answer".to_string());
    lines.push("- `print()` — ALWAYS print to see results".to_string());
    lines.push("- Standard libraries available (import as needed)".to_string());
    lines.push("Plus any user-provided tools with their descriptions.".to_string());

    lines.push(String::new());
    lines.push("## Guidelines".to_string());
    lines.push("Response format contract:".to_string());
    lines.push("- Output code only.".to_string());
    lines.push("- No prose.".to_string());
    lines.push("- No markdown fences.".to_string());
    lines.push("- If needed, put reasoning only in Python comments.".to_string());
    lines.push(String::new());
    lines.push("1. EXPLORE FIRST - Look at your data before processing it.".to_string());
    lines.push("2. ITERATE - Write small code snippets, observe, decide next steps.".to_string());
    lines.push("3. VERIFY BEFORE SUBMITTING - If results seem wrong, reconsider.".to_string());
    lines.push(
        "4. STRING OPS FOR SYNTAX, llm_query FOR MEANING — they interleave freely. The mistake is using one where the other belongs."
            .to_string(),
    );
    lines.push(
        "5. ACCUMULATE UNDERSTANDING — use named variables to build your evolving model of the problem, not just cache results."
            .to_string(),
    );
    lines.push(
        "6. SUBMIT ONLY AFTER SEEING OUTPUTS — verify your answer looks right before calling SUBMIT.".to_string(),
    );

    lines.push(String::new());
    lines.push("## Constraints".to_string());
    lines.push("- Soft checks use ⚠. Hard assertions use ❌.".to_string());
    let mut any_constraints = false;
    for field in schema.output_fields() {
        for constraint in field.constraints {
            any_constraints = true;
            let marker = match constraint.kind {
                ConstraintKind::Check => "⚠ soft",
                ConstraintKind::Assert => "❌ hard",
            };
            lines.push(format!(
                "- `{}`: {marker} - {} ({})",
                field.lm_name, constraint.label, constraint.expression
            ));
        }
    }
    if !any_constraints {
        lines.push("- No explicit soft checks or hard assertions for this signature.".to_string());
    }

    lines.push(String::new());
    lines.push(PATTERNS_BLOCK.to_string());

    while lines.last().is_some_and(String::is_empty) {
        lines.pop();
    }

    lines.join("\n")
}

pub(super) fn render_extract_instruction<S: Signature>(
    instruction_override: Option<&str>,
) -> String {
    let schema = SignatureSchema::of::<S>();
    let task = instruction_override
        .unwrap_or_else(|| schema.instruction())
        .trim();

    [
        "The following REPL session was generated for this task:",
        task,
        "",
        "Based on the execution history, extract the final outputs. Review what was computed and provide the best answer from the trajectory.",
    ]
    .join("\n")
}

fn is_simple_output_type(type_ir: &crate::TypeIR) -> bool {
    match type_ir {
        TypeGeneric::Union(union, _) => match union.view() {
            UnionTypeViewGeneric::Optional(inner) => is_simple_output_type(inner),
            _ => false,
        },
        TypeGeneric::List(inner, _) => is_simple_output_type(inner),
        TypeGeneric::Primitive(..)
        | TypeGeneric::Enum { .. }
        | TypeGeneric::Literal(..)
        | TypeGeneric::Top(..) => true,
        _ => is_primitive_input_type(type_ir),
    }
}

fn enum_variants_line(
    type_ir: &crate::TypeIR,
    output_format: &OutputFormatContent,
) -> Option<String> {
    let enum_name = match type_ir {
        TypeGeneric::Enum { name, .. } => Some(name.as_str()),
        TypeGeneric::Union(union, _) => match union.view() {
            UnionTypeViewGeneric::Optional(inner) => match inner {
                TypeGeneric::Enum { name, .. } => Some(name.as_str()),
                _ => None,
            },
            _ => None,
        },
        _ => None,
    }?;

    let enm = output_format.enums.get(enum_name)?;
    let variants = enm
        .values
        .iter()
        .map(|(name, _)| name.rendered_name().to_string())
        .collect::<Vec<_>>();
    if variants.is_empty() {
        None
    } else {
        Some(variants.join(" | "))
    }
}

#[cfg(test)]
mod tests {
    use crate::BamlType;
    use crate::Signature;

    use super::*;

    #[derive(Signature, Clone, Debug)]
    /// Solve the query against the corpus.
    struct PromptSig {
        #[input]
        papers: Vec<String>,

        #[input]
        question: String,

        #[output]
        #[assert("this.len() > 0", label = "non_empty")]
        answer: String,
    }

    #[derive(Clone, Debug)]
    #[BamlType]
    enum FailureMode {
        Ignorance,
        DiscoveryFailure,
    }

    #[derive(Signature, Clone, Debug)]
    struct OutputFormatSig {
        #[input]
        question: String,

        #[output]
        tags: Vec<String>,

        #[output]
        mode: FailureMode,

        #[output]
        /// Line one.
        /// - top bullet
        ///   - nested bullet
        notes: String,
    }

    #[test]
    fn includes_new_core_sections() {
        let rendered = render_action_instruction::<PromptSig>(
            &RlmConfig::default(),
            None,
            "Variable: `papers`",
        );

        assert!(rendered.contains("## Task"));
        assert!(rendered.contains("## Input Variables"));
        assert!(rendered.contains("## Output Schema"));
        assert!(rendered.contains("## Available Tools"));
        assert!(rendered.contains("## Guidelines"));
        assert!(rendered.contains("## Constraints"));
        assert!(rendered.contains("## Analysis Patterns"));
        assert!(rendered.contains("No markdown fences"));
        assert!(rendered.contains("SUBMIT safely for long answers"));
    }

    #[test]
    fn system_message_sections_are_in_locked_order() {
        let rendered = render_action_instruction::<PromptSig>(
            &RlmConfig::default(),
            None,
            "Variable: `papers`",
        );

        let idx_task = rendered.find("## Task").expect("task section");
        let idx_inputs = rendered
            .find("## Input Variables")
            .expect("input variables section");
        let idx_output = rendered
            .find("## Output Schema")
            .expect("output schema section");
        let idx_tools = rendered.find("## Available Tools").expect("tools section");
        let idx_guidelines = rendered.find("## Guidelines").expect("guidelines section");
        let idx_constraints = rendered
            .find("## Constraints")
            .expect("constraints section");
        let idx_patterns = rendered
            .find("## Analysis Patterns")
            .expect("patterns section");

        assert!(idx_task < idx_inputs);
        assert!(idx_inputs < idx_output);
        assert!(idx_output < idx_tools);
        assert!(idx_tools < idx_guidelines);
        assert!(idx_guidelines < idx_constraints);
        assert!(idx_constraints < idx_patterns);
    }

    #[test]
    fn extract_instruction_includes_task_and_extraction_guidance() {
        let rendered = render_extract_instruction::<PromptSig>(None);

        assert!(rendered.contains("The following REPL session was generated for this task:"));
        assert!(rendered.contains("Solve the query against the corpus."));
        assert!(rendered.contains("Based on the execution history, extract the final outputs."));
    }

    #[test]
    fn output_section_skips_schema_for_simple_list_and_enum_and_shows_enum_values() {
        let rendered =
            render_action_instruction::<OutputFormatSig>(&RlmConfig::default(), None, "");

        let tags_block = rendered
            .split("1. `tags` (list[string])")
            .nth(1)
            .and_then(|tail| tail.split("2. `mode` (FailureMode)").next())
            .expect("tags block");
        assert!(!tags_block.contains("Schema:"));

        let mode_block = rendered
            .split("2. `mode` (FailureMode)")
            .nth(1)
            .and_then(|tail| tail.split("3. `notes` (string)").next())
            .expect("mode block");
        assert!(!mode_block.contains("Schema:"));
        assert!(mode_block.contains("Valid values: Ignorance | DiscoveryFailure"));
    }

    #[test]
    fn output_docstrings_preserve_leading_whitespace() {
        let rendered =
            render_action_instruction::<OutputFormatSig>(&RlmConfig::default(), None, "");
        assert!(rendered.contains("   - top bullet"));
        assert!(rendered.contains("- nested bullet"));
    }
}
