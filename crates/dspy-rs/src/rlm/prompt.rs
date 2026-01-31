#![cfg(feature = "rlm")]

/// DSPy-compatible action instructions template for RLM prompts.
///
/// Placeholder requirements:
/// - `{inputs}`: comma-separated backticked variable names, e.g. `` `issues`, `query` ``.
/// - `{output_fields}`: bullet list of output fields, one per line, formatted as:
///   `- name: type` with optional inline description/constraints (e.g. `- summary: string  # short summary`).
/// - `{final_output_names}`: comma-separated output assignments, e.g. `summary=summary, count=count`.
/// - `{max_llm_calls}`: integer from `RlmConfig.max_llm_calls`.
///
/// The line beginning with `5. MINIMIZE RETYPING (INPUTS & OUTPUTS)` must be kept verbatim.
pub const ACTION_INSTRUCTIONS_TEMPLATE: &str = r#"You are tasked with producing the following outputs given the inputs {inputs}:
{output_fields}

You have access to a Python REPL environment. Write Python code and it will be executed. You will see the output, then write more code based on what you learned. This is an iterative process.

Available:
- Variables: {inputs} (your input data)
- `llm_query(prompt)` - query a sub-LLM (~500K char capacity) for semantic analysis
- `llm_query_batched(prompts)` - query multiple prompts concurrently (much faster for multiple queries)
- `print()` - ALWAYS print to see results
- `SUBMIT({final_output_names})` - submit final output when done
- Standard libraries: re, json, collections, math, etc.

IMPORTANT: This is ITERATIVE. Each code block you write will execute, you'll see the output, then you decide what to do next. Do NOT try to solve everything in one step.

1. EXPLORE FIRST - Look at your data before processing it. Print samples, check types/lengths, understand the structure.
2. ITERATE - Write small code snippets, observe outputs, then decide next steps. State persists between iterations.
3. VERIFY BEFORE SUBMITTING - If results seem wrong (zeros, empty, unexpected), reconsider your approach.
4. USE llm_query FOR SEMANTICS - String matching finds WHERE things are; llm_query understands WHAT things mean.
5. MINIMIZE RETYPING (INPUTS & OUTPUTS) - When values are long, precise, or error-prone (IDs, numbers, code, quotes), re-access them via variables and parse/compute in code instead of retyping. Use small, targeted prints to sanity-check, but avoid manual copying when variables can carry the exact value.
6. SUBMIT ONLY AFTER SEEING OUTPUTS - SUBMIT ends the current run immediately. If you need to inspect printed output, run it in one step, review the result, then call SUBMIT in a later step.

You have max {max_llm_calls} sub-LLM calls. When done, call SUBMIT() with your output.
"#;

use std::collections::HashSet;

use crate::{ConstraintKind, OutputFormatContent, Signature, StreamingMode, TypeIR};

/// Generate the typed RLM preamble with variable descriptions and output schema.
pub fn generate_typed_preamble<S: Signature>(variable_descriptions: &str) -> String {
    let instruction = S::instruction();
    let output_schema = generate_output_schema_description::<S>();

    format!(
        r#"You are tasked with a computation that requires structured output.

## Task
{instruction}

## Input Variables
{variable_descriptions}

## Output Schema
Call SUBMIT() with the following fields when you have your answer:

{output_schema}

## Available Commands

1. Python code in ```repl``` blocks:
   - Access input variables directly by name
   - `llm_query(prompt)` - Query a sub-LLM for semantic analysis (~500K char capacity)
   - `llm_query_batched(prompts)` - Batch query multiple prompts concurrently
   - `SUBMIT(field1=value1, ...)` - Submit your final answer (validates against schema)
   - `print()` - Print intermediate results

2. SUBMIT validates your output and returns:
   - "✓ SUBMIT successful!" on valid output
   - Detailed error messages if validation fails (fix and retry)

## Guidelines

1. EXPLORE FIRST - Examine input variables before processing
2. ITERATE - Write small code snippets, observe outputs, adjust
3. USE llm_query FOR SEMANTICS - String matching finds WHERE; llm_query understands WHAT
4. VERIFY BEFORE SUBMITTING - Check results look sensible
5. SUBMIT when ready - It validates types and constraints automatically

## Constraints

- Soft checks (⚠): Violations are reported but output is accepted
- Hard asserts (❌): Violations require you to fix and resubmit

Example:
```repl
# Explore the data
print(type(trajectories), len(trajectories))
print(trajectories[0])

# Process
results = []
for t in trajectories[:5]:
    analysis = llm_query(f"Summarize: {{t}}")
    results.append(analysis)

# Submit
SUBMIT(summary=results[0], count=len(results))
```
"#
    )
}

pub fn generate_output_schema_description<S: Signature>() -> String {
    let mut desc = String::new();
    desc.push_str("SUBMIT(\n");

    for field in S::output_fields() {
        let type_ir = (field.type_ir)();
        let type_name = format_type_for_prompt(&type_ir);

        desc.push_str(&format!("    {}={},", field.name, type_name));
        if !field.description.is_empty() {
            desc.push_str(&format!("  # {}", field.description));
        }
        desc.push('\n');

        for constraint in field.constraints {
            let icon = match constraint.kind {
                ConstraintKind::Check => "⚠",
                ConstraintKind::Assert => "❌",
            };
            desc.push_str(&format!(
                "        {} {}: {}\n",
                icon, constraint.label, constraint.expression
            ));
        }
    }

    desc.push_str(")\n");
    desc
}

fn format_type_for_prompt(type_ir: &TypeIR) -> String {
    type_ir.diagnostic_repr().to_string()
}

const SHAPE_MAX_DEPTH: usize = 3;
const SHAPE_MAX_FIELDS: usize = 20;

/// Render a BAML-style shape description for a type.
pub fn format_baml_shape(output_format: &OutputFormatContent, type_ir: &TypeIR) -> String {
    let mut visited = HashSet::new();
    format_shape_inner(output_format, type_ir, SHAPE_MAX_DEPTH, &mut visited)
}

fn format_shape_inner(
    output_format: &OutputFormatContent,
    type_ir: &TypeIR,
    depth: usize,
    visited: &mut HashSet<String>,
) -> String {
    match type_ir {
        TypeIR::Class { name, mode, .. } => {
            format_class_shape(output_format, name, mode, depth, visited)
        }
        TypeIR::RecursiveTypeAlias { name, .. } => {
            if let Some(next) = output_format.structural_recursive_aliases.get(name) {
                if visited.contains(name) {
                    return format!("{name} {{ ... }}");
                }
                format_shape_inner(output_format, next, depth, visited)
            } else {
                name.clone()
            }
        }
        _ => format_type_name(output_format, type_ir),
    }
}

fn format_class_shape(
    output_format: &OutputFormatContent,
    name: &str,
    mode: &StreamingMode,
    depth: usize,
    visited: &mut HashSet<String>,
) -> String {
    if depth == 0 || visited.contains(name) {
        return format!("{name} {{ ... }}");
    }

    let class = output_format
        .classes
        .get(&(name.to_string(), *mode))
        .or_else(|| output_format.classes.get(&(name.to_string(), StreamingMode::NonStreaming)))
        .or_else(|| output_format.classes.get(&(name.to_string(), StreamingMode::Streaming)));

    let Some(class) = class else {
        return name.to_string();
    };

    visited.insert(name.to_string());

    let mut lines = Vec::new();
    lines.push(format!("{name} {{"));

    let mut field_iter = class.fields.iter();
    for (idx, (field_name, field_type, field_desc, _, _)) in field_iter
        .by_ref()
        .take(SHAPE_MAX_FIELDS)
        .enumerate()
    {
        let field_type_string = format_field_type(output_format, field_type, depth - 1, visited);
        let mut field_lines = field_type_string.lines();
        let first = field_lines.next().unwrap_or_default();
        let mut line = format!("  {}: {}", field_name.rendered_name(), first);
        if let Some(desc) = field_desc {
            if !desc.is_empty() {
                line.push_str(&format!(" // {desc}"));
            }
        }
        lines.push(line);

        for continuation in field_lines {
            lines.push(format!("    {continuation}"));
        }

        if idx + 1 == SHAPE_MAX_FIELDS {
            break;
        }
    }

    let remaining = class.fields.len().saturating_sub(SHAPE_MAX_FIELDS);
    if remaining > 0 {
        lines.push(format!("  ... (+{remaining} more)"));
    }

    lines.push("}".to_string());
    visited.remove(name);
    lines.join("\n")
}

fn format_field_type(
    output_format: &OutputFormatContent,
    type_ir: &TypeIR,
    depth: usize,
    visited: &mut HashSet<String>,
) -> String {
    match type_ir {
        TypeIR::Class { .. } | TypeIR::RecursiveTypeAlias { .. } => {
            format_shape_inner(output_format, type_ir, depth, visited)
        }
        _ => format_type_name(output_format, type_ir),
    }
}

fn format_type_name(output_format: &OutputFormatContent, type_ir: &TypeIR) -> String {
    match type_ir {
        TypeIR::Primitive(type_value, _) => type_value.basename().to_string(),
        TypeIR::Enum { name, .. } => output_format
            .enums
            .get(name)
            .map(|enm| {
                let values: Vec<String> = enm
                    .values
                    .iter()
                    .map(|(value, _, _)| format!("\"{}\"", value.rendered_name()))
                    .collect();
                values.join(" | ")
            })
            .unwrap_or_else(|| name.clone()),
        TypeIR::Literal(literal, _) => match literal {
            crate::baml_bridge::baml_types::LiteralValue::String(_) => "string".to_string(),
            crate::baml_bridge::baml_types::LiteralValue::Int(_) => "int".to_string(),
            crate::baml_bridge::baml_types::LiteralValue::Bool(_) => "bool".to_string(),
        },
        TypeIR::Class { name, .. } => name.clone(),
        TypeIR::List(inner, _) => format!("list[{}]", format_type_name(output_format, inner)),
        TypeIR::Map(key, value, _) => format!(
            "map<{}, {}>",
            format_type_name(output_format, key),
            format_type_name(output_format, value)
        ),
        TypeIR::Tuple(items, _) => {
            let parts: Vec<String> = items
                .iter()
                .map(|item| format_type_name(output_format, item))
                .collect();
            format!("({})", parts.join(", "))
        }
        TypeIR::Union(union, _) => {
            let parts: Vec<String> = union
                .iter_include_null()
                .into_iter()
                .map(|item| format_type_name(output_format, item))
                .collect();
            parts.join(" | ")
        }
        TypeIR::RecursiveTypeAlias { name, .. } => name.clone(),
        _ => type_ir.diagnostic_repr().to_string(),
    }
}
