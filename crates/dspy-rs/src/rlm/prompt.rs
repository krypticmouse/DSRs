#![cfg(feature = "rlm")]

/// DSPy-compatible action instructions template for RLM prompts.
///
/// Placeholder requirements:
/// - `{inputs}`: comma-separated backticked variable names, e.g. `` `issues`, `query` ``.
/// - `{output_fields}`: bullet list of output fields, one per line, formatted as:
///   `- name: type` with optional inline description/constraints (e.g. `- summary: string  # short summary`).
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
- `SUBMIT(...)` - submit final output when done
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

use crate::{ConstraintKind, Signature, TypeIR};

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

fn generate_output_schema_description<S: Signature>() -> String {
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
