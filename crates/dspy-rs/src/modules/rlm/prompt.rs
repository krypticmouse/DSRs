use crate::{ConstraintKind, Signature, SignatureSchema};

use super::RlmConfig;

const PATTERNS_BLOCK: &str = r#"## Patterns

Write programs that route data through llm_query:

# Filter a collection
relevant = [x for x, r in zip(items, llm_query_batched(
    [f"Is {x.title} about {topic}? yes/no" for x in items]
)) if 'yes' in r.lower()]

# Extract details, then synthesize
findings = llm_query_batched([f"Key finding: {x}" for x in relevant])
answer = llm_query(f"Synthesize:\n" + "\n---\n".join(findings))

# Handle large inputs by chunking
chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]
parts = llm_query_batched([f"Summarize: {c}" for c in chunks])
summary = llm_query(f"Combine:\n" + "\n".join(parts))"#;

pub(super) fn render_action_instruction<S: Signature>(
    config: &RlmConfig,
    instruction_override: Option<&str>,
) -> String {
    let schema = SignatureSchema::of::<S>();
    let task = instruction_override
        .unwrap_or_else(|| schema.instruction())
        .trim();

    let input_names = if schema.input_fields().is_empty() {
        "(none)".to_string()
    } else {
        schema
            .input_fields()
            .iter()
            .map(|field| field.rust_name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    };

    let output_assignments = if schema.output_fields().is_empty() {
        "".to_string()
    } else {
        schema
            .output_fields()
            .iter()
            .map(|field| format!("{}=...", field.lm_name))
            .collect::<Vec<_>>()
            .join(", ")
    };

    let mut lines = vec![
        format!("You work in a Python REPL. {task}"),
        String::new(),
        format!(
            "Your inputs - {input_names} - are available as variables in memory."
        ),
        "You'll see metadata about each on your first turn. The full data is larger than your context window - interact with it by writing code.".to_string(),
        format!("When done, call SUBMIT({output_assignments})."),
        String::new(),
        "Each turn: write code, see what happens, decide what's next. Don't try to solve everything at once.".to_string(),
        "Variables persist - store your findings, build on them. Anything not in a variable is lost.".to_string(),
        String::new(),
        format!(
            "All REPL output is limited to {} characters. Full content is always in the variable - pass it to llm_query() to analyze anything beyond what you can see directly.",
            config.max_output_chars
        ),
        String::new(),
        "## Tools".to_string(),
        String::new(),
        "llm_query(prompt)".to_string(),
        "  Delegate analysis to a sub-model with ~500K character capacity.".to_string(),
        "  This is your primary tool. Your context window is small; sub-models are large.".to_string(),
        "  Code finds WHERE things are; llm_query understands WHAT they mean.".to_string(),
        String::new(),
        "llm_query_batched(prompts)".to_string(),
        "  Concurrent batch queries. When analyzing a collection, process all items in parallel, not one at a time.".to_string(),
        String::new(),
        format!("SUBMIT({output_assignments})"),
        "  Validates against the output schema below. If validation fails, you see detailed errors and can fix and retry.".to_string(),
        "  - Print final values before calling SUBMIT".to_string(),
        "  - Derive outputs from variables, don't retype literals".to_string(),
        "  - If outputs look empty, zero, or too short: investigate first".to_string(),
        String::new(),
        "print()".to_string(),
        "  You only see what you print. No output means no feedback.".to_string(),
        String::new(),
        "Standard Python: re, json, collections, math, itertools, etc.".to_string(),
        String::new(),
        "## Output Contract".to_string(),
        String::new(),
    ];

    for field in schema.output_fields() {
        lines.push(format!(
            "{}: {}",
            field.lm_name,
            field.type_ir.diagnostic_repr()
        ));
        if !field.docs.trim().is_empty() {
            lines.push(format!("  {}", field.docs.trim()));
        }
        for constraint in field.constraints {
            let marker = match constraint.kind {
                ConstraintKind::Check => "soft",
                ConstraintKind::Assert => "hard",
            };
            lines.push(format!(
                "  {marker}: {} ({})",
                constraint.label, constraint.expression
            ));
        }
        lines.push(String::new());
    }

    lines.push(PATTERNS_BLOCK.to_string());
    lines.push(String::new());
    lines.push("## Budget".to_string());
    lines.push(String::new());
    lines.push(format!(
        "{} turns. {} sub-model calls.",
        config.max_iterations, config.max_llm_calls
    ));

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

#[cfg(test)]
mod tests {
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

    #[test]
    fn includes_core_sections_and_schema_substitutions() {
        let config = RlmConfig {
            max_iterations: 7,
            max_llm_calls: 11,
            max_output_chars: 1234,
            enable_extraction_fallback: true,
        };

        let rendered = render_action_instruction::<PromptSig>(&config, None);

        assert!(
            rendered.contains("You work in a Python REPL. Solve the query against the corpus.")
        );
        assert!(rendered.contains("Your inputs - papers, question -"));
        assert!(rendered.contains("SUBMIT(answer=...)"));
        assert!(rendered.contains("## Tools"));
        assert!(rendered.contains("llm_query(prompt)"));
        assert!(rendered.contains("llm_query_batched(prompts)"));
        assert!(rendered.contains("## Output Contract"));
        assert!(rendered.contains("answer: string"));
        assert!(rendered.contains("hard: non_empty"));
        assert!(rendered.contains("## Patterns"));
        assert!(rendered.contains("## Budget"));
        assert!(rendered.contains("7 turns. 11 sub-model calls."));
    }

    #[test]
    fn instruction_override_replaces_task_sentence() {
        let rendered = render_action_instruction::<PromptSig>(
            &RlmConfig::default(),
            Some("Custom task sentence."),
        );

        assert!(rendered.contains("You work in a Python REPL. Custom task sentence."));
        assert!(!rendered.contains("Solve the query against the corpus."));
    }

    #[test]
    fn extract_instruction_includes_task_and_extraction_guidance() {
        let rendered = render_extract_instruction::<PromptSig>(None);

        assert!(rendered.contains("The following REPL session was generated for this task:"));
        assert!(rendered.contains("Solve the query against the corpus."));
        assert!(rendered.contains("Based on the execution history, extract the final outputs."));
    }

    #[test]
    fn extract_instruction_uses_override_when_present() {
        let rendered = render_extract_instruction::<PromptSig>(Some("Custom extraction task."));

        assert!(rendered.contains("Custom extraction task."));
        assert!(!rendered.contains("Solve the query against the corpus."));
    }
}
