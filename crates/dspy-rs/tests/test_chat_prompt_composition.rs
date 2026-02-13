use dspy_rs::{ChatAdapter, Example, Signature};

#[derive(Signature, Clone, Debug)]
/// Answer the prompt using the provided context.
struct PromptPartsSig {
    #[input(desc = "User question")]
    question: String,

    #[input(desc = "Retrieved context")]
    context: String,

    #[output(desc = "Final answer")]
    answer: String,

    #[output(desc = "Confidence score")]
    confidence: f64,
}

#[derive(Signature, Clone, Debug)]
struct EmptyInstructionSig {
    #[input]
    topic: String,

    #[output]
    summary: String,
}

fn find_required(haystack: &str, needle: &str) -> usize {
    haystack
        .find(needle)
        .unwrap_or_else(|| panic!("missing `{needle}` in:\n{haystack}"))
}

fn response_instruction_line(message: &str) -> &str {
    message
        .lines()
        .find(|line| line.starts_with("Respond with the corresponding output fields"))
        .expect("response instruction line")
}

#[test]
fn system_prompt_includes_all_sections_in_order_with_boundaries() {
    let adapter = ChatAdapter;
    let system = adapter
        .format_system_message_typed::<PromptPartsSig>()
        .expect("system prompt should format");

    let descriptions_idx = find_required(&system, "Your input fields are:");
    let structure_idx = find_required(
        &system,
        "All interactions will be structured in the following way, with the appropriate values filled in.",
    );
    let instructions_idx = find_required(&system, "Respond with the corresponding output fields");
    let objective_idx = find_required(&system, "In adhering to this structure, your objective is:");

    assert!(descriptions_idx < structure_idx);
    assert!(structure_idx < instructions_idx);
    assert!(instructions_idx < objective_idx);

    assert!(
        system.contains(
            "[[ ## completed ## ]]\n\nRespond with the corresponding output fields, starting with the field",
        ),
        "field-structure and response-instruction boundary missing:\n{system}"
    );
    assert!(
        system.contains(
            "and then ending with the marker for `[[ ## completed ## ]]`.\n\nIn adhering to this structure, your objective is:",
        ),
        "response-instruction and objective boundary missing:\n{system}"
    );

    assert_eq!(
        system
            .matches("Respond with the corresponding output fields")
            .count(),
        1
    );
}

#[test]
fn system_prompt_field_descriptions_and_structure_are_present() {
    let adapter = ChatAdapter;
    let system = adapter
        .format_system_message_typed::<PromptPartsSig>()
        .expect("system prompt should format");

    assert!(system.contains("`question` (string): User question"));
    assert!(system.contains("`context` (string): Retrieved context"));
    assert!(system.contains("`answer` (string): Final answer"));
    assert!(system.contains("`confidence` (float): Confidence score"));

    assert!(system.contains("[[ ## question ## ]]"));
    assert!(system.contains("[[ ## context ## ]]"));
    assert!(system.contains("[[ ## answer ## ]]"));
    assert!(system.contains("[[ ## confidence ## ]]"));
    assert!(system.contains("Output field `answer` should be of type: string"));
    assert!(system.contains("Output field `confidence` should be of type: float"));
    assert!(system.contains("[[ ## completed ## ]]"));
}

#[test]
fn response_instruction_line_orders_output_fields() {
    let adapter = ChatAdapter;
    let system = adapter
        .format_system_message_typed::<PromptPartsSig>()
        .expect("system prompt should format");
    let line = response_instruction_line(&system);

    let answer_idx = find_required(line, "[[ ## answer ## ]]");
    let confidence_idx = find_required(line, "[[ ## confidence ## ]]");
    assert!(answer_idx < confidence_idx);
    assert!(line.contains("[[ ## completed ## ]]"));
}

#[test]
fn instruction_override_is_used_in_objective_section() {
    let adapter = ChatAdapter;
    let override_instruction = "Follow the rubric.\nCite the context.";
    let system = adapter
        .format_system_message_typed_with_instruction::<PromptPartsSig>(Some(override_instruction))
        .expect("system prompt should format with override");

    assert!(system.contains("In adhering to this structure, your objective is:"));
    assert!(system.contains("        Follow the rubric."));
    assert!(system.contains("        Cite the context."));
    assert!(!system.contains("Answer the prompt using the provided context."));
}

#[test]
fn empty_instruction_uses_generated_fallback_objective() {
    let adapter = ChatAdapter;
    let system = adapter
        .format_system_message_typed::<EmptyInstructionSig>()
        .expect("system prompt should format");

    assert!(system.contains("In adhering to this structure, your objective is:"));
    assert!(system.contains("Given the fields `topic`, produce the fields `summary`."));
}

#[test]
fn typed_and_schema_system_builders_match() {
    let adapter = ChatAdapter;
    let typed = adapter
        .format_system_message_typed_with_instruction::<PromptPartsSig>(Some("Override objective"))
        .expect("typed system prompt");
    let schema = adapter
        .build_system(PromptPartsSig::schema(), Some("Override objective"))
        .expect("schema system prompt");

    assert_eq!(typed, schema);
}

#[test]
fn typed_and_schema_user_builders_match_and_append_requirements() {
    let adapter = ChatAdapter;
    let input = PromptPartsSigInput {
        question: "What is the capital of France?".to_string(),
        context: "Facts: Paris is the capital city of France.".to_string(),
    };

    let typed = adapter.format_user_message_typed::<PromptPartsSig>(&input);
    let schema = adapter.format_input(PromptPartsSig::schema(), &input);
    assert_eq!(typed, schema);

    assert!(typed.contains("[[ ## question ## ]]"));
    assert!(typed.contains("What is the capital of France?"));
    assert!(typed.contains("[[ ## context ## ]]"));
    assert!(typed.contains("Facts: Paris is the capital city of France."));

    let context_idx = find_required(&typed, "Facts: Paris is the capital city of France.");
    let instruction_idx = find_required(&typed, "Respond with the corresponding output fields");
    assert!(context_idx < instruction_idx);
    assert_eq!(
        typed
            .matches("Respond with the corresponding output fields")
            .count(),
        1
    );
    assert!(
        typed
            .trim_end()
            .ends_with("and then ending with the marker for `[[ ## completed ## ]]`.")
    );
}

#[test]
fn demo_format_composes_user_and_assistant_parts() {
    let adapter = ChatAdapter;
    let demo = Example::<PromptPartsSig>::new(
        PromptPartsSigInput {
            question: "Question?".to_string(),
            context: "Context.".to_string(),
        },
        PromptPartsSigOutput {
            answer: "Answer.".to_string(),
            confidence: 0.8,
        },
    );

    let (user_msg, assistant_msg) = adapter.format_demo_typed::<PromptPartsSig>(&demo);

    assert!(user_msg.contains("[[ ## question ## ]]"));
    assert!(user_msg.contains("[[ ## context ## ]]"));
    assert!(user_msg.contains("Respond with the corresponding output fields"));
    assert!(user_msg.contains("[[ ## answer ## ]]"));
    assert!(user_msg.contains("[[ ## confidence ## ]]"));

    assert!(assistant_msg.contains("[[ ## answer ## ]]"));
    assert!(assistant_msg.contains("[[ ## confidence ## ]]"));
    assert!(assistant_msg.trim_end().ends_with("[[ ## completed ## ]]"));
}

#[test]
fn typed_and_schema_assistant_builders_match_and_end_with_completed_marker() {
    let adapter = ChatAdapter;
    let output = PromptPartsSigOutput {
        answer: "Paris".to_string(),
        confidence: 0.9,
    };

    let typed = adapter.format_assistant_message_typed::<PromptPartsSig>(&output);
    let schema = adapter.format_output(PromptPartsSig::schema(), &output);
    assert_eq!(typed, schema);

    let answer_idx = find_required(&typed, "[[ ## answer ## ]]");
    let confidence_idx = find_required(&typed, "[[ ## confidence ## ]]");
    assert!(answer_idx < confidence_idx);
    assert!(typed.trim_end().ends_with("[[ ## completed ## ]]"));
}
