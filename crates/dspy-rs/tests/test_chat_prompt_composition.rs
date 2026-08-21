//! Prompt composition through the def lane — the one prompt path since
//! Predict routes through the IR interpreter.

use dspy_rs::ir::SignatureDef;
use dspy_rs::{ChatAdapter, Demo, Signature};
use serde_json::Value;

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

fn json_map<T: serde::Serialize>(value: &T) -> serde_json::Map<String, Value> {
    match serde_json::to_value(value).expect("serializable") {
        Value::Object(map) => map,
        other => panic!("expected object, got {other:?}"),
    }
}

fn system_prompt<S: Signature>(instruction_override: Option<&str>) -> String {
    ChatAdapter.build_system_def(
        SignatureDef::of::<S>(),
        SignatureDef::types_of::<S>(),
        instruction_override,
    )
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
    let system = system_prompt::<PromptPartsSig>(None);

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
    let system = system_prompt::<PromptPartsSig>(None);

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
    let system = system_prompt::<PromptPartsSig>(None);
    let line = response_instruction_line(&system);

    let answer_idx = find_required(line, "[[ ## answer ## ]]");
    let confidence_idx = find_required(line, "[[ ## confidence ## ]]");
    assert!(answer_idx < confidence_idx);
    assert!(line.contains("[[ ## completed ## ]]"));
}

#[test]
fn instruction_override_is_used_in_objective_section() {
    let override_instruction = "Follow the rubric.\nCite the context.";
    let system = system_prompt::<PromptPartsSig>(Some(override_instruction));

    assert!(system.contains("In adhering to this structure, your objective is:"));
    assert!(system.contains("        Follow the rubric."));
    assert!(system.contains("        Cite the context."));
    assert!(!system.contains("Answer the prompt using the provided context."));
}

#[test]
fn empty_instruction_uses_generated_fallback_objective() {
    let system = system_prompt::<EmptyInstructionSig>(None);

    assert!(system.contains("In adhering to this structure, your objective is:"));
    assert!(system.contains("Given the fields `topic`, produce the fields `summary`."));
}

#[test]
fn user_builder_appends_requirements() {
    let adapter = ChatAdapter;
    let input = PromptPartsSigInput {
        question: "What is the capital of France?".to_string(),
        context: "Facts: Paris is the capital city of France.".to_string(),
    };

    let user = adapter.format_input_def(SignatureDef::of::<PromptPartsSig>(), &json_map(&input));

    assert!(user.contains("[[ ## question ## ]]"));
    assert!(user.contains("What is the capital of France?"));
    assert!(user.contains("[[ ## context ## ]]"));
    assert!(user.contains("Facts: Paris is the capital city of France."));

    let context_idx = find_required(&user, "Facts: Paris is the capital city of France.");
    let instruction_idx = find_required(&user, "Respond with the corresponding output fields");
    assert!(context_idx < instruction_idx);
    assert_eq!(
        user.matches("Respond with the corresponding output fields")
            .count(),
        1
    );
    assert!(
        user.trim_end()
            .ends_with("and then ending with the marker for `[[ ## completed ## ]]`.")
    );
}

#[test]
fn demo_format_composes_user_and_assistant_parts() {
    let adapter = ChatAdapter;
    let demo = Demo::<PromptPartsSig>::new(
        PromptPartsSigInput {
            question: "Question?".to_string(),
            context: "Context.".to_string(),
        },
        PromptPartsSigOutput {
            answer: "Answer.".to_string(),
            confidence: 0.8,
        },
    );

    let def = SignatureDef::of::<PromptPartsSig>();
    let user_msg = adapter.format_input_def(def, &json_map(&demo.input));
    let assistant_msg = adapter.format_output_def(def, &json_map(&demo.output));

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
fn assistant_builder_orders_fields_and_ends_with_completed_marker() {
    let adapter = ChatAdapter;
    let output = PromptPartsSigOutput {
        answer: "Paris".to_string(),
        confidence: 0.9,
    };

    let assistant =
        adapter.format_output_def(SignatureDef::of::<PromptPartsSig>(), &json_map(&output));

    let answer_idx = find_required(&assistant, "[[ ## answer ## ]]");
    let confidence_idx = find_required(&assistant, "[[ ## confidence ## ]]");
    assert!(answer_idx < confidence_idx);
    assert!(assistant.trim_end().ends_with("[[ ## completed ## ]]"));
}
