use dspy_rs::{ChatAdapter, Message, Signature};

#[derive(Signature, Clone, Debug)]
/// Provide an answer using aliases.
struct AliasSignature {
    #[input(desc = "Primary question")]
    #[alias = "question_text"]
    question: String,

    #[output(desc = "Final response")]
    #[alias = "final_answer"]
    answer: String,
}

#[test]
fn typed_alias_is_used_in_prompt_and_user_message() {
    let adapter = ChatAdapter;
    let system = adapter
        .format_system_message::<AliasSignature>()
        .expect("system message");

    assert!(system.contains("Your input fields are:"));
    assert!(system.contains("- question_text: string"));
    assert!(system.contains("Your output fields are:"));
    assert!(system.contains("- final_answer: string"));
    assert!(!system.contains("- question:"));
    assert!(!system.contains("- answer:"));

    let input = AliasSignatureInput {
        question: "Hello".to_string(),
    };
    let user = adapter
        .format_user_message::<AliasSignature>(&input)
        .expect("user message");
    assert!(user.contains("[[ ## question_text ## ]]"));
    assert!(user.contains("Hello"));
    assert!(!user.contains("[[ ## question ## ]]"));
}

#[test]
fn typed_alias_parses_output_and_maps_to_rust_name() {
    let adapter = ChatAdapter;
    let response = Message::assistant("[[ ## final_answer ## ]]\nHi\n\n[[ ## completed ## ]]");
    let (output, metas) = adapter
        .parse_response_typed::<AliasSignature>(&response)
        .expect("parse response");

    assert_eq!(output.answer, "Hi");
    assert!(metas.contains_key("answer"));
    let meta = metas.get("answer").expect("meta for answer");
    assert_eq!(meta.raw_text, "Hi");
}
