use dspy_rs::{ChatAdapter, Message, Signature};

#[derive(Signature, Clone, Debug, PartialEq)]
struct BasicSignature {
    #[input]
    problem: String,

    #[output]
    answer: String,
}

#[test]
fn chat_adapter_formats_typed_system_prompt() {
    let adapter = ChatAdapter;
    let system = adapter
        .format_system_message_typed::<BasicSignature>()
        .expect("system prompt should format");

    assert!(system.contains("Your input fields are:"));
    assert!(system.contains("`problem`"));
    assert!(system.contains("Your output fields are:"));
    assert!(system.contains("`answer`"));
    assert!(system.contains("[[ ## completed ## ]]"));
}

#[test]
fn chat_adapter_formats_user_and_assistant_messages() {
    let adapter = ChatAdapter;

    let user = adapter.format_user_message_typed::<BasicSignature>(&BasicSignatureInput {
        problem: "What is the capital of France?".to_string(),
    });
    let assistant = adapter.format_assistant_message_typed::<BasicSignature>(&BasicSignatureOutput {
        answer: "Paris".to_string(),
    });

    assert!(user.contains("[[ ## problem ## ]]"));
    assert!(user.contains("What is the capital of France?"));

    assert!(assistant.contains("[[ ## answer ## ]]"));
    assert!(assistant.contains("Paris"));
    assert!(assistant.contains("[[ ## completed ## ]]"));
}

#[test]
fn chat_adapter_parses_typed_response() {
    let adapter = ChatAdapter;
    let response = Message::assistant("[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]");

    let (output, field_meta) = adapter
        .parse_response_typed::<BasicSignature>(&response)
        .expect("typed response should parse");

    assert_eq!(output.answer, "Paris");
    assert_eq!(field_meta.get("answer").map(|meta| meta.raw_text.as_str()), Some("Paris"));
}

#[test]
fn parse_sections_accepts_non_word_field_names() {
    let sections = ChatAdapter::parse_sections(
        "[[ ## detail.note ## ]]\nhello\n\n[[ ## completed ## ]]\n",
    );

    assert_eq!(sections.get("detail.note").map(String::as_str), Some("hello"));
}
