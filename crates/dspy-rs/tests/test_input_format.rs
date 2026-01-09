use dspy_rs::{BamlType, ChatAdapter, Signature};

#[derive(BamlType, Clone, Debug)]
struct Document {
    #[baml(alias = "docText")]
    text: String,
}

#[derive(Signature, Clone, Debug)]
/// Provide an answer with supporting context.
struct FormatSig {
    #[input]
    question: String,

    #[input]
    #[format("yaml")]
    context: Vec<Document>,

    #[output]
    answer: String,
}

#[test]
fn typed_input_format_yaml_preserves_aliases() {
    let adapter = ChatAdapter;
    let input = FormatSigInput {
        question: "What is YAML?".to_string(),
        context: vec![Document {
            text: "Hello".to_string(),
        }],
    };

    let message = adapter.format_user_message_typed::<FormatSig>(&input);

    assert!(message.contains("[[ ## context ## ]]"));
    assert!(message.contains("docText: Hello"));
    assert!(message.contains("What is YAML?"));
}
