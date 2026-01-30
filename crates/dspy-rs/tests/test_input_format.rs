use dspy_rs::{BamlType, ChatAdapter, Signature};
#[cfg(feature = "rlm")]
use dspy_rs::RlmType;

#[cfg_attr(feature = "rlm", dspy_rs::rlm_type)]
#[cfg_attr(not(feature = "rlm"), derive(BamlType))]
#[derive(Clone, Debug)]
struct Document {
    #[baml(alias = "docText")]
    text: String,
}

#[derive(Signature, Clone, Debug)]
/// Provide an answer with supporting context using the default format.
struct DefaultFormatSig {
    #[input]
    question: String,

    #[input]
    context: Vec<Document>,

    #[output]
    answer: String,
}

fn extract_field(message: &str, field_name: &str) -> String {
    let start_marker = format!("[[ ## {field_name} ## ]]");
    let start_pos = message
        .find(&start_marker)
        .unwrap_or_else(|| panic!("missing marker: {field_name}"));
    let after_marker = start_pos + start_marker.len();
    let remaining = &message[after_marker..];
    let end_pos = remaining.find("[[ ##").unwrap_or(remaining.len());
    remaining[..end_pos].trim().to_string()
}

#[test]
fn typed_input_default_string_is_raw() {
    let adapter = ChatAdapter;
    let input = DefaultFormatSigInput {
        question: "Raw string".to_string(),
        context: vec![Document {
            text: "Hello".to_string(),
        }],
    };

    let message = adapter.format_user_message_typed::<DefaultFormatSig>(&input);
    let question_value = extract_field(&message, "question");

    assert_eq!(question_value, "Raw string");
}

#[test]
fn typed_input_default_non_string_is_json() {
    let adapter = ChatAdapter;
    let input = DefaultFormatSigInput {
        question: "Default JSON".to_string(),
        context: vec![Document {
            text: "Hello".to_string(),
        }],
    };

    let message = adapter.format_user_message_typed::<DefaultFormatSig>(&input);
    let context_value = extract_field(&message, "context");
    let parsed: serde_json::Value = serde_json::from_str(&context_value).expect("valid JSON");
    let first = parsed
        .as_array()
        .and_then(|items| items.first())
        .and_then(|value| value.as_object())
        .expect("expected array with object");
    assert_eq!(first.get("docText").and_then(|v| v.as_str()), Some("Hello"));
}
