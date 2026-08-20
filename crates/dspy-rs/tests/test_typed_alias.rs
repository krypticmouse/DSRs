use dspy_rs::ir::SignatureDef;
use dspy_rs::{ChatAdapter, Message, Signature};
use serde_json::Value;

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

fn json_map<T: serde::Serialize>(value: &T) -> serde_json::Map<String, Value> {
    match serde_json::to_value(value).expect("serializable") {
        Value::Object(map) => map,
        other => panic!("expected object, got {other:?}"),
    }
}

#[test]
fn typed_alias_is_used_in_prompt_and_user_message() {
    let adapter = ChatAdapter;
    let system = adapter.build_system_def(
        SignatureDef::of::<AliasSignature>(),
        SignatureDef::types_of::<AliasSignature>(),
        None,
    );

    assert!(system.contains("[[ ## question_text ## ]]"));
    assert!(system.contains("[[ ## final_answer ## ]]"));
    assert!(!system.contains("[[ ## question ## ]]"));
    assert!(!system.contains("[[ ## answer ## ]]"));
    assert!(system.contains("`question_text` (string): Primary question"));
    assert!(system.contains("`final_answer` (string): Final response"));

    let input = AliasSignatureInput {
        question: "Hello".to_string(),
    };
    let user = adapter.format_input_def(SignatureDef::of::<AliasSignature>(), &json_map(&input));
    assert!(user.contains("[[ ## question_text ## ]]"));
    assert!(user.contains("Hello"));
    assert!(!user.contains("[[ ## question ## ]]"));
}

#[test]
fn typed_alias_parses_output_and_maps_to_rust_name() {
    let adapter = ChatAdapter;
    let response = Message::assistant("[[ ## final_answer ## ]]\nHi\n\n[[ ## completed ## ]]");
    let (output_map, metas) = adapter
        .parse_output_def(
            SignatureDef::of::<AliasSignature>(),
            SignatureDef::types_of::<AliasSignature>(),
            &response,
        )
        .expect("parse response");

    let output: AliasSignatureOutput =
        serde_json::from_value(Value::Object(output_map)).expect("typed assembly");
    assert_eq!(output.answer, "Hi");
    assert!(metas.contains_key("answer"));
    let meta = metas.get("answer").expect("meta for answer");
    assert_eq!(meta.raw_text, "Hi");
}
