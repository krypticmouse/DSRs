//! Golden prompt tests: the exact bytes of the `[[ ## field ## ]]` protocol.
//!
//! These render through the def lane ([`SignatureDef::of`] + the `*_def`
//! adapter methods) — the one prompt path since Predict routes through the IR
//! interpreter. The expected strings are the historical static-lane bytes and
//! must never change silently.

use dspy_rs::ir::SignatureDef;
use dspy_rs::{ChatAdapter, Demo, Signature};
use serde_json::Value;

#[derive(Signature, Clone, Debug)]
struct GoldenSig {
    #[input]
    question: String,

    #[output]
    answer: String,
}

fn json_map<T: serde::Serialize>(value: &T) -> serde_json::Map<String, Value> {
    match serde_json::to_value(value).expect("serializable") {
        Value::Object(map) => map,
        other => panic!("expected object, got {other:?}"),
    }
}

#[test]
fn golden_system_prompt_is_stable() {
    let adapter = ChatAdapter;
    let system = adapter.build_system_def(
        SignatureDef::of::<GoldenSig>(),
        SignatureDef::types_of::<GoldenSig>(),
        None,
    );

    let expected = concat!(
        "Your input fields are:\n",
        "1. `question` (string)\n",
        "\n",
        "Your output fields are:\n",
        "1. `answer` (string)\n",
        "\n",
        "All interactions will be structured in the following way, with the appropriate values filled in.\n",
        "\n",
        "[[ ## question ## ]]\n",
        "question\n",
        "\n",
        "[[ ## answer ## ]]\n",
        "Output field `answer` should be of type: string\n",
        "\n",
        "[[ ## completed ## ]]\n",
        "\n",
        "Respond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.\n",
        "\n",
        "In adhering to this structure, your objective is: \n",
        "        Given the fields `question`, produce the fields `answer`.",
    );

    assert_eq!(system, expected);
}

#[test]
fn golden_user_prompt_is_stable() {
    let adapter = ChatAdapter;
    let input = GoldenSigInput {
        question: "What is 2+2?".to_string(),
    };
    let user = adapter.format_input_def(SignatureDef::of::<GoldenSig>(), &json_map(&input));

    let expected = concat!(
        "[[ ## question ## ]]\n",
        "What is 2+2?\n",
        "\n",
        "Respond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.",
    );

    assert_eq!(user, expected);
}

#[test]
fn golden_assistant_prompt_is_stable() {
    let adapter = ChatAdapter;
    let output = GoldenSigOutput {
        answer: "4".to_string(),
    };
    let assistant = adapter.format_output_def(SignatureDef::of::<GoldenSig>(), &json_map(&output));

    let expected = concat!(
        "[[ ## answer ## ]]\n",
        "4\n",
        "\n",
        "[[ ## completed ## ]]\n",
    );
    assert_eq!(assistant, expected);
}

#[test]
fn golden_demo_messages_are_stable() {
    let adapter = ChatAdapter;
    let demo = Demo::<GoldenSig>::new(
        GoldenSigInput {
            question: "What is 2+2?".to_string(),
        },
        GoldenSigOutput {
            answer: "4".to_string(),
        },
    );

    let def = SignatureDef::of::<GoldenSig>();
    let user = adapter.format_input_def(def, &json_map(&demo.input));
    let assistant = adapter.format_output_def(def, &json_map(&demo.output));

    let expected_user = concat!(
        "[[ ## question ## ]]\n",
        "What is 2+2?\n",
        "\n",
        "Respond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.",
    );
    let expected_assistant = concat!(
        "[[ ## answer ## ]]\n",
        "4\n",
        "\n",
        "[[ ## completed ## ]]\n",
    );

    assert_eq!(user, expected_user);
    assert_eq!(assistant, expected_assistant);
}
