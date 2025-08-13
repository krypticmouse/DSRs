use schemars::JsonSchema;
use std::collections::HashMap;

use dspy_rs::adapter::base::Adapter;
use dspy_rs::adapter::chat::ChatAdapter;
use dspy_rs::data::example::Example;
use dspy_rs::field::{In, Out};
use dspy_rs::providers::chat::Chat;
use dspy_rs::providers::dummy_lm::DummyLM;
use dspy_rs::{Signature, sign};

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_chat_adapter() {
    let signature = sign! {
        (problem: String) -> answer: String
    };

    let mut lm = DummyLM::default();
    let adapter = ChatAdapter;

    let messages: Chat = adapter.format(
        &signature,
        Example::new(
            HashMap::from([(
                "problem".to_string(),
                "What is the capital of France?".to_string(),
            )]),
            vec!["problem".to_string()],
            vec!["answer".to_string()],
        ),
    );

    let json_value = messages.to_json();
    let json = json_value.as_array().unwrap();

    assert_eq!(messages.len(), 2);
    assert_eq!(json[0]["role"], "system");
    assert_eq!(json[1]["role"], "user");

    assert_eq!(
        json[0]["content"],
        "Your input fields are:\n1. `problem` (String)\n\nYour output fields are:\n1. `answer` (String)\n\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## problem ## ]]\nproblem\n\n[[ ## answer ## ]]\nanswer\n\n[[ ## completed ## ]]\n\nIn adhering to this structure, your objective is:\n\tGiven the fields `problem`, produce the fields `answer`."
    );
    assert_eq!(
        json[1]["content"],
        "[[ ## problem ## ]]\nWhat is the capital of France?\n\nRespond with the corresponding output fields, starting with the field `answer`, and then ending with the marker for `completed`."
    );

    let response = lm
        .call(
            &messages,
            "[[ ## answer ## ]]\n150 degrees\n\n[[ ## completed ## ]]",
            "test",
        )
        .await
        .unwrap();
    let output = adapter.parse_response(&signature, response);

    assert_eq!(output.data.len(), 1);
    assert_eq!(output.data.get("answer").unwrap(), "150 degrees");
}

#[allow(dead_code)]
#[derive(Signature)]
struct TestSignature {
    ///You are a helpful assistant that can answer questions. You will be given a problem and a hint. You will need to use the hint to answer the problem. You will then need to provide the reasoning and the answer.
    pub problem: In<String>,
    pub hint: In<String>,
    pub reasoning: Out<String>,
    pub answer: Out<String>,
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_chat_adapter_with_multiple_fields() {
    let signature = TestSignature::new();

    let mut lm = DummyLM::default();
    let adapter = ChatAdapter;

    let messages: Chat = adapter.format(
        &signature,
        Example::new(
            HashMap::from([
                (
                    "problem".to_string(),
                    "What is the capital of France?".to_string(),
                ),
                (
                    "hint".to_string(),
                    "The capital of France is Paris.".to_string(),
                ),
            ]),
            vec!["problem".to_string(), "hint".to_string()],
            vec!["reasoning".to_string(), "answer".to_string()],
        ),
    );

    let json_value = messages.to_json();
    let json = json_value.as_array().unwrap();

    assert_eq!(messages.len(), 2);
    assert_eq!(json[0]["role"], "system");
    assert_eq!(json[1]["role"], "user");

    assert_eq!(
        json[0]["content"],
        "Your input fields are:\n1. `problem` (String)\n2. `hint` (String)\n\nYour output fields are:\n1. `reasoning` (String)\n2. `answer` (String)\n\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## problem ## ]]\nproblem\n\n[[ ## hint ## ]]\nhint\n\n[[ ## reasoning ## ]]\nreasoning\n\n[[ ## answer ## ]]\nanswer\n\n[[ ## completed ## ]]\n\nIn adhering to this structure, your objective is:\n\tYou are a helpful assistant that can answer questions. You will be given a problem and a hint. You will need to use the hint to answer the problem. You will then need to provide the reasoning and the answer."
    );
    assert_eq!(
        json[1]["content"],
        "[[ ## problem ## ]]\nWhat is the capital of France?\n\n[[ ## hint ## ]]\nThe capital of France is Paris.\n\nRespond with the corresponding output fields, starting with the field `reasoning`, then `answer`, and then ending with the marker for `completed`."
    );

    let response = lm
        .call(
            &messages,
            "[[ ## reasoning ## ]]\nThe capital of France is Paris.\n\n[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]",
            "test",
        )
        .await
        .unwrap();
    let output = adapter.parse_response(&signature, response);

    assert_eq!(output.data.len(), 2);
    assert_eq!(
        output.data.get("reasoning").unwrap(),
        "The capital of France is Paris."
    );
    assert_eq!(output.data.get("answer").unwrap(), "Paris");
}

#[allow(dead_code)]
#[derive(JsonSchema)]
struct TestOutput {
    pub reasoning: String,
    pub rating: i8,
}

#[allow(dead_code)]
#[derive(Signature)]
struct TestSignature2 {
    pub problem: In<String>,
    pub hint: In<i8>,
    pub output: Out<TestOutput>,
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_chat_adapter_with_multiple_fields_and_output_schema() {
    let signature = TestSignature2::new();

    let mut lm = DummyLM::default();
    let adapter = ChatAdapter;

    let messages: Chat = adapter.format(
        &signature,
        Example::new(
            HashMap::from([
                (
                    "problem".to_string(),
                    "What is the capital of France?".to_string(),
                ),
                (
                    "hint".to_string(),
                    "The capital of France is Paris.".to_string(),
                ),
            ]),
            vec!["problem".to_string(), "hint".to_string()],
            vec!["output".to_string()],
        ),
    );

    let json_value = messages.to_json();
    let json = json_value.as_array().unwrap();

    assert_eq!(messages.len(), 2);
    assert_eq!(json[0]["role"], "system");
    assert_eq!(json[1]["role"], "user");

    assert_eq!(
        json[0]["content"],
        "Your input fields are:\n1. `problem` (String)\n2. `hint` (i8)\n\nYour output fields are:\n1. `output` (TestOutput)\n\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## problem ## ]]\nproblem\n\n[[ ## hint ## ]]\nhint\t# note: the value you produce must be a single i8 value\n\n[[ ## output ## ]]\noutput\t# note: the value you produce must adhere to the JSON schema: {\"rating\":{\"format\":\"int8\",\"maximum\":127,\"minimum\":-128,\"type\":\"integer\"},\"reasoning\":{\"type\":\"string\"}}\n\n[[ ## completed ## ]]\n\nIn adhering to this structure, your objective is:\n\tGiven the fields `problem`, `hint`, produce the fields `output`."
    );
    assert_eq!(
        json[1]["content"],
        "[[ ## problem ## ]]\nWhat is the capital of France?\n\n[[ ## hint ## ]]\nThe capital of France is Paris.\n\nRespond with the corresponding output fields, starting with the field `output` (must be formatted as valid Rust TestOutput), and then ending with the marker for `completed`."
    );

    let response = lm
        .call(
            &messages,
            "[[ ## output ## ]]\n{\"reasoning\": \"The capital of France is Paris.\", \"rating\": 5}\n\n[[ ## completed ## ]]",
            "test",
        )
        .await
        .unwrap();
    let output = adapter.parse_response(&signature, response);

    assert_eq!(output.data.len(), 1);

    let parsed_output: serde_json::Value =
        serde_json::from_str("{\"reasoning\": \"The capital of France is Paris.\", \"rating\": 5}")
            .unwrap();
    assert_eq!(
        output.data.get("output").unwrap()["reasoning"],
        parsed_output["reasoning"]
    );
    assert_eq!(
        output.data.get("output").unwrap()["rating"],
        parsed_output["rating"]
    );
}
