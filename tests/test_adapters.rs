use indexmap::IndexMap;
use openrouter_rs::types::Role;
use std::collections::HashMap;

use dspy_rs::adapter::base::Adapter;
use dspy_rs::adapter::chat_adapter::ChatAdapter;
use dspy_rs::clients::chat::Chat;
use dspy_rs::clients::dummy_lm::DummyLM;
use dspy_rs::data::example::Example;
use dspy_rs::field::{In, Out};
use dspy_rs::signature::Signature;

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_chat_adapter() {
    let signature = Signature::builder()
        .name("test".to_string())
        .instruction("Given the fields `problem`, produce the fields `answer`.".to_string())
        .input_fields(IndexMap::from([("problem".to_string(), In::default())]))
        .output_fields(IndexMap::from([("answer".to_string(), Out::default())]))
        .build()
        .unwrap();

    let mut lm = DummyLM::default();
    let adapter = ChatAdapter;

    let messages: Chat = adapter.format(
        signature.clone(),
        Example::new(
            HashMap::from([(
                "problem".to_string(),
                "What is the capital of France?".to_string(),
            )]),
            vec!["problem".to_string()],
            vec!["answer".to_string()],
        ),
    );
    assert_eq!(messages.len(), 2);
    assert_eq!(
        messages.messages[0].role.to_string(),
        Role::System.to_string()
    );
    assert_eq!(
        messages.messages[1].role.to_string(),
        Role::User.to_string()
    );

    assert_eq!(
        messages.messages[0].content.to_string(),
        "Your input fields are:\n1. `problem`\n\nYour output fields are:\n1. `answer`\n\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## problem ## ]]\nproblem\n\n[[ ## answer ## ]]\nanswer\n\n[[ ## completed ## ]]\n\nIn adhering to this structure, your objective is:\n\tGiven the fields `problem`, produce the fields `answer`."
    );
    assert_eq!(
        messages.messages[1].content.to_string(),
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
    let output = adapter.parse_response(signature.clone(), response);

    assert_eq!(output.data.len(), 1);
    assert_eq!(output.data.get("answer").unwrap().as_str(), "150 degrees");
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_chat_adapter_with_multiple_fields() {
    let signature = Signature::builder()
        .name("test".to_string())
        .instruction("You are a helpful assistant that can answer questions. You will be given a problem and a hint. You will need to use the hint to answer the problem. You will then need to provide the reasoning and the answer.".to_string())
        .input_fields(IndexMap::from([
            ("problem".to_string(), In::default()),
            ("hint".to_string(), In::default()),
        ]))
        .output_fields(IndexMap::from([
            ("reasoning".to_string(), Out::default()),
            ("answer".to_string(), Out::default()),
        ]))
        .build()
        .unwrap();

    let mut lm = DummyLM::default();
    let adapter = ChatAdapter;

    let messages: Chat = adapter.format(
        signature.clone(),
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
    assert_eq!(messages.len(), 2);
    assert_eq!(
        messages.messages[0].role.to_string(),
        Role::System.to_string()
    );
    assert_eq!(
        messages.messages[1].role.to_string(),
        Role::User.to_string()
    );

    assert_eq!(
        messages.messages[0].content.to_string(),
        "Your input fields are:\n1. `problem`\n2. `hint`\n\nYour output fields are:\n1. `reasoning`\n2. `answer`\n\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## problem ## ]]\nproblem\n\n[[ ## hint ## ]]\nhint\n\n[[ ## reasoning ## ]]\nreasoning\n\n[[ ## answer ## ]]\nanswer\n\n[[ ## completed ## ]]\n\nIn adhering to this structure, your objective is:\n\tYou are a helpful assistant that can answer questions. You will be given a problem and a hint. You will need to use the hint to answer the problem. You will then need to provide the reasoning and the answer."
    );
    assert_eq!(
        messages.messages[1].content.to_string(),
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
    let output = adapter.parse_response(signature.clone(), response);

    assert_eq!(output.data.len(), 2);
    assert_eq!(
        output.data.get("reasoning").unwrap(),
        "The capital of France is Paris."
    );
    assert_eq!(output.data.get("answer").unwrap().as_str(), "Paris");
}
