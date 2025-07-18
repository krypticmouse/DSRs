use indexmap::IndexMap;
use openrouter_rs::types::Role;
use std::collections::HashMap;

use dsrs::adapter::base::Adapter;
use dsrs::adapter::chat_adapter::ChatAdapter;
use dsrs::clients::chat::Chat;
use dsrs::clients::dummy_lm::DummyLM;
use dsrs::signature::field::Field;
use dsrs::signature::signature::Signature;

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_chat_adapter() {
    let signature = Signature::builder()
        .instruction("Given the fields `problem`, produce the fields `answer`.".to_string())
        .input_fields(IndexMap::from([(
            "problem".to_string(),
            Field::InputField {
                prefix: "".to_string(),
                desc: "".to_string(),
                format: None,
                output_type: "String".to_string(),
            },
        )]))
        .output_fields(IndexMap::from([(
            "answer".to_string(),
            Field::OutputField {
                prefix: "".to_string(),
                desc: "".to_string(),
                format: None,
                output_type: "String".to_string(),
            },
        )]))
        .build()
        .unwrap();

    let mut lm = DummyLM::default();
    let adapter = ChatAdapter;

    let messages: Chat = adapter.format(
        &signature,
        HashMap::from([(
            "problem".to_string(),
            "What is the capital of France?".to_string(),
        )]),
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
        "Your input fields are:\n1. `problem` (String)\n\nYour output fields are:\n1. `answer` (String)\n\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## problem ## ]]\nproblem\n\n[[ ## answer ## ]]\nanswer\n\n[[ ## completed ## ]]\n\nIn adhering to this structure, your objective is:\n\tGiven the fields `problem`, produce the fields `answer`."
    );
    assert_eq!(
        messages.messages[1].content.to_string(),
        "[[ ## problem ## ]]\nWhat is the capital of France?\n\nRespond with the corresponding output fields, starting with the field `answer`, and then ending with the marker for `completed`."
    );

    let response = lm
        .call(
            &messages,
            "[[ ## answer ## ]]\n150 degrees\n\n[[ ## completed ## ]]".to_string(),
            "test".to_string(),
        )
        .await
        .unwrap();
    let output = adapter.parse_response(&signature, response);

    assert_eq!(output.data.len(), 1);
    assert_eq!(output.data.get("answer").unwrap().as_str(), "150 degrees");
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_chat_adapter_with_multiple_fields() {
    let signature = Signature::builder()
        .instruction("You are a helpful assistant that can answer questions. You will be given a problem and a hint. You will need to use the hint to answer the problem. You will then need to provide the reasoning and the answer.".to_string())
        .input_fields(IndexMap::from([
            ("problem".to_string(), Field::InputField { prefix: "".to_string(), desc: "".to_string(), format: None, output_type: "String".to_string() }),
            ("hint".to_string(), Field::InputField { prefix: "".to_string(), desc: "".to_string(), format: None, output_type: "String".to_string() }),
        ]))
        .output_fields(IndexMap::from([
            ("reasoning".to_string(), Field::OutputField { prefix: "".to_string(), desc: "".to_string(), format: None, output_type: "String".to_string() }),
            ("answer".to_string(), Field::OutputField { prefix: "".to_string(), desc: "".to_string(), format: None, output_type: "String".to_string() }),
        ]))
        .build()
        .unwrap();

    let mut lm = DummyLM::default();
    let adapter = ChatAdapter;

    let messages: Chat = adapter.format(
        &signature,
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
        "Your input fields are:\n1. `problem` (String)\n2. `hint` (String)\n\nYour output fields are:\n1. `reasoning` (String)\n2. `answer` (String)\n\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## problem ## ]]\nproblem\n\n[[ ## hint ## ]]\nhint\n\n[[ ## reasoning ## ]]\nreasoning\n\n[[ ## answer ## ]]\nanswer\n\n[[ ## completed ## ]]\n\nIn adhering to this structure, your objective is:\n\tYou are a helpful assistant that can answer questions. You will be given a problem and a hint. You will need to use the hint to answer the problem. You will then need to provide the reasoning and the answer."
    );
    assert_eq!(
        messages.messages[1].content.to_string(),
        "[[ ## problem ## ]]\nWhat is the capital of France?\n\n[[ ## hint ## ]]\nThe capital of France is Paris.\n\nRespond with the corresponding output fields, starting with the field `reasoning`, then `answer`, and then ending with the marker for `completed`."
    );

    let response = lm.call(&messages, "[[ ## reasoning ## ]]\nThe capital of France is Paris.\n\n[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]".to_string(), "test".to_string()).await.unwrap();
    let output = adapter.parse_response(&signature, response);

    assert_eq!(output.data.len(), 2);
    assert_eq!(
        output.data.get("reasoning").unwrap(),
        "The capital of France is Paris."
    );
    assert_eq!(output.data.get("answer").unwrap().as_str(), "Paris");
}
