use indexmap::IndexMap;
use std::collections::HashMap;

use dspy_rs::clients::dummy_lm::DummyLM;
use dspy_rs::programs::dummy_predictor::DummyPredict;
use dspy_rs::signature::field::Field;
use dspy_rs::signature::signature::Signature;

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_predictor() {
    let mut signature = Signature::builder()
        .name("QASignature".to_string())
        .instruction("You are a helpful assistant.".to_string())
        .input_fields(IndexMap::from([(
            "question".to_string(),
            Field::InputField {
                prefix: "".to_string(),
                desc: "The question to answer".to_string(),
                format: None,
                output_type: "String".to_string(),
            },
        )]))
        .output_fields(IndexMap::from([(
            "answer".to_string(),
            Field::OutputField {
                prefix: "".to_string(),
                desc: "The answer to the question".to_string(),
                format: None,
                output_type: "String".to_string(),
            },
        )]))
        .build()
        .unwrap();

    let predictor = DummyPredict {
        signature: &mut signature,
    };
    let inputs = HashMap::from([(
        "question".to_string(),
        "What is the capital of France?".to_string(),
    )]);

    let lm = DummyLM::default();

    let outputs = predictor
        .forward(
            inputs,
            "[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]".to_string(),
            Some(lm),
            None,
        )
        .await;
    assert_eq!(outputs.data.get("answer").unwrap().as_str(), "Paris");
}
