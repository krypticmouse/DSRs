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
        .name("QASignature")
        .instruction("You are a helpful assistant.".to_string())
        .input_fields(IndexMap::from([(
            "question".to_string(),
            Field::In("The question to answer"),
        )]))
        .output_fields(IndexMap::from([(
            "answer".to_string(),
            Field::Out("The answer to the question"),
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
            "[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]",
            Some(lm),
            None,
        )
        .await;
    assert_eq!(outputs.data.get("answer").unwrap().as_str(), "Paris");
}
