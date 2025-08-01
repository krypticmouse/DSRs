use indexmap::IndexMap;
use std::collections::HashMap;

use dspy_rs::clients::dummy_lm::DummyLM;
use dspy_rs::data::example::Example;
use dspy_rs::field::{In, Out};
use dspy_rs::programs::dummy_predictor::DummyPredict;
use dspy_rs::signature::Signature;

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_predictor() {
    let signature = Signature::builder()
        .name("QASignature".to_string())
        .instruction("You are a helpful assistant.".to_string())
        .input_fields(IndexMap::from([("question".to_string(), In::default())]))
        .output_fields(IndexMap::from([("answer".to_string(), Out::default())]))
        .build()
        .unwrap();

    let predictor = DummyPredict {
        signature: signature.clone(),
    };
    let inputs = Example::new(
        HashMap::from([(
            "question".to_string(),
            "What is the capital of France?".to_string(),
        )]),
        vec!["question".to_string()],
        vec!["answer".to_string()],
    );

    let lm = DummyLM::default();

    let outputs = predictor
        .forward(
            inputs.clone(),
            "[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]",
            Some(lm),
            None,
        )
        .await;
    assert_eq!(outputs.data.get("answer").unwrap().as_str(), "Paris");
}
