use std::collections::HashMap;

use dspy_rs::Signature;
use dspy_rs::data::example::Example;
use dspy_rs::field::{In, Out};
use dspy_rs::module::dummy_predictor::DummyPredict;
use dspy_rs::providers::dummy_lm::DummyLM;

#[allow(dead_code)]
#[derive(Signature)]
struct QASignature {
    /// You are a helpful assistant.
    pub question: In<String>,
    pub answer: Out<String>,
}

#[cfg_attr(miri, ignore)] // Miri doesn't support tokio's I/O driver
#[tokio::test]
async fn test_predictor() {
    let signature = QASignature::new();

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
    assert_eq!(outputs.data.get("answer").unwrap(), "Paris");
}
