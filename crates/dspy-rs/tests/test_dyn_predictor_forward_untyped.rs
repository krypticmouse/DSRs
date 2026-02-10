use std::sync::LazyLock;

use dspy_rs::__macro_support::bamltype::facet;
use dspy_rs::{
    BamlType, BamlValue, ChatAdapter, LM, LMClient, Predict, PredictError, Signature,
    TestCompletionModel, configure, named_parameters,
};
use rig::completion::AssistantContent;
use rig::message::Text;
use tokio::sync::Mutex;

static SETTINGS_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

fn response_with_fields(fields: &[(&str, &str)]) -> String {
    let mut response = String::new();
    for (name, value) in fields {
        response.push_str(&format!("[[ ## {name} ## ]]\n{value}\n\n"));
    }
    response.push_str("[[ ## completed ## ]]\n");
    response
}

fn text_response(text: impl Into<String>) -> AssistantContent {
    AssistantContent::Text(Text { text: text.into() })
}

async fn configure_test_lm(responses: Vec<String>) {
    unsafe {
        std::env::set_var("OPENAI_API_KEY", "test");
    }

    let client = TestCompletionModel::new(responses.into_iter().map(text_response));
    let lm = LM::builder()
        .model("openai:gpt-4o-mini".to_string())
        .build()
        .await
        .unwrap()
        .with_client(LMClient::Test(client))
        .await
        .unwrap();

    configure(lm, ChatAdapter {});
}

#[derive(Signature, Clone, Debug, PartialEq, facet::Facet)]
#[facet(crate = facet)]
struct QA {
    #[input]
    question: String,

    #[output]
    answer: String,
}

#[derive(facet::Facet)]
#[facet(crate = facet)]
struct Wrapper {
    predictor: Predict<QA>,
}

#[derive(Signature, Clone, Debug, PartialEq, facet::Facet)]
#[facet(crate = facet)]
struct QAWithConfidence {
    #[input]
    question: String,

    #[output]
    answer: String,

    #[output]
    #[check("this >= 0.0 and this <= 1.0", label = "valid_confidence")]
    confidence: f32,
}

#[derive(facet::Facet)]
#[facet(crate = facet)]
struct ConfidenceWrapper {
    predictor: Predict<QAWithConfidence>,
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn dyn_predictor_forward_untyped_returns_baml_and_metadata() {
    let _lock = SETTINGS_LOCK.lock().await;
    let response = response_with_fields(&[("answer", "Paris")]);
    configure_test_lm(vec![response.clone(), response]).await;

    let mut module = Wrapper {
        predictor: Predict::<QA>::new(),
    };
    let input = QAInput {
        question: "What is the capital of France?".to_string(),
    };
    let untyped_input = input.to_baml_value();

    let untyped = {
        let mut params = named_parameters(&mut module).expect("walker should find predictor");
        let (_, predictor) = params
            .iter_mut()
            .find(|(name, _)| name == "predictor")
            .expect("predictor should exist");
        predictor
            .forward_untyped(untyped_input)
            .await
            .expect("untyped call should succeed")
    };
    let typed = module
        .predictor
        .call(input)
        .await
        .expect("typed call should succeed");

    let (untyped_output, untyped_meta) = untyped.into_parts();
    let (typed_output, typed_meta) = typed.into_parts();

    let untyped_output = QAOutput::try_from_baml_value(untyped_output)
        .expect("untyped output should roundtrip to QAOutput");
    assert_eq!(untyped_output.answer, typed_output.answer);
    assert!(!untyped_meta.raw_response.is_empty());
    assert_eq!(untyped_meta.raw_response, typed_meta.raw_response);
}

#[tokio::test]
async fn dyn_predictor_forward_untyped_reports_conversion_error_with_original_payload() {
    let predictor = Predict::<QA>::new();
    let input = BamlValue::Int(42);

    let err = predictor
        .forward_untyped(input.clone())
        .await
        .expect_err("invalid untyped input should fail before LM call");

    match err {
        PredictError::Conversion { parsed, .. } => assert_eq!(parsed, input),
        other => panic!("expected PredictError::Conversion, got {other:?}"),
    }
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn dyn_predictor_forward_untyped_preserves_field_metadata_and_checks() {
    let _lock = SETTINGS_LOCK.lock().await;
    let response = response_with_fields(&[("answer", "Paris"), ("confidence", "1.5")]);
    configure_test_lm(vec![response.clone(), response]).await;

    let mut module = ConfidenceWrapper {
        predictor: Predict::<QAWithConfidence>::new(),
    };
    let typed_input = QAWithConfidenceInput {
        question: "What is the capital of France?".to_string(),
    };

    let untyped = {
        let mut params = named_parameters(&mut module).expect("walker should find predictor");
        let (_, predictor) = params
            .iter_mut()
            .find(|(name, _)| name == "predictor")
            .expect("predictor should exist");
        predictor
            .forward_untyped(typed_input.to_baml_value())
            .await
            .expect("untyped call should succeed")
    };
    let typed = module
        .predictor
        .call(typed_input)
        .await
        .expect("typed call should succeed");

    let (_, untyped_meta) = untyped.into_parts();
    let (_, typed_meta) = typed.into_parts();

    assert_eq!(untyped_meta.raw_response, typed_meta.raw_response);
    assert_eq!(untyped_meta.field_raw("answer"), typed_meta.field_raw("answer"));
    assert_eq!(
        untyped_meta.field_raw("confidence"),
        typed_meta.field_raw("confidence")
    );
    assert_eq!(
        untyped_meta.has_failed_checks(),
        typed_meta.has_failed_checks()
    );

    let untyped_checks = untyped_meta.field_checks("confidence");
    let typed_checks = typed_meta.field_checks("confidence");
    assert_eq!(untyped_checks.len(), typed_checks.len());
    assert!(
        untyped_checks
            .iter()
            .zip(typed_checks.iter())
            .all(|(left, right)| left.label == right.label && left.passed == right.passed)
    );
}
