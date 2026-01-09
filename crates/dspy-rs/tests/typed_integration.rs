use dspy_rs::{
    configure, ChatAdapter, LM, LMClient, ParseError, Predict, PredictError, Signature,
    TestCompletionModel,
};
use rig::completion::AssistantContent;
use rig::message::Text;
use std::sync::{LazyLock, Mutex};

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

async fn configure_test_lm(responses: Vec<String>) -> TestCompletionModel {
    unsafe {
        std::env::set_var("OPENAI_API_KEY", "test");
    }

    let client = TestCompletionModel::new(responses.into_iter().map(|text| text_response(text)));
    let lm = LM::builder()
        .model("openai:gpt-4o-mini".to_string())
        .build()
        .await
        .unwrap()
        .with_client(LMClient::Test(client.clone()))
        .await
        .unwrap();

    configure(lm, ChatAdapter {});

    client
}

#[derive(Signature, Clone, Debug, PartialEq)]
/// Answer questions with confidence.
struct QA {
    #[input]
    question: String,

    #[output]
    answer: String,

    #[output]
    #[check("this >= 0.0 and this <= 1.0", label = "valid_confidence")]
    confidence: f32,
}

#[derive(Signature, Clone, Debug, PartialEq)]
/// Answer questions with confidence, enforcing a strict range.
struct StrictQA {
    #[input]
    question: String,

    #[output]
    answer: String,

    #[output]
    #[assert("this >= 0.0 and this <= 1.0", label = "confidence_range")]
    confidence: f32,
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn typed_prediction_happy_path_with_metadata() {
    let _lock = SETTINGS_LOCK.lock().unwrap();
    let response = response_with_fields(&[("answer", "Paris"), ("confidence", "0.9")]);
    let _client = configure_test_lm(vec![response]).await;

    let predict = Predict::<QA>::new();
    let input = QAInput {
        question: "What is the capital of France?".to_string(),
    };

    let result = predict.call_with_meta(input).await.unwrap();

    assert_eq!(result.output.answer, "Paris");
    assert!((result.output.confidence - 0.9).abs() < 1e-6);
    assert!(result.field_raw("answer").is_some());
    assert!(result.field_raw("confidence").is_some());

    let checks = result.field_checks("confidence");
    assert!(checks
        .iter()
        .any(|check| check.label == "valid_confidence" && check.passed));
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn typed_prediction_check_failure_is_recorded() {
    let _lock = SETTINGS_LOCK.lock().unwrap();
    let response = response_with_fields(&[("answer", "Paris"), ("confidence", "1.5")]);
    let _client = configure_test_lm(vec![response]).await;

    let predict = Predict::<QA>::new();
    let input = QAInput {
        question: "What is the capital of France?".to_string(),
    };

    let result = predict.call_with_meta(input).await.unwrap();

    let checks = result.field_checks("confidence");
    let check = checks
        .iter()
        .find(|check| check.label == "valid_confidence")
        .expect("check constraint should be recorded");
    assert!(!check.passed);
    assert!(result.has_failed_checks());
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn typed_prediction_missing_field_surfaces_error() {
    let _lock = SETTINGS_LOCK.lock().unwrap();
    let response = response_with_fields(&[("answer", "Paris")]);
    let _client = configure_test_lm(vec![response]).await;

    let predict = Predict::<QA>::new();
    let input = QAInput {
        question: "What is the capital of France?".to_string(),
    };

    let err = match predict.call_with_meta(input).await {
        Ok(_) => panic!("expected missing field error"),
        Err(err) => err,
    };
    match err {
        PredictError::Parse { source, .. } => match source {
            ParseError::Multiple { errors, .. } => {
                assert!(errors.iter().any(|error| {
                    matches!(
                        error,
                        ParseError::MissingField { field, .. } if field == "confidence"
                    )
                }));
            }
            other => panic!("unexpected parse error: {other:?}"),
        },
        other => panic!("unexpected error type: {other:?}"),
    }
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn typed_prediction_assert_failure_raises_error() {
    let _lock = SETTINGS_LOCK.lock().unwrap();
    let response = response_with_fields(&[("answer", "Paris"), ("confidence", "1.5")]);
    let _client = configure_test_lm(vec![response]).await;

    let predict = Predict::<StrictQA>::new();
    let input = StrictQAInput {
        question: "What is the capital of France?".to_string(),
    };

    let err = match predict.call_with_meta(input).await {
        Ok(_) => panic!("expected assert failure error"),
        Err(err) => err,
    };
    match err {
        PredictError::Parse { source, .. } => match source {
            ParseError::Multiple { errors, .. } => {
                assert!(errors.iter().any(|error| {
                    matches!(
                        error,
                        ParseError::CoercionFailed { field, .. } if field == "confidence"
                    )
                }));
            }
            other => panic!("unexpected parse error: {other:?}"),
        },
        other => panic!("unexpected error type: {other:?}"),
    }
}
