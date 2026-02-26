#![cfg(feature = "rlm")]

use dspy_rs::modules::rlm::PyO3Runtime;
use dspy_rs::{
    ChatAdapter, LM, LMClient, PredictError, Rlm, Signature, TestCompletionModel, configure,
};
use rig::completion::AssistantContent;
use rig::message::Text;
use std::sync::{Arc, LazyLock};
use tokio::sync::Mutex;

static SETTINGS_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

fn text_response(text: impl Into<String>) -> AssistantContent {
    AssistantContent::Text(Text { text: text.into() })
}

fn response_with_fields(fields: &[(&str, &str)]) -> String {
    let mut response = String::new();
    for (name, value) in fields {
        response.push_str(&format!("[[ ## {name} ## ]]\n{value}\n\n"));
    }
    response.push_str("[[ ## completed ## ]]\n");
    response
}

async fn build_test_lm_with_client(responses: Vec<String>) -> (LM, TestCompletionModel) {
    let client = TestCompletionModel::new(responses.into_iter().map(text_response));
    let lm = temp_env::async_with_vars(
        [("OPENAI_API_KEY", Some("test"))],
        LM::builder()
            .model("openai:gpt-4o-mini".to_string())
            .temperature(0.0)
            .build(),
    )
    .await
    .expect("build lm")
    .with_client(LMClient::Test(client.clone()))
    .await
    .expect("install test client");
    (lm, client)
}

async fn configure_test_lm(responses: Vec<String>) -> LM {
    let (lm, _) = build_test_lm_with_client(responses).await;
    configure(lm.clone(), ChatAdapter::new());
    lm
}

async fn configure_test_lm_with_client(responses: Vec<String>) -> (LM, TestCompletionModel) {
    let (lm, client) = build_test_lm_with_client(responses).await;
    configure(lm.clone(), ChatAdapter::new());
    (lm, client)
}

#[derive(Signature, Clone, Debug, PartialEq)]
struct RlmFallbackSig {
    #[input]
    prompt: String,
    #[output]
    answer: String,
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn rlm_fallback_extractor_runs_after_finalization_failure_and_uses_repl_history() {
    let _lock = SETTINGS_LOCK.lock().await;
    let (lm, client) = configure_test_lm_with_client(vec![
        "x = 40 + 2\nprint(f'x={x}')".to_string(),
        "print('still working')".to_string(),
        "print('final turn still no submit')".to_string(),
        response_with_fields(&[("answer", "from-fallback")]),
    ])
    .await;

    let rlm = Rlm::<RlmFallbackSig>::builder()
        .runtime(Arc::new(PyO3Runtime))
        .sub_lm(Arc::new(lm))
        .max_iterations(3)
        .enable_extraction_fallback(true)
        .build();

    let predicted = rlm
        .call(RlmFallbackSigInput {
            prompt: "never submit; let fallback extract".to_string(),
        })
        .await
        .expect("fallback extraction should produce typed output");

    assert_eq!(predicted.answer, "from-fallback");
    let raw = &predicted.metadata().raw_response;
    assert!(raw.contains("x = 40 + 2"));
    assert!(raw.contains("print('final turn still no submit')"));
    assert!(raw.contains("[[ ## answer ## ]]"));

    let last_request = client.last_request().expect("expected extraction request");
    let request_debug = format!("{last_request:?}");
    assert!(request_debug.contains("[[ ## repl_history ## ]]"));
    assert!(request_debug.contains("=== Turn 1 ==="));
    assert!(request_debug.contains("Code:"));
    assert!(request_debug.contains("Output:"));
    assert!(request_debug.contains("x = 40 + 2"));
    assert!(request_debug.contains("[[ ## variables_info ## ]]"));
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn rlm_without_extraction_fallback_returns_max_iterations_error() {
    let _lock = SETTINGS_LOCK.lock().await;
    let lm = configure_test_lm(vec![
        "print('turn1')".to_string(),
        "print('turn2')".to_string(),
    ])
    .await;

    let rlm = Rlm::<RlmFallbackSig>::builder()
        .runtime(Arc::new(PyO3Runtime))
        .sub_lm(Arc::new(lm))
        .max_iterations(2)
        .enable_extraction_fallback(false)
        .build();

    let err = rlm
        .call(RlmFallbackSigInput {
            prompt: "never submit".to_string(),
        })
        .await
        .expect_err("expected max-iteration failure when fallback is disabled");
    match err {
        PredictError::Module { source, .. } => {
            assert!(
                source.to_string().contains("max iterations reached (2)"),
                "unexpected error: {source}"
            );
        }
        other => panic!("expected module error, got: {other}"),
    }
}
