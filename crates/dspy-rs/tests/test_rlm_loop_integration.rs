#![cfg(feature = "rlm")]

use dspy_rs::modules::rlm::PyO3Runtime;
use dspy_rs::{ChatAdapter, LM, LMClient, Rlm, Signature, TestCompletionModel, configure};
use rig::completion::AssistantContent;
use rig::message::Text;
use std::sync::{Arc, LazyLock};
use tokio::sync::Mutex;

static SETTINGS_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

fn text_response(text: impl Into<String>) -> AssistantContent {
    AssistantContent::Text(Text { text: text.into() })
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
struct RlmLoopSig {
    #[input]
    prompt: String,
    #[output]
    answer: String,
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn rlm_recovers_from_empty_action_then_submits() {
    let _lock = SETTINGS_LOCK.lock().await;
    let lm = configure_test_lm(vec![
        String::new(),
        "SUBMIT(answer='recovered')".to_string(),
    ])
    .await;

    let rlm = Rlm::<RlmLoopSig>::builder()
        .runtime(Arc::new(PyO3Runtime))
        .sub_lm(Arc::new(lm))
        .max_iterations(3)
        .enable_extraction_fallback(false)
        .build();

    let predicted = rlm
        .call(RlmLoopSigInput {
            prompt: "return recovered".to_string(),
        })
        .await
        .expect("rlm call should recover and submit");

    assert_eq!(predicted.answer, "recovered");
    assert!(
        predicted
            .metadata()
            .raw_response
            .contains("SUBMIT(answer='recovered')")
    );
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn rlm_invalid_submit_retries_then_accepts_valid_submit() {
    let _lock = SETTINGS_LOCK.lock().await;
    let lm = configure_test_lm(vec![
        "SUBMIT(answer=123)".to_string(),
        "SUBMIT(answer='fixed')".to_string(),
    ])
    .await;

    let rlm = Rlm::<RlmLoopSig>::builder()
        .runtime(Arc::new(PyO3Runtime))
        .sub_lm(Arc::new(lm))
        .max_iterations(3)
        .enable_extraction_fallback(false)
        .build();

    let predicted = rlm
        .call(RlmLoopSigInput {
            prompt: "return fixed".to_string(),
        })
        .await
        .expect("rlm call should retry after invalid submit");

    assert_eq!(predicted.answer, "fixed");
    assert!(
        predicted
            .metadata()
            .raw_response
            .contains("SUBMIT(answer=123)")
    );
    assert!(
        predicted
            .metadata()
            .raw_response
            .contains("SUBMIT(answer='fixed')")
    );
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn rlm_v3_demo_recovers_empty_then_python_error_then_finalization_submit() {
    let _lock = SETTINGS_LOCK.lock().await;
    let (lm, client) = configure_test_lm_with_client(vec![
        String::new(),
        "if True print('x')".to_string(),
        "SUBMIT(answer='finalized')".to_string(),
    ])
    .await;

    let rlm = Rlm::<RlmLoopSig>::builder()
        .runtime(Arc::new(PyO3Runtime))
        .sub_lm(Arc::new(lm))
        .max_iterations(3)
        .max_output_chars(500)
        .enable_extraction_fallback(false)
        .build();

    let predicted = rlm
        .call(RlmLoopSigInput {
            prompt: "finalize with best answer".to_string(),
        })
        .await
        .expect("rlm should recover and submit on finalization turn");
    assert_eq!(predicted.answer, "finalized");

    let last_request = client
        .last_request()
        .expect("expected final request to be captured");
    let request_debug = format!("{last_request:?}");
    assert!(
        request_debug.contains("SyntaxError"),
        "finalization turn should include prior python error feedback"
    );
    assert!(
        request_debug.contains("⚠ LAST TURN — you MUST call SUBMIT() now with your best answer."),
        "finalization directive should be present on last repair turn"
    );
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn rlm_feedback_carries_truncation_marker_with_configured_budget() {
    let _lock = SETTINGS_LOCK.lock().await;
    let (lm, client) = configure_test_lm_with_client(vec![
        "print('abcdefghijklmnopqrstuvwxyz0123456789')".to_string(),
        "SUBMIT(answer='done')".to_string(),
    ])
    .await;

    let rlm = Rlm::<RlmLoopSig>::builder()
        .runtime(Arc::new(PyO3Runtime))
        .sub_lm(Arc::new(lm))
        .max_iterations(2)
        .max_output_chars(10)
        .enable_extraction_fallback(false)
        .build();

    let predicted = rlm
        .call(RlmLoopSigInput {
            prompt: "test truncation".to_string(),
        })
        .await
        .expect("rlm should truncate feedback and still submit");
    assert_eq!(predicted.answer, "done");

    let last_request = client
        .last_request()
        .expect("expected request carrying truncated feedback");
    let request_debug = format!("{last_request:?}");
    assert!(request_debug.contains("... [STDOUT TRUNCATED: Exceeded 10 char threshold]"));
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test(flavor = "multi_thread")]
async fn rlm_sub_lm_tools_persist_state_and_decrement_budget_across_turns() {
    let _lock = SETTINGS_LOCK.lock().await;
    let (_action_lm, action_client) = configure_test_lm_with_client(vec![
        "single = llm_query('single')\nbatch = llm_query_batched(['left', 'right'])\ncounter = 40 + len(batch)".to_string(),
        "try:\n    llm_query('should_fail')\n    budget_state = 'not_exhausted'\nexcept Exception as err:\n    budget_state = 'exhausted' if 'budget exhausted' in str(err) else f'unexpected:{err}'\nSUBMIT(answer=f'{counter}:{budget_state}:{single}')".to_string(),
    ])
    .await;
    let (sub_lm, _) = build_test_lm_with_client(vec![
        "single-ok".to_string(),
        "batch-a".to_string(),
        "batch-b".to_string(),
    ])
    .await;

    let rlm = Rlm::<RlmLoopSig>::builder()
        .runtime(Arc::new(PyO3Runtime))
        .sub_lm(Arc::new(sub_lm))
        .max_iterations(2)
        .max_llm_calls(3)
        .enable_extraction_fallback(false)
        .build();

    let predicted = rlm
        .call(RlmLoopSigInput {
            prompt: "Use both sub-LM helpers, then submit on turn two.".to_string(),
        })
        .await
        .expect("rlm should complete with persisted state and enforced budget");

    assert_eq!(predicted.answer, "42:exhausted:single-ok");
    assert!(
        predicted
            .metadata()
            .raw_response
            .contains("llm_query_batched")
    );

    let last_request = action_client
        .last_request()
        .expect("expected second-turn request with feedback");
    let request_debug = format!("{last_request:?}");
    assert!(
        request_debug.contains("Budget: 1 turn remaining | 0 sub-LLM calls remaining"),
        "second turn should see depleted sub-LM budget"
    );
}
