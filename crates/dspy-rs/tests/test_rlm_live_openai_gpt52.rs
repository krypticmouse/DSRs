#![cfg(feature = "rlm")]

use dspy_rs::modules::rlm::PyO3Runtime;
use dspy_rs::{ChatAdapter, LM, Rlm, Signature, configure};
use std::sync::{Arc, LazyLock};
use tokio::sync::Mutex;

static SETTINGS_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

#[derive(Signature, Clone, Debug, PartialEq)]
/// Return executable Python only and call SUBMIT with the final typed answer.
struct LiveMathProblem {
    #[input]
    problem: String,

    #[output]
    answer: i64,
}

#[tokio::test]
#[ignore] // Requires network access + OPENAI_API_KEY
async fn live_rlm_v1_openai_responses_gpt52_end_to_end() {
    let _lock = SETTINGS_LOCK.lock().await;
    let _ = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set for live test");

    let lm = LM::builder()
        .model("openai-responses:gpt-5.2".to_string())
        .temperature(0.0)
        .max_tokens(512)
        .build()
        .await
        .expect("failed to build live LM");

    configure(lm.clone(), ChatAdapter::new());

    let rlm = Rlm::<LiveMathProblem>::builder()
        .runtime(Arc::new(PyO3Runtime))
        .sub_lm(Arc::new(lm))
        .max_iterations(12)
        .max_llm_calls(8)
        .enable_extraction_fallback(false)
        .build();

    let predicted = rlm
        .call(LiveMathProblemInput {
            problem:
                "Compute 12 * 13. Respond with executable Python only (no markdown, no prose). \
Immediately call SUBMIT(answer=<integer>). Example shape:\nresult = 12 * 13\nSUBMIT(answer=result)"
                    .to_string(),
        })
        .await
        .expect("live RLM call failed");

    assert_eq!(predicted.answer, 156);
    assert!(
        predicted.metadata().raw_response.contains("SUBMIT("),
        "expected SUBMIT path evidence in raw response"
    );
}
