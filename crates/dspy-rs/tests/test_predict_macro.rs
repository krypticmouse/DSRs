//! `#[predict]` / `#[cot]` attribute macros: bodyless fns as LM calls.

use dspy_rs::{ChatAdapter, LM, LMClient, TestCompletionModel, configure, cot, fx, predict};
use rig::completion::AssistantContent;
use rig::message::Text;
use std::sync::LazyLock;
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

async fn install_test_lm(responses: Vec<String>) -> TestCompletionModel {
    let client = TestCompletionModel::new(responses.into_iter().map(text_response));
    let lm = temp_env::async_with_vars(
        [("OPENAI_API_KEY", Some("test"))],
        LM::builder().model("openai:gpt-4o-mini".to_string()).build(),
    )
    .await
    .unwrap()
    .with_client(LMClient::Test(client.clone()))
    .await
    .unwrap();
    configure(lm, ChatAdapter);
    client
}

#[predict]
/// MACRO-INSTRUCTION answer accurately.
fn answer(question: String) -> String;

#[predict]
/// Summarize the document for the audience.
fn summarize(document: String, audience: String) -> String;

#[cot]
/// Solve the math problem.
fn solve(problem: String) -> String;

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn predict_macro_calls_lm_with_doc_instruction() {
    let _lock = SETTINGS_LOCK.lock().await;
    let client = install_test_lm(vec![response_with_fields(&[("answer", "Paris")])]).await;

    let out = answer("Capital of France?".to_string())
        .await
        .expect("macro fn should succeed");
    assert_eq!(out.answer, "Paris");

    let preamble = client.last_request().unwrap().preamble.unwrap_or_default();
    assert!(
        preamble.contains("MACRO-INSTRUCTION"),
        "doc comment should become the instruction"
    );
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn predict_macro_supports_multiple_inputs_and_params_override() {
    let _lock = SETTINGS_LOCK.lock().await;
    let client = install_test_lm(vec![
        response_with_fields(&[("summarize", "short version")]),
        response_with_fields(&[("summarize", "tuned version")]),
    ])
    .await;

    let out = summarize("long doc".to_string(), "experts".to_string())
        .await
        .expect("multi-input macro fn should succeed");
    assert_eq!(out.summarize, "short version");

    // Params slot = fn name.
    let mut params = fx::Params::new();
    params.set_instruction("summarize", "PARAMS-MARKER be terse");
    let out = fx::with_params(
        params,
        summarize("long doc".to_string(), "experts".to_string()),
    )
    .await
    .expect("params-scoped macro fn should succeed");
    assert_eq!(out.summarize, "tuned version");
    let preamble = client.last_request().unwrap().preamble.unwrap_or_default();
    assert!(preamble.contains("PARAMS-MARKER"));
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn cot_macro_adds_reasoning_field() {
    let _lock = SETTINGS_LOCK.lock().await;
    let _client = install_test_lm(vec![response_with_fields(&[
        ("reasoning", "2 and 2 make 4"),
        ("solve", "4"),
    ])])
    .await;

    let out = solve("2+2?".to_string()).await.expect("cot fn should succeed");
    assert_eq!(out.reasoning, "2 and 2 make 4");
    assert_eq!(out.solve, "4"); // through WithReasoning's Deref
}

#[test]
fn generated_signature_types_are_referenceable() {
    // The module namespace gives Example/metric code a nameable signature.
    let input = answer::SigInput {
        question: "q".to_string(),
    };
    let _example = dspy_rs::Example::<answer::Sig>::new(
        input,
        answer::SigOutput {
            answer: "a".to_string(),
        },
    );
}
