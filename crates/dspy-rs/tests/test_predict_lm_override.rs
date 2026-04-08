use dspy_rs::{ChatAdapter, LM, LMClient, Predict, Signature, TestCompletionModel, configure};
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

async fn make_test_lm(responses: Vec<String>) -> (LM, TestCompletionModel) {
    let client = TestCompletionModel::new(responses.into_iter().map(text_response));
    let lm = temp_env::async_with_vars(
        [("OPENAI_API_KEY", Some("test"))],
        LM::builder()
            .model("openai:gpt-4o-mini".to_string())
            .build(),
    )
    .await
    .unwrap()
    .with_client(LMClient::Test(client.clone()))
    .await
    .unwrap();
    (lm, client)
}

#[derive(Signature, Clone, Debug, PartialEq)]
/// Answer questions.
struct QA {
    #[input]
    question: String,

    #[output]
    answer: String,
}

/// A Predict with a per-instance LM uses that LM instead of the global.
#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn predict_uses_per_instance_lm_over_global() {
    let _lock = SETTINGS_LOCK.lock().await;

    // Configure the global LM — its response says "global"
    let global_response = response_with_fields(&[("answer", "from-global")]);
    let (global_lm, _global_client) = make_test_lm(vec![global_response]).await;
    configure(global_lm, ChatAdapter {});

    // Build a per-instance LM — its response says "override"
    let override_response = response_with_fields(&[("answer", "from-override")]);
    let (override_lm, _override_client) = make_test_lm(vec![override_response]).await;

    // Predict with per-instance LM override
    let predict = Predict::<QA>::builder()
        .lm(override_lm)
        .build();

    let result = predict
        .call(QAInput {
            question: "Which LM?".to_string(),
        })
        .await
        .expect("call should succeed");

    assert_eq!(result.answer, "from-override");
}

/// A Predict without an LM override still uses the global (backward compat).
#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn predict_without_override_uses_global() {
    let _lock = SETTINGS_LOCK.lock().await;

    let global_response = response_with_fields(&[("answer", "from-global")]);
    let (global_lm, _) = make_test_lm(vec![global_response]).await;
    configure(global_lm, ChatAdapter {});

    // No .lm() call — should use global
    let predict = Predict::<QA>::new();

    let result = predict
        .call(QAInput {
            question: "Which LM?".to_string(),
        })
        .await
        .expect("call should succeed");

    assert_eq!(result.answer, "from-global");
}
