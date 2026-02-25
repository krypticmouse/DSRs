use dspy_rs::{
    ChatAdapter, LM, LMClient, ParseError, Predict, PredictError, Signature, TestCompletionModel,
    configure,
};
use rig::completion::AssistantContent;
use rig::message::Text;
use std::sync::LazyLock;
use tokio::sync::Mutex;

static SETTINGS_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

fn text_response(text: impl Into<String>) -> AssistantContent {
    AssistantContent::Text(Text { text: text.into() })
}

fn structured_response(fields: &[(&str, &str)]) -> String {
    let mut response = String::new();
    for (name, value) in fields {
        response.push_str(&format!("[[ ## {name} ## ]]\n{value}\n\n"));
    }
    response.push_str("[[ ## completed ## ]]\n");
    response
}

async fn configure_test_lm(responses: Vec<String>) {
    let client = TestCompletionModel::new(responses.into_iter().map(text_response));
    let lm = temp_env::async_with_vars(
        [("OPENAI_API_KEY", Some("test"))],
        LM::builder()
            .model("openai:gpt-4o-mini".to_string())
            .build(),
    )
    .await
    .unwrap()
    .with_client(LMClient::Test(client))
    .await
    .unwrap();
    configure(lm, ChatAdapter::new());
}

#[derive(Signature, Clone, Debug, PartialEq)]
/// Generate executable Python code for the task.
struct RlmActionLike {
    #[input]
    task: String,

    #[output]
    code: String,
}

#[derive(Signature, Clone, Debug, PartialEq)]
/// Answer the question.
struct QaLike {
    #[input]
    question: String,

    #[output]
    answer: String,
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn passthrough_adapter_maps_entire_response_to_code() {
    let _lock = SETTINGS_LOCK.lock().await;
    configure_test_lm(vec![r#"print("ok")"#.to_string()]).await;

    let predict = Predict::<RlmActionLike>::new().adapter(ChatAdapter::passthrough());
    let result = predict
        .call(RlmActionLikeInput {
            task: "print ok".to_string(),
        })
        .await
        .expect("passthrough parse should succeed");

    assert_eq!(result.into_inner().code, r#"print("ok")"#);
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn passthrough_adapter_extracts_fenced_code() {
    let _lock = SETTINGS_LOCK.lock().await;
    configure_test_lm(vec!["```python\nprint('hello')\n```\n".to_string()]).await;

    let predict = Predict::<RlmActionLike>::new().adapter(ChatAdapter::passthrough());
    let result = predict
        .call(RlmActionLikeInput {
            task: "say hello".to_string(),
        })
        .await
        .expect("fenced passthrough parse should succeed");

    assert_eq!(result.into_inner().code, "print('hello')");
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn passthrough_whitespace_response_surfaces_parse_error_with_chat() {
    let _lock = SETTINGS_LOCK.lock().await;
    configure_test_lm(vec!["   \n\t".to_string()]).await;

    let predict = Predict::<RlmActionLike>::new().adapter(ChatAdapter::passthrough());
    let err = predict
        .call(RlmActionLikeInput {
            task: "do something".to_string(),
        })
        .await
        .expect_err("whitespace passthrough response should fail parse");

    match err {
        PredictError::Parse {
            source: ParseError::ExtractionFailed { .. },
            raw_response,
            chat,
            ..
        } => {
            assert!(raw_response.trim().is_empty());
            assert!(
                !chat.is_empty(),
                "parse error should carry conversation chat for recovery"
            );
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn per_predict_adapter_selection_allows_mixed_dialects() {
    let _lock = SETTINGS_LOCK.lock().await;
    configure_test_lm(vec![
        "print(2 + 2)".to_string(),
        structured_response(&[("answer", "4")]),
    ])
    .await;

    let action_predict = Predict::<RlmActionLike>::new().adapter(ChatAdapter::passthrough());
    let extract_predict = Predict::<QaLike>::new();

    let action = action_predict
        .call(RlmActionLikeInput {
            task: "math".to_string(),
        })
        .await
        .expect("passthrough action parse should succeed");
    assert_eq!(action.code, "print(2 + 2)");

    let extract = extract_predict
        .call(QaLikeInput {
            question: "2 + 2".to_string(),
        })
        .await
        .expect("default chat parse should succeed");
    assert_eq!(extract.answer, "4");
}
