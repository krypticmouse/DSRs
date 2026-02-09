use std::sync::LazyLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use dspy_rs::{
    ChatAdapter, LM, LMClient, Module, ReAct, Signature, TestCompletionModel, configure,
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

#[derive(Signature, Clone, Debug)]
struct QA {
    #[input]
    question: String,

    #[output]
    answer: String,
}

type QAOutput = __QAOutput;

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn react_builder_executes_tool_loop_and_extracts_output() {
    let _lock = SETTINGS_LOCK.lock().await;

    let action_1 = response_with_fields(&[
        ("thought", "Need lookup"),
        ("action", "search"),
        ("action_input", "{\"query\":\"capital of france\"}"),
    ]);
    let action_2 = response_with_fields(&[
        ("thought", "Done"),
        ("action", "finish"),
        ("action_input", "Use gathered observation"),
    ]);
    let extract = response_with_fields(&[("output", "{\"answer\":\"Paris\"}")]);

    configure_test_lm(vec![action_1, action_2, extract]).await;

    let calls = std::sync::Arc::new(AtomicUsize::new(0));
    let calls_for_tool = calls.clone();

    let react = ReAct::<QA>::builder()
        .max_steps(3)
        .tool("search", "Search docs", move |args| {
            let calls_for_tool = calls_for_tool.clone();
            async move {
                calls_for_tool.fetch_add(1, Ordering::SeqCst);
                format!("observation:{args}")
            }
        })
        .build();

    let outcome = react
        .forward(QAInput {
            question: "What is the capital of France?".to_string(),
        })
        .await;

    let (result, metadata) = outcome.into_parts();
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "tool execution count mismatch; metadata raw_response: {}",
        metadata.raw_response
    );
    assert!(
        metadata
            .tool_executions
            .iter()
            .any(|entry| entry.contains("observation:")),
        "expected observation execution in metadata; got {:?}",
        metadata.tool_executions
    );

    let result: QAOutput = result
        .map_err(|err| format!("{err:?}"))
        .expect("react call should succeed");
    assert_eq!(result.answer, "Paris");
}
