use std::sync::LazyLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use dspy_rs::{
    ChatAdapter, LM, LMClient, Module, ReAct, Signature, TestCompletionModel, configure,
};
use rig::completion::AssistantContent;
use rig::message::Text;
use serde_json::Value;
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

fn parse_calculator_args(args: &str) -> (i64, i64) {
    let value: Value =
        serde_json::from_str(args).unwrap_or_else(|_| serde_json::json!({ "a": 0, "b": 0 }));
    let a = value.get("a").and_then(Value::as_i64).unwrap_or(0);
    let b = value.get("b").and_then(Value::as_i64).unwrap_or(0);
    (a, b)
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

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn react_builder_executes_multi_tool_calculator_loop_and_extracts_output() {
    let _lock = SETTINGS_LOCK.lock().await;

    let action_1 = response_with_fields(&[
        ("thought", "Need to add first"),
        ("action", "add"),
        ("action_input", "{\"a\":17,\"b\":5}"),
    ]);
    let action_2 = response_with_fields(&[
        ("thought", "Now multiply the intermediate result"),
        ("action", "multiply"),
        ("action_input", "{\"a\":22,\"b\":3}"),
    ]);
    let action_3 = response_with_fields(&[
        ("thought", "Done"),
        ("action", "finish"),
        ("action_input", "66"),
    ]);
    let extract = response_with_fields(&[("output", "{\"answer\":\"66\"}")]);

    configure_test_lm(vec![action_1, action_2, action_3, extract]).await;

    let add_calls = std::sync::Arc::new(AtomicUsize::new(0));
    let multiply_calls = std::sync::Arc::new(AtomicUsize::new(0));
    let add_calls_for_tool = add_calls.clone();
    let multiply_calls_for_tool = multiply_calls.clone();

    let react = ReAct::<QA>::builder()
        .max_steps(4)
        .tool("add", "Adds two integers {a,b}", move |args| {
            let add_calls = add_calls_for_tool.clone();
            async move {
                add_calls.fetch_add(1, Ordering::SeqCst);
                let (a, b) = parse_calculator_args(&args);
                (a + b).to_string()
            }
        })
        .tool("multiply", "Multiplies two integers {a,b}", move |args| {
            let multiply_calls = multiply_calls_for_tool.clone();
            async move {
                multiply_calls.fetch_add(1, Ordering::SeqCst);
                let (a, b) = parse_calculator_args(&args);
                (a * b).to_string()
            }
        })
        .build();

    let outcome = react
        .forward(QAInput {
            question: "Compute (17 + 5) * 3 using tools.".to_string(),
        })
        .await;

    let (result, metadata) = outcome.into_parts();
    assert_eq!(
        add_calls.load(Ordering::SeqCst),
        1,
        "add tool execution count mismatch; metadata raw_response: {}",
        metadata.raw_response
    );
    assert_eq!(
        multiply_calls.load(Ordering::SeqCst),
        1,
        "multiply tool execution count mismatch; metadata raw_response: {}",
        metadata.raw_response
    );
    let tool_names: Vec<String> = metadata
        .tool_calls
        .iter()
        .map(|call| call.function.name.clone())
        .collect();
    assert!(
        tool_names.iter().any(|name| name == "add")
            && tool_names.iter().any(|name| name == "multiply"),
        "expected add and multiply in tool call trajectory; got {:?}",
        tool_names
    );
    assert!(
        metadata
            .tool_executions
            .iter()
            .any(|entry| entry.contains("Step 1"))
            && metadata
                .tool_executions
                .iter()
                .any(|entry| entry.contains("Step 2"))
            && metadata
                .tool_executions
                .iter()
                .any(|entry| entry.contains("Step 3")),
        "expected full multi-step trajectory in metadata; got {:?}",
        metadata.tool_executions
    );
    assert!(
        metadata
            .tool_executions
            .iter()
            .any(|entry| entry.contains("Observation: 22"))
            && metadata
                .tool_executions
                .iter()
                .any(|entry| entry.contains("Observation: 66")),
        "expected calculator observations in trajectory; got {:?}",
        metadata.tool_executions
    );

    let result: QAOutput = result
        .map_err(|err| format!("{err:?}"))
        .expect("react call should succeed");
    assert_eq!(result.answer, "66");
}
