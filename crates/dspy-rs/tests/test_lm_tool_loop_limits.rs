use dspy_rs::core::lm::ToolLoopError;
use dspy_rs::{Chat, LM, LMClient, Message, TestCompletionModel, ToolLoopMode};
use rig::completion::{AssistantContent, ToolDefinition};
use rig::tool::{Tool, ToolDyn};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Clone)]
struct Counter(Arc<AtomicUsize>);

impl Tool for Counter {
    const NAME: &'static str = "counter";
    type Error = std::convert::Infallible;
    type Args = serde_json::Value;
    type Output = String;

    async fn definition(&self, _: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.into(),
            description: "Count executions".into(),
            parameters: serde_json::json!({"type": "object"}),
        }
    }

    async fn call(&self, _: Self::Args) -> Result<String, Self::Error> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok("counted".into())
    }
}

fn tool_call(id: usize) -> AssistantContent {
    AssistantContent::tool_call(format!("call_{id}"), "counter", serde_json::json!({}))
}

async fn lm(limit: u32, responses: Vec<AssistantContent>) -> (LM, TestCompletionModel) {
    let client = TestCompletionModel::new(responses);
    let lm = LM::builder()
        .base_url("http://localhost:11434/v1".into())
        .model("test".into())
        .max_tool_iterations(limit)
        .max_retries(0)
        .build()
        .await
        .unwrap()
        .with_client(LMClient::Test(client.clone()))
        .await
        .unwrap();
    (lm, client)
}

fn tools(count: &Arc<AtomicUsize>) -> Vec<Arc<dyn ToolDyn>> {
    vec![Arc::new(Counter(Arc::clone(count)))]
}

fn chat() -> Chat {
    Chat::new(vec![Message::user("Use the counter, then finish")])
}

struct FailingTool;
impl Tool for FailingTool {
    const NAME: &'static str = "fails";
    type Error = std::io::Error;
    type Args = serde_json::Value;
    type Output = String;
    async fn definition(&self, _: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.into(),
            description: "Fail for the error-chain test".into(),
            parameters: serde_json::json!({"type":"object"}),
        }
    }
    async fn call(&self, _: Self::Args) -> Result<String, Self::Error> {
        Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "fixture tool failed",
        ))
    }
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn tool_execution_failure_keeps_name_context_and_original_cause() {
    let (lm, _) = lm(
        1,
        vec![AssistantContent::tool_call(
            "call",
            "fails",
            serde_json::json!({}),
        )],
    )
    .await;
    let error = lm
        .call(chat(), vec![Arc::new(FailingTool)])
        .await
        .unwrap_err();
    let context = error.downcast_ref::<ToolLoopError>().unwrap();
    assert!(
        matches!(context, ToolLoopError::Execution {name, context, ..}
        if name == "fails" && context == "initial")
    );
    assert!(error.chain().any(|source| {
        source
            .downcast_ref::<std::io::Error>()
            .is_some_and(|error| error.kind() == std::io::ErrorKind::InvalidInput)
    }));
    assert!(format!("{error:#}").contains("fixture tool failed"));
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn missing_tool_feedback_remains_recoverable_without_executing_other_tools() {
    let (lm, _) = lm(
        1,
        vec![
            AssistantContent::tool_call("call", "missing", serde_json::json!({})),
            AssistantContent::text("done"),
        ],
    )
    .await;
    let count = Arc::new(AtomicUsize::new(0));
    let response = lm.call(chat(), tools(&count)).await.unwrap();
    assert_eq!(response.output.content(), "done");
    assert_eq!(count.load(Ordering::SeqCst), 0);
    assert!(response.events.iter().any(|event| matches!(event,
        dspy_rs::SpanEvent::ToolRun {name, error: Some(error), ..}
        if name == "missing" && error == "Tool 'missing' not found")));
}

#[rstest::rstest]
#[case(1)]
#[case(2)]
#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn final_completion_is_allowed_after_last_tool_round(#[case] limit: u32) {
    let mut responses = (0..limit as usize).map(tool_call).collect::<Vec<_>>();
    responses.push(AssistantContent::text("done"));
    let (lm, _) = lm(limit, responses).await;
    let count = Arc::new(AtomicUsize::new(0));
    let response = lm.call(chat(), tools(&count)).await.unwrap();
    assert_eq!(response.output.content(), "done");
    assert_eq!(count.load(Ordering::SeqCst), limit as usize);
    assert_eq!(response.tool_calls.len(), limit as usize);
    assert_eq!(response.chat.len(), 2 * limit as usize + 2);
}

#[rstest::rstest]
#[case(0)]
#[case(1)]
#[case(2)]
#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn exhausted_limit_prevents_additional_tool_side_effects(#[case] limit: u32) {
    let responses = (0..=limit as usize).map(tool_call).collect();
    let (lm, client) = lm(limit, responses).await;
    let count = Arc::new(AtomicUsize::new(0));
    let error = lm.call(chat(), tools(&count)).await.unwrap_err();
    assert!(error.to_string().contains("Max tool iterations reached"));
    assert!(matches!(error.downcast_ref::<ToolLoopError>(),
        Some(ToolLoopError::IterationLimit { limit: actual }) if *actual == limit as usize));
    assert_eq!(count.load(Ordering::SeqCst), limit as usize);
    assert_eq!(
        client.last_request().unwrap().chat_history.iter().count(),
        2 * limit as usize + 1,
    );
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn zero_limit_allows_text_responses() {
    let (lm, _) = lm(0, vec![AssistantContent::text("done")]).await;
    let count = Arc::new(AtomicUsize::new(0));
    assert_eq!(
        lm.call(chat(), tools(&count))
            .await
            .unwrap()
            .output
            .content(),
        "done"
    );
    assert_eq!(count.load(Ordering::SeqCst), 0);
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn caller_managed_loops_return_tool_calls_with_zero_auto_limit() {
    let (lm, _) = lm(0, vec![tool_call(0)]).await;
    let count = Arc::new(AtomicUsize::new(0));
    let response = lm
        .call_with_tool_loop_mode(chat(), tools(&count), ToolLoopMode::CallerManaged)
        .await
        .unwrap();
    assert_eq!(response.tool_calls.len(), 1);
    assert!(response.tool_executions.is_empty());
    assert_eq!(count.load(Ordering::SeqCst), 0);
}
