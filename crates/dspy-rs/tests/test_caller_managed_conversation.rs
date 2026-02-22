//! CallerManaged + tools + conversation flow test.
//!
//! This is the RLM critical path: the caller controls tool execution and
//! manages the conversation loop, not the LM layer's auto tool loop.

use dspy_rs::{
    ChatAdapter, LM, LMClient, Message, Predict, Role, Signature, TestCompletionModel,
    ToolLoopMode, configure,
};
use rig::completion::AssistantContent;
use rig::message::{Text, ToolCall, ToolFunction};
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

async fn build_test_lm(responses: Vec<AssistantContent>) -> (LM, TestCompletionModel) {
    let client = TestCompletionModel::new(responses);
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
/// Code execution signature for RLM-style interaction.
struct CodeExec {
    #[input]
    prompt: String,

    #[output]
    result: String,
}

/// The full RLM-style loop:
/// 1. Predict builds initial chat → calls LM → model requests a tool call
/// 2. CallerManaged mode: LM returns the tool call without executing it
/// 3. Caller manually executes the tool, then calls Predict forward with prior chat history
/// 4. LM returns the final text answer
///
/// This is the exact pattern RLM will use for Python REPL interaction.
#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn caller_managed_tool_loop_with_conversation() {
    let _lock = SETTINGS_LOCK.lock().await;

    // Response 1: model wants to call a tool (returned as text since TestCompletionModel
    // only supports single-content responses via AssistantContent)
    let tool_call_response =
        text_response("[[ ## result ## ]]\nNeed to execute code first\n\n[[ ## completed ## ]]\n");
    // Response 2: after seeing tool result, model gives final answer
    let final_response = text_response(response_with_fields(&[("result", "42")]));

    let (lm, _client) = build_test_lm(vec![tool_call_response, final_response]).await;
    configure(lm, ChatAdapter {});

    let predict = Predict::<CodeExec>::new();
    let input = CodeExecInput {
        prompt: "Calculate 6 * 7".to_string(),
    };

    // Turn 1
    let first_result = predict
        .forward(input, None)
        .await
        .expect("first turn forward should succeed");
    let chat = first_result.chat().clone();
    assert_eq!(
        first_result.into_inner().result,
        "Need to execute code first"
    );

    // Turn 2: continue with prior chat and typed follow-up
    let follow_up = CodeExecInput {
        prompt: "Tool output: 42".to_string(),
    };
    let second_result = predict
        .forward(follow_up, Some(chat))
        .await
        .expect("second turn forward should succeed");
    let final_chat = second_result.chat().clone();
    assert_eq!(second_result.into_inner().result, "42");

    // Verify chat grew across turns
    assert!(
        final_chat.len() >= 5,
        "chat should have system + user + asst + user + asst, got {}",
        final_chat.len()
    );

    // Verify turn ordering
    assert_eq!(final_chat.messages[0].role, Role::System);
    assert_eq!(final_chat.messages[1].role, Role::User);
    assert_eq!(final_chat.messages[2].role, Role::Assistant);
    assert_eq!(final_chat.messages[3].role, Role::User); // caller's tool result
    assert_eq!(final_chat.messages[4].role, Role::Assistant); // final answer
}

/// Tests the LM-level CallerManaged mode directly: when a tool call is requested
/// with CallerManaged mode, the LM returns the tool calls without executing them
/// and the caller controls what happens next.
#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn lm_caller_managed_returns_tool_calls_in_chat_history() {
    let _lock = SETTINGS_LOCK.lock().await;

    // Model responds with a tool call
    let tool_call_content = AssistantContent::ToolCall(ToolCall::new(
        "tc-1".to_string(),
        ToolFunction {
            name: "python_repl".to_string(),
            arguments: serde_json::json!({"code": "print(6 * 7)"}),
        },
    ));

    let (lm, _client) = build_test_lm(vec![tool_call_content]).await;

    let chat = dspy_rs::Chat::new(vec![Message::user("Run some code")]);
    let response = lm
        .call(chat, vec![], ToolLoopMode::CallerManaged)
        .await
        .expect("caller-managed call should succeed");

    // Tool calls returned but NOT executed
    assert_eq!(response.tool_calls.len(), 1);
    assert_eq!(response.tool_calls[0].function.name, "python_repl");
    assert!(
        response.tool_executions.is_empty(),
        "CallerManaged should not execute tools"
    );

    // Chat history should contain the tool call message
    assert!(
        response.chat.messages.iter().any(|m| m.has_tool_calls()),
        "chat history should include the tool call message"
    );
}

/// Multi-turn with parse failure on second turn verifies that errors
/// include the correct raw_response from the continuation, not the first turn.
#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn parse_failure_on_second_turn_includes_correct_raw_response() {
    let _lock = SETTINGS_LOCK.lock().await;

    let good_response = text_response(response_with_fields(&[("result", "first answer")]));
    // Second response is malformed — no field markers
    let bad_response = text_response("This response has no field markers at all.");

    let (lm, _client) = build_test_lm(vec![good_response, bad_response]).await;
    configure(lm, ChatAdapter {});

    let predict = Predict::<CodeExec>::new();
    let input = CodeExecInput {
        prompt: "test".to_string(),
    };

    // Turn 1: succeeds
    let first_result = predict.forward(input, None).await.expect("turn 1");
    let chat = first_result.chat().clone();
    assert_eq!(first_result.into_inner().result, "first answer");

    // Turn 2: should fail with parse error containing the bad response
    let follow_up = CodeExecInput {
        prompt: "follow up".to_string(),
    };
    let err = predict
        .forward(follow_up, Some(chat))
        .await
        .expect_err("second turn should fail");

    match err {
        dspy_rs::PredictError::Parse {
            raw_response,
            source,
            ..
        } => {
            assert!(
                raw_response.contains("no field markers"),
                "raw_response should be from the second turn, got: {}",
                raw_response
            );
            // The error should mention the missing field
            let fields = source.fields();
            assert!(
                !fields.is_empty() || source.field().is_some(),
                "parse error should identify which field(s) failed"
            );
        }
        other => panic!(
            "expected PredictError::Parse, got: {:?}",
            std::mem::discriminant(&other)
        ),
    }
}
