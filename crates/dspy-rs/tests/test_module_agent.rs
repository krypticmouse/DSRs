//! RFC 0003 M-2/M-3, agent lane: `#[tool]` (host implementation + metadata),
//! `#[agent]` (StepDef + standalone static-lane execution), and a `#[module]`
//! whose body lowers an agent step to a first-class `AgentLoop` node with the
//! tool bound and the capability ceiling self-granted.

use std::sync::LazyLock;

use dspy_rs::ir::StepKind;
use dspy_rs::trace::{SpanEvent, capture};
use dspy_rs::{
    LM, LMClient, TestCompletionModel, agent, configure, module, tool,
};
use rig::completion::AssistantContent;
use rig::message::{Text, ToolCall, ToolFunction};
use serde_json::json;
use tokio::sync::Mutex;

static SETTINGS_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

fn text_fields(fields: &[(&str, &str)]) -> AssistantContent {
    let mut response = String::new();
    for (name, value) in fields {
        response.push_str(&format!("[[ ## {name} ## ]]\n{value}\n\n"));
    }
    response.push_str("[[ ## completed ## ]]\n");
    AssistantContent::Text(Text { text: response })
}

fn tool_call(name: &str, args: serde_json::Value) -> AssistantContent {
    AssistantContent::ToolCall(ToolCall::new(
        format!("tc-{name}"),
        ToolFunction {
            name: name.to_string(),
            arguments: args,
        },
    ))
}

async fn make_test_lm(responses: Vec<AssistantContent>) -> (LM, TestCompletionModel) {
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

// ---------------------------------------------------------------------------
// Steps
// ---------------------------------------------------------------------------

/// Uppercase text.
#[tool(caps("demo:shout"))]
fn shout(text: String) -> String {
    text.to_uppercase()
}

/// Research the question. Use the shout tool when volume is needed.
#[agent(tools(shout), max_turns = 3, budget(tokens = 50_000, on_exhausted = finalize))]
fn research(question: String) -> String;

/// Record the final report.
#[tool]
fn submit(report: String) -> String {
    report
}

/// Compile a report; call submit when done.
#[agent(tools(shout, submit), stop_tools(submit), max_turns = 4)]
fn report(question: String) -> String;

/// Answer immediately — one turn, fail when exhausted.
#[agent(tools(shout), max_turns = 1)]
fn quick(question: String) -> String;

#[dspy_rs::Schema]
#[derive(Debug)]
pub struct AOut {
    pub answer: String,
}

#[module(caps("demo:shout"))]
async fn agentic(question: String) -> Result<AOut, dspy_rs::ir::RunError> {
    let researcher = research(question).await?;
    Ok(AOut {
        answer: researcher.research,
    })
}

// ---------------------------------------------------------------------------
// M-2: metadata
// ---------------------------------------------------------------------------

#[test]
fn tool_metadata_and_definition() {
    let tool = shout::__dsrs_tool();
    assert_eq!(tool.name, "shout");
    assert_eq!(tool.desc, "Uppercase text.");
    assert_eq!(tool.caps, ["demo:shout"]);
    assert_eq!(tool.sig.inputs.len(), 1);
    assert_eq!(tool.sig.inputs[0].name.as_ref(), "text");
    assert_eq!(tool.sig.outputs[0].name.as_ref(), "shout");

    // The original fn stays plain Rust.
    assert_eq!(shout("hi".to_string()), "HI");
}

#[test]
fn agent_step_metadata() {
    let step = research::__dsrs_step();
    assert_eq!(step.name, "research");
    assert_eq!(step.kind, StepKind::Agent);
    let agent = step.agent.expect("agent steps carry opts");
    assert_eq!(agent.tools.len(), 1);
    assert_eq!(agent.tools[0].name, "shout");
    assert_eq!(agent.max_turns, Some(3));
    assert_eq!(agent.budget.max_tokens, Some(50_000));
    assert!(agent.stop_tools.is_empty());
}

// ---------------------------------------------------------------------------
// M-3: the lowered artifact
// ---------------------------------------------------------------------------

#[test]
fn module_lowers_agent_step_and_tool() {
    let program = agentic::program();
    let printed = program.to_dsrs();
    assert!(printed.contains("caps { demo:shout }"), "ceiling printed:\n{printed}");
    assert!(printed.contains("tool shout"), "host tool declared:\n{printed}");
    assert!(printed.contains("agent "), "agent loop node:\n{printed}");
    assert!(
        program.param_id("tool.shout.desc").is_some(),
        "tool description is an optimizable gene"
    );
    assert!(program.param_id("researcher.context").is_some());
    assert!(agentic::OPACITY.is_empty(), "no holes in this module");
}

// ---------------------------------------------------------------------------
// Execution: module (interpreter AgentLoop) then standalone (static lane)
// ---------------------------------------------------------------------------

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn agent_runs_in_module_and_standalone() {
    let _lock = SETTINGS_LOCK.lock().await;
    let (lm, client) = make_test_lm(vec![
        // Module run: one tool turn, then a parseable answer.
        tool_call("shout", json!({"text": "hi"})),
        text_fields(&[("research", "DONE")]),
    ])
    .await;
    configure(lm);

    let (out, trace) = capture(|| agentic("how loud?".to_string())).await;
    let out = out.expect("module agent run succeeds");
    assert_eq!(out.answer, "DONE");

    // One span for the loop, with the ToolRun event inside it.
    assert_eq!(trace.components, vec!["researcher"]);
    let events = &trace.spans[0].events;
    assert!(
        events.iter().any(|event| matches!(
            event,
            SpanEvent::ToolRun { name, error: None, .. } if name == "shout"
        )),
        "the host tool executed inside the loop: {events:?}"
    );

    // Standalone: the same fn, the same 1-node AgentLoop program, same tool
    // binding — and the same attribute options (see the dedicated tests below).
    client.push_response(tool_call("shout", json!({"text": "again"})));
    client.push_response(text_fields(&[("research", "LOUDER")]));
    let predicted = research("standalone?".to_string())
        .await
        .expect("standalone agent call succeeds");
    assert_eq!(predicted.research, "LOUDER");
}

// ---------------------------------------------------------------------------
// Standalone path honors the `#[agent(...)]` options (phase 4)
// ---------------------------------------------------------------------------

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn standalone_honors_stop_tools() {
    let _lock = SETTINGS_LOCK.lock().await;
    let (lm, _client) = make_test_lm(vec![
        // A `submit` call must end the loop with its args as the raw final
        // output — no tool execution, no second LM turn (the queue has none).
        tool_call("submit", json!({"report": "FINAL"})),
    ])
    .await;
    configure(lm);

    let predicted = report("summarize".to_string())
        .await
        .expect("stop tool ends the loop with its args as the output");
    assert_eq!(predicted.report, "FINAL");
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn standalone_honors_max_turns() {
    let _lock = SETTINGS_LOCK.lock().await;
    let (lm, _client) = make_test_lm(vec![
        // One tool turn spends the whole `max_turns = 1` allowance; the old
        // ignored-options path would take a second LM turn and fail on the
        // empty response queue instead.
        tool_call("shout", json!({"text": "stall"})),
    ])
    .await;
    configure(lm);

    let err = quick("now".to_string())
        .await
        .expect_err("one tool turn exhausts max_turns = 1");
    let message = format!("{err:?}");
    assert!(
        message.contains("budget exhausted"),
        "max_turns bounded the loop: {message}"
    );
    assert!(
        !message.contains("queue is empty"),
        "the loop must not take a second LM turn: {message}"
    );
}
