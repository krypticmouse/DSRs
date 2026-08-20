//! Conversation surface (RFC 0004 §1–2): conversation-in/conversation-out
//! turns through the interpreter — chat growth and per-turn spans, opening
//! rendering parity with the map-in path, the caller-managed suspend/resume
//! loop, stop-tool and budget parity across dispatching and suspending modes,
//! and replay of recorded conversation runs.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use dspy_rs::ir::{
    self, Budget, BudgetPolicy, ConversationTurn, FieldType as T, Interpreter, NodeBudget, Program,
    ProgramBuilder, RunError, RuntimeEnv, SignatureDef,
};
use dspy_rs::trace::{ReplayMode, SpanEvent, capture, replay};
use dspy_rs::{LM, LMClient, LMConfig, Message, Role, TestCompletionModel};
use rig::completion::{AssistantContent, ToolDefinition};
use rig::message::{Text, ToolCall, ToolFunction};
use serde_json::json;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn fields(pairs: &[(&str, &str)]) -> String {
    let mut out = String::new();
    for (name, value) in pairs {
        out.push_str(&format!("[[ ## {name} ## ]]\n{value}\n\n"));
    }
    out.push_str("[[ ## completed ## ]]\n");
    out
}

fn text(content: impl Into<String>) -> AssistantContent {
    AssistantContent::Text(Text {
        text: content.into(),
    })
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

async fn canned_lm(responses: Vec<AssistantContent>) -> (Arc<LM>, TestCompletionModel) {
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
    (Arc::new(lm), client)
}

fn config() -> LMConfig {
    LMConfig {
        model: "openai:gpt-4o-mini".to_string(),
        ..LMConfig::default()
    }
}

fn obj(pairs: &[(&str, serde_json::Value)]) -> serde_json::Map<String, serde_json::Value> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.clone()))
        .collect()
}

/// A 1-leaf `predict` program — what `Predict<S>` compiles to.
fn qa_program() -> Program {
    let mut b = ProgramBuilder::new("qa");
    b.model("m", config());
    let main_sig = b.sig(
        SignatureDef::build("Main")
            .input("question", T::String)
            .output("answer", T::String)
            .finish()
            .unwrap(),
    );
    let qa = b.sig(
        SignatureDef::build("QA")
            .instruction("Answer the question.")
            .input("question", T::String)
            .output("answer", T::String)
            .finish()
            .unwrap(),
    );
    let assistant = ir::predict("assistant", qa).bind("question", ir::input("question"));
    b.main(
        main_sig,
        ir::seq([assistant]).out("answer", ir::out("assistant", "answer")),
    )
    .unwrap()
}

/// A tool the loop can dispatch, counting executions so caller-managed runs
/// can assert it was never invoked.
#[derive(Clone)]
struct CountingSearch {
    calls: Arc<AtomicUsize>,
}

#[derive(Debug)]
struct CountingSearchError;

impl std::fmt::Display for CountingSearchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "counting search error")
    }
}

impl std::error::Error for CountingSearchError {}

impl rig::tool::Tool for CountingSearch {
    const NAME: &'static str = "search";
    type Error = CountingSearchError;
    type Args = serde_json::Value;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "host-side definition (ignored: the IR declares the interface)"
                .to_string(),
            parameters: json!({"type": "object", "additionalProperties": true}),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(format!(
            "results for {}: dsrs is a rust dspy",
            args.get("query").and_then(|v| v.as_str()).unwrap_or("?")
        ))
    }
}

/// A 1-leaf `agent` program with a `search` host tool. `budget` lands on the
/// node; `with_stop` adds a `submit` stop tool.
fn agent_program(budget: Option<NodeBudget>, with_stop: bool) -> Program {
    let mut b = ProgramBuilder::new("agentic");
    b.model("m", config());
    let main_sig = b.sig(
        SignatureDef::build("Main")
            .input("question", T::String)
            .output("answer", T::String)
            .finish()
            .unwrap(),
    );
    let research = b.sig(
        SignatureDef::build("Research")
            .instruction("Research and answer.")
            .input("question", T::String)
            .output("answer", T::String)
            .finish()
            .unwrap(),
    );
    let search_sig = b.sig(
        SignatureDef::build("Search")
            .input("query", T::String)
            .output("results", T::String)
            .finish()
            .unwrap(),
    );
    let search = b.host_tool("search", "Web search", search_sig, &[]);
    let mut tools = vec![search];
    let mut stop = Vec::new();
    if with_stop {
        let submit_sig = b.sig(
            SignatureDef::build("Submit")
                .input("answer", T::String)
                .output("ok", T::String)
                .finish()
                .unwrap(),
        );
        let submit = b.host_tool("submit", "Submit the final answer", submit_sig, &[]);
        tools.push(submit);
        stop.push(submit);
    }
    let mut researcher = ir::agent("researcher", research)
        .bind("question", ir::input("question"))
        .tools(tools)
        .stop_tools(stop)
        .max_turns(4);
    if let Some(budget) = budget {
        researcher = researcher.budget(budget);
    }
    b.main(
        main_sig,
        ir::seq([researcher]).out("answer", ir::out("researcher", "answer")),
    )
    .unwrap()
}

async fn load_agent(program: Program, lm: Arc<LM>, counter: &Arc<AtomicUsize>) -> Interpreter {
    let mut env = RuntimeEnv::new().bind_model("m", lm).bind_host_tool(
        "search",
        Arc::new(CountingSearch {
            calls: Arc::clone(counter),
        }),
    );
    if program
        .tools
        .iter()
        .any(|(_, tool)| program.syms.get(tool.name) == "submit")
    {
        env = env.bind_host_tool(
            "submit",
            Arc::new(CountingSearch {
                calls: Arc::clone(counter),
            }),
        );
    }
    Interpreter::load(program, env).await.unwrap()
}

// ---------------------------------------------------------------------------
// Seam 1: conversation-in/conversation-out
// ---------------------------------------------------------------------------

#[tokio::test]
async fn multi_turn_conversation_grows_chat_and_records_one_span_per_turn() {
    let (lm, _client) = canned_lm(vec![
        text(fields(&[("answer", "42")])),
        text(fields(&[("answer", "yes, exactly 42")])),
    ])
    .await;
    let interp = Interpreter::load(qa_program(), RuntimeEnv::new().bind_model("m", lm))
        .await
        .unwrap();

    let (result, trace) = capture(|| async {
        // Turn 1: opening — empty chat plus the typed input.
        let (run1, mut chat) = interp
            .run_conversation(
                dspy_rs::Chat::new(Vec::new()),
                Some(obj(&[("question", json!("what is 6*7?"))])),
                None,
                Budget::unlimited(),
            )
            .await?;
        assert_eq!(run1.output["answer"], "42");
        assert_eq!(chat.len(), 3);
        assert_eq!(chat.messages[0].role, Role::System);
        assert_eq!(chat.messages[1].role, Role::User);
        assert_eq!(chat.messages[2].role, Role::Assistant);

        // Turn 2: continuation — the caller appends the follow-up.
        chat.push_message(Message::user("are you sure?"));
        let (run2, chat) = interp
            .run_conversation(chat, None, None, Budget::unlimited())
            .await?;
        assert_eq!(run2.output["answer"], "yes, exactly 42");
        assert_eq!(chat.len(), 5);
        assert_eq!(chat.messages[4].role, Role::Assistant);
        Ok::<_, RunError>(())
    })
    .await;
    result.unwrap();

    // A turn is not a run: one span per turn, seq increments per component.
    assert_eq!(trace.components, vec!["assistant"]);
    assert_eq!(trace.spans.len(), 2);
    let opening = &trace.spans[0];
    assert_eq!(opening.seq, 0);
    assert_eq!(opening.input.as_ref().unwrap()["question"], "what is 6*7?");
    assert!(
        opening.prefix.is_some(),
        "rendered opening prefix is interned"
    );
    assert_eq!(opening.output.as_ref().unwrap()["answer"], "42");
    let continuation = &trace.spans[1];
    assert_eq!(continuation.seq, 1);
    assert!(
        continuation.prefix.is_none(),
        "caller-owned chat has no prefix split"
    );
    assert_eq!(continuation.suffix.len(), 4, "full chat recorded as suffix");
    assert!(continuation.input.is_none());
}

#[tokio::test]
async fn typed_continuation_appends_the_formatted_input_turn() {
    let (lm, client) = canned_lm(vec![
        text(fields(&[("answer", "first")])),
        text(fields(&[("answer", "second")])),
    ])
    .await;
    let interp = Interpreter::load(qa_program(), RuntimeEnv::new().bind_model("m", lm))
        .await
        .unwrap();

    let (_, mut chat) = interp
        .run_conversation(
            dspy_rs::Chat::new(Vec::new()),
            Some(obj(&[("question", json!("first question"))])),
            None,
            Budget::unlimited(),
        )
        .await
        .unwrap();
    let before = chat.len();
    (_, chat) = interp
        .run_conversation(
            chat,
            Some(obj(&[("question", json!("second question"))])),
            None,
            Budget::unlimited(),
        )
        .await
        .unwrap();
    // input rendered as the next user turn + the assistant reply.
    assert_eq!(chat.len(), before + 2);
    let last = client.last_request().unwrap();
    let sent = format!("{:?}", last.chat_history);
    assert!(sent.contains("second question"));
}

#[tokio::test]
async fn conversation_opening_matches_the_map_in_rendering() {
    let (lm, _client) = canned_lm(vec![
        text(fields(&[("answer", "a")])),
        text(fields(&[("answer", "b")])),
    ])
    .await;
    let interp = Interpreter::load(qa_program(), RuntimeEnv::new().bind_model("m", lm))
        .await
        .unwrap();
    let input = obj(&[("question", json!("what is 6*7?"))]);

    // Map-in evaluation and the conversation opening must hash identically —
    // same rendered prompt, same model config, same replay key.
    let (result, map_trace) =
        capture(|| interp.run_collecting(input.clone(), None, Budget::unlimited())).await;
    result.unwrap();
    let (result, conv_trace) = capture(|| {
        interp.run_conversation(
            dspy_rs::Chat::new(Vec::new()),
            Some(input.clone()),
            None,
            Budget::unlimited(),
        )
    })
    .await;
    result.unwrap();
    assert_eq!(
        map_trace.spans[0].request_hash, conv_trace.spans[0].request_hash,
        "conversation opening renders byte-identically to the map-in path"
    );

    // And `conversation_opening` returns exactly the prompt the turn sent.
    let opening = interp.conversation_opening(&input, None).unwrap();
    let recorded = conv_trace.prompt(&conv_trace.spans[0]);
    assert_eq!(format!("{:?}", opening.messages), format!("{recorded:?}"));
}

#[tokio::test]
async fn conversation_surface_refuses_multi_node_programs_and_empty_turns() {
    // question → drafter → checker: two leaves, no conversation to own.
    let mut b = ProgramBuilder::new("pipeline");
    b.model("m", config());
    let main_sig = b.sig(
        SignatureDef::build("Main")
            .input("question", T::String)
            .output("verdict", T::String)
            .finish()
            .unwrap(),
    );
    let qa = b.sig(
        SignatureDef::build("QA")
            .input("question", T::String)
            .output("answer", T::String)
            .finish()
            .unwrap(),
    );
    let check = b.sig(
        SignatureDef::build("Check")
            .input("answer", T::String)
            .output("verdict", T::String)
            .finish()
            .unwrap(),
    );
    let drafter = ir::predict("drafter", qa).bind("question", ir::input("question"));
    let checker = ir::predict("checker", check).bind("answer", ir::out("drafter", "answer"));
    let program = b
        .main(
            main_sig,
            ir::seq([drafter, checker]).out("verdict", ir::out("checker", "verdict")),
        )
        .unwrap();

    let (lm, _client) = canned_lm(vec![]).await;
    let two_leaves = Interpreter::load(program, RuntimeEnv::new().bind_model("m", Arc::clone(&lm)))
        .await
        .unwrap();
    let err = two_leaves
        .run_conversation(
            dspy_rs::Chat::new(Vec::new()),
            Some(obj(&[("question", json!("q"))])),
            None,
            Budget::unlimited(),
        )
        .await
        .unwrap_err();
    assert!(matches!(err, RunError::Input { .. }));

    // An empty chat with no input has nothing to send.
    let one_leaf = Interpreter::load(qa_program(), RuntimeEnv::new().bind_model("m", lm))
        .await
        .unwrap();
    let err = one_leaf
        .run_conversation(
            dspy_rs::Chat::new(Vec::new()),
            None,
            None,
            Budget::unlimited(),
        )
        .await
        .unwrap_err();
    assert!(matches!(err, RunError::Input { .. }));
}

// ---------------------------------------------------------------------------
// Seam 2: caller-managed suspend/resume
// ---------------------------------------------------------------------------

#[tokio::test]
async fn caller_managed_turn_suspends_on_tool_calls_and_resumes_with_results() {
    let (lm, _client) = canned_lm(vec![
        tool_call("search", json!({"query": "dsrs"})),
        text(fields(&[("answer", "dsrs is a rust dspy")])),
    ])
    .await;
    let executed = Arc::new(AtomicUsize::new(0));
    let interp = load_agent(agent_program(None, false), lm, &executed).await;

    let (result, trace) = capture(|| async {
        let turn = interp
            .run_conversation_caller_managed(
                dspy_rs::Chat::new(Vec::new()),
                Some(obj(&[("question", json!("what is dsrs?"))])),
                None,
                Budget::unlimited(),
            )
            .await?;
        let suspension = match turn {
            ConversationTurn::Suspended(suspension) => suspension,
            ConversationTurn::Complete { .. } => panic!("expected a suspension"),
        };
        assert_eq!(suspension.calls().len(), 1);
        assert_eq!(suspension.calls()[0].function.name, "search");
        assert!(
            suspension.chat().messages.last().unwrap().has_tool_calls(),
            "the assistant tool-call turn is already in the conversation"
        );

        // The caller executes the tool and feeds the result back.
        let turn = interp
            .resume_conversation(
                suspension,
                vec!["caller says: dsrs is a rust dspy".to_string()],
            )
            .await?;
        let (run, chat) = match turn {
            ConversationTurn::Complete { run, chat } => (run, chat),
            ConversationTurn::Suspended(_) => panic!("expected completion"),
        };
        assert_eq!(run.output["answer"], "dsrs is a rust dspy");
        assert_eq!(run.leaves.len(), 1);
        assert_eq!(run.leaves[0].tool_calls.len(), 1);
        assert_eq!(
            run.leaves[0].tool_executions,
            vec!["caller says: dsrs is a rust dspy".to_string()]
        );
        // ... conversation shape: tool-result turn then the final answer.
        let roles: Vec<Role> = chat.messages.iter().map(|m| m.role).collect();
        assert_eq!(
            roles,
            vec![
                Role::System,
                Role::User,
                Role::Assistant,
                Role::User,
                Role::Assistant
            ]
        );
        assert!(chat.messages[3].has_tool_results());
        Ok::<_, RunError>(())
    })
    .await;
    result.unwrap();

    // The interpreter never dispatched the tool.
    assert_eq!(executed.load(Ordering::SeqCst), 0);

    // One span for the whole suspended-and-resumed turn, with the same
    // exchange/tool_run/exchange stream a dispatched loop records.
    assert_eq!(trace.spans.len(), 1);
    let span = &trace.spans[0];
    let kinds: Vec<&str> = span
        .events
        .iter()
        .map(|event| match event {
            SpanEvent::Exchange { .. } => "exchange",
            SpanEvent::ToolRun { .. } => "tool_run",
            _ => "other",
        })
        .collect();
    assert_eq!(kinds, vec!["exchange", "tool_run", "exchange"]);
    match &span.events[1] {
        SpanEvent::ToolRun {
            name,
            result,
            error,
            ..
        } => {
            assert_eq!(name, "search");
            assert_eq!(result, "caller says: dsrs is a rust dspy");
            assert!(error.is_none());
        }
        other => panic!("expected ToolRun, got {other:?}"),
    }
    assert_eq!(
        span.output.as_ref().unwrap()["answer"],
        "dsrs is a rust dspy"
    );
}

#[tokio::test]
async fn dropping_a_suspension_cancels_its_span() {
    let (lm, _client) = canned_lm(vec![tool_call("search", json!({"query": "dsrs"}))]).await;
    let executed = Arc::new(AtomicUsize::new(0));
    let interp = load_agent(agent_program(None, false), lm, &executed).await;

    let (_, trace) = capture(|| async {
        let turn = interp
            .run_conversation_caller_managed(
                dspy_rs::Chat::new(Vec::new()),
                Some(obj(&[("question", json!("q"))])),
                None,
                Budget::unlimited(),
            )
            .await
            .unwrap();
        assert!(matches!(turn, ConversationTurn::Suspended(_)));
        drop(turn);
    })
    .await;

    assert_eq!(trace.spans.len(), 1);
    let error = trace.spans[0].error.as_ref().expect("span closed as error");
    assert_eq!(error.kind, dspy_rs::trace::SpanErrorKind::Cancelled);
}

#[tokio::test]
async fn suspending_and_dispatching_modes_record_identical_spans() {
    let question = obj(&[("question", json!("what is dsrs?"))]);

    // Dispatching mode: the bound tool executes.
    let (lm, _client) = canned_lm(vec![
        tool_call("search", json!({"query": "dsrs"})),
        text(fields(&[("answer", "dsrs is a rust dspy")])),
    ])
    .await;
    let executed = Arc::new(AtomicUsize::new(0));
    let interp = load_agent(agent_program(None, false), lm, &executed).await;
    let (result, dispatched) = capture(|| {
        interp.run_conversation(
            dspy_rs::Chat::new(Vec::new()),
            Some(question.clone()),
            None,
            Budget::unlimited(),
        )
    })
    .await;
    let (dispatched_run, dispatched_chat) = result.unwrap();
    assert_eq!(executed.load(Ordering::SeqCst), 1);
    let dispatched_result = match &dispatched.spans[0].events[1] {
        SpanEvent::ToolRun { result, .. } => result.clone(),
        other => panic!("expected ToolRun, got {other:?}"),
    };

    // Suspending mode over the same exchanges, feeding the exact result the
    // dispatched tool produced.
    let (lm, _client) = canned_lm(vec![
        tool_call("search", json!({"query": "dsrs"})),
        text(fields(&[("answer", "dsrs is a rust dspy")])),
    ])
    .await;
    let executed = Arc::new(AtomicUsize::new(0));
    let interp = load_agent(agent_program(None, false), lm, &executed).await;
    let (result, suspended) = capture(|| async {
        let turn = interp
            .run_conversation_caller_managed(
                dspy_rs::Chat::new(Vec::new()),
                Some(question.clone()),
                None,
                Budget::unlimited(),
            )
            .await?;
        let ConversationTurn::Suspended(suspension) = turn else {
            panic!("expected a suspension");
        };
        interp
            .resume_conversation(suspension, vec![dispatched_result.clone()])
            .await
    })
    .await;
    let ConversationTurn::Complete { run, chat } = result.unwrap() else {
        panic!("expected completion");
    };
    assert_eq!(executed.load(Ordering::SeqCst), 0);

    // Same output, same conversation, same span identity and event stream.
    assert_eq!(run.output, dispatched_run.output);
    assert_eq!(
        format!("{:?}", chat.messages),
        format!("{:?}", dispatched_chat.messages)
    );
    let a = &dispatched.spans[0];
    let b = &suspended.spans[0];
    assert_eq!(a.request_hash, b.request_hash);
    assert_eq!(a.usage.total_tokens, b.usage.total_tokens);
    assert_eq!(a.usage.prompt_tokens, b.usage.prompt_tokens);
    assert_eq!(a.usage.completion_tokens, b.usage.completion_tokens);
    assert_eq!(a.events.len(), b.events.len());
    for (left, right) in a.events.iter().zip(b.events.iter()) {
        match (left, right) {
            (SpanEvent::Exchange { message: l, .. }, SpanEvent::Exchange { message: r, .. }) => {
                assert_eq!(format!("{l:?}"), format!("{r:?}"))
            }
            (
                SpanEvent::ToolRun {
                    name: ln,
                    result: lr,
                    error: le,
                    ..
                },
                SpanEvent::ToolRun {
                    name: rn,
                    result: rr,
                    error: re,
                    ..
                },
            ) => {
                assert_eq!(ln, rn);
                assert_eq!(lr, rr);
                assert_eq!(le, re);
            }
            (left, right) => panic!("event streams diverge: {left:?} vs {right:?}"),
        }
    }
}

#[tokio::test]
async fn stop_tool_completes_the_turn_in_both_modes_without_suspending() {
    let stop_call = || tool_call("submit", json!({"answer": "42"}));

    let (lm, _client) = canned_lm(vec![stop_call()]).await;
    let executed = Arc::new(AtomicUsize::new(0));
    let interp = load_agent(agent_program(None, true), lm, &executed).await;
    let (dispatched, _) = capture(|| {
        interp.run_conversation(
            dspy_rs::Chat::new(Vec::new()),
            Some(obj(&[("question", json!("q"))])),
            None,
            Budget::unlimited(),
        )
    })
    .await;
    let (run, _) = dispatched.unwrap();
    assert_eq!(run.output["answer"], "42");

    let (lm, _client) = canned_lm(vec![stop_call()]).await;
    let interp = load_agent(agent_program(None, true), lm, &executed).await;
    let (turn, trace) = capture(|| {
        interp.run_conversation_caller_managed(
            dspy_rs::Chat::new(Vec::new()),
            Some(obj(&[("question", json!("q"))])),
            None,
            Budget::unlimited(),
        )
    })
    .await;
    let ConversationTurn::Complete { run, .. } = turn.unwrap() else {
        panic!("a stop-tool call completes the turn — it never suspends");
    };
    assert_eq!(run.output["answer"], "42");
    // Stop tools are never executed, in either mode.
    assert_eq!(executed.load(Ordering::SeqCst), 0);
    assert!(
        trace.spans[0]
            .events
            .iter()
            .all(|event| matches!(event, SpanEvent::Exchange { .. })),
        "no ToolRun events for a stop call"
    );
}

#[tokio::test]
async fn budget_exhaustion_is_identical_across_modes() {
    let budget = NodeBudget {
        max_lm_calls: Some(1),
        max_tokens: None,
        deadline_ms: None,
        on_exhausted: BudgetPolicy::Fail,
    };

    // Dispatching: the tool executes, then the next loop turn is refused.
    let (lm, _client) = canned_lm(vec![tool_call("search", json!({"query": "dsrs"}))]).await;
    let executed = Arc::new(AtomicUsize::new(0));
    let interp = load_agent(agent_program(Some(budget.clone()), false), lm, &executed).await;
    let (result, dispatched) = capture(|| {
        interp.run_conversation(
            dspy_rs::Chat::new(Vec::new()),
            Some(obj(&[("question", json!("q"))])),
            None,
            Budget::unlimited(),
        )
    })
    .await;
    let dispatched_err = result.unwrap_err();
    assert!(matches!(dispatched_err, RunError::Budget { .. }));

    // Suspending: the resume hits the same refusal at the same point.
    let (lm, _client) = canned_lm(vec![tool_call("search", json!({"query": "dsrs"}))]).await;
    let interp = load_agent(agent_program(Some(budget), false), lm, &executed).await;
    let (result, suspended) = capture(|| async {
        let turn = interp
            .run_conversation_caller_managed(
                dspy_rs::Chat::new(Vec::new()),
                Some(obj(&[("question", json!("q"))])),
                None,
                Budget::unlimited(),
            )
            .await?;
        let ConversationTurn::Suspended(suspension) = turn else {
            panic!("expected a suspension");
        };
        interp
            .resume_conversation(suspension, vec!["result".to_string()])
            .await
    })
    .await;
    let suspended_err = result.unwrap_err();
    assert!(matches!(suspended_err, RunError::Budget { .. }));
    assert_eq!(dispatched_err.to_string(), suspended_err.to_string());

    // Both spans closed as the same error with one recorded exchange.
    let a = &dispatched.spans[0];
    let b = &suspended.spans[0];
    assert_eq!(
        a.error.as_ref().map(|e| (e.kind, e.message.clone())),
        b.error.as_ref().map(|e| (e.kind, e.message.clone())),
    );
    assert_eq!(
        a.events
            .iter()
            .filter(|e| matches!(e, SpanEvent::Exchange { .. }))
            .count(),
        b.events
            .iter()
            .filter(|e| matches!(e, SpanEvent::Exchange { .. }))
            .count(),
    );
}

// ---------------------------------------------------------------------------
// Replay of conversation runs
// ---------------------------------------------------------------------------

#[tokio::test]
async fn conversation_runs_replay_turn_by_turn() {
    let input = obj(&[("question", json!("what is 6*7?"))]);

    // Record a 2-turn conversation live.
    let (lm, _client) = canned_lm(vec![
        text(fields(&[("answer", "42")])),
        text(fields(&[("answer", "yes, exactly 42")])),
    ])
    .await;
    let interp = Interpreter::load(qa_program(), RuntimeEnv::new().bind_model("m", lm))
        .await
        .unwrap();
    let (result, recording) = capture(|| async {
        let (run1, mut chat) = interp
            .run_conversation(
                dspy_rs::Chat::new(Vec::new()),
                Some(input.clone()),
                None,
                Budget::unlimited(),
            )
            .await?;
        chat.push_message(Message::user("are you sure?"));
        let (run2, chat) = interp
            .run_conversation(chat, None, None, Budget::unlimited())
            .await?;
        Ok::<_, RunError>((run1.output, run2.output, chat))
    })
    .await;
    let (live1, live2, live_chat) = result.unwrap();

    // Replay strictly against an LM with no responses left: every turn must
    // be served from the recording, and the chats must rebuild identically.
    let (empty_lm, _client) = canned_lm(vec![]).await;
    let replayed = Interpreter::load(qa_program(), RuntimeEnv::new().bind_model("m", empty_lm))
        .await
        .unwrap();
    let (result, report) = replay(&recording, ReplayMode::Strict, || async {
        let (run1, mut chat) = replayed
            .run_conversation(
                dspy_rs::Chat::new(Vec::new()),
                Some(input.clone()),
                None,
                Budget::unlimited(),
            )
            .await?;
        chat.push_message(Message::user("are you sure?"));
        let (run2, chat) = replayed
            .run_conversation(chat, None, None, Budget::unlimited())
            .await?;
        Ok::<_, RunError>((run1.output, run2.output, chat))
    })
    .await;
    let (served1, served2, served_chat) = result.unwrap();

    assert_eq!(report.served, 2);
    assert_eq!(report.live, 0);
    assert_eq!(served1, live1);
    assert_eq!(served2, live2);
    assert_eq!(
        format!("{:?}", served_chat.messages),
        format!("{:?}", live_chat.messages)
    );
}

#[tokio::test]
async fn caller_managed_turns_never_suspend_under_replay() {
    let input = obj(&[("question", json!("what is dsrs?"))]);

    // Record a suspended-and-resumed turn live.
    let (lm, _client) = canned_lm(vec![
        tool_call("search", json!({"query": "dsrs"})),
        text(fields(&[("answer", "dsrs is a rust dspy")])),
    ])
    .await;
    let executed = Arc::new(AtomicUsize::new(0));
    let interp = load_agent(agent_program(None, false), lm, &executed).await;
    let (result, recording) = capture(|| async {
        let turn = interp
            .run_conversation_caller_managed(
                dspy_rs::Chat::new(Vec::new()),
                Some(input.clone()),
                None,
                Budget::unlimited(),
            )
            .await?;
        let ConversationTurn::Suspended(suspension) = turn else {
            panic!("expected a suspension");
        };
        interp
            .resume_conversation(suspension, vec!["tool says 42".to_string()])
            .await
    })
    .await;
    let ConversationTurn::Complete { run: live_run, .. } = result.unwrap() else {
        panic!("expected completion");
    };

    // Under replay the whole turn is served — tool effects are baked into the
    // recorded span, so the caller never sees a suspension.
    let (empty_lm, _client) = canned_lm(vec![]).await;
    let replayed = load_agent(agent_program(None, false), empty_lm, &executed).await;
    let (turn, report) = replay(&recording, ReplayMode::Strict, || {
        replayed.run_conversation_caller_managed(
            dspy_rs::Chat::new(Vec::new()),
            Some(input.clone()),
            None,
            Budget::unlimited(),
        )
    })
    .await;
    let ConversationTurn::Complete { run, chat } = turn.unwrap() else {
        panic!("served turns never suspend");
    };
    assert_eq!(report.served, 1);
    assert_eq!(run.output, live_run.output);
    assert_eq!(
        run.leaves[0].tool_executions,
        vec!["tool says 42".to_string()]
    );
    // The served chat carries the recorded tool-call turn and result.
    assert!(chat.messages.iter().any(|m| m.has_tool_calls()));
    assert!(chat.messages.iter().any(|m| m.has_tool_results()));
}
