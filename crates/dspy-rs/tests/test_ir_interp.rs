//! IR-3 (RFC 0002 §3): interpreter end-to-end over canned LM responses —
//! every node kind, overlay read-through at render time, trace capture with
//! program leaf names, budget metering, and load-time refusals.

use std::sync::Arc;

use dspy_rs::ir::{
    self, Budget, FieldType as T, Interpreter, LoadError, Overlay, ParamValue, Program,
    ProgramBuilder, RunError, RuntimeEnv, SignatureDef,
};
use dspy_rs::trace::{SpanEvent, capture};
use dspy_rs::typesys::{EnumDef, EnumValueDef, TypeTable};
use dspy_rs::{LM, LMClient, LMConfig, TestCompletionModel};
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

/// question → drafter (QA) → checker (Check) → verdict.
fn seq_program() -> Program {
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
            .instruction("Answer the question.")
            .input("question", T::String)
            .output("answer", T::String)
            .finish()
            .unwrap(),
    );
    let check = b.sig(
        SignatureDef::build("Check")
            .instruction("Judge the answer.")
            .input("answer", T::String)
            .output("verdict", T::String)
            .finish()
            .unwrap(),
    );
    let drafter = ir::predict("drafter", qa).bind("question", ir::input("question"));
    let checker = ir::predict("checker", check).bind("answer", ir::out("drafter", "answer"));
    b.main(
        main_sig,
        ir::seq([drafter, checker]).out("verdict", ir::out("checker", "verdict")),
    )
    .unwrap()
}

// ---------------------------------------------------------------------------
// Seq + trace capture
// ---------------------------------------------------------------------------

#[tokio::test]
async fn seq_pipeline_end_to_end_with_trace() {
    let (lm, client) = canned_lm(vec![
        text(fields(&[("answer", "42")])),
        text(fields(&[("verdict", "correct")])),
    ])
    .await;
    let interp = Interpreter::load(seq_program(), RuntimeEnv::new().bind_model("m", lm))
        .await
        .unwrap();

    let (result, trace) = capture(|| {
        interp.run(
            obj(&[("question", json!("what is 6*7?"))]),
            None,
            Budget::unlimited(),
        )
    })
    .await;
    let output = result.unwrap();
    assert_eq!(output["verdict"], "correct");

    // Component names are the program-unique leaf names.
    assert_eq!(trace.components, vec!["drafter", "checker"]);
    assert_eq!(trace.spans.len(), 2);
    let drafter = &trace.spans[0];
    assert_eq!(trace.component_name(drafter.component), "drafter");
    assert_eq!(drafter.seq, 0);
    assert_eq!(drafter.input.as_ref().unwrap()["question"], "what is 6*7?");
    assert_eq!(drafter.output.as_ref().unwrap()["answer"], "42");
    assert!(drafter.prefix.is_some(), "rendered prefix is interned");
    let checker = &trace.spans[1];
    assert_eq!(trace.component_name(checker.component), "checker");
    assert_eq!(checker.input.as_ref().unwrap()["answer"], "42");
    assert_eq!(checker.output.as_ref().unwrap()["verdict"], "correct");

    // The checker prompt was rendered from the drafter's parsed output.
    let last = client.last_request().unwrap();
    assert!(last.preamble.unwrap().contains("Judge the answer."));
}

#[tokio::test]
async fn missing_input_is_rejected_before_any_call() {
    let (lm, _client) = canned_lm(vec![]).await;
    let interp = Interpreter::load(seq_program(), RuntimeEnv::new().bind_model("m", lm))
        .await
        .unwrap();
    let err = interp
        .run(obj(&[]), None, Budget::unlimited())
        .await
        .unwrap_err();
    assert!(matches!(err, RunError::Input { .. }));
}

// ---------------------------------------------------------------------------
// ForkJoin
// ---------------------------------------------------------------------------

#[tokio::test]
async fn fork_join_runs_branches_concurrently() {
    let mut b = ProgramBuilder::new("forked");
    let left_m = b.model("left_m", config());
    let right_m = b.model("right_m", config());
    let main_sig = b.sig(
        SignatureDef::build("Main")
            .input("question", T::String)
            .output("a", T::String)
            .output("b", T::String)
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
    let fork = ir::fork([
        ir::predict("left", qa)
            .model(left_m)
            .bind("question", ir::input("question")),
        ir::predict("right", qa)
            .model(right_m)
            .bind("question", ir::input("question")),
    ])
    .join("a", ir::out("left", "answer"))
    .join("b", ir::out("right", "answer"))
    .named("both");
    let program = b
        .main(
            main_sig,
            ir::seq([fork])
                .out("a", ir::out("both", "a"))
                .out("b", ir::out("both", "b")),
        )
        .unwrap();

    let (left_lm, _) = canned_lm(vec![text(fields(&[("answer", "from-left")]))]).await;
    let (right_lm, _) = canned_lm(vec![text(fields(&[("answer", "from-right")]))]).await;
    let interp = Interpreter::load(
        program,
        RuntimeEnv::new()
            .bind_model("left_m", left_lm)
            .bind_model("right_m", right_lm),
    )
    .await
    .unwrap();

    let (result, trace) =
        capture(|| interp.run(obj(&[("question", json!("q"))]), None, Budget::unlimited())).await;
    let output = result.unwrap();
    assert_eq!(output["a"], "from-left");
    assert_eq!(output["b"], "from-right");
    // Both branch spans recorded on the shared capture scope.
    let mut components: Vec<&str> = trace.components.iter().map(String::as_str).collect();
    components.sort();
    assert_eq!(components, vec!["left", "right"]);
}

// ---------------------------------------------------------------------------
// Route
// ---------------------------------------------------------------------------

#[tokio::test]
async fn route_selects_the_matching_arm() {
    let mut b = ProgramBuilder::new("routed");
    b.model("m", config());
    let mut types = TypeTable::default();
    types.enums.insert(
        "Severity".to_string(),
        EnumDef {
            internal_name: "Severity".to_string(),
            rendered_name: "Severity".to_string(),
            docs: None,
            values: ["Low", "High"]
                .iter()
                .map(|name| EnumValueDef {
                    name: (*name).to_string(),
                    rendered_name: (*name).to_string(),
                    docs: None,
                })
                .collect(),
        },
    );
    b.add_types(&types);
    let main_sig = b.sig(
        SignatureDef::build("Main")
            .input("ticket", T::String)
            .output("reply", T::String)
            .finish()
            .unwrap(),
    );
    let classify = b.sig(
        SignatureDef::build("Classify")
            .input("ticket", T::String)
            .output("severity", T::Enum("Severity".to_string()))
            .finish()
            .unwrap(),
    );
    let reply = b.sig(
        SignatureDef::build("Reply")
            .input("ticket", T::String)
            .output("reply", T::String)
            .finish()
            .unwrap(),
    );
    let classifier = ir::predict("classifier", classify).bind("ticket", ir::input("ticket"));
    let router = ir::route(ir::out("classifier", "severity"))
        .arm(
            "Low",
            ir::predict("low_reply", reply).bind("ticket", ir::input("ticket")),
        )
        .arm(
            "High",
            ir::predict("high_reply", reply).bind("ticket", ir::input("ticket")),
        )
        .named("router");
    let program = b
        .main(
            main_sig,
            ir::seq([classifier, router]).out("reply", ir::out("router", "reply")),
        )
        .unwrap();

    let (lm, _client) = canned_lm(vec![
        text(fields(&[("severity", "High")])),
        text(fields(&[("reply", "escalating now")])),
    ])
    .await;
    let interp = Interpreter::load(program, RuntimeEnv::new().bind_model("m", lm))
        .await
        .unwrap();

    let (result, trace) = capture(|| {
        interp.run(
            obj(&[("ticket", json!("prod down"))]),
            None,
            Budget::unlimited(),
        )
    })
    .await;
    assert_eq!(result.unwrap()["reply"], "escalating now");
    // Only the selected arm ran.
    assert!(trace.components.contains(&"high_reply".to_string()));
    assert!(!trace.components.contains(&"low_reply".to_string()));
}

// ---------------------------------------------------------------------------
// Retry
// ---------------------------------------------------------------------------

#[tokio::test]
async fn retry_intercepts_parse_failures_with_feedback() {
    let mut b = ProgramBuilder::new("retried");
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
            .input("question", T::String)
            .output("answer", T::String)
            .finish()
            .unwrap(),
    );
    let node = ir::retry(
        ir::predict("answerer", qa).bind("question", ir::input("question")),
        2,
    )
    .feedback(true);
    let program = b
        .main(
            main_sig,
            ir::seq([node.named("attempted")]).out("answer", ir::out("attempted", "answer")),
        )
        .unwrap();

    let (lm, client) = canned_lm(vec![
        text("no field markers here at all"),
        text(fields(&[("answer", "second try")])),
    ])
    .await;
    let interp = Interpreter::load(program, RuntimeEnv::new().bind_model("m", lm))
        .await
        .unwrap();

    let (result, trace) =
        capture(|| interp.run(obj(&[("question", json!("q"))]), None, Budget::unlimited())).await;
    assert_eq!(result.unwrap()["answer"], "second try");

    // Each attempt is a fresh span: (component, seq) disambiguates.
    let spans: Vec<_> = trace.for_component("answerer").collect();
    assert_eq!(spans.len(), 2);
    assert_eq!(spans[0].seq, 0);
    assert!(spans[0].error.is_some(), "first attempt failed to parse");
    assert_eq!(spans[1].seq, 1);
    assert_eq!(spans[1].output.as_ref().unwrap()["answer"], "second try");

    // The corrective user turn reached the second request.
    let last = format!("{:?}", client.last_request().unwrap());
    assert!(last.contains("could not be parsed"));
}

// ---------------------------------------------------------------------------
// Bounded Loop
// ---------------------------------------------------------------------------

#[tokio::test]
async fn bounded_loop_carries_values_and_stops_on_condition() {
    let mut b = ProgramBuilder::new("looped");
    b.model("m", config());
    let main_sig = b.sig(
        SignatureDef::build("Main")
            .input("question", T::String)
            .input("draft", T::String)
            .output("final", T::String)
            .finish()
            .unwrap(),
    );
    let improve = b.sig(
        SignatureDef::build("Improve")
            .input("question", T::String)
            .input("draft", T::String)
            .output("draft", T::String)
            .output("keep_going", T::Bool)
            .finish()
            .unwrap(),
    );
    let improver = ir::predict("improver", improve)
        .bind("question", ir::input("question"))
        .bind("draft", ir::carried("draft"));
    let looped = ir::loop_(
        ir::seq([improver]).out("draft", ir::out("improver", "draft")),
        5,
    )
    .while_(ir::out("improver", "keep_going"))
    .carry("draft", ir::out("improver", "draft"))
    .out("final", ir::out("improver", "draft"))
    .named("refinement");
    let program = b
        .main(
            main_sig,
            ir::seq([looped]).out("final", ir::out("refinement", "final")),
        )
        .unwrap();

    let (lm, client) = canned_lm(vec![
        text(fields(&[("draft", "v2"), ("keep_going", "true")])),
        text(fields(&[("draft", "v3"), ("keep_going", "false")])),
    ])
    .await;
    let interp = Interpreter::load(program, RuntimeEnv::new().bind_model("m", lm))
        .await
        .unwrap();

    let (result, trace) = capture(|| {
        interp.run(
            obj(&[("question", json!("q")), ("draft", json!("v1"))]),
            None,
            Budget::unlimited(),
        )
    })
    .await;
    assert_eq!(result.unwrap()["final"], "v3");

    // Two iterations: (component, seq) = (improver, 0), (improver, 1).
    let spans: Vec<_> = trace.for_component("improver").collect();
    assert_eq!(spans.len(), 2);
    assert_eq!(spans[0].input.as_ref().unwrap()["draft"], "v1");
    assert_eq!(spans[1].input.as_ref().unwrap()["draft"], "v2");

    // Iteration 2's prompt carried iteration 1's output.
    let last = format!("{:?}", client.last_request().unwrap());
    assert!(last.contains("v2"));
}

// ---------------------------------------------------------------------------
// Refine
// ---------------------------------------------------------------------------

#[tokio::test]
async fn refine_reruns_child_with_judge_feedback() {
    let mut b = ProgramBuilder::new("refined");
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
            .input("question", T::String)
            .input("hint", T::String)
            .output("answer", T::String)
            .finish()
            .unwrap(),
    );
    let judge_sig = b.sig(
        SignatureDef::build("Judge")
            .input("answer", T::String)
            .output("score", T::Float)
            .output("feedback", T::String)
            .finish()
            .unwrap(),
    );
    let child = ir::predict("writer", qa)
        .bind("question", ir::input("question"))
        .bind("hint", ir::lit(""));
    let judge = ir::predict("judge", judge_sig).bind("answer", ir::out("writer", "answer"));
    let refined = ir::refine(child, judge, "hint")
        .threshold(0.8)
        .max_rounds(2)
        .named("loop");
    let program = b
        .main(
            main_sig,
            ir::seq([refined]).out("answer", ir::out("loop", "answer")),
        )
        .unwrap();

    let (lm, client) = canned_lm(vec![
        text(fields(&[("answer", "first draft")])),
        text(fields(&[("score", "0.2"), ("feedback", "cite a source")])),
        text(fields(&[("answer", "sourced answer")])),
        text(fields(&[("score", "0.9"), ("feedback", "good")])),
    ])
    .await;
    let interp = Interpreter::load(program, RuntimeEnv::new().bind_model("m", lm))
        .await
        .unwrap();

    let (result, trace) =
        capture(|| interp.run(obj(&[("question", json!("q"))]), None, Budget::unlimited())).await;
    assert_eq!(result.unwrap()["answer"], "sourced answer");

    let writers: Vec<_> = trace.for_component("writer").collect();
    assert_eq!(writers.len(), 2);
    // Round 2's child input carried the judge's feedback in `hint`.
    assert_eq!(writers[1].input.as_ref().unwrap()["hint"], "cite a source");
    assert_eq!(trace.for_component("judge").count(), 2);
    let _ = client;
}

// ---------------------------------------------------------------------------
// AgentLoop
// ---------------------------------------------------------------------------

struct SearchTool;

#[derive(Debug)]
struct SearchToolError;
impl std::fmt::Display for SearchToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "search tool error")
    }
}
impl std::error::Error for SearchToolError {}

impl rig::tool::Tool for SearchTool {
    const NAME: &'static str = "search";
    type Error = SearchToolError;
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
        Ok(format!(
            "results for {}: dsrs is a rust dspy",
            args.get("query").and_then(|v| v.as_str()).unwrap_or("?")
        ))
    }
}

fn agent_program() -> Program {
    let mut b = ProgramBuilder::new("agentic");
    b.cap("net:search");
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
            .output("results", T::List(Box::new(T::String)))
            .finish()
            .unwrap(),
    );
    let search = b.host_tool(
        "search",
        "Web search; returns result snippets with URLs",
        search_sig,
        &["net:search"],
    );
    let researcher = ir::agent("researcher", research)
        .bind("question", ir::input("question"))
        .tools([search])
        .max_turns(4);
    b.main(
        main_sig,
        ir::seq([researcher]).out("answer", ir::out("researcher", "answer")),
    )
    .unwrap()
}

#[tokio::test]
async fn agent_loop_is_one_span_with_tool_events() {
    let (lm, client) = canned_lm(vec![
        tool_call("search", json!({"query": "dsrs"})),
        text(fields(&[("answer", "dsrs is a rust dspy")])),
    ])
    .await;
    let env = RuntimeEnv::new()
        .bind_model("m", lm)
        .bind_host_tool("search", Arc::new(SearchTool))
        .grant("net:search");
    let interp = Interpreter::load(agent_program(), env).await.unwrap();

    let (result, trace) = capture(|| {
        interp.run(
            obj(&[("question", json!("what is dsrs?"))]),
            None,
            Budget::unlimited(),
        )
    })
    .await;
    assert_eq!(result.unwrap()["answer"], "dsrs is a rust dspy");

    // One span, N events — the RFC 0001 attribution unit.
    assert_eq!(trace.components, vec!["researcher"]);
    assert_eq!(trace.spans.len(), 1);
    let span = &trace.spans[0];
    assert_eq!(
        span.output.as_ref().unwrap()["answer"],
        "dsrs is a rust dspy"
    );

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
            assert!(result.contains("dsrs is a rust dspy"));
            assert!(error.is_none());
        }
        other => panic!("expected ToolRun, got {other:?}"),
    }

    // The tool definition the model saw came from the IR: interface projected
    // from the declared signature, description from the ToolDesc param.
    let last = client.last_request().unwrap();
    assert_eq!(last.tools.len(), 1);
    assert_eq!(last.tools[0].name, "search");
    assert_eq!(
        last.tools[0].description,
        "Web search; returns result snippets with URLs"
    );
    assert_eq!(
        last.tools[0].parameters["properties"]["query"]["type"],
        "string"
    );
}

// ---------------------------------------------------------------------------
// Hole via sandbox
// ---------------------------------------------------------------------------

fn hole_program(js: &str) -> Program {
    let mut b = ProgramBuilder::new("holey");
    b.model("m", config());
    let main_sig = b.sig(
        SignatureDef::build("Main")
            .input("question", T::String)
            .output("shout", T::String)
            .finish()
            .unwrap(),
    );
    let shout_sig = b.sig(
        SignatureDef::build("Shout")
            .input("question", T::String)
            .output("shout", T::String)
            .finish()
            .unwrap(),
    );
    let shouter = ir::hole("shouter", shout_sig, js, &[]).bind("question", ir::input("question"));
    b.main(
        main_sig,
        ir::seq([shouter]).out("shout", ir::out("shouter", "shout")),
    )
    .unwrap()
}

#[tokio::test]
async fn hole_executes_in_the_sandbox() {
    let (lm, _client) = canned_lm(vec![]).await;
    let sandbox = Arc::new(dsrs_tools::QuickJsExecutor::new());
    let env = RuntimeEnv::new().bind_model("m", lm).with_sandbox(sandbox);
    let program = hole_program("(a) => ({shout: a.question.toUpperCase()})");
    let interp = Interpreter::load(program, env).await.unwrap();

    let (result, trace) = capture(|| {
        interp.run(
            obj(&[("question", json!("hello"))]),
            None,
            Budget::unlimited(),
        )
    })
    .await;
    assert_eq!(result.unwrap()["shout"], "HELLO");

    // One span whose model is the reserved sandbox config, one ToolRun event.
    assert_eq!(trace.components, vec!["shouter"]);
    let span = &trace.spans[0];
    assert_eq!(trace.model(span).model, "sandbox:quickjs");
    assert_eq!(span.events.len(), 1);
    assert!(matches!(
        &span.events[0],
        SpanEvent::ToolRun { error: None, .. }
    ));
    assert_eq!(span.output.as_ref().unwrap()["shout"], "HELLO");
}

#[tokio::test]
async fn hole_that_does_not_compile_fails_the_load() {
    let (lm, _client) = canned_lm(vec![]).await;
    let sandbox = Arc::new(dsrs_tools::QuickJsExecutor::new());
    let env = RuntimeEnv::new().bind_model("m", lm).with_sandbox(sandbox);
    let program = hole_program("this is not javascript ((");
    let err = Interpreter::load(program, env).await.unwrap_err();
    assert!(matches!(err, LoadError::Register { ref at, .. } if at == "shouter"));
}

#[tokio::test]
async fn hole_without_sandbox_fails_the_load() {
    let (lm, _client) = canned_lm(vec![]).await;
    let program = hole_program("(a) => ({shout: a.question})");
    let err = Interpreter::load(program, RuntimeEnv::new().bind_model("m", lm))
        .await
        .unwrap_err();
    assert!(matches!(err, LoadError::SandboxMissing));
}

// ---------------------------------------------------------------------------
// Overlay read-through in the interpreter
// ---------------------------------------------------------------------------

#[tokio::test]
async fn overlays_read_through_at_render_time_without_mutation() {
    let mut b = ProgramBuilder::new("overlaid");
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
            .instruction("Answer plainly.")
            .input("question", T::String)
            .output("answer", T::String)
            .finish()
            .unwrap(),
    );
    let node = ir::predict("answerer", qa).bind("question", ir::input("question"));
    let program = b
        .main(
            main_sig,
            ir::seq([node]).out("answer", ir::out("answerer", "answer")),
        )
        .unwrap();

    let (lm, client) = canned_lm(vec![
        text(fields(&[("answer", "a")])),
        text(fields(&[("answer", "b")])),
        text(fields(&[("answer", "c")])),
    ])
    .await;
    let interp = Interpreter::load(program, RuntimeEnv::new().bind_model("m", lm))
        .await
        .unwrap();
    let program = Arc::clone(interp.program());

    let slot = program
        .slot_of::<ir::Instruction>("answerer.instruction")
        .unwrap();
    let demos_slot = program.slot_of::<ir::Demos>("answerer.demos").unwrap();

    // Two overlays evaluated against ONE program instance.
    let mut overlay_a = Overlay::new(&program);
    overlay_a.set_instruction(slot, "OVERLAY A: be poetic.");
    let mut overlay_b = Overlay::new(&program);
    overlay_b.set_instruction(slot, "OVERLAY B: be blunt.");
    overlay_b.set_demos(
        demos_slot,
        vec![ir::DemoRow {
            input: obj(&[("question", json!("demo q"))]),
            output: obj(&[("answer", json!("demo a"))]),
        }],
    );

    interp
        .run(
            obj(&[("question", json!("q"))]),
            Some(Arc::new(overlay_a)),
            Budget::unlimited(),
        )
        .await
        .unwrap();
    let request_a = client.last_request().unwrap();
    assert!(
        request_a
            .preamble
            .as_ref()
            .unwrap()
            .contains("OVERLAY A: be poetic.")
    );
    assert!(
        !request_a
            .preamble
            .as_ref()
            .unwrap()
            .contains("Answer plainly.")
    );

    interp
        .run(
            obj(&[("question", json!("q"))]),
            Some(Arc::new(overlay_b)),
            Budget::unlimited(),
        )
        .await
        .unwrap();
    let request_b = client.last_request().unwrap();
    assert!(
        request_b
            .preamble
            .as_ref()
            .unwrap()
            .contains("OVERLAY B: be blunt.")
    );
    // The demo rows rendered into the prefix turns.
    assert!(format!("{:?}", request_b.chat_history).contains("demo q"));

    // Base run (no overlay): the program's incumbent values, untouched.
    interp
        .run(obj(&[("question", json!("q"))]), None, Budget::unlimited())
        .await
        .unwrap();
    let request_base = client.last_request().unwrap();
    assert!(
        request_base
            .preamble
            .as_ref()
            .unwrap()
            .contains("Answer plainly.")
    );
    match &program.params[slot.id].default {
        ParamValue::Instruction { text } => assert_eq!(text, "Answer plainly."),
        other => panic!("unexpected {other:?}"),
    }
}

#[tokio::test]
async fn stale_overlay_is_refused() {
    let (lm, _client) = canned_lm(vec![]).await;
    let interp = Interpreter::load(seq_program(), RuntimeEnv::new().bind_model("m", lm))
        .await
        .unwrap();
    let mut stale = Overlay::default();
    stale.base = 0xdead_beef;
    let err = interp
        .run(
            obj(&[("question", json!("q"))]),
            Some(Arc::new(stale)),
            Budget::unlimited(),
        )
        .await
        .unwrap_err();
    assert!(matches!(err, RunError::Overlay { .. }));
}

// ---------------------------------------------------------------------------
// Budgets and load-time refusals
// ---------------------------------------------------------------------------

#[tokio::test]
async fn budget_gates_lm_calls_pre_call() {
    let (lm, _client) = canned_lm(vec![text(fields(&[("answer", "42")]))]).await;
    let interp = Interpreter::load(seq_program(), RuntimeEnv::new().bind_model("m", lm))
        .await
        .unwrap();
    let err = interp
        .run(
            obj(&[("question", json!("q"))]),
            None,
            Budget {
                max_lm_calls: Some(1),
                ..Budget::unlimited()
            },
        )
        .await
        .unwrap_err();
    // The first call fits; the second leaf is refused before calling.
    assert!(matches!(err, RunError::Budget { ref at } if &**at == "checker"));
}

#[tokio::test]
async fn caps_exceeding_grants_refuse_the_load() {
    let (lm, _client) = canned_lm(vec![]).await;
    let env = RuntimeEnv::new()
        .bind_model("m", lm)
        .bind_host_tool("search", Arc::new(SearchTool));
    // agent_program declares net:search; the env grants nothing.
    let err = Interpreter::load(agent_program(), env).await.unwrap_err();
    assert!(matches!(
        err,
        LoadError::CapsExceedGrants { ref missing } if missing == &["net:search".to_string()]
    ));
}

#[tokio::test]
async fn unbound_host_tool_refuses_the_load() {
    let (lm, _client) = canned_lm(vec![]).await;
    let env = RuntimeEnv::new().bind_model("m", lm).grant("net:search");
    let err = Interpreter::load(agent_program(), env).await.unwrap_err();
    assert!(matches!(err, LoadError::HostToolUnbound { ref name } if name == "search"));
}

// ---------------------------------------------------------------------------
// Overlay model swap
// ---------------------------------------------------------------------------

#[tokio::test]
async fn overlay_can_swap_the_model_ref() {
    let mut b = ProgramBuilder::new("modelswap");
    let m1 = b.model("m1", config());
    let m2 = b.model("m2", config());
    let main_sig = b.sig(
        SignatureDef::build("Main")
            .input("question", T::String)
            .output("answer", T::String)
            .finish()
            .unwrap(),
    );
    let node = ir::predict("answerer", main_sig)
        .model(m1)
        .bind("question", ir::input("question"));
    let program = b
        .main(
            main_sig,
            ir::seq([node]).out("answer", ir::out("answerer", "answer")),
        )
        .unwrap();
    let _ = m2;

    let (lm1, client1) = canned_lm(vec![text(fields(&[("answer", "from m1")]))]).await;
    let (lm2, client2) = canned_lm(vec![text(fields(&[("answer", "from m2")]))]).await;
    let interp = Interpreter::load(
        program,
        RuntimeEnv::new()
            .bind_model("m1", lm1)
            .bind_model("m2", lm2),
    )
    .await
    .unwrap();
    let program = Arc::clone(interp.program());

    let out = interp
        .run(obj(&[("question", json!("q"))]), None, Budget::unlimited())
        .await
        .unwrap();
    assert_eq!(out["answer"], "from m1");
    assert!(client1.last_request().is_some());

    let mut overlay = Overlay::new(&program);
    let id = program.param_id("answerer.model").unwrap();
    overlay
        .set(&program, id, ParamValue::ModelRef { model: m2 })
        .unwrap();
    let out = interp
        .run(
            obj(&[("question", json!("q"))]),
            Some(Arc::new(overlay)),
            Budget::unlimited(),
        )
        .await
        .unwrap();
    assert_eq!(out["answer"], "from m2");
    assert!(client2.last_request().is_some());
}

// ---------------------------------------------------------------------------
// The RFC §4.3 worked example, end to end
// ---------------------------------------------------------------------------

/// CoT draft → tool-using agent loop → typed hole, all three leaf kinds in
/// one program, run through the interpreter.
#[tokio::test]
async fn worked_example_runs_end_to_end() {
    let mut b = ProgramBuilder::new("qa");
    b.cap("net:search");
    let fast = b.model("fast", config());
    let deep = b.model("deep", config());

    let main_sig = b.sig(
        SignatureDef::build("Main")
            .input("question", T::String)
            .output("answer", T::String)
            .output("sources", T::List(Box::new(T::String)))
            .finish()
            .unwrap(),
    );
    let draft = b.sig(
        SignatureDef::build("Draft")
            .instruction("Draft a thorough, factual answer.")
            .input("question", T::String)
            .output("answer", T::String)
            .finish()
            .unwrap(),
    );
    let research = b.sig(
        SignatureDef::build("Research")
            .instruction("Verify the draft against sources; collect URLs.")
            .input("question", T::String)
            .input("draft", T::String)
            .output("evidence", T::List(Box::new(T::String)))
            .finish()
            .unwrap(),
    );
    let cite = b.sig(
        SignatureDef::build("CiteCheck")
            .input("draft", T::String)
            .input("evidence", T::List(Box::new(T::String)))
            .output("answer", T::String)
            .output("sources", T::List(Box::new(T::String)))
            .finish()
            .unwrap(),
    );
    let search_sig = b.sig(
        SignatureDef::build("Search")
            .input("query", T::String)
            .output("results", T::List(Box::new(T::String)))
            .finish()
            .unwrap(),
    );
    let search = b.host_tool(
        "search",
        "Web search; returns result snippets with URLs",
        search_sig,
        &["net:search"],
    );

    let drafter = ir::cot("drafter", draft)
        .model(deep)
        .bind("question", ir::input("question"));
    let researcher = ir::agent("researcher", research)
        .model(fast)
        .bind("question", ir::input("question"))
        .bind("draft", ir::out("drafter", "answer"))
        .tools([search])
        .max_turns(6);
    let checker = ir::hole(
        "checker",
        cite,
        r#"(a) => ({
            answer: a.draft,
            sources: a.evidence.filter(e => e.startsWith("http")),
        })"#,
        &[],
    )
    .bind("draft", ir::out("drafter", "answer"))
    .bind("evidence", ir::out("researcher", "evidence"));

    let program = b
        .main(
            main_sig,
            ir::seq([drafter, researcher, checker])
                .out("answer", ir::out("checker", "answer"))
                .out("sources", ir::out("checker", "sources")),
        )
        .unwrap();

    // deep drives the CoT drafter; fast drives the agent loop.
    let (deep_lm, _) = canned_lm(vec![text(fields(&[
        ("reasoning", "The capital question is factual."),
        ("answer", "Paris is the capital of France."),
    ]))])
    .await;
    let (fast_lm, _) = canned_lm(vec![
        tool_call("search", json!({"query": "capital of France"})),
        text(fields(&[(
            "evidence",
            r#"["http://wiki/paris", "unsourced claim"]"#,
        )])),
    ])
    .await;

    let env = RuntimeEnv::new()
        .bind_model("deep", deep_lm)
        .bind_model("fast", fast_lm)
        .bind_host_tool("search", Arc::new(SearchTool))
        .with_sandbox(Arc::new(dsrs_tools::QuickJsExecutor::new()))
        .grant("net:search");
    let interp = Interpreter::load(program, env).await.unwrap();

    let (result, trace) = capture(|| {
        interp.run(
            obj(&[("question", json!("capital of France?"))]),
            None,
            Budget::unlimited(),
        )
    })
    .await;
    let output = result.unwrap();
    assert_eq!(output["answer"], "Paris is the capital of France.");
    assert_eq!(output["sources"], json!(["http://wiki/paris"]));

    // Three leaves, three spans, in dataflow order.
    assert_eq!(trace.components, vec!["drafter", "researcher", "checker"]);
    let drafter_span = trace.for_component("drafter").next().unwrap();
    assert_eq!(
        drafter_span.output.as_ref().unwrap()["reasoning"],
        "The capital question is factual."
    );
    let researcher_span = trace.for_component("researcher").next().unwrap();
    assert!(
        researcher_span
            .events
            .iter()
            .any(|event| matches!(event, SpanEvent::ToolRun { name, .. } if name == "search"))
    );
    // The agent saw the drafter's answer as its bound input.
    assert_eq!(
        researcher_span.input.as_ref().unwrap()["draft"],
        "Paris is the capital of France."
    );
    let checker_span = trace.for_component("checker").next().unwrap();
    assert_eq!(trace.model(checker_span).model, "sandbox:quickjs");
}
