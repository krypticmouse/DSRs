//! RFC 0001 §1's reserved `param_ids` column + RFC 0002 §3.3's
//! `Trace::attach_program`: joining span components to a program's global
//! `ParamId`s — one addressing story across traces, overlays, and slots.

use std::collections::BTreeSet;
use std::sync::Arc;

use dspy_rs::ir::{
    self, Budget, FieldType as T, Interpreter, Program, ProgramBuilder, RuntimeEnv, SignatureDef,
};
use dspy_rs::trace::capture;
use dspy_rs::{LM, LMClient, LMConfig, TestCompletionModel, Trace};
use rig::completion::{AssistantContent, ToolDefinition};
use rig::message::{Text, ToolCall, ToolFunction};
use serde_json::json;

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
            description: "ignored".to_string(),
            parameters: json!({"type": "object", "additionalProperties": true}),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok("http://wiki/source".to_string())
    }
}

/// All three leaf kinds — predict, agent loop, hole — so the join covers
/// every slot layout.
fn three_leaf_program() -> Program {
    let mut b = ProgramBuilder::new("qa");
    b.cap("net:search");
    b.model("m", config());
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
            .input("question", T::String)
            .output("answer", T::String)
            .finish()
            .unwrap(),
    );
    let research = b.sig(
        SignatureDef::build("Research")
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
    let search = b.host_tool("search", "Web search", search_sig, &["net:search"]);

    let drafter = ir::predict("drafter", draft).bind("question", ir::input("question"));
    let researcher = ir::agent("researcher", research)
        .bind("question", ir::input("question"))
        .bind("draft", ir::out("drafter", "answer"))
        .tools([search])
        .max_turns(4);
    let checker = ir::hole(
        "checker",
        cite,
        r#"(a) => ({answer: a.draft, sources: a.evidence})"#,
        &[],
    )
    .bind("draft", ir::out("drafter", "answer"))
    .bind("evidence", ir::out("researcher", "evidence"));

    b.main(
        main_sig,
        ir::seq([drafter, researcher, checker])
            .out("answer", ir::out("checker", "answer"))
            .out("sources", ir::out("checker", "sources")),
    )
    .unwrap()
}

async fn captured_run() -> (Arc<Program>, Trace) {
    let (lm, _) = canned_lm(vec![
        text(fields(&[("answer", "Paris.")])),
        tool_call("search", json!({"query": "capital of France"})),
        text(fields(&[("evidence", r#"["http://wiki/paris"]"#)])),
    ])
    .await;
    let env = RuntimeEnv::new()
        .bind_model("m", lm)
        .bind_host_tool("search", Arc::new(SearchTool))
        .with_sandbox(Arc::new(dsrs_tools::QuickJsExecutor::new()))
        .grant("net:search");
    let interp = Interpreter::load(three_leaf_program(), env).await.unwrap();
    let program = Arc::clone(interp.program());

    let (result, trace) = capture(|| {
        interp.run(
            obj(&[("question", json!("capital of France?"))]),
            None,
            Budget::unlimited(),
        )
    })
    .await;
    result.unwrap();
    (program, trace)
}

fn paths_of(program: &Program, ids: &[ir::ParamId]) -> BTreeSet<String> {
    ids.iter()
        .map(|&id| program.param_path(id).to_string())
        .collect()
}

// ---------------------------------------------------------------------------
// The join
// ---------------------------------------------------------------------------

#[tokio::test]
async fn attach_program_resolves_the_join_for_every_leaf_span() {
    let (program, mut trace) = captured_run().await;
    assert!(trace.param_ids.is_empty(), "column starts empty");

    trace.attach_program(&program);

    // Parallel column, one entry per interned component.
    assert_eq!(trace.param_ids.len(), trace.components.len());

    // Every leaf span joins: its component's entry is present, and every
    // joined id's canonical path is prefixed by the component name — the
    // "one addressing story" invariant.
    assert!(!trace.spans.is_empty());
    for span in &trace.spans {
        let component = trace.component_name(span.component).to_string();
        let ids = trace.param_ids[span.component.0 as usize]
            .as_ref()
            .unwrap_or_else(|| panic!("component `{component}` did not join"));
        assert!(!ids.is_empty());
        for &id in ids {
            assert!(
                program.param_path(id).starts_with(&format!("{component}.")),
                "{} does not belong to `{component}`",
                program.param_path(id)
            );
            // The joined ids are live: they resolve back through the program.
            assert_eq!(program.param_id(program.param_path(id)), Some(id));
        }
    }

    // Exact slot layouts per leaf kind.
    let entry = |name: &str| {
        let id = trace.component_id(name).unwrap();
        paths_of(&program, trace.param_ids[id.0 as usize].as_ref().unwrap())
    };
    assert_eq!(
        entry("drafter"),
        [
            "drafter.instruction",
            "drafter.demos",
            "drafter.model",
            "drafter.render",
        ]
        .into_iter()
        .map(String::from)
        .collect()
    );
    assert_eq!(
        entry("researcher"),
        [
            "researcher.instruction",
            "researcher.demos",
            "researcher.model",
            "researcher.context",
            "researcher.tool_set",
        ]
        .into_iter()
        .map(String::from)
        .collect(),
        "tool-owned slots (tool.search.desc) must not leak into a leaf's entry"
    );
    assert_eq!(
        entry("checker"),
        ["checker.code"].into_iter().map(String::from).collect()
    );
}

#[tokio::test]
async fn components_unknown_to_the_program_stay_none() {
    let (program, mut trace) = captured_run().await;
    // A static-lane component name the program has no leaf for.
    trace.components.push("pipeline.judge".to_string());
    trace.attach_program(&program);
    assert_eq!(trace.param_ids.len(), trace.components.len());
    assert!(trace.param_ids.last().unwrap().is_none());
}

// ---------------------------------------------------------------------------
// Wire format: additive, no version bump
// ---------------------------------------------------------------------------

#[tokio::test]
async fn param_ids_survive_the_jsonl_round_trip_and_stay_additive() {
    let (program, mut trace) = captured_run().await;

    // Unattached traces serialize without the column at all.
    let plain = trace.to_jsonl().unwrap();
    assert!(!plain.lines().next().unwrap().contains("param_ids"));
    assert!(Trace::from_jsonl(&plain).unwrap().param_ids.is_empty());

    trace.attach_program(&program);
    let jsonl = trace.to_jsonl().unwrap();
    assert!(jsonl.lines().next().unwrap().contains("param_ids"));
    let restored = Trace::from_jsonl(&jsonl).unwrap();
    assert_eq!(restored.meta.v, trace.meta.v, "no format version bump");
    assert_eq!(restored.param_ids, trace.param_ids);
}
