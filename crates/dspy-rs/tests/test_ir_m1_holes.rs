//! RFC 0003 stage M-1 (`hole-integrity`): the `HoleImpl` split (sandboxed vs
//! extern/host holes), the non-degenerate hole `request_hash` preimage, and
//! interpreter-lane replay — predicts, agent loops, and holes served from a
//! recorded trace with divergence detection on changed hole implementations.

use std::sync::Arc;

use dspy_rs::ir::{
    self, Budget, CodeK, Interpreter, LoadError, Overlay, Program, ProgramBuilder, RunError,
    RuntimeEnv, SignatureDef,
};
use dspy_rs::ir::FieldType as T;
use dspy_rs::trace::JsonMap;
use dspy_rs::{LM, LMClient, LMConfig, ReplayMode, TestCompletionModel, capture, replay};
use rig::completion::AssistantContent;
use rig::message::Text;
use serde_json::json;

// ---------------------------------------------------------------------------
// Fixtures (mirrors test_ir_interp.rs)
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

fn obj(pairs: &[(&str, serde_json::Value)]) -> JsonMap {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.clone()))
        .collect()
}

/// question → shouter (extern hole) → shout.
fn extern_program() -> Program {
    let mut b = ProgramBuilder::new("externed");
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
    let shouter = ir::extern_hole("shouter", shout_sig, 0xdead_beef, &[])
        .bind("question", ir::input("question"));
    b.main(
        main_sig,
        ir::seq([shouter]).out("shout", ir::out("shouter", "shout")),
    )
    .unwrap()
}

/// question → drafter (Predict) → shouter (sandboxed JS hole) → shout.
fn predict_then_hole_program(js: &str) -> Program {
    let mut b = ProgramBuilder::new("mixed");
    b.model("m", config());
    let main_sig = b.sig(
        SignatureDef::build("Main")
            .input("question", T::String)
            .output("shout", T::String)
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
    let shout_sig = b.sig(
        SignatureDef::build("Shout")
            .input("answer", T::String)
            .output("shout", T::String)
            .finish()
            .unwrap(),
    );
    let drafter = ir::predict("drafter", qa).bind("question", ir::input("question"));
    let shouter = ir::hole("shouter", shout_sig, js, &[]).bind("answer", ir::out("drafter", "answer"));
    b.main(
        main_sig,
        ir::seq([drafter, shouter]).out("shout", ir::out("shouter", "shout")),
    )
    .unwrap()
}

// ---------------------------------------------------------------------------
// Extern holes: text round-trip, binding, refusal
// ---------------------------------------------------------------------------

#[test]
fn extern_hole_round_trips_through_text() {
    let program = extern_program();
    let printed = program.to_dsrs();
    assert!(
        printed.contains("extern \"00000000deadbeef\""),
        "canonical text should carry the extern hash:\n{printed}"
    );

    let reparsed = Program::from_dsrs(&printed).unwrap();
    assert_eq!(reparsed.meta.program_hash, program.meta.program_hash);
    assert_eq!(reparsed.to_dsrs(), printed);
}

#[test]
fn extern_hash_must_be_hex() {
    let program = extern_program();
    let printed = program.to_dsrs();
    let mangled = printed.replace("extern \"00000000deadbeef\"", "extern \"not-a-hash\"");
    let err = Program::from_dsrs(&mangled).unwrap_err();
    assert!(
        err.to_string().contains("extern"),
        "expected an extern-hash parse error, got: {err}"
    );
}

#[tokio::test]
async fn host_hole_binds_and_executes() {
    let (lm, _client) = canned_lm(vec![]).await;
    let env = RuntimeEnv::new()
        .bind_model("m", lm)
        .bind_host_hole("shouter", |input: JsonMap| async move {
            let q = input
                .get("question")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_uppercase();
            Ok(json!({ "shout": q }))
        });
    // No sandbox in the env: extern holes must not require one.
    let interp = Interpreter::load(extern_program(), env).await.unwrap();

    let (result, trace) = capture(|| {
        interp.run(
            obj(&[("question", json!("hello"))]),
            None,
            Budget::unlimited(),
        )
    })
    .await;
    assert_eq!(result.unwrap()["shout"], "HELLO");

    assert_eq!(trace.components, vec!["shouter"]);
    let span = &trace.spans[0];
    assert_eq!(trace.model(span).model, "host:extern");
    assert_ne!(span.request_hash, 0, "hole spans must carry a real preimage");
    assert_eq!(span.output.as_ref().unwrap()["shout"], "HELLO");
}

#[tokio::test]
async fn unbound_host_hole_fails_the_load() {
    let (lm, _client) = canned_lm(vec![]).await;
    let err = Interpreter::load(extern_program(), RuntimeEnv::new().bind_model("m", lm))
        .await
        .unwrap_err();
    assert!(matches!(err, LoadError::HostHoleUnbound { ref name } if name == "shouter"));
}

// ---------------------------------------------------------------------------
// Hole request_hash: no longer degenerate
// ---------------------------------------------------------------------------

#[tokio::test]
async fn distinct_holes_record_distinct_request_hashes() {
    // Two sandboxed holes with different code and different inputs: before
    // M-1 every hole span hashed identically (empty prompt + constant
    // sandbox config); now the preimage is impl ++ input ++ caps.
    let mut b = ProgramBuilder::new("two_holes");
    b.model("m", config());
    let main_sig = b.sig(
        SignatureDef::build("Main")
            .input("question", T::String)
            .output("a", T::String)
            .output("b", T::String)
            .finish()
            .unwrap(),
    );
    let echo = b.sig(
        SignatureDef::build("EchoA")
            .input("question", T::String)
            .output("a", T::String)
            .finish()
            .unwrap(),
    );
    let echo_b = b.sig(
        SignatureDef::build("EchoB")
            .input("question", T::String)
            .output("b", T::String)
            .finish()
            .unwrap(),
    );
    let first = ir::hole("first", echo, "(x) => ({a: x.question})", &[])
        .bind("question", ir::input("question"));
    let second = ir::hole("second", echo_b, "(x) => ({b: x.question + '!'})", &[])
        .bind("question", ir::input("question"));
    let program = b
        .main(
            main_sig,
            ir::seq([first, second])
                .out("a", ir::out("first", "a"))
                .out("b", ir::out("second", "b")),
        )
        .unwrap();

    let (lm, _client) = canned_lm(vec![]).await;
    let env = RuntimeEnv::new()
        .bind_model("m", lm)
        .with_sandbox(Arc::new(dsrs_tools::QuickJsExecutor::new()));
    let interp = Interpreter::load(program, env).await.unwrap();

    let (result, trace) = capture(|| {
        interp.run(obj(&[("question", json!("hi"))]), None, Budget::unlimited())
    })
    .await;
    result.unwrap();

    assert_eq!(trace.spans.len(), 2);
    assert_ne!(trace.spans[0].request_hash, 0);
    assert_ne!(trace.spans[1].request_hash, 0);
    assert_ne!(
        trace.spans[0].request_hash, trace.spans[1].request_hash,
        "different implementations must hash differently"
    );
}

// ---------------------------------------------------------------------------
// Interpreter-lane replay
// ---------------------------------------------------------------------------

#[tokio::test]
async fn strict_replay_serves_predicts_and_holes_with_zero_live_calls() {
    let js = "(x) => ({shout: x.answer.toUpperCase()})";

    // Live run, recorded.
    let (lm, _client) = canned_lm(vec![text(fields(&[("answer", "hello")]))]).await;
    let env = RuntimeEnv::new()
        .bind_model("m", lm)
        .with_sandbox(Arc::new(dsrs_tools::QuickJsExecutor::new()));
    let interp = Interpreter::load(predict_then_hole_program(js), env)
        .await
        .unwrap();
    let (result, trace) = capture(|| {
        interp.run(
            obj(&[("question", json!("say hello"))]),
            None,
            Budget::unlimited(),
        )
    })
    .await;
    let baseline = result.unwrap();
    assert_eq!(baseline["shout"], "HELLO");

    // Replay against a fresh interpreter with an EMPTY canned queue and a
    // zero-call budget: any live LM call would fail, any budget reservation
    // would fail — proving served calls construct nothing and spend nothing.
    let (lm2, _client2) = canned_lm(vec![]).await;
    let env2 = RuntimeEnv::new()
        .bind_model("m", lm2)
        .with_sandbox(Arc::new(dsrs_tools::QuickJsExecutor::new()));
    let interp2 = Interpreter::load(predict_then_hole_program(js), env2)
        .await
        .unwrap();
    let zero_budget = Budget {
        max_lm_calls: Some(0),
        ..Budget::default()
    };
    let (replayed, report) = replay(&trace, ReplayMode::Strict, || {
        interp2.run(obj(&[("question", json!("say hello"))]), None, zero_budget)
    })
    .await;
    assert_eq!(replayed.unwrap(), baseline);
    assert_eq!(report.served, 2, "predict + hole both served");
    assert_eq!(report.live, 0);
}

#[tokio::test]
async fn changed_hole_code_diverges_and_runs_live() {
    let js = "(x) => ({shout: x.answer.toUpperCase()})";

    let (lm, _client) = canned_lm(vec![text(fields(&[("answer", "hello")]))]).await;
    let env = RuntimeEnv::new()
        .bind_model("m", lm)
        .with_sandbox(Arc::new(dsrs_tools::QuickJsExecutor::new()));
    let interp = Interpreter::load(predict_then_hole_program(js), env)
        .await
        .unwrap();
    let (result, trace) = capture(|| {
        interp.run(
            obj(&[("question", json!("say hello"))]),
            None,
            Budget::unlimited(),
        )
    })
    .await;
    result.unwrap();

    // Counterfactual: mutate the hole's Code gene through an overlay. The
    // predict prefix replays free; the hole's preimage changes, so it (and
    // only it) runs live under UntilDivergence.
    let (lm2, _client2) = canned_lm(vec![]).await;
    let env2 = RuntimeEnv::new()
        .bind_model("m", lm2)
        .with_sandbox(Arc::new(dsrs_tools::QuickJsExecutor::new()));
    let interp2 = Interpreter::load(predict_then_hole_program(js), env2)
        .await
        .unwrap();
    let program = Arc::clone(interp2.program());
    let mut overlay = Overlay::new(&program);
    let slot = program
        .slot_of::<CodeK>("shouter.code")
        .expect("shouter.code is a Code slot");
    overlay.set_code(slot, "(x) => ({shout: x.answer + '?!'})".to_string());

    let (replayed, report) = replay(&trace, ReplayMode::UntilDivergence, || {
        interp2.run(
            obj(&[("question", json!("say hello"))]),
            Some(Arc::new(overlay)),
            Budget::unlimited(),
        )
    })
    .await;
    let out = replayed.unwrap();
    assert_eq!(out["shout"], "hello?!", "mutated code ran live");
    assert_eq!(report.served, 1, "the predict prefix was served");
    assert_eq!(report.live, 1, "only the mutated hole went live");
    assert!(report.divergence.is_some());
}

#[tokio::test]
async fn strict_replay_refuses_changed_host_hole_hash() {
    // Record with one extern implementation hash…
    let (lm, _client) = canned_lm(vec![]).await;
    let env = RuntimeEnv::new()
        .bind_model("m", lm)
        .bind_host_hole("shouter", |input: JsonMap| async move {
            let q = input
                .get("question")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_uppercase();
            Ok(json!({ "shout": q }))
        });
    let interp = Interpreter::load(extern_program(), env).await.unwrap();
    let (result, trace) = capture(|| {
        interp.run(
            obj(&[("question", json!("hello"))]),
            None,
            Budget::unlimited(),
        )
    })
    .await;
    result.unwrap();

    // …then replay a program whose extern hash differs: the implementation
    // changed underneath the recording, and strict mode must refuse.
    let mut b = ProgramBuilder::new("externed");
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
    let shouter = ir::extern_hole("shouter", shout_sig, 0xfeed_f00d, &[])
        .bind("question", ir::input("question"));
    let changed = b
        .main(
            main_sig,
            ir::seq([shouter]).out("shout", ir::out("shouter", "shout")),
        )
        .unwrap();

    let (lm2, _client2) = canned_lm(vec![]).await;
    let env2 = RuntimeEnv::new()
        .bind_model("m", lm2)
        .bind_host_hole("shouter", |input: JsonMap| async move {
            Ok(json!({ "shout": input.get("question").cloned().unwrap_or_default() }))
        });
    let interp2 = Interpreter::load(changed, env2).await.unwrap();

    let (replayed, report) = replay(&trace, ReplayMode::Strict, || {
        interp2.run(
            obj(&[("question", json!("hello"))]),
            None,
            Budget::unlimited(),
        )
    })
    .await;
    let err = replayed.unwrap_err();
    assert!(
        matches!(err, RunError::Replay { ref at, .. } if &**at == "shouter"),
        "expected a replay refusal at the hole, got: {err}"
    );
    assert!(report.divergence.is_some());
    assert_eq!(report.served, 0);
}
