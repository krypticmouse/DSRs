//! IR-7 CLI coverage (RFC 0002 §6.2): `check`/`fmt` through the exact
//! library functions the binary calls, and the serving host end-to-end on an
//! ephemeral port with canned LM responses (`TestCompletionModel` pre-bound
//! into `RuntimeEnv` — the same injection point a production host uses for
//! real models).

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use dspy_rs::ir::RuntimeEnv;
use dspy_rs::{LM, LMClient, LMConfig, TestCompletionModel};
use dsrs_cli::check::check_file;
use dsrs_cli::fmt::{FmtOutcome, fmt_file};
use dsrs_cli::serve::{self, ServeConfig};
use rig::completion::AssistantContent;
use rig::message::Text;
use serde_json::{Value, json};

fn fixture(name: &str) -> PathBuf {
    Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../dspy-rs/tests/fixtures"
    ))
    .join(name)
}

// ---------------------------------------------------------------------------
// check
// ---------------------------------------------------------------------------

#[test]
fn check_reports_a_valid_program() {
    let report = check_file(fixture("qa.dsrs")).expect("golden fixture validates");
    assert_eq!(report.name, "qa");
    assert!(report.program_hash != 0);
    assert!(report.nodes >= 3, "qa has drafter/researcher/checker + seq");
    assert_eq!(report.caps, vec!["net:search".to_string()]);
    let line = report.to_string();
    assert!(line.starts_with("ok: program `qa` ("), "{line}");
    assert!(line.contains("caps { net:search }"), "{line}");
}

#[test]
fn check_rejects_with_path_and_position() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("broken.dsrs");
    std::fs::write(&path, "dsrs 1\nprogram broken\nwidget Oops { }\n").expect("write");
    let err = check_file(&path).expect_err("syntax error");
    let message = err.to_string();
    assert!(message.contains("broken.dsrs"), "{message}");
    assert!(message.contains("line 3, column 1"), "{message}");
    assert!(message.contains("unknown top-level keyword"), "{message}");
}

// ---------------------------------------------------------------------------
// fmt
// ---------------------------------------------------------------------------

#[test]
fn fmt_canonicalizes_and_is_idempotent() {
    let golden = std::fs::read_to_string(fixture("qa.dsrs")).expect("golden");
    let scrambled = std::fs::read_to_string(fixture("qa_scrambled.dsrs")).expect("scrambled");
    assert_ne!(golden, scrambled);

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("qa.dsrs");
    std::fs::write(&path, &scrambled).expect("write");

    // Print mode leaves the file alone.
    let FmtOutcome::Canonical(text) = fmt_file(&path, false).expect("formats") else {
        panic!("expected canonical text");
    };
    assert_eq!(text, golden, "canonical form is the golden fixture");
    assert_eq!(
        std::fs::read_to_string(&path).expect("read"),
        scrambled,
        "print mode must not write"
    );

    // Write mode rewrites, then reports Unchanged (idempotence).
    assert!(matches!(
        fmt_file(&path, true).expect("formats"),
        FmtOutcome::Rewrote
    ));
    assert_eq!(std::fs::read_to_string(&path).expect("read"), golden);
    assert!(matches!(
        fmt_file(&path, true).expect("formats"),
        FmtOutcome::Unchanged
    ));
}

#[test]
fn fmt_fails_on_invalid_input() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("nope.dsrs");
    std::fs::write(&path, "not a program").expect("write");
    let err = fmt_file(&path, true).expect_err("parse failure");
    assert!(err.to_string().contains("nope.dsrs"), "{err:#}");
}

// ---------------------------------------------------------------------------
// serve
// ---------------------------------------------------------------------------

const ECHO: &str = r#"dsrs 1
program echo

model m = "openai:gpt-4o-mini"

sig Main {
  in question: string
  out answer: string
}

main: Main = seq {
  answerer = predict Main (question = $.question)
  out { answer = answerer.answer }
}
"#;

fn canned_text(fields: &[(&str, &str)]) -> AssistantContent {
    let mut out = String::new();
    for (name, value) in fields {
        out.push_str(&format!("[[ ## {name} ## ]]\n{value}\n\n"));
    }
    out.push_str("[[ ## completed ## ]]\n");
    AssistantContent::Text(Text { text: out })
}

/// A live LM whose provider client is replaced by a canned test model — the
/// serving host binds it by model name through `RuntimeEnv`, exactly as a
/// production host would pre-bind a shared client.
async fn canned_lm(responses: Vec<AssistantContent>) -> (Arc<LM>, TestCompletionModel) {
    let client = TestCompletionModel::new(responses);
    let lm = LM::from_config(LMConfig {
        model: "openai:gpt-4o-mini".to_string(),
        api_key: Some("canned-test-key".to_string()),
        ..LMConfig::default()
    })
    .await
    .expect("LM from config")
    .with_client(LMClient::Test(client.clone()))
    .await
    .expect("swap in test client");
    (Arc::new(lm), client)
}

fn write_program(dir: &tempfile::TempDir, name: &str, text: &str) -> PathBuf {
    let path = dir.path().join(name);
    std::fs::write(&path, text).expect("write program");
    path
}

async fn spawn_server(app: Arc<serve::App>) -> SocketAddr {
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local addr");
    tokio::spawn(async move {
        axum::serve(listener, serve::router(app)).await.expect("serve");
    });
    addr
}

#[tokio::test]
async fn serve_end_to_end_on_an_ephemeral_port() {
    let dir = tempfile::tempdir().expect("tempdir");
    let program = write_program(&dir, "echo.dsrs", ECHO);
    let (lm, client) = canned_lm(vec![
        canned_text(&[("answer", "42")]),
        canned_text(&[("answer", "still 42")]),
    ])
    .await;

    let config = ServeConfig {
        program,
        overlay: None,
        allow: vec![],
    };
    let app = serve::load(&config, RuntimeEnv::new().bind_model("m", lm))
        .await
        .expect("load");
    let addr = spawn_server(app).await;
    let http = reqwest::Client::new();
    let base = format!("http://{addr}");

    // GET /healthz
    let health: Value = http
        .get(format!("{base}/healthz"))
        .send()
        .await
        .expect("healthz")
        .json()
        .await
        .expect("json");
    assert_eq!(health["status"], "ok");
    assert_eq!(health["program"], "echo");
    assert_eq!(health["program_hash"].as_str().map(str::len), Some(16));

    // GET /schema — the main SignatureDef + TypeTable serde forms.
    let schema: Value = http
        .get(format!("{base}/schema"))
        .send()
        .await
        .expect("schema")
        .json()
        .await
        .expect("json");
    assert_eq!(schema["program"], "echo");
    assert_eq!(schema["signature"]["name"], "Main");
    assert_eq!(schema["signature"]["inputs"][0]["name"], "question");
    assert_eq!(schema["signature"]["outputs"][0]["name"], "answer");

    // GET /program — canonical .dsrs text.
    let text = http
        .get(format!("{base}/program"))
        .send()
        .await
        .expect("program")
        .text()
        .await
        .expect("text");
    assert!(text.starts_with("dsrs 1"), "{text}");
    assert!(text.contains("program echo"), "{text}");

    // POST /run — output map.
    let run: Value = http
        .post(format!("{base}/run"))
        .json(&json!({"question": "what is the answer?"}))
        .send()
        .await
        .expect("run")
        .json()
        .await
        .expect("json");
    assert_eq!(run["output"]["answer"], "42");

    // POST /run?trace=1 — output plus the trace JSONL artifact.
    let traced_response = http
        .post(format!("{base}/run?trace=1"))
        .json(&json!({"question": "again?"}))
        .send()
        .await
        .expect("run traced");
    assert_eq!(traced_response.status(), 200);
    let traced: Value = traced_response.json().await.expect("json");
    assert_eq!(traced["output"]["answer"], "still 42");
    let jsonl = traced["trace_jsonl"].as_str().expect("trace_jsonl string");
    assert!(jsonl.lines().count() >= 2, "header + span:\n{jsonl}");
    assert!(jsonl.contains("answerer"), "leaf name in trace:\n{jsonl}");

    // Two LM calls total, both served from the canned client.
    assert!(client.last_request().is_some());

    // POST /run with a missing input field — 400 with the interpreter's
    // input-surface message.
    let bad = http
        .post(format!("{base}/run"))
        .json(&json!({}))
        .send()
        .await
        .expect("bad run");
    assert_eq!(bad.status(), 400);
    let bad: Value = bad.json().await.expect("json");
    assert!(
        bad["error"]
            .as_str()
            .unwrap_or_default()
            .contains("missing input field `question`"),
        "{bad}"
    );

    // POST /run with a non-object body — 400.
    let non_object = http
        .post(format!("{base}/run"))
        .json(&json!([1, 2, 3]))
        .send()
        .await
        .expect("non-object run");
    assert_eq!(non_object.status(), 400);
}

#[tokio::test]
async fn serve_applies_a_named_overlay() {
    let dir = tempfile::tempdir().expect("tempdir");
    let program = write_program(&dir, "echo.dsrs", ECHO);
    let overlay_path = dir.path().join("candidate.ovl.json");
    std::fs::write(
        &overlay_path,
        serde_json::to_string_pretty(&json!({
            "answerer.instruction": {"k": "instruction", "text": "OVERLAY-INSTRUCTION-MARKER"}
        }))
        .expect("overlay json"),
    )
    .expect("write overlay");

    let (lm, client) = canned_lm(vec![canned_text(&[("answer", "overlaid")])]).await;
    let config = ServeConfig {
        program,
        overlay: Some(overlay_path),
        allow: vec![],
    };
    let app = serve::load(&config, RuntimeEnv::new().bind_model("m", lm))
        .await
        .expect("load with overlay");
    let addr = spawn_server(app).await;

    let run: Value = reqwest::Client::new()
        .post(format!("http://{addr}/run"))
        .json(&json!({"question": "q"}))
        .send()
        .await
        .expect("run")
        .json()
        .await
        .expect("json");
    assert_eq!(run["output"]["answer"], "overlaid");

    // Overlay read-through reached the rendered prompt.
    let request = client.last_request().expect("one LM call");
    let rendered = format!("{request:?}");
    assert!(
        rendered.contains("OVERLAY-INSTRUCTION-MARKER"),
        "overlay instruction must reach the prompt: {rendered}"
    );
}

#[tokio::test]
async fn serve_refuses_ungranted_caps_and_accepts_allow() {
    let capped = r#"dsrs 1
program capped

caps { net:probe }

model m = "openai:gpt-4o-mini"

sig Main {
  in question: string
  out answer: string
}

main: Main = seq {
  answerer = predict Main (question = $.question)
  out { answer = answerer.answer }
}
"#;
    let dir = tempfile::tempdir().expect("tempdir");
    let program = write_program(&dir, "capped.dsrs", capped);

    // No grants: load refuses, naming the missing capability.
    let (lm, _client) = canned_lm(vec![]).await;
    let config = ServeConfig {
        program: program.clone(),
        overlay: None,
        allow: vec![],
    };
    let err = serve::load(&config, RuntimeEnv::new().bind_model("m", lm))
        .await
        .expect_err("caps exceed grants");
    assert!(err.to_string().contains("net:probe"), "{err:#}");

    // --allow closes the gap.
    let (lm, _client) = canned_lm(vec![]).await;
    let config = ServeConfig {
        program,
        overlay: None,
        allow: vec!["net:probe".to_string()],
    };
    serve::load(&config, RuntimeEnv::new().bind_model("m", lm))
        .await
        .expect("granted load succeeds");
}

#[tokio::test]
async fn serve_rejects_stale_overlays() {
    let dir = tempfile::tempdir().expect("tempdir");
    let program = write_program(&dir, "echo.dsrs", ECHO);
    let overlay_path = dir.path().join("stale.ovl.json");
    std::fs::write(
        &overlay_path,
        serde_json::to_string(&json!({
            "no_such_leaf.instruction": {"k": "instruction", "text": "x"}
        }))
        .expect("json"),
    )
    .expect("write overlay");

    let (lm, _client) = canned_lm(vec![]).await;
    let config = ServeConfig {
        program,
        overlay: Some(overlay_path),
        allow: vec![],
    };
    let err = serve::load(&config, RuntimeEnv::new().bind_model("m", lm))
        .await
        .expect_err("unknown param path refused at load");
    assert!(
        format!("{err:#}").contains("no_such_leaf.instruction"),
        "{err:#}"
    );
}
