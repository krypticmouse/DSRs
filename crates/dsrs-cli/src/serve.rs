//! `dsrs serve`: the serving host (RFC 0002 §6.2, adapted — see below).
//!
//! Startup is fail-fast, before binding the port: parse → optional overlay
//! (named `to_named` form, verified against the program's hash and slot
//! kinds) → `Interpreter::load` with capability grants from `--allow`
//! (refused if `caps ⊄ grants`, printing the missing set), models bound from
//! env-held secrets via `LM::from_config`, holes/sandboxed tools compiled and
//! registered.
//!
//! Endpoints (this stage's surface; the RFC sketch's `/v1/*`, OpenAPI,
//! trace-ring and canary hooks are deferred):
//!
//! - `POST /run[?trace=1]` — JSON input map → `{"output": …}`; with
//!   `trace=1`, the request runs inside an RFC 0001 capture scope and the
//!   response adds `"trace_jsonl"` (the exact `.trace.jsonl` artifact text,
//!   `param_ids` attached). Input errors are 400, everything else 500, always
//!   `{"error": …}`.
//! - `GET /schema` — the program's external interface: the main
//!   `SignatureDef` and `TypeTable` in their serde forms (the artifact's own
//!   vocabulary, not a lossy OpenAPI projection).
//! - `GET /program` — the canonical `.dsrs` text (`Program::to_dsrs`).
//! - `GET /healthz` — `{"status":"ok", program, program_hash}`.
//!
//! Host tools cannot be served: a program declaring `ToolKind::Host` needs a
//! embedding host that supplies the binding (`include_program!` + your own
//! code). `load` surfaces that refusal with a hint rather than a bare
//! "unbound" error.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use axum::Router;
use axum::extract::{Query, State};
use axum::http::{StatusCode, header};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use dspy_rs::ir::{Budget, Interpreter, LoadError, Overlay, ParamValue, Program, RunError};
use dspy_rs::trace::{JsonMap, capture};
use serde_json::{Value, json};

/// What `dsrs serve` needs to know before binding a port.
#[derive(Debug, Clone, Default)]
pub struct ServeConfig {
    /// The `.dsrs` artifact to serve.
    pub program: PathBuf,
    /// Optional overlay in the named (`Overlay::to_named`) JSON form:
    /// `{"<param path>": <ParamValue>, …}`, applied read-through on every run.
    pub overlay: Option<PathBuf>,
    /// Capability grants (`--allow net:search`), checked against the
    /// program's `caps` ceiling at load.
    pub allow: Vec<String>,
}

/// A loaded serving app: shared interpreter + precomputed responses.
pub struct App {
    interp: Interpreter,
    overlay: Option<Arc<Overlay>>,
    canonical: String,
    schema: Value,
    health: Value,
}

impl App {
    pub fn program(&self) -> &Arc<Program> {
        self.interp.program()
    }
}

impl std::fmt::Debug for App {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("App")
            .field("program", &self.program().meta.name)
            .field("overlay", &self.overlay.is_some())
            .finish_non_exhaustive()
    }
}

/// Loads the program and builds the app state. `env` carries host-supplied
/// bindings — the CLI passes `RuntimeEnv::new()` (models constructed from
/// their artifact configs, secrets from provider env vars); tests pre-bind
/// canned models. Grants from `config.allow` and a QuickJS sandbox (when none
/// was supplied) are added here.
pub async fn load(config: &ServeConfig, env: dspy_rs::ir::RuntimeEnv) -> anyhow::Result<Arc<App>> {
    let program = Program::load_dsrs(&config.program)?;

    let overlay = match &config.overlay {
        Some(path) => {
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("failed to read overlay `{}`", path.display()))?;
            let named: BTreeMap<String, ParamValue> = serde_json::from_str(&text)
                .with_context(|| {
                    format!(
                        "overlay `{}` is not a named overlay JSON object \
                         ({{\"<param path>\": <value>, ...}})",
                        path.display()
                    )
                })?;
            Some(Arc::new(Overlay::from_named(&program, named).with_context(
                || format!("overlay `{}` does not apply to this program", path.display()),
            )?))
        }
        None => None,
    };

    let canonical = program.to_dsrs();
    let hash = format!("{:016x}", program.meta.program_hash);
    let sig = &program.sigs[program.sig];
    let schema = json!({
        "program": program.meta.name,
        "program_hash": hash,
        "signature": sig,
        "types": program.types,
    });
    let health = json!({
        "status": "ok",
        "program": program.meta.name,
        "program_hash": hash,
    });

    let mut env = env;
    for cap in &config.allow {
        env = env.grant(cap);
    }
    if env.sandbox.is_none() {
        env = env.with_sandbox(Arc::new(dsrs_tools::QuickJsExecutor::new()));
    }

    let interp = Interpreter::load(program, env).await.map_err(|err| {
        if let LoadError::HostToolUnbound { name } = &err {
            anyhow::anyhow!(
                "host tool `{name}` is not bound in the runtime environment\n\
                 hint: `dsrs serve` cannot bind host tools — embed the program \
                 (dspy_rs::include_program!) and supply the binding, or make the \
                 tool sandboxed (`js` fence in the artifact)"
            )
        } else {
            anyhow::Error::from(err)
        }
    })?;

    Ok(Arc::new(App {
        interp,
        overlay,
        canonical,
        schema,
        health,
    }))
}

/// The HTTP surface over a loaded [`App`]. Split from [`serve`] so
/// integration tests drive the exact production router on an ephemeral port.
pub fn router(app: Arc<App>) -> Router {
    Router::new()
        .route("/run", post(run_handler))
        .route("/schema", get(schema_handler))
        .route("/program", get(program_handler))
        .route("/healthz", get(healthz_handler))
        .with_state(app)
}

/// Loads and serves until the process ends. Prints the bound address on
/// stderr (port 0 binds an ephemeral port).
pub async fn serve(config: &ServeConfig, host: &str, port: u16) -> anyhow::Result<()> {
    let app = load(config, dspy_rs::ir::RuntimeEnv::new()).await?;
    let listener = tokio::net::TcpListener::bind((host, port))
        .await
        .with_context(|| format!("failed to bind {host}:{port}"))?;
    let addr = listener.local_addr()?;
    eprintln!(
        "dsrs: serving `{}` ({:016x}) at http://{addr}",
        app.program().meta.name,
        app.program().meta.program_hash
    );
    axum::serve(listener, router(app)).await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn healthz_handler(State(app): State<Arc<App>>) -> axum::Json<Value> {
    axum::Json(app.health.clone())
}

async fn schema_handler(State(app): State<Arc<App>>) -> axum::Json<Value> {
    axum::Json(app.schema.clone())
}

async fn program_handler(State(app): State<Arc<App>>) -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
        app.canonical.clone(),
    )
}

async fn run_handler(
    State(app): State<Arc<App>>,
    Query(query): Query<HashMap<String, String>>,
    body: axum::Json<Value>,
) -> (StatusCode, axum::Json<Value>) {
    let Some(input) = body.0.as_object().cloned() else {
        return (
            StatusCode::BAD_REQUEST,
            axum::Json(json!({"error": "request body must be a JSON object of input fields"})),
        );
    };
    let want_trace = matches!(
        query.get("trace").map(String::as_str),
        Some("1") | Some("true")
    );
    if want_trace {
        run_traced(&app, input).await
    } else {
        match app
            .interp
            .run(input, app.overlay.clone(), Budget::unlimited())
            .await
        {
            Ok(output) => (StatusCode::OK, axum::Json(json!({"output": output}))),
            Err(err) => run_error(&err),
        }
    }
}

async fn run_traced(app: &Arc<App>, input: JsonMap) -> (StatusCode, axum::Json<Value>) {
    let (result, mut trace) = capture(|| {
        app.interp
            .run(input, app.overlay.clone(), Budget::unlimited())
    })
    .await;
    match result {
        Ok(output) => {
            trace.attach_program(app.interp.program());
            match trace.to_jsonl() {
                Ok(jsonl) => (
                    StatusCode::OK,
                    axum::Json(json!({"output": output, "trace_jsonl": jsonl})),
                ),
                Err(err) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    axum::Json(json!({"error": format!("trace serialization failed: {err}")})),
                ),
            }
        }
        Err(err) => run_error(&err),
    }
}

/// Input-surface rejections are the caller's fault (400); everything else —
/// LM failures, parse failures, budget, routing — is a server-side 500. The
/// error text is the interpreter's own (`at`-qualified) message.
fn run_error(err: &RunError) -> (StatusCode, axum::Json<Value>) {
    let status = match err {
        RunError::Input { .. } => StatusCode::BAD_REQUEST,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    (status, axum::Json(json!({"error": err.to_string()})))
}
