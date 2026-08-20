//! The IR interpreter (RFC 0002 §3): async evaluation of a loaded [`Program`].
//!
//! - **Loading** ([`Interpreter::load`]) front-loads every check: validation,
//!   capability grants, model binding, host-tool binding, and the full
//!   dsrs-tools lifecycle (parse → compile → register) for every sandboxed
//!   tool and hole. A hole that doesn't compile fails the *load*, not the call.
//! - **Run state** is a call-scoped `Cx`, never stored on the interpreter —
//!   concurrent runs (and concurrent overlays) share one `Arc<Program>`.
//! - **Overlay read-through**: instruction/demos/model/context/code resolve
//!   through the optional [`Overlay`](crate::ir::params::Overlay) at render
//!   time. The program is never mutated.
//! - **Spans** (RFC 0001): each leaf evaluation emits one span through the
//!   ambient capture scope, `component` = the program-unique leaf name. An
//!   `AgentLoop` is one span with N `Exchange`/`ToolRun` events; a `Hole` is
//!   one span whose model is the reserved `sandbox:quickjs` config.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Instant;

use cranelift_entity::SecondaryMap;
use futures::future::BoxFuture;
use indexmap::IndexMap;
use serde_json::{Value, json};

use crate::adapter::chat::ChatAdapter;
use crate::core::FieldMeta;
use crate::ir::graph::{
    AgentLoopNode, Binding, BudgetPolicy, CapSet, HoleImpl, HoleNode, ModelId, Node, NodeId,
    PortRef, PredictNode, Program, ToolId, ToolKind,
};
use crate::ir::params::{ContextPolicy, DemoRow, Overlay, ParamId, ParamValue};
use crate::ir::sig::SignatureDef;
use crate::ir::validate::{ValidateError, json_matches_type};
use crate::trace::{JsonMap, SpanEvent, SpanOutcome, SpanRequest, begin_span};
use crate::typesys::{FieldType, TypeTable};
use crate::{Chat, LM, LMConfig, LmError, LmUsage, Message, Role, ToolLoopMode, ToolSet};

// ---------------------------------------------------------------------------
// Budgets
// ---------------------------------------------------------------------------

/// Run-level spend limits. `None` = unlimited.
#[derive(Clone, Copy, Debug, Default)]
pub struct Budget {
    pub max_lm_calls: Option<u32>,
    pub max_tokens: Option<u64>,
    pub deadline: Option<Instant>,
}

impl Budget {
    pub fn unlimited() -> Self {
        Self::default()
    }
}

/// Budget reservation failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("budget exhausted")]
pub struct Exhausted;

/// Check-before-call metering: calls and deadline are hard-gated pre-call;
/// token budgets are soft (checked against accumulated usage, since usage is
/// only known post-hoc). Reservation recurses into the parent — an
/// `AgentLoop` chains a child meter under the run meter.
#[derive(Debug)]
pub struct BudgetMeter {
    limits: Budget,
    lm_calls: AtomicU32,
    tokens: AtomicU64,
    parent: Option<Arc<BudgetMeter>>,
}

impl BudgetMeter {
    pub fn new(limits: Budget) -> Self {
        Self {
            limits,
            lm_calls: AtomicU32::new(0),
            tokens: AtomicU64::new(0),
            parent: None,
        }
    }

    /// A child meter whose reservations also count against `parent`.
    pub fn child(parent: &Arc<BudgetMeter>, limits: Budget) -> Self {
        Self {
            parent: Some(Arc::clone(parent)),
            ..Self::new(limits)
        }
    }

    fn check_soft(&self) -> Result<(), Exhausted> {
        if let Some(deadline) = self.limits.deadline
            && Instant::now() >= deadline
        {
            return Err(Exhausted);
        }
        if let Some(max) = self.limits.max_tokens
            && self.tokens.load(Ordering::Relaxed) >= max
        {
            return Err(Exhausted);
        }
        Ok(())
    }

    /// Reserves one LM call against this meter and every ancestor.
    pub fn try_reserve_call(&self) -> Result<(), Exhausted> {
        self.check_soft()?;
        if let Some(max) = self.limits.max_lm_calls {
            self.lm_calls
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |calls| {
                    (calls < max).then_some(calls + 1)
                })
                .map_err(|_| Exhausted)?;
        } else {
            self.lm_calls.fetch_add(1, Ordering::Relaxed);
        }
        if let Some(parent) = &self.parent
            && let Err(exhausted) = parent.try_reserve_call()
        {
            self.lm_calls.fetch_sub(1, Ordering::Relaxed);
            return Err(exhausted);
        }
        Ok(())
    }

    /// Records observed usage on this meter and every ancestor.
    pub fn record_usage(&self, usage: &LmUsage) {
        self.tokens.fetch_add(usage.total_tokens, Ordering::Relaxed);
        if let Some(parent) = &self.parent {
            parent.record_usage(usage);
        }
    }

    pub fn lm_calls(&self) -> u32 {
        self.lm_calls.load(Ordering::Relaxed)
    }

    pub fn tokens(&self) -> u64 {
        self.tokens.load(Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// Leaf metadata
// ---------------------------------------------------------------------------

/// Parse/coercion metadata from one successful `Predict`-leaf evaluation,
/// collected by [`Interpreter::run_collecting`] in execution order.
///
/// This is the interpreter-side twin of the static lane's
/// [`CallMetadata`](crate::CallMetadata): the same [`FieldMeta`] type (jsonish
/// coercion [`Flag`](crate::Flag)s + `#[check]` [`ConstraintResult`](crate::ConstraintResult)s)
/// that `ChatAdapter::parse_output_with_meta` produces, so a typed `Predict<S>`
/// routed through the interpreter loses none of `Predicted`'s metadata contract.
///
/// Scope and semantics:
/// - Only `Predict` leaves report. `Hole` leaves make no LM call and parse no
///   delimited text; `AgentLoop` leaves are a single multi-turn span whose
///   final output may come from stop-tool args (no parse metadata exists).
/// - Only *successful* evaluations report — a failed attempt inside `Retry`
///   propagates its error and leaves no outcome; the succeeding attempt
///   reports one. A leaf re-evaluated by `Refine`/`Loop` reports once per
///   evaluation.
/// - A replay-served leaf reports raw text and usage from the recorded span
///   with **empty** `field_meta`, matching the static lane ("served
///   predictions carry no per-field parse metadata").
#[derive(Debug, Clone)]
pub struct LeafOutcome {
    /// The program-unique leaf name (the trace span `component`).
    pub name: String,
    /// The full text the LM returned, before parsing.
    pub raw_response: String,
    /// Per-field parse details keyed by canonical field name
    /// (`FieldDef::name`) — raw section text, coercion flags, check results.
    pub field_meta: IndexMap<String, FieldMeta>,
    /// Token usage for this leaf's LM call.
    pub usage: LmUsage,
    /// Stable hash of the redacted model config used — identical to the
    /// trace's [`ModelEntry::config_hash`](crate::trace::ModelEntry) for the
    /// same config.
    pub model_config_hash: u64,
}

/// Program output plus per-leaf metadata, returned by
/// [`Interpreter::run_collecting`].
#[derive(Debug, Clone)]
pub struct RunOutput {
    /// The program's output map — exactly what [`Interpreter::run`] returns.
    pub output: JsonMap,
    /// One entry per successful `Predict`-leaf evaluation, in execution
    /// order. `ForkJoin` branches append in declared branch order.
    pub leaves: Vec<LeafOutcome>,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("program failed validation")]
    Invalid(#[from] ValidateError),
    #[error("program caps exceed host grants: missing {missing:?}")]
    CapsExceedGrants { missing: Vec<String> },
    #[error("model `{name}` could not be bound: {message}")]
    Model { name: String, message: String },
    #[error("host tool `{name}` is not bound in the runtime environment")]
    HostToolUnbound { name: String },
    #[error("host (extern) hole `{name}` is not bound in the runtime environment")]
    HostHoleUnbound { name: String },
    #[error("program contains sandboxed code but the environment has no sandbox executor")]
    SandboxMissing,
    #[error("sandboxed code at `{at}` failed to register")]
    Register {
        at: String,
        #[source]
        source: dsrs_tools::RegisterError,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum RunError {
    #[error("lm error at `{at}`: {source}")]
    Lm {
        at: Box<str>,
        #[source]
        source: LmError,
    },
    #[error("parse error at `{at}`")]
    Parse { at: Box<str>, raw: String },
    #[error("tool `{tool}` failed at `{at}`: {message}")]
    Tool {
        at: Box<str>,
        tool: Box<str>,
        message: String,
    },
    #[error("hole `{at}` failed")]
    Hole {
        at: Box<str>,
        #[source]
        source: dsrs_tools::ExecError,
    },
    #[error("capability `{cap}` denied at `{at}`")]
    CapabilityDenied { at: Box<str>, cap: Box<str> },
    #[error("budget exhausted at `{at}`")]
    Budget { at: Box<str> },
    #[error("route unmatched at `{at}`: `{value}`")]
    Route { at: Box<str>, value: String },
    #[error("cancelled")]
    Cancelled,
    /// Additive to RFC 0002: overlay minted against a different program.
    #[error("overlay minted against program {expected:016x}, run against {got:016x}")]
    Overlay { expected: u64, got: u64 },
    /// Additive to RFC 0002: run input rejected against the program signature.
    #[error("invalid input at `{at}`: {message}")]
    Input { at: Box<str>, message: String },
    /// Additive to RFC 0002: interpreter invariant violation.
    #[error("internal interpreter error at `{at}`: {message}")]
    Internal { at: Box<str>, message: String },
    /// RFC 0003 M-1: a strict replay scope refused this call.
    #[error("replay refused at `{at}`")]
    Replay {
        at: Box<str>,
        #[source]
        source: crate::trace::ReplayError,
    },
}

impl RunError {
    /// Whether `Retry`/`Refine` may intercept this error (RFC 0002 §3.4:
    /// `Lm | Parse | Tool | Hole`; `Budget`/`CapabilityDenied` are not
    /// retryable).
    pub fn retryable(&self) -> bool {
        matches!(
            self,
            Self::Lm { .. } | Self::Parse { .. } | Self::Tool { .. } | Self::Hole { .. }
        )
    }
}

// ---------------------------------------------------------------------------
// Runtime environment
// ---------------------------------------------------------------------------

/// What the host supplies at load: live models, host tool bindings, the
/// sandbox, and the capability grants. Secrets never transit the artifact —
/// model configs are bound to clients here, from host-held keys/env vars.
#[derive(Default)]
pub struct RuntimeEnv {
    /// Pre-bound live models by declared model name (`"fast"`). Models not
    /// bound here are constructed from their `ModelDef.config` at load.
    pub models: HashMap<String, Arc<LM>>,
    /// Host tool bindings by tool name, consulted once at load.
    pub host_tools: HashMap<String, Arc<dyn rig::tool::ToolDyn>>,
    /// Host (extern) hole bindings by leaf name, consulted once at load
    /// (RFC 0003 §4.1). The fn receives the hole's resolved input map and
    /// returns a JSON value coerced against the hole's output signature.
    pub host_holes: HashMap<String, HostHoleFn>,
    /// The sandbox executing holes and sandboxed tools (QuickJS in v1).
    /// Required iff the program carries sandboxed code.
    pub sandbox: Option<Arc<dyn dsrs_tools::Executor>>,
    /// What this host permits. `program.caps ⊄ grants` ⇒ load refused.
    pub grants: CapSet,
    /// Code Mode: when set, every `AgentLoop` presents its non-stop tools as
    /// a single sandboxed `run_js` tool (with this sandbox config) instead of
    /// N JSON tools; the model writes JavaScript that calls them as globals.
    ///
    /// This is a `RuntimeEnv` binding option, not a `ToolKind` variant, by
    /// design: code mode is a host presentation/execution strategy, not
    /// program semantics — the same artifact (same tools, same genes, same
    /// program hash) must run identically either way, so it cannot live in
    /// the serialized program. `ToolKind` is also per-tool, while code mode
    /// collapses a whole loop's tool surface; and leaving the closed enum
    /// untouched keeps the `.dsrs` text format stable.
    #[cfg(feature = "code-mode")]
    pub code_mode: Option<dsrs_tools::SandboxConfig>,
}

impl RuntimeEnv {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn bind_model(mut self, name: &str, lm: Arc<LM>) -> Self {
        self.models.insert(name.to_string(), lm);
        self
    }

    pub fn bind_host_tool(mut self, name: &str, tool: Arc<dyn rig::tool::ToolDyn>) -> Self {
        self.host_tools.insert(name.to_string(), tool);
        self
    }

    /// Binds a native implementation for an extern hole by leaf name.
    pub fn bind_host_hole<F, Fut>(mut self, name: &str, f: F) -> Self
    where
        F: Fn(JsonMap) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Value, String>> + Send + 'static,
    {
        self.host_holes
            .insert(name.to_string(), Arc::new(move |input| Box::pin(f(input))));
        self
    }

    pub fn with_sandbox(mut self, sandbox: Arc<dyn dsrs_tools::Executor>) -> Self {
        self.sandbox = Some(sandbox);
        self
    }

    pub fn grant(mut self, cap: &str) -> Self {
        self.grants.insert(cap);
        self
    }

    /// Enables Code Mode for every `AgentLoop` (see
    /// [`code_mode`](Self::code_mode)): tools are presented to the model as a
    /// JS API behind one `run_js` tool executing under `config`.
    #[cfg(feature = "code-mode")]
    pub fn with_code_mode(mut self, config: dsrs_tools::SandboxConfig) -> Self {
        self.code_mode = Some(config);
        self
    }
}

// ---------------------------------------------------------------------------
// Interpreter
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum ToolExec {
    Host(Arc<dyn rig::tool::ToolDyn>),
    Sandboxed,
}

/// A native extern-hole implementation: resolved input map in, JSON value out
/// (coerced against the hole's output signature by the interpreter).
pub type HostHoleFn =
    Arc<dyn Fn(JsonMap) -> BoxFuture<'static, Result<Value, String>> + Send + Sync>;

/// A loaded, executable program: validated graph + bound models/tools +
/// registered sandbox code. Cheap to share; run state never lives here.
pub struct Interpreter {
    program: Arc<Program>,
    models: SecondaryMap<ModelId, Option<Arc<LM>>>,
    tool_exec: SecondaryMap<ToolId, Option<ToolExec>>,
    /// Extern-hole bindings by leaf name, verified complete at load.
    host_holes: HashMap<String, HostHoleFn>,
    sandbox: Option<Arc<dyn dsrs_tools::Executor>>,
    /// Code-hash → registered sandbox tool name. Seeded at load with every
    /// default code gene; overlay code variants register on first use.
    registered: tokio::sync::Mutex<HashMap<u64, String>>,
    /// Code Mode sandbox config (see [`RuntimeEnv::code_mode`]).
    #[cfg(feature = "code-mode")]
    code_mode: Option<dsrs_tools::SandboxConfig>,
}

impl std::fmt::Debug for Interpreter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Interpreter")
            .field("program", &self.program.meta.name)
            .field("program_hash", &self.program.meta.program_hash)
            .finish_non_exhaustive()
    }
}

impl Interpreter {
    /// Load-time checks, in order — nothing lazy, nothing at call time:
    /// 1. [`Program::validate`] (§2.3 rules)
    /// 2. `program.caps ⊆ env.grants` (WIT rule: no ambient authority)
    /// 3. every model bindable (pre-bound by name, else client-constructible)
    /// 4. every `ToolKind::Host` name bound
    /// 5. every sandboxed tool and hole registered through the dsrs-tools
    ///    lifecycle (parse → compile → register). A hole that doesn't compile
    ///    fails the LOAD, not the call.
    pub async fn load(program: Program, env: RuntimeEnv) -> Result<Self, LoadError> {
        program.validate()?;

        if !program.caps.is_subset(&env.grants) {
            return Err(LoadError::CapsExceedGrants {
                missing: program.caps.missing_from(&env.grants),
            });
        }

        let mut models: SecondaryMap<ModelId, Option<Arc<LM>>> = SecondaryMap::new();
        for (id, def) in program.models.iter() {
            let lm = match env.models.get(&*def.name) {
                Some(lm) => Arc::clone(lm),
                None => Arc::new(LM::from_config(def.config.clone()).await.map_err(|err| {
                    LoadError::Model {
                        name: def.name.to_string(),
                        message: err.to_string(),
                    }
                })?),
            };
            models[id] = Some(lm);
        }

        let mut tool_exec: SecondaryMap<ToolId, Option<ToolExec>> = SecondaryMap::new();
        let mut sandboxed: Vec<(String, ParamId, crate::ir::graph::SigId)> = Vec::new();
        for (id, tool) in program.tools.iter() {
            let name = program.syms.get(tool.name).to_string();
            match tool.kind {
                ToolKind::Host => {
                    let bound = env
                        .host_tools
                        .get(&name)
                        .cloned()
                        .ok_or(LoadError::HostToolUnbound { name: name.clone() })?;
                    tool_exec[id] = Some(ToolExec::Host(bound));
                }
                ToolKind::Sandboxed { code } => {
                    sandboxed.push((name, code, tool.sig));
                    tool_exec[id] = Some(ToolExec::Sandboxed);
                }
            }
        }
        let mut host_holes: HashMap<String, HostHoleFn> = HashMap::new();
        for (_, node) in program.nodes.iter() {
            if let Node::Hole(hole) = node {
                let name = program.syms.get(hole.name).to_string();
                match hole.imp {
                    HoleImpl::Sandboxed { code } => sandboxed.push((name, code, hole.sig)),
                    HoleImpl::Host { .. } => {
                        let bound = env
                            .host_holes
                            .get(&name)
                            .cloned()
                            .ok_or(LoadError::HostHoleUnbound { name: name.clone() })?;
                        host_holes.insert(name, bound);
                    }
                }
            }
        }

        // Code Mode: JS-identifier collisions among a loop's non-stop tool
        // names are a load-time refusal (nothing lazy, nothing at call time).
        #[cfg(feature = "code-mode")]
        if env.code_mode.is_some() {
            for (_, node) in program.nodes.iter() {
                let Node::AgentLoop(agent) = node else {
                    continue;
                };
                let mut seen: HashMap<String, String> = HashMap::new();
                for &tool_id in agent.tools.iter() {
                    if agent.stop.stop_tools.contains(&tool_id) {
                        continue;
                    }
                    let name = program.syms.get(program.tools[tool_id].name).to_string();
                    let js_name = dsrs_tools::js_identifier(&name);
                    if let Some(previous) = seen.insert(js_name.clone(), name.clone()) {
                        return Err(LoadError::Register {
                            at: program.syms.get(agent.name).to_string(),
                            source: dsrs_tools::RegisterError::InvalidCapability {
                                name: js_name.clone(),
                                reason: format!(
                                    "tool names `{previous}` and `{name}` both mangle to the JS identifier `{js_name}`"
                                ),
                            },
                        });
                    }
                }
            }
        }

        let sandbox = env.sandbox;
        let mut registered = HashMap::new();
        if !sandboxed.is_empty() {
            let executor = sandbox.as_ref().ok_or(LoadError::SandboxMissing)?;
            for (base, code, sig) in sandboxed {
                let ParamValue::Code { source, hash, .. } = &program.params[code].default else {
                    unreachable!("validated: code slots hold Code values");
                };
                let name = register_code(
                    executor.as_ref(),
                    &base,
                    source,
                    *hash,
                    &program.sigs[sig],
                    &program.types,
                )
                .await
                .map_err(|source| LoadError::Register {
                    at: base.clone(),
                    source,
                })?;
                registered.insert(*hash, name);
            }
        }

        Ok(Self {
            program: Arc::new(program),
            models,
            tool_exec,
            host_holes,
            sandbox,
            registered: tokio::sync::Mutex::new(registered),
            #[cfg(feature = "code-mode")]
            code_mode: env.code_mode,
        })
    }

    pub fn program(&self) -> &Arc<Program> {
        &self.program
    }

    /// Evaluates the program on `input`, reading parameters through `overlay`
    /// (never mutating the program), metering spend against `budget`.
    pub async fn run(
        &self,
        input: JsonMap,
        overlay: Option<Arc<Overlay>>,
        budget: Budget,
    ) -> Result<JsonMap, RunError> {
        self.run_inner(input, overlay, budget, false)
            .await
            .map(|out| out.output)
    }

    /// Like [`run`](Interpreter::run), additionally collecting a
    /// [`LeafOutcome`] per successful `Predict`-leaf evaluation (raw response,
    /// per-field flags and check results, usage, model config hash). See
    /// [`LeafOutcome`] for the exact scope and semantics.
    pub async fn run_collecting(
        &self,
        input: JsonMap,
        overlay: Option<Arc<Overlay>>,
        budget: Budget,
    ) -> Result<RunOutput, RunError> {
        self.run_inner(input, overlay, budget, true).await
    }

    async fn run_inner(
        &self,
        input: JsonMap,
        overlay: Option<Arc<Overlay>>,
        budget: Budget,
        collect: bool,
    ) -> Result<RunOutput, RunError> {
        if let Some(overlay) = &overlay
            && overlay.base != self.program.meta.program_hash
        {
            return Err(RunError::Overlay {
                expected: overlay.base,
                got: self.program.meta.program_hash,
            });
        }

        // Input surface check against the program's external signature.
        let sig = &self.program.sigs[self.program.sig];
        for field in sig.inputs.iter() {
            match input.get(&*field.name) {
                Some(value) => {
                    if !json_matches_type(value, &field.ty, &self.program.types) {
                        return Err(RunError::Input {
                            at: "$".into(),
                            message: format!(
                                "field `{}` does not match its declared type",
                                field.name
                            ),
                        });
                    }
                }
                None if field.ty.is_optional() => {}
                None => {
                    return Err(RunError::Input {
                        at: "$".into(),
                        message: format!("missing input field `{}`", field.name),
                    });
                }
            }
        }

        let mut cx = Cx {
            overlay,
            meter: Arc::new(BudgetMeter::new(budget)),
            frames: SecondaryMap::new(),
            inputs: vec![input],
            feedback: None,
            refine_feedback: None,
            leaves: collect.then(Vec::new),
        };
        let output = self.eval(self.program.root, &mut cx).await?;
        Ok(RunOutput {
            output,
            leaves: cx.leaves.unwrap_or_default(),
        })
    }

    // -- node dispatch --------------------------------------------------------

    fn eval<'a>(&'a self, id: NodeId, cx: &'a mut Cx) -> BoxFuture<'a, Result<JsonMap, RunError>> {
        Box::pin(async move {
            let out = match &self.program.nodes[id] {
                Node::Predict(n) => self.eval_predict(id, n, cx).await?,
                Node::AgentLoop(n) => self.eval_agent(id, n, cx).await?,
                Node::Hole(n) => self.eval_hole(id, n, cx).await?,
                Node::Seq(n) => {
                    for &child in n.body.iter() {
                        self.eval(child, cx).await?;
                    }
                    self.resolve_exports(&n.out, cx)?
                }
                Node::ForkJoin(n) => {
                    let branch_cxs: Vec<(NodeId, Cx)> = n
                        .branches
                        .iter()
                        .map(|&branch| (branch, cx.branch()))
                        .collect();
                    let futures =
                        branch_cxs
                            .into_iter()
                            .map(|(branch, mut branch_cx)| async move {
                                self.eval(branch, &mut branch_cx).await?;
                                Ok::<_, RunError>((branch_cx.frames, branch_cx.leaves))
                            });
                    // First error aborts siblings: try_join_all drops the
                    // remaining futures, whose open spans record Cancelled
                    // via guard-drop (RFC 0001 §3.1).
                    let all_frames = futures::future::try_join_all(futures).await?;
                    for (frames, leaves) in all_frames {
                        for (node, out) in frames.iter() {
                            if let Some(out) = out
                                && cx.frames[node].is_none()
                            {
                                cx.frames[node] = Some(out.clone());
                            }
                        }
                        // Branch outcomes merge in declared branch order.
                        if let (Some(collected), Some(branch_leaves)) =
                            (cx.leaves.as_mut(), leaves)
                        {
                            collected.extend(branch_leaves);
                        }
                    }
                    self.resolve_exports(&n.join, cx)?
                }
                Node::Route(n) => {
                    let at = self.program.node_display(id);
                    let value = self.resolve_port(&at, &n.on, cx)?;
                    let discriminant = match value.as_str() {
                        Some(s) => s.to_string(),
                        None => value.to_string(),
                    };
                    let arm = n
                        .arms
                        .iter()
                        .find(|(variant, _)| {
                            self.program.syms.get(*variant) == discriminant.as_str()
                        })
                        .map(|(_, arm)| *arm)
                        .or(n.default);
                    match arm {
                        Some(arm) => self.eval(arm, cx).await?,
                        None => {
                            return Err(RunError::Route {
                                at: at.into(),
                                value: discriminant,
                            });
                        }
                    }
                }
                Node::Retry(n) => {
                    let mut attempt = 0u32;
                    loop {
                        match self.eval(n.child, cx).await {
                            Ok(out) => break out,
                            Err(err) if err.retryable() && attempt + 1 < n.max_attempts.get() => {
                                attempt += 1;
                                if n.feedback && matches!(err, RunError::Parse { .. }) {
                                    cx.feedback = Some(format!(
                                        "Your previous response could not be parsed: {err}. \
                                         Respond again, following the required \
                                         `[[ ## field ## ]]` output format exactly."
                                    ));
                                }
                                if n.backoff_ms > 0 {
                                    tokio::time::sleep(std::time::Duration::from_millis(
                                        n.backoff_ms as u64,
                                    ))
                                    .await;
                                }
                            }
                            Err(err) => return Err(err),
                        }
                    }
                }
                Node::Refine(n) => {
                    let saved = cx.refine_feedback.take();
                    let result = self.eval_refine(n, cx).await;
                    cx.refine_feedback = saved;
                    result?
                }
                Node::Loop(n) => {
                    let enclosing = cx.inputs.last().cloned().unwrap_or_default();
                    let mut carry = JsonMap::new();
                    let at = self.program.node_display(id);
                    let mut result = None;
                    for iter in 0..n.max_iters.get() {
                        let mut frame = enclosing.clone();
                        frame.extend(carry.clone());
                        cx.inputs.push(frame);
                        let body = self.eval(n.body, cx).await;
                        if body.is_err() {
                            cx.inputs.pop();
                        }
                        body?;
                        // Rebind carried values from this iteration.
                        let mut next = JsonMap::new();
                        for b in n.carry.iter() {
                            let name = self.program.syms.get(b.dst).to_string();
                            next.insert(name, self.resolve_port(&at, &b.src, cx)?);
                        }
                        let continue_ = match &n.while_ {
                            Some(port) => {
                                self.resolve_port(&at, port, cx)?.as_bool().unwrap_or(false)
                            }
                            None => true,
                        };
                        cx.inputs.pop();
                        carry = next;
                        if !continue_ || iter + 1 == n.max_iters.get() {
                            // Exports resolve in the final iteration's scope
                            // (body outputs + final carry visible).
                            let mut frame = enclosing.clone();
                            frame.extend(carry.clone());
                            cx.inputs.push(frame);
                            let out = self.resolve_exports(&n.out, cx);
                            cx.inputs.pop();
                            result = Some(out?);
                            break;
                        }
                    }
                    result.expect("bounded loops always take the exit branch")
                }
            };
            cx.frames[id] = Some(out.clone());
            Ok(out)
        })
    }

    async fn eval_refine(
        &self,
        n: &crate::ir::graph::RefineNode,
        cx: &mut Cx,
    ) -> Result<JsonMap, RunError> {
        for round in 0..n.max_rounds.get() {
            let out = self.eval(n.child, cx).await?;
            let judge_out = self.eval(n.judge, cx).await?;
            let score = judge_out
                .get("score")
                .and_then(Value::as_f64)
                .unwrap_or(f64::NEG_INFINITY);
            if score >= n.threshold || round + 1 == n.max_rounds.get() {
                return Ok(out);
            }
            let feedback = judge_out.get("feedback").cloned().unwrap_or(Value::Null);
            let field = self.program.syms.get(n.feedback_field).to_string();
            cx.refine_feedback = Some((n.child, field, feedback));
        }
        unreachable!("refine rounds are bounded and return inside the loop")
    }

    // -- leaves ---------------------------------------------------------------

    async fn eval_predict(
        &self,
        id: NodeId,
        n: &PredictNode,
        cx: &mut Cx,
    ) -> Result<JsonMap, RunError> {
        let p = &*self.program;
        let at = p.syms.get(n.name).to_string();
        let def = &p.sigs[n.sig];
        let input = self.resolve_bindings(&at, Some(id), &n.binding, cx)?;

        let instruction = self.p_text(cx, n.instruction);
        let demos = self.p_demos(cx, n.demos);
        let (prefix, mut suffix) = render_prompt(def, &p.types, &instruction, &demos, &input, None);
        if let Some(feedback) = cx.feedback.take() {
            suffix.push(Message::user(feedback));
        }
        let lm = self.p_model(&at, cx, n.model)?;

        let guard = begin_span(SpanRequest {
            component: &at,
            prefix: Some(&prefix),
            suffix: &suffix,
            input: Some(input.clone()),
            model: &lm.config,
            request_hash: None,
        });

        let mut messages = prefix;
        messages.extend(suffix);

        // Replay scope (RFC 0001 §4d/e, RFC 0003 M-1): intercept above the
        // LM. A served call reserves no budget, constructs no client, and
        // reaches no provider.
        match crate::trace::replay::intercept(&at, &lm.config, &messages) {
            Some(crate::trace::replay::ReplayDirective::Serve(span)) => {
                let output = span
                    .output
                    .clone()
                    .expect("replay serves only spans with parsed output");
                // Parity with the static lane's `serve_recorded_span`: served
                // predictions carry no per-field parse metadata — the
                // recording stores the parsed output, not the parser's
                // field-level bookkeeping.
                if let Some(leaves) = cx.leaves.as_mut() {
                    leaves.push(LeafOutcome {
                        name: at.clone(),
                        raw_response: span.raw_output.clone().unwrap_or_default(),
                        field_meta: IndexMap::new(),
                        usage: span.usage,
                        model_config_hash: crate::trace::ModelEntry::from_config(&lm.config)
                            .config_hash,
                    });
                }
                if let Some(guard) = guard {
                    guard.finish(SpanOutcome {
                        events: span.events.clone(),
                        raw_output: span.raw_output.clone(),
                        output: Some(output.clone()),
                        usage: span.usage,
                        error: None,
                    });
                }
                return Ok(output);
            }
            Some(crate::trace::replay::ReplayDirective::Refuse(err)) => {
                if let Some(guard) = guard {
                    guard.finish(span_error(
                        crate::trace::SpanErrorKind::Lm,
                        err.to_string(),
                        Vec::new(),
                        None,
                        LmUsage::default(),
                    ));
                }
                return Err(RunError::Replay {
                    at: at.into(),
                    source: err,
                });
            }
            Some(crate::trace::replay::ReplayDirective::Live) | None => {}
        }

        if cx.meter.try_reserve_call().is_err() {
            if let Some(guard) = guard {
                guard.finish(span_error(
                    crate::trace::SpanErrorKind::Lm,
                    "budget exhausted".to_string(),
                    Vec::new(),
                    None,
                    LmUsage::default(),
                ));
            }
            return Err(RunError::Budget {
                at: at.clone().into(),
            });
        }

        let response = match lm.call(Chat::new(messages), Vec::new()).await {
            Ok(response) => response,
            Err(err) => {
                if let Some(guard) = guard {
                    guard.finish(span_error(
                        crate::trace::SpanErrorKind::Lm,
                        err.to_string(),
                        Vec::new(),
                        None,
                        LmUsage::default(),
                    ));
                }
                return Err(RunError::Lm {
                    at: at.into(),
                    source: LmError::Provider {
                        provider: lm.config.model.clone(),
                        message: err.to_string(),
                        source: None,
                    },
                });
            }
        };
        cx.meter.record_usage(&response.usage);

        let raw = response.output.content();
        match ChatAdapter.parse_output_def(def, &p.types, &response.output) {
            Ok((output, metas)) => {
                if let Some(leaves) = cx.leaves.as_mut() {
                    leaves.push(LeafOutcome {
                        name: at.clone(),
                        raw_response: raw.clone(),
                        field_meta: metas,
                        usage: response.usage,
                        model_config_hash: crate::trace::ModelEntry::from_config(&lm.config)
                            .config_hash,
                    });
                }
                if let Some(guard) = guard {
                    guard.finish(SpanOutcome {
                        events: response.events,
                        raw_output: Some(raw),
                        output: Some(output.clone()),
                        usage: response.usage,
                        error: None,
                    });
                }
                Ok(output)
            }
            Err(err) => {
                if let Some(guard) = guard {
                    guard.finish(span_error(
                        crate::trace::SpanErrorKind::Parse,
                        err.to_string(),
                        response.events,
                        Some(raw.clone()),
                        response.usage,
                    ));
                }
                Err(RunError::Parse { at: at.into(), raw })
            }
        }
    }

    async fn eval_hole(&self, id: NodeId, n: &HoleNode, cx: &mut Cx) -> Result<JsonMap, RunError> {
        let p = &*self.program;
        let at = p.syms.get(n.name).to_string();
        let def = &p.sigs[n.sig];
        let input = self.resolve_bindings(&at, Some(id), &n.binding, cx)?;

        // Resolve the implementation up front — its hash is part of the
        // replay preimage (RFC 0003 §4.4).
        let imp = match &n.imp {
            HoleImpl::Sandboxed { code } => match self.p_value(cx, *code) {
                ParamValue::Code { source, hash, .. } => ResolvedHoleImpl::Sandboxed {
                    source: source.clone(),
                    hash: *hash,
                },
                other => {
                    return Err(RunError::Internal {
                        at: at.into(),
                        message: format!("code slot resolved to {:?}", other.kind()),
                    });
                }
            },
            HoleImpl::Host { hash } => ResolvedHoleImpl::Host { hash: *hash },
        };
        let request_hash = hole_request_hash(&imp, &input, &n.caps);

        let pseudo_model = LMConfig {
            model: match &imp {
                ResolvedHoleImpl::Sandboxed { .. } => "sandbox:quickjs".to_string(),
                ResolvedHoleImpl::Host { .. } => "host:extern".to_string(),
            },
            ..LMConfig::default()
        };
        let guard = begin_span(SpanRequest {
            component: &at,
            prefix: None,
            suffix: &[],
            input: Some(input.clone()),
            model: &pseudo_model,
            request_hash: Some(request_hash),
        });

        // Replay (RFC 0003 M-1): a hole keys on (impl ++ input ++ caps); a
        // served span returns the recorded, already-coerced output without
        // touching the sandbox or the host fn.
        match crate::trace::replay::intercept_hashed(&at, request_hash) {
            Some(crate::trace::replay::ReplayDirective::Serve(span)) => {
                let output = span
                    .output
                    .clone()
                    .expect("replay serves only spans with parsed output");
                if let Some(guard) = guard {
                    guard.finish(SpanOutcome {
                        events: span.events.clone(),
                        raw_output: span.raw_output.clone(),
                        output: Some(output.clone()),
                        usage: span.usage,
                        error: None,
                    });
                }
                return Ok(output);
            }
            Some(crate::trace::replay::ReplayDirective::Refuse(err)) => {
                if let Some(guard) = guard {
                    guard.finish(span_error(
                        crate::trace::SpanErrorKind::Tool,
                        err.to_string(),
                        Vec::new(),
                        None,
                        LmUsage::default(),
                    ));
                }
                return Err(RunError::Replay {
                    at: at.into(),
                    source: err,
                });
            }
            Some(crate::trace::replay::ReplayDirective::Live) | None => {}
        }

        let started = Instant::now();
        let result = self.execute_hole_impl(&at, &imp, &input, def).await;
        let duration_us = started.elapsed().as_micros() as u64;

        match result {
            Ok((invoked_as, value)) => {
                let raw = value.to_string();
                let event = SpanEvent::ToolRun {
                    id: String::new(),
                    name: invoked_as,
                    args: Value::Object(input.clone()),
                    result: raw.clone(),
                    duration_us,
                    error: None,
                };
                match coerce_outputs(&at, def, &p.types, &value) {
                    Ok(output) => {
                        if let Some(guard) = guard {
                            guard.finish(SpanOutcome {
                                events: vec![event],
                                raw_output: Some(raw),
                                output: Some(output.clone()),
                                usage: LmUsage::default(),
                                error: None,
                            });
                        }
                        Ok(output)
                    }
                    Err(err) => {
                        if let Some(guard) = guard {
                            guard.finish(span_error(
                                crate::trace::SpanErrorKind::Parse,
                                err.to_string(),
                                vec![event],
                                Some(raw),
                                LmUsage::default(),
                            ));
                        }
                        Err(err)
                    }
                }
            }
            Err(err) => {
                if let Some(guard) = guard {
                    guard.finish(span_error(
                        crate::trace::SpanErrorKind::Tool,
                        err.to_string(),
                        Vec::new(),
                        None,
                        LmUsage::default(),
                    ));
                }
                Err(err)
            }
        }
    }

    /// Executes a hole's resolved implementation. Returns the name it was
    /// invoked as (the registered sandbox tool name, or the leaf name for
    /// host holes) and the raw JSON result.
    async fn execute_hole_impl(
        &self,
        at: &str,
        imp: &ResolvedHoleImpl,
        input: &JsonMap,
        def: &SignatureDef,
    ) -> Result<(String, Value), RunError> {
        match imp {
            ResolvedHoleImpl::Sandboxed { source, hash } => {
                // The capability gate is constructive (§3.5): caps were
                // checked at build/load and the sandbox only carries granted
                // host functions — undeclared authority is unreachable here.
                let sandbox = self.sandbox.as_ref().ok_or_else(|| RunError::Internal {
                    at: at.into(),
                    message: "hole evaluated without a sandbox (load should have refused)".into(),
                })?;
                let name = self
                    .ensure_registered(at, sandbox.as_ref(), source, *hash, def)
                    .await?;
                let value = sandbox
                    .execute(dsrs_tools::ToolInvocation::new(
                        name.clone(),
                        Value::Object(input.clone()),
                    ))
                    .await
                    .map_err(|source| RunError::Hole {
                        at: at.into(),
                        source,
                    })?;
                Ok((name, value))
            }
            ResolvedHoleImpl::Host { .. } => {
                let host = self.host_holes.get(at).ok_or_else(|| RunError::Internal {
                    at: at.into(),
                    message: "host hole not bound (load should have refused)".into(),
                })?;
                let value = host(input.clone()).await.map_err(|message| RunError::Hole {
                    at: at.into(),
                    source: dsrs_tools::ExecError::Internal { message },
                })?;
                Ok((at.to_string(), value))
            }
        }
    }

    async fn eval_agent(
        &self,
        id: NodeId,
        n: &AgentLoopNode,
        cx: &mut Cx,
    ) -> Result<JsonMap, RunError> {
        let p = &*self.program;
        let at = p.syms.get(n.name).to_string();
        let def = &p.sigs[n.sig];
        let input = self.resolve_bindings(&at, Some(id), &n.binding, cx)?;

        let instruction = self.p_text(cx, n.instruction);
        let demos = self.p_demos(cx, n.demos);
        let policy = self.p_context(cx, n.context_policy);
        let (prefix, mut suffix) = render_prompt(
            def,
            &p.types,
            &instruction,
            &demos,
            &input,
            policy.playbook.as_deref(),
        );
        if let Some(feedback) = cx.feedback.take() {
            suffix.push(Message::user(feedback));
        }
        let lm = self.p_model(&at, cx, n.model)?;

        // Tool surface: definitions from declared signatures with
        // overlay-resolved descriptions (ToolDesc is a first-class gene), and
        // overlay-resolved code for sandboxed tools.
        let mut definitions = Vec::with_capacity(n.tools.len());
        let mut by_name: HashMap<String, ToolId> = HashMap::new();
        let mut sandbox_code: HashMap<ToolId, (String, u64)> = HashMap::new();
        for &tool_id in n.tools.iter() {
            let tool = &p.tools[tool_id];
            let name = p.syms.get(tool.name).to_string();
            definitions.push(rig::completion::ToolDefinition {
                name: name.clone(),
                description: self.p_text(cx, tool.desc),
                parameters: input_schema_of(&p.sigs[tool.sig], &p.types),
            });
            by_name.insert(name, tool_id);
            if let ToolKind::Sandboxed { code } = tool.kind
                && let ParamValue::Code { source, hash, .. } = self.p_value(cx, code)
            {
                sandbox_code.insert(tool_id, (source.clone(), *hash));
            }
        }
        let stop_names: Vec<String> = n
            .stop
            .stop_tools
            .iter()
            .map(|&t| p.syms.get(p.tools[t].name).to_string())
            .collect();
        // Code Mode: collapse the non-stop tool surface into one `run_js`
        // definition (stop tools stay individual — the loop must see their
        // calls by name to end).
        #[cfg(feature = "code-mode")]
        let code_mode = self
            .build_code_mode_surface(&at, n, &mut definitions, &sandbox_code, &stop_names)
            .await?;
        let toolset = ToolSet::from_definitions(definitions);

        let meter = Arc::new(BudgetMeter::child(&cx.meter, node_budget(&n.budget)));

        let guard = begin_span(SpanRequest {
            component: &at,
            prefix: Some(&prefix),
            suffix: &suffix,
            input: Some(input.clone()),
            model: &lm.config,
            request_hash: None,
        });

        let prefix_len = prefix.len();
        let mut messages = prefix;
        messages.extend(suffix);
        let chat = Chat::new(messages);

        // Replay (RFC 0003 M-1): an agent loop is one span keyed on its
        // opening prompt; serving it returns the recorded final output with
        // every tool effect baked in — nothing re-executes.
        match crate::trace::replay::intercept(&at, &lm.config, &chat.messages) {
            Some(crate::trace::replay::ReplayDirective::Serve(span)) => {
                let output = span
                    .output
                    .clone()
                    .expect("replay serves only spans with parsed output");
                if let Some(guard) = guard {
                    guard.finish(SpanOutcome {
                        events: span.events.clone(),
                        raw_output: span.raw_output.clone(),
                        output: Some(output.clone()),
                        usage: span.usage,
                        error: None,
                    });
                }
                return Ok(output);
            }
            Some(crate::trace::replay::ReplayDirective::Refuse(err)) => {
                if let Some(guard) = guard {
                    guard.finish(span_error(
                        crate::trace::SpanErrorKind::Lm,
                        err.to_string(),
                        Vec::new(),
                        None,
                        LmUsage::default(),
                    ));
                }
                return Err(RunError::Replay {
                    at: at.into(),
                    source: err,
                });
            }
            Some(crate::trace::replay::ReplayDirective::Live) | None => {}
        }

        let loop_cx = AgentLoopCx {
            at: &at,
            n,
            def,
            lm: &lm,
            toolset: &toolset,
            by_name: &by_name,
            sandbox_code: &sandbox_code,
            stop_names: &stop_names,
            prefix_len,
            meter: &meter,
            run_meter: &cx.meter,
            policy: &policy,
            #[cfg(feature = "code-mode")]
            code_mode: code_mode.as_ref(),
        };
        let mut events: Vec<SpanEvent> = Vec::new();
        let mut usage = LmUsage::default();
        let outcome = self
            .agent_loop(&loop_cx, chat, &mut events, &mut usage)
            .await;

        match outcome {
            Ok((output, raw)) => {
                if let Some(guard) = guard {
                    guard.finish(SpanOutcome {
                        events,
                        raw_output: Some(raw),
                        output: Some(output.clone()),
                        usage,
                        error: None,
                    });
                }
                Ok(output)
            }
            Err(err) => {
                if let Some(guard) = guard {
                    let kind = match &err {
                        RunError::Parse { .. } => crate::trace::SpanErrorKind::Parse,
                        RunError::Tool { .. } => crate::trace::SpanErrorKind::Tool,
                        _ => crate::trace::SpanErrorKind::Lm,
                    };
                    guard.finish(span_error(kind, err.to_string(), events, None, usage));
                }
                Err(err)
            }
        }
    }

    async fn agent_loop(
        &self,
        lc: &AgentLoopCx<'_>,
        mut chat: Chat,
        events: &mut Vec<SpanEvent>,
        usage: &mut LmUsage,
    ) -> Result<(JsonMap, String), RunError> {
        let types = &self.program.types;

        for turn in 0..lc.n.stop.max_turns.get() {
            if lc.meter.try_reserve_call().is_err() {
                return match lc.n.budget.on_exhausted {
                    BudgetPolicy::Fail => Err(RunError::Budget { at: lc.at.into() }),
                    BudgetPolicy::Finalize => self.finalize(lc, chat, events, usage).await,
                };
            }

            chat = truncate_history(chat, lc.prefix_len, lc.policy);
            let response = lm_call_toolset(lc.lm, chat, lc.toolset, lc.at).await?;
            lc.meter.record_usage(&response.usage);
            *usage = *usage + response.usage;
            events.extend(response.events.clone());
            chat = response.chat;

            if !response.tool_calls.is_empty() {
                // Stop-tool check: its args become the raw final output.
                if let Some(call) = response
                    .tool_calls
                    .iter()
                    .find(|call| lc.stop_names.contains(&call.function.name))
                {
                    let args = call.function.arguments.clone();
                    let output = coerce_outputs(lc.at, lc.def, types, &args)?;
                    return Ok((output, args.to_string()));
                }

                let mut blocks = Vec::with_capacity(response.tool_calls.len());
                for call in &response.tool_calls {
                    let started = Instant::now();
                    let (result, error) = self.execute_agent_tool(lc, call).await;
                    events.push(SpanEvent::ToolRun {
                        id: call.id.clone(),
                        name: call.function.name.clone(),
                        args: call.function.arguments.clone(),
                        result: result.clone(),
                        duration_us: started.elapsed().as_micros() as u64,
                        error,
                    });
                    blocks.push(tool_result_block(call, result));
                }
                chat.push_message(Message::with_content(Role::User, blocks));
                continue;
            }

            // Text turn.
            if lc.n.stop.until_parse {
                let raw = response.output.content();
                match ChatAdapter.parse_output_def(lc.def, types, &response.output) {
                    Ok((output, _)) => return Ok((output, raw)),
                    Err(err) if turn + 1 < lc.n.stop.max_turns.get() => {
                        chat.push_message(Message::user(format!(
                            "Your response could not be parsed: {err}. Respond again, \
                             producing every output field in the required \
                             `[[ ## field ## ]]` format."
                        )));
                    }
                    Err(_) => {
                        return Err(RunError::Parse {
                            at: lc.at.into(),
                            raw,
                        });
                    }
                }
            }
        }

        // Turns exhausted without an accepted answer.
        match lc.n.budget.on_exhausted {
            BudgetPolicy::Fail => Err(RunError::Budget { at: lc.at.into() }),
            BudgetPolicy::Finalize => self.finalize(lc, chat, events, usage).await,
        }
    }

    /// One forced tool-less closing round-trip ("wrap up now"), reserved
    /// against the RUN meter — the node budget is spent by definition.
    async fn finalize(
        &self,
        lc: &AgentLoopCx<'_>,
        mut chat: Chat,
        events: &mut Vec<SpanEvent>,
        usage: &mut LmUsage,
    ) -> Result<(JsonMap, String), RunError> {
        lc.run_meter
            .try_reserve_call()
            .map_err(|_| RunError::Budget { at: lc.at.into() })?;
        chat.push_message(Message::user(
            "Budget exhausted — wrap up now. Produce the final output fields in the required \
             `[[ ## field ## ]]` format, without calling any tools.",
        ));
        let response = lc
            .lm
            .call(chat, Vec::new())
            .await
            .map_err(|err| RunError::Lm {
                at: lc.at.into(),
                source: LmError::Provider {
                    provider: lc.lm.config.model.clone(),
                    message: err.to_string(),
                    source: None,
                },
            })?;
        lc.run_meter.record_usage(&response.usage);
        *usage = *usage + response.usage;
        events.extend(response.events.clone());
        let raw = response.output.content();
        let output = ChatAdapter
            .parse_output_def(lc.def, &self.program.types, &response.output)
            .map_err(|_| RunError::Parse {
                at: lc.at.into(),
                raw: raw.clone(),
            })?
            .0;
        Ok((output, raw))
    }

    /// Executes one agent tool call. Failures are conversational: the error
    /// text goes back to the model (LATM-style repair), never up the tree.
    async fn execute_agent_tool(
        &self,
        lc: &AgentLoopCx<'_>,
        call: &rig::message::ToolCall,
    ) -> (String, Option<String>) {
        #[cfg(feature = "code-mode")]
        let outcome: Result<String, String> = match lc.code_mode {
            Some(surface) if call.function.name == dsrs_tools::RUN_JS_TOOL_NAME => {
                execute_code_mode_script(surface, &call.function.arguments).await
            }
            _ => self.dispatch_agent_tool(lc, call).await,
        };
        #[cfg(not(feature = "code-mode"))]
        let outcome = self.dispatch_agent_tool(lc, call).await;
        let (mut text, error) = match outcome {
            Ok(text) => (text, None),
            Err(message) => (message.clone(), Some(message)),
        };
        if let Some(max) = lc.policy.tool_result_max_bytes {
            let max = max as usize;
            if text.len() > max {
                let mut cut = max;
                while cut > 0 && !text.is_char_boundary(cut) {
                    cut -= 1;
                }
                text.truncate(cut);
                text.push_str("… [truncated]");
            }
        }
        (text, error)
    }

    /// Routes one tool call to its bound executor (host `ToolDyn` or the
    /// sandbox), by declared name.
    async fn dispatch_agent_tool(
        &self,
        lc: &AgentLoopCx<'_>,
        call: &rig::message::ToolCall,
    ) -> Result<String, String> {
        let name = call.function.name.as_str();
        match lc.by_name.get(name) {
            None => Err(format!("Tool '{name}' not found")),
            Some(&tool_id) => match &self.tool_exec[tool_id] {
                Some(ToolExec::Host(tool)) => tool
                    .call(call.function.arguments.to_string())
                    .await
                    .map_err(|err| format!("tool `{name}` failed: {err}")),
                Some(ToolExec::Sandboxed) => self.execute_sandboxed_tool(lc, tool_id, call).await,
                None => Err(format!("tool `{name}` is not bound")),
            },
        }
    }

    async fn execute_sandboxed_tool(
        &self,
        lc: &AgentLoopCx<'_>,
        tool_id: ToolId,
        call: &rig::message::ToolCall,
    ) -> Result<String, String> {
        let sandbox = self
            .sandbox
            .as_ref()
            .ok_or_else(|| "sandbox unavailable".to_string())?;
        let (source, hash) = lc
            .sandbox_code
            .get(&tool_id)
            .ok_or_else(|| "sandboxed tool has no code".to_string())?;
        let tool = &self.program.tools[tool_id];
        let registered_name = self
            .ensure_registered(
                lc.at,
                sandbox.as_ref(),
                source,
                *hash,
                &self.program.sigs[tool.sig],
            )
            .await
            .map_err(|err| err.to_string())?;
        sandbox
            .execute(dsrs_tools::ToolInvocation::new(
                registered_name,
                call.function.arguments.clone(),
            ))
            .await
            .map(|value| value.to_string())
            .map_err(|err| err.to_llm_json())
    }

    /// Builds one agent loop's Code Mode surface: wraps every non-stop tool
    /// as a sandbox [`Capability`](dsrs_tools::Capability) (host tools call
    /// straight through `ToolDyn`; sandboxed tools route through the
    /// executor under their registered names) and replaces their definitions
    /// with a single `run_js` definition whose description is generated from
    /// the *overlay-resolved* tool descriptions — `ToolDesc` genes keep
    /// flowing into the surface the model sees. Returns `None` when code
    /// mode is off or the loop has no non-stop tools.
    #[cfg(feature = "code-mode")]
    async fn build_code_mode_surface(
        &self,
        at: &str,
        n: &AgentLoopNode,
        definitions: &mut Vec<rig::completion::ToolDefinition>,
        sandbox_code: &HashMap<ToolId, (String, u64)>,
        stop_names: &[String],
    ) -> Result<Option<CodeModeSurface>, RunError> {
        let Some(config) = self.code_mode else {
            return Ok(None);
        };
        let p = &*self.program;
        let internal = |message: String| RunError::Internal {
            at: at.into(),
            message,
        };
        let mut kept = Vec::new();
        let mut apis = Vec::new();
        let mut capabilities = Vec::new();
        // `definitions[i]` was built from `n.tools[i]` — same order.
        for (&tool_id, definition) in n.tools.iter().zip(definitions.iter()) {
            if stop_names.contains(&definition.name) {
                kept.push(definition.clone());
                continue;
            }
            // Collisions were refused at load; names here are unique.
            let js_name = dsrs_tools::js_identifier(&definition.name);
            let capability = match &self.tool_exec[tool_id] {
                Some(ToolExec::Host(tool)) => dsrs_tools::Capability::wrap_tool(
                    js_name.clone(),
                    definition.description.clone(),
                    definition.name.clone(),
                    Arc::clone(tool),
                ),
                Some(ToolExec::Sandboxed) => {
                    let sandbox = Arc::clone(self.sandbox.as_ref().ok_or_else(|| {
                        internal("sandbox unavailable (load should have refused)".to_string())
                    })?);
                    let (source, hash) = sandbox_code.get(&tool_id).ok_or_else(|| {
                        internal(format!("sandboxed tool `{}` has no code", definition.name))
                    })?;
                    let registered = self
                        .ensure_registered(
                            at,
                            sandbox.as_ref(),
                            source,
                            *hash,
                            &p.sigs[p.tools[tool_id].sig],
                        )
                        .await?;
                    let tool_name = definition.name.clone();
                    dsrs_tools::Capability::new(
                        js_name.clone(),
                        definition.description.clone(),
                        move |args| {
                            let sandbox = Arc::clone(&sandbox);
                            let registered = registered.clone();
                            let tool_name = tool_name.clone();
                            async move {
                                sandbox
                                    .execute(dsrs_tools::ToolInvocation::new(registered, args))
                                    .await
                                    .map_err(|err| format!("tool `{tool_name}` failed: {err}"))
                            }
                        },
                    )
                }
                None => {
                    return Err(internal(format!(
                        "tool `{}` is not bound (load should have refused)",
                        definition.name
                    )));
                }
            };
            apis.push(dsrs_tools::ToolApi {
                js_name,
                description: definition.description.clone(),
                parameters: definition.parameters.clone(),
            });
            capabilities.push(capability);
        }
        if apis.is_empty() {
            return Ok(None);
        }
        let run_js = rig::completion::ToolDefinition {
            name: dsrs_tools::RUN_JS_TOOL_NAME.to_string(),
            description: dsrs_tools::code_mode_description(&apis),
            parameters: dsrs_tools::run_js_parameters(),
        };
        *definitions = std::iter::once(run_js).chain(kept).collect();
        Ok(Some(CodeModeSurface {
            capabilities,
            config,
        }))
    }

    // -- param + port resolution ----------------------------------------------

    fn p_value<'a>(&'a self, cx: &'a Cx, id: ParamId) -> &'a ParamValue {
        match &cx.overlay {
            Some(overlay) => overlay.resolve(&self.program, id),
            None => &self.program.params[id].default,
        }
    }

    fn p_text(&self, cx: &Cx, id: ParamId) -> String {
        match self.p_value(cx, id) {
            ParamValue::Instruction { text } | ParamValue::ToolDesc { text } => text.clone(),
            other => panic!("text slot resolved to {:?}", other.kind()),
        }
    }

    fn p_demos(&self, cx: &Cx, id: ParamId) -> Vec<DemoRow> {
        match self.p_value(cx, id) {
            ParamValue::Demos { rows } => rows.clone(),
            other => panic!("demos slot resolved to {:?}", other.kind()),
        }
    }

    fn p_context(&self, cx: &Cx, id: ParamId) -> ContextPolicy {
        match self.p_value(cx, id) {
            ParamValue::ContextPolicy { policy } => policy.clone(),
            other => panic!("context slot resolved to {:?}", other.kind()),
        }
    }

    fn p_model(&self, at: &str, cx: &Cx, id: ParamId) -> Result<Arc<LM>, RunError> {
        let model = match self.p_value(cx, id) {
            ParamValue::ModelRef { model } => *model,
            other => {
                return Err(RunError::Internal {
                    at: at.into(),
                    message: format!("model slot resolved to {:?}", other.kind()),
                });
            }
        };
        self.models[model]
            .as_ref()
            .cloned()
            .ok_or_else(|| RunError::Internal {
                at: at.into(),
                message: format!("model {model} not bound (load should have refused)"),
            })
    }

    async fn ensure_registered(
        &self,
        at: &str,
        sandbox: &dyn dsrs_tools::Executor,
        source: &str,
        hash: u64,
        sig: &SignatureDef,
    ) -> Result<String, RunError> {
        {
            let registered = self.registered.lock().await;
            if let Some(name) = registered.get(&hash) {
                return Ok(name.clone());
            }
        }
        let name = register_code(sandbox, at, source, hash, sig, &self.program.types)
            .await
            .map_err(|err| RunError::Hole {
                at: at.into(),
                source: dsrs_tools::ExecError::Internal {
                    message: format!("overlay code failed to register: {err}"),
                },
            })?;
        self.registered.lock().await.insert(hash, name.clone());
        Ok(name)
    }

    fn resolve_bindings(
        &self,
        at: &str,
        node: Option<NodeId>,
        binds: &[Binding],
        cx: &Cx,
    ) -> Result<JsonMap, RunError> {
        let mut map = JsonMap::new();
        for b in binds {
            let name = self.program.syms.get(b.dst).to_string();
            map.insert(name, self.resolve_port(at, &b.src, cx)?);
        }
        // Refine feedback injection targets one leaf's input field.
        if let (Some(node), Some((target, field, value))) = (node, &cx.refine_feedback)
            && *target == node
        {
            map.insert(field.clone(), value.clone());
        }
        Ok(map)
    }

    fn resolve_exports(&self, binds: &[Binding], cx: &Cx) -> Result<JsonMap, RunError> {
        let mut map = JsonMap::new();
        for b in binds {
            let name = self.program.syms.get(b.dst).to_string();
            map.insert(name, self.resolve_port("out", &b.src, cx)?);
        }
        Ok(map)
    }

    fn resolve_port(&self, at: &str, port: &PortRef, cx: &Cx) -> Result<Value, RunError> {
        match port {
            PortRef::Lit(value) => Ok(value.clone()),
            // `^field` and `$.field` share the merged scope frame at runtime:
            // loop iteration frames are (enclosing inputs ⊕ carry), and the
            // v1 shadow rule makes the distinction purely one of intent.
            PortRef::Input(sym) | PortRef::Carried(sym) => {
                let name = self.program.syms.get(*sym);
                Ok(cx
                    .inputs
                    .last()
                    .and_then(|frame| frame.get(name))
                    .cloned()
                    .unwrap_or(Value::Null))
            }
            PortRef::Out { node, field } => {
                let name = self.program.syms.get(*field);
                let frame = cx.frames[*node]
                    .as_ref()
                    .ok_or_else(|| RunError::Internal {
                        at: at.into(),
                        message: format!(
                            "port references {} before it produced output",
                            self.program.node_display(*node)
                        ),
                    })?;
                Ok(frame.get(name).cloned().unwrap_or(Value::Null))
            }
        }
    }
}

/// Borrowed context for one agent-loop invocation — keeps `agent_loop` and
/// its helpers at a sane arity.
struct AgentLoopCx<'a> {
    at: &'a str,
    n: &'a AgentLoopNode,
    def: &'a SignatureDef,
    lm: &'a Arc<LM>,
    toolset: &'a ToolSet,
    by_name: &'a HashMap<String, ToolId>,
    sandbox_code: &'a HashMap<ToolId, (String, u64)>,
    stop_names: &'a [String],
    prefix_len: usize,
    meter: &'a Arc<BudgetMeter>,
    run_meter: &'a Arc<BudgetMeter>,
    policy: &'a ContextPolicy,
    #[cfg(feature = "code-mode")]
    code_mode: Option<&'a CodeModeSurface>,
}

/// One agent loop's Code Mode surface: the wrapped tool capabilities and the
/// sandbox config `run_js` scripts execute under.
#[cfg(feature = "code-mode")]
struct CodeModeSurface {
    capabilities: Vec<dsrs_tools::Capability>,
    config: dsrs_tools::SandboxConfig,
}

/// Executes one `run_js` call against the loop's Code Mode surface. Failures
/// are conversational, like every agent tool failure.
#[cfg(feature = "code-mode")]
async fn execute_code_mode_script(
    surface: &CodeModeSurface,
    args: &Value,
) -> Result<String, String> {
    let Some(code) = args.get("code").and_then(Value::as_str) else {
        return Err(format!(
            "{} requires a string `code` argument",
            dsrs_tools::RUN_JS_TOOL_NAME
        ));
    };
    dsrs_tools::run_script(code, surface.capabilities.clone(), surface.config)
        .await
        .map(|value| value.to_string())
        .map_err(|err| err.to_llm_json())
}

async fn lm_call_toolset(
    lm: &Arc<LM>,
    chat: Chat,
    toolset: &ToolSet,
    at: &str,
) -> Result<crate::core::lm::LMResponse, RunError> {
    lm.call_with_toolset(chat, toolset, ToolLoopMode::CallerManaged)
        .await
        .map_err(|err| RunError::Lm {
            at: at.into(),
            source: LmError::Provider {
                provider: lm.config.model.clone(),
                message: err.to_string(),
                source: None,
            },
        })
}

// ---------------------------------------------------------------------------
// Rendering and coercion helpers
// ---------------------------------------------------------------------------

/// Renders the (prefix, suffix) message split for a leaf call: prefix =
/// system + demo turns, suffix = the live user turn. Instruction and demos
/// arrive overlay-resolved.
fn render_prompt(
    def: &SignatureDef,
    types: &TypeTable,
    instruction: &str,
    demos: &[DemoRow],
    input: &JsonMap,
    playbook: Option<&str>,
) -> (Vec<Message>, Vec<Message>) {
    let adapter = ChatAdapter;
    let mut system = adapter.build_system_def(def, types, Some(instruction));
    if let Some(playbook) = playbook {
        system.push_str("\n\n");
        system.push_str(playbook);
    }
    let mut prefix = Vec::with_capacity(1 + demos.len() * 2);
    prefix.push(Message::system(system));
    for row in demos {
        prefix.push(Message::user(adapter.format_input_def(def, &row.input)));
        prefix.push(Message::assistant(
            adapter.format_output_def(def, &row.output),
        ));
    }
    let suffix = vec![Message::user(adapter.format_input_def(def, input))];
    (prefix, suffix)
}

fn span_error(
    kind: crate::trace::SpanErrorKind,
    message: String,
    events: Vec<SpanEvent>,
    raw_output: Option<String>,
    usage: LmUsage,
) -> SpanOutcome {
    SpanOutcome {
        events,
        raw_output,
        output: None,
        usage,
        error: Some(crate::trace::SpanError { kind, message }),
    }
}

/// Coerces an already-JSON value (hole result, stop-tool args) into the
/// signature's output map. Structural, with the standard widenings.
fn coerce_outputs(
    at: &str,
    def: &SignatureDef,
    types: &TypeTable,
    value: &Value,
) -> Result<JsonMap, RunError> {
    let Some(object) = value.as_object() else {
        return Err(RunError::Parse {
            at: at.into(),
            raw: value.to_string(),
        });
    };
    let mut output = JsonMap::new();
    for field in def.outputs.iter() {
        match object.get(&*field.name) {
            Some(v) if json_matches_type(v, &field.ty, types) => {
                output.insert(field.name.to_string(), v.clone());
            }
            None if field.ty.is_optional() => {
                output.insert(field.name.to_string(), Value::Null);
            }
            _ => {
                return Err(RunError::Parse {
                    at: at.into(),
                    raw: value.to_string(),
                });
            }
        }
    }
    Ok(output)
}

/// Projects a tool/hole signature's *input* side to a JSON Schema object —
/// the declared interface is the schema the model sees. Public because
/// `#[tool]`-generated `rig::tool::Tool` impls build their definitions
/// through it (RFC 0003 M-2).
pub fn input_schema_of(def: &SignatureDef, types: &TypeTable) -> Value {
    let mut properties = serde_json::Map::new();
    let mut required = Vec::new();
    for field in def.inputs.iter() {
        let mut schema = json_schema_of(&field.ty, types);
        if let Some(docs) = &field.docs
            && let Some(obj) = schema.as_object_mut()
        {
            obj.insert("description".to_string(), json!(docs.as_ref()));
        }
        properties.insert(field.lm_name.to_string(), schema);
        if !field.ty.is_optional() {
            required.push(json!(field.lm_name.as_ref()));
        }
    }
    json!({
        "type": "object",
        "properties": properties,
        "required": required,
    })
}

fn json_schema_of(ty: &FieldType, types: &TypeTable) -> Value {
    match ty {
        FieldType::String => json!({"type": "string"}),
        FieldType::Int => json!({"type": "integer"}),
        FieldType::Float => json!({"type": "number"}),
        FieldType::Bool => json!({"type": "boolean"}),
        FieldType::Literal(s) => json!({"type": "string", "const": s}),
        FieldType::List(inner) => json!({"type": "array", "items": json_schema_of(inner, types)}),
        FieldType::Optional(inner) => json_schema_of(inner, types),
        FieldType::Map(_, value) => {
            json!({"type": "object", "additionalProperties": json_schema_of(value, types)})
        }
        FieldType::Class(token) => match types.classes.get(token) {
            Some(def) => {
                let mut properties = serde_json::Map::new();
                let mut required = Vec::new();
                for field in &def.fields {
                    properties.insert(
                        field.rendered_name.clone(),
                        json_schema_of(&field.field_type, types),
                    );
                    if !field.field_type.is_optional() {
                        required.push(json!(field.rendered_name));
                    }
                }
                json!({"type": "object", "properties": properties, "required": required})
            }
            None => json!({"type": "object"}),
        },
        FieldType::Enum(token) => match types.enums.get(token) {
            Some(def) => {
                let values: Vec<&str> = def.values.iter().map(|v| v.name.as_str()).collect();
                json!({"type": "string", "enum": values})
            }
            None => json!({"type": "string"}),
        },
        FieldType::Union(items) => {
            let any_of: Vec<Value> = items.iter().map(|i| json_schema_of(i, types)).collect();
            json!({"anyOf": any_of})
        }
    }
}

fn node_budget(budget: &crate::ir::graph::NodeBudget) -> Budget {
    Budget {
        max_lm_calls: budget.max_lm_calls,
        max_tokens: budget.max_tokens,
        deadline: budget
            .deadline_ms
            .map(|ms| Instant::now() + std::time::Duration::from_millis(ms)),
    }
}

/// `ContextPolicy.max_history_turns`: keep the rendered prefix plus the
/// trailing N conversation messages.
fn truncate_history(chat: Chat, prefix_len: usize, policy: &ContextPolicy) -> Chat {
    let Some(max_turns) = policy.max_history_turns else {
        return chat;
    };
    let max_turns = max_turns as usize;
    let tail = chat.messages.len().saturating_sub(prefix_len);
    if tail <= max_turns {
        return chat;
    }
    let mut messages = chat.messages;
    let mut kept: Vec<Message> = messages.drain(..prefix_len).collect();
    // `messages` is now the conversation tail; keep its last `max_turns`.
    let skip = messages.len() - max_turns;
    kept.extend(messages.into_iter().skip(skip));
    Chat::new(kept)
}

/// Builds a tool-result content block for the conversation.
fn tool_result_block(call: &rig::message::ToolCall, result: String) -> crate::ContentBlock {
    use rig::OneOrMany;
    use rig::message::UserContent;
    let content = match &call.call_id {
        Some(call_id) => UserContent::tool_result_with_call_id(
            call.id.clone(),
            call_id.clone(),
            OneOrMany::one(result.into()),
        ),
        None => UserContent::tool_result(call.id.clone(), OneOrMany::one(result.into())),
    };
    match content {
        UserContent::ToolResult(tool_result) => crate::ContentBlock::tool_result(tool_result),
        _ => unreachable!("tool_result constructors return ToolResult"),
    }
}

/// A hole's implementation, resolved through the overlay (sandboxed code may
/// be a candidate's Code gene; host holes are fixed by their content hash).
enum ResolvedHoleImpl {
    Sandboxed { source: String, hash: u64 },
    Host { hash: u64 },
}

/// The hole replay preimage (RFC 0003 §4.4): impl discriminant ++ impl/code
/// hash ++ canonical input (sorted keys) ++ sorted caps. Holes render no
/// prompt, so the config+prompt hash every other leaf uses would be identical
/// for every hole in every program — this preimage is what makes hole spans
/// individually addressable by replay and divergence detection.
fn hole_request_hash(imp: &ResolvedHoleImpl, input: &JsonMap, caps: &CapSet) -> u64 {
    use std::hash::Hasher as _;
    let mut hasher = crate::utils::hash::StableHasher::new();
    let (discriminant, hash) = match imp {
        ResolvedHoleImpl::Sandboxed { hash, .. } => (0u8, *hash),
        ResolvedHoleImpl::Host { hash } => (1u8, *hash),
    };
    hasher.write_u8(discriminant);
    hasher.write_u64(hash);
    let mut fields: Vec<(&String, &Value)> = input.iter().collect();
    fields.sort_by_key(|(key, _)| *key);
    for (key, value) in fields {
        hasher.write(key.as_bytes());
        hasher.write(value.to_string().as_bytes());
    }
    for cap in caps.iter() {
        hasher.write(cap.as_bytes());
    }
    hasher.finish()
}

/// Registers a code gene in the sandbox under a content-addressed name
/// (`<base>-<hash hex>`), deduplicating against an already-registered tool.
async fn register_code(
    sandbox: &dyn dsrs_tools::Executor,
    base: &str,
    source: &str,
    hash: u64,
    sig: &SignatureDef,
    types: &TypeTable,
) -> Result<String, dsrs_tools::RegisterError> {
    let sanitized: String = base
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .take(40)
        .collect();
    let name = format!("{sanitized}-{hash:016x}");
    if sandbox.tool(&name).is_some() {
        // Content-addressed: same name ⇒ same source hash ⇒ already usable.
        return Ok(name);
    }
    let params = input_schema_of(sig, types);
    let tool_source = dsrs_tools::ToolSource::new(
        name.clone(),
        format!("dsrs code gene `{base}`"),
        params,
        source,
    );
    sandbox.register(tool_source).await?;
    Ok(name)
}

// ---------------------------------------------------------------------------
// Run state
// ---------------------------------------------------------------------------

/// Call-scoped run state; never stored on the interpreter. Concurrent runs
/// (with different overlays) share one `Arc<Program>`.
struct Cx {
    overlay: Option<Arc<Overlay>>,
    meter: Arc<BudgetMeter>,
    /// Completed node outputs — `PortRef::Out` resolution.
    frames: SecondaryMap<NodeId, Option<JsonMap>>,
    /// Scope-input stack (`$.field`): program input at root, merged carry
    /// frames inside loops.
    inputs: Vec<JsonMap>,
    /// Retry corrective turn, consumed by the next leaf evaluation.
    feedback: Option<String>,
    /// Refine feedback injection: (child leaf, input field, value).
    refine_feedback: Option<(NodeId, String, Value)>,
    /// `Some` when the caller asked for per-leaf metadata
    /// ([`Interpreter::run_collecting`]); successful `Predict` leaves push
    /// here in execution order. `None` = plain `run`, zero collection cost.
    leaves: Option<Vec<LeafOutcome>>,
}

impl Cx {
    /// A branch context for `ForkJoin`: shared overlay/meter, snapshotted
    /// frames and inputs, branch-local feedback. Branch-local leaf collection
    /// when the parent collects; merged back in branch order at the join.
    fn branch(&self) -> Cx {
        Cx {
            overlay: self.overlay.clone(),
            meter: Arc::clone(&self.meter),
            frames: self.frames.clone(),
            inputs: self.inputs.clone(),
            feedback: None,
            refine_feedback: None,
            leaves: self.leaves.as_ref().map(|_| Vec::new()),
        }
    }
}
