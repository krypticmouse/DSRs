//! The Rust builder frontend (RFC 0002 §4.3–4.4): constructs the same runtime
//! [`Program`] value the text parser will. `to_ir()` is **total by
//! construction** — the API has no method that accepts a closure, a `dyn Tool`
//! without a [`ToolDef`](crate::ir::ToolDef), an arbitrary `Module`, or an
//! unbounded loop; the lossy inputs are unrepresentable.
//!
//! ```no_run
//! use dspy_rs::ir::{self, FieldType as T, SignatureDef};
//! use dspy_rs::LMConfig;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut b = ir::ProgramBuilder::new("qa");
//! let fast = b.model("fast", LMConfig::default());
//! let qa = b.sig(
//!     SignatureDef::build("QA")
//!         .input("question", T::String)
//!         .output("answer", T::String)
//!         .finish()?,
//! );
//! let answerer = ir::predict("answerer", qa)
//!     .model(fast)
//!     .bind("question", ir::input("question"));
//! let program = b.main(qa, ir::seq([answerer]).out("answer", ir::out("answerer", "answer")))?;
//! # Ok(())
//! # }
//! ```
//!
//! Ports are *name-based* at the builder surface (`ir::out("drafter",
//! "answer")` ≙ the text form `drafter.answer`) and resolve to entity ids at
//! [`ProgramBuilder::main`] — a dangling reference is a [`BuildError`], never
//! a panic.

use std::collections::HashMap;
use std::num::NonZeroU32;

use cranelift_entity::{EntityRef, PrimaryMap};

use crate::LMConfig;
use crate::core::Signature;
use crate::ir::graph::{
    AgentLoopNode, Binding, CapSet, ForkJoinNode, HoleImpl, HoleNode, Interner, LoopNode, ModelDef,
    ModelId,
    Node, NodeBudget, NodeId, PortRef, PredictNode, Program, ProgramMeta, RefineNode, RetryNode,
    RouteNode, SeqNode, SigId, StopSpec, ToolDef, ToolId, ToolKind,
};
use crate::ir::params::{
    CodeLang, ContextPolicy, DemoRow, ParamId, ParamKind, ParamOwner, ParamSlot, ParamValue,
    RenderMode,
};
use crate::ir::sig::{FieldDef, SignatureDef};
use crate::ir::validate::ValidateError;
use crate::typesys::FieldType;

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum BuildError {
    #[error("port references unknown node `{name}` (dangling PortRef)")]
    UnknownNode { name: String },
    #[error("`{at}` has no model: set one explicitly or declare exactly one program model")]
    MissingModel { at: String },
    #[error("duplicate step name `{name}`")]
    DuplicateStepName { name: String },
    #[error(transparent)]
    Invalid(#[from] ValidateError),
}

// ---------------------------------------------------------------------------
// Ports
// ---------------------------------------------------------------------------

/// A name-based port, resolved to a [`PortRef`] at `main()`.
#[derive(Clone, Debug)]
pub enum Port {
    /// `$.field`.
    Input(String),
    /// `node.field`.
    Out { node: String, field: String },
    /// `^field`.
    Carried(String),
    /// JSON literal.
    Lit(serde_json::Value),
}

/// `$.field` — the enclosing scope's input.
pub fn input(field: &str) -> Port {
    Port::Input(field.to_string())
}

/// `node.field` — an earlier node's output. `node` accepts a name string or a
/// leaf spec (anything [`AsNodeName`]).
pub fn out(node: impl AsNodeName, field: &str) -> Port {
    Port::Out {
        node: node.node_name().to_string(),
        field: field.to_string(),
    }
}

/// `^field` — the previous iteration's carried value (Loop bodies only).
pub fn carried(field: &str) -> Port {
    Port::Carried(field.to_string())
}

/// A JSON literal port.
pub fn lit(value: impl Into<serde_json::Value>) -> Port {
    Port::Lit(value.into())
}

/// Anything that names a node: a string, or a leaf/step spec.
pub trait AsNodeName {
    fn node_name(&self) -> &str;
}
impl AsNodeName for &str {
    fn node_name(&self) -> &str {
        self
    }
}
impl AsNodeName for String {
    fn node_name(&self) -> &str {
        self
    }
}
impl AsNodeName for &NodeSpec {
    fn node_name(&self) -> &str {
        self.step_name().unwrap_or("")
    }
}

// ---------------------------------------------------------------------------
// Node specs
// ---------------------------------------------------------------------------

/// An unregistered node: the builder-side mirror of [`Node`] with name-based
/// ports and inline children. Constructed by [`predict`], [`cot`], [`agent`],
/// [`hole`], [`seq`], [`fork`], [`route`], [`retry`], [`refine`], [`loop_`].
#[derive(Clone, Debug)]
pub struct NodeSpec {
    kind: SpecKind,
    /// Optional step name for containers (leaves carry their own names).
    name: Option<String>,
}

#[derive(Clone, Debug)]
enum SpecKind {
    Predict {
        name: String,
        sig: SigId,
        cot: bool,
        model: Option<ModelId>,
        instruction: Option<String>,
        demos: Vec<DemoRow>,
        render: RenderMode,
        binds: Vec<(String, Port)>,
    },
    Agent {
        name: String,
        sig: SigId,
        model: Option<ModelId>,
        instruction: Option<String>,
        demos: Vec<DemoRow>,
        tools: Vec<ToolId>,
        /// The `tool_set` gene's default selection; `None` = the full
        /// declared `tools` list.
        tool_set: Option<Vec<ToolId>>,
        stop_tools: Vec<ToolId>,
        max_turns: NonZeroU32,
        until_parse: bool,
        budget: NodeBudget,
        context: ContextPolicy,
        binds: Vec<(String, Port)>,
    },
    Hole {
        name: String,
        sig: SigId,
        imp: HoleSpecImpl,
        caps: Vec<String>,
        binds: Vec<(String, Port)>,
    },
    Seq {
        body: Vec<NodeSpec>,
        out: Vec<(String, Port)>,
    },
    Fork {
        branches: Vec<NodeSpec>,
        join: Vec<(String, Port)>,
    },
    Route {
        on: Port,
        arms: Vec<(String, NodeSpec)>,
        default: Option<Box<NodeSpec>>,
    },
    Retry {
        child: Box<NodeSpec>,
        max_attempts: NonZeroU32,
        backoff_ms: u32,
        feedback: bool,
    },
    Refine {
        child: Box<NodeSpec>,
        judge: Box<NodeSpec>,
        threshold: f64,
        max_rounds: NonZeroU32,
        feedback_field: String,
    },
    Loop {
        body: Box<NodeSpec>,
        max_iters: NonZeroU32,
        while_: Option<Port>,
        carry: Vec<(String, Port)>,
        out: Vec<(String, Port)>,
    },
}

impl NodeSpec {
    /// The leaf name (for leaves) or step name (for named containers).
    pub fn step_name(&self) -> Option<&str> {
        match &self.kind {
            SpecKind::Predict { name, .. }
            | SpecKind::Agent { name, .. }
            | SpecKind::Hole { name, .. } => Some(name),
            _ => self.name.as_deref(),
        }
    }

    /// Names a container step so later nodes can reference its exports
    /// (`name.field`). Leaves already carry their own names.
    pub fn named(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }

    /// Binds an input field (leaves only; no-op recorded for containers is
    /// rejected at `main()` by validation).
    pub fn bind(mut self, field: &str, port: Port) -> Self {
        match &mut self.kind {
            SpecKind::Predict { binds, .. }
            | SpecKind::Agent { binds, .. }
            | SpecKind::Hole { binds, .. } => binds.push((field.to_string(), port)),
            _ => panic!("bind() applies to leaf specs (predict/cot/agent/hole)"),
        }
        self
    }

    /// Overrides the instruction (Predict/Agent). Default: the signature's.
    pub fn instruction(mut self, text: &str) -> Self {
        match &mut self.kind {
            SpecKind::Predict { instruction, .. } | SpecKind::Agent { instruction, .. } => {
                *instruction = Some(text.to_string());
            }
            _ => panic!("instruction() applies to predict/cot/agent specs"),
        }
        self
    }

    /// Seeds demo rows (Predict/Agent).
    pub fn demos(mut self, rows: Vec<DemoRow>) -> Self {
        match &mut self.kind {
            SpecKind::Predict { demos, .. } | SpecKind::Agent { demos, .. } => *demos = rows,
            _ => panic!("demos() applies to predict/cot/agent specs"),
        }
        self
    }

    /// Sets the render mode's default (Predict only): the marker protocol
    /// vs. bare rendering. Bare is refused by validation unless the leaf's
    /// signature has exactly one non-optional `String` output.
    pub fn render(mut self, mode: RenderMode) -> Self {
        match &mut self.kind {
            SpecKind::Predict { render, .. } => *render = mode,
            _ => panic!("render() applies to predict/cot specs"),
        }
        self
    }

    /// Sets the model ref (Predict/Agent).
    pub fn model(mut self, model: ModelId) -> Self {
        match &mut self.kind {
            SpecKind::Predict { model: slot, .. } | SpecKind::Agent { model: slot, .. } => {
                *slot = Some(model);
            }
            _ => panic!("model() applies to predict/cot/agent specs"),
        }
        self
    }

    /// Declares the agent's tools.
    pub fn tools(mut self, ids: impl IntoIterator<Item = ToolId>) -> Self {
        match &mut self.kind {
            SpecKind::Agent { tools, .. } => tools.extend(ids),
            _ => panic!("tools() applies to agent specs"),
        }
        self
    }

    /// Seeds the `tool_set` gene: which declared tools the loop carries at
    /// run time. Default (absent): the full declared `tools` list. Must be a
    /// subset of the declared tools — validation refuses anything else.
    pub fn tool_set(mut self, ids: impl IntoIterator<Item = ToolId>) -> Self {
        match &mut self.kind {
            SpecKind::Agent { tool_set, .. } => *tool_set = Some(ids.into_iter().collect()),
            _ => panic!("tool_set() applies to agent specs"),
        }
        self
    }

    /// Declares stop tools — calling one ends the loop, its args become the
    /// raw final output.
    pub fn stop_tools(mut self, ids: impl IntoIterator<Item = ToolId>) -> Self {
        match &mut self.kind {
            SpecKind::Agent { stop_tools, .. } => stop_tools.extend(ids),
            _ => panic!("stop_tools() applies to agent specs"),
        }
        self
    }

    /// Bounds the agent loop (mandatory; default 8).
    pub fn max_turns(mut self, turns: u32) -> Self {
        match &mut self.kind {
            SpecKind::Agent { max_turns, .. } => {
                *max_turns = NonZeroU32::new(turns).expect("max_turns must be > 0");
            }
            _ => panic!("max_turns() applies to agent specs"),
        }
        self
    }

    /// Whether a parseable assistant turn ends the loop (default true).
    pub fn until_parse(mut self, value: bool) -> Self {
        match &mut self.kind {
            SpecKind::Agent { until_parse, .. } => *until_parse = value,
            _ => panic!("until_parse() applies to agent specs"),
        }
        self
    }

    /// Node-level budget (Agent).
    pub fn budget(mut self, budget: NodeBudget) -> Self {
        match &mut self.kind {
            SpecKind::Agent { budget: slot, .. } => *slot = budget,
            _ => panic!("budget() applies to agent specs"),
        }
        self
    }

    /// Context policy (Agent).
    pub fn context(mut self, policy: ContextPolicy) -> Self {
        match &mut self.kind {
            SpecKind::Agent { context, .. } => *context = policy,
            _ => panic!("context() applies to agent specs"),
        }
        self
    }

    /// Exports a field from a `seq`/`loop` scope.
    pub fn out(mut self, field: &str, port: Port) -> Self {
        match &mut self.kind {
            SpecKind::Seq { out, .. } | SpecKind::Loop { out, .. } => {
                out.push((field.to_string(), port));
            }
            _ => panic!("out() applies to seq/loop specs"),
        }
        self
    }

    /// Exports a field from a `fork` join.
    pub fn join(mut self, field: &str, port: Port) -> Self {
        match &mut self.kind {
            SpecKind::Fork { join, .. } => join.push((field.to_string(), port)),
            _ => panic!("join() applies to fork specs"),
        }
        self
    }

    /// Adds a route arm.
    pub fn arm(mut self, variant: &str, node: NodeSpec) -> Self {
        match &mut self.kind {
            SpecKind::Route { arms, .. } => arms.push((variant.to_string(), node)),
            _ => panic!("arm() applies to route specs"),
        }
        self
    }

    /// Sets the route default (`else ->`).
    pub fn default_arm(mut self, node: NodeSpec) -> Self {
        match &mut self.kind {
            SpecKind::Route { default, .. } => *default = Some(Box::new(node)),
            _ => panic!("default_arm() applies to route specs"),
        }
        self
    }

    /// Retry backoff between attempts.
    pub fn backoff_ms(mut self, ms: u32) -> Self {
        match &mut self.kind {
            SpecKind::Retry { backoff_ms, .. } => *backoff_ms = ms,
            _ => panic!("backoff_ms() applies to retry specs"),
        }
        self
    }

    /// On parse failure, feed the error back as a corrective user turn.
    pub fn feedback(mut self, value: bool) -> Self {
        match &mut self.kind {
            SpecKind::Retry { feedback, .. } => *feedback = value,
            _ => panic!("feedback() applies to retry specs"),
        }
        self
    }

    /// Refine acceptance threshold (judge `score` must reach it).
    pub fn threshold(mut self, value: f64) -> Self {
        match &mut self.kind {
            SpecKind::Refine { threshold, .. } => *threshold = value,
            _ => panic!("threshold() applies to refine specs"),
        }
        self
    }

    /// Bounds refine rounds.
    pub fn max_rounds(mut self, rounds: u32) -> Self {
        match &mut self.kind {
            SpecKind::Refine { max_rounds, .. } => {
                *max_rounds = NonZeroU32::new(rounds).expect("max_rounds must be > 0");
            }
            _ => panic!("max_rounds() applies to refine specs"),
        }
        self
    }

    /// Loop condition: continue while this bool port is true.
    pub fn while_(mut self, port: Port) -> Self {
        match &mut self.kind {
            SpecKind::Loop { while_, .. } => *while_ = Some(port),
            _ => panic!("while_() applies to loop specs"),
        }
        self
    }

    /// Loop carry: next-iteration value of `field` from this iteration.
    pub fn carry(mut self, field: &str, port: Port) -> Self {
        match &mut self.kind {
            SpecKind::Loop { carry, .. } => carry.push((field.to_string(), port)),
            _ => panic!("carry() applies to loop specs"),
        }
        self
    }
}

/// The output field `cot` prepends to the base signature (RFC 0002 §4.2: `cot`
/// is signature sugar). The text-format printer detects exactly this field to
/// re-sugar augmented Predicts, so builder and parser must share it.
pub(crate) fn cot_reasoning_field() -> FieldDef {
    FieldDef::new("reasoning", FieldType::String)
        .with_docs("Think step by step to reach the answer.")
}

/// One LM call over `sig`.
pub fn predict(name: &str, sig: SigId) -> NodeSpec {
    NodeSpec {
        kind: SpecKind::Predict {
            name: name.to_string(),
            sig,
            cot: false,
            model: None,
            instruction: None,
            demos: Vec::new(),
            render: RenderMode::Markers,
            binds: Vec::new(),
        },
        name: None,
    }
}

/// Chain-of-thought sugar: a Predict over `sig.augmented_with([reasoning])`.
pub fn cot(name: &str, sig: SigId) -> NodeSpec {
    let mut spec = predict(name, sig);
    if let SpecKind::Predict { cot, .. } = &mut spec.kind {
        *cot = true;
    }
    spec
}

/// The LLM+tool loop.
pub fn agent(name: &str, sig: SigId) -> NodeSpec {
    NodeSpec {
        kind: SpecKind::Agent {
            name: name.to_string(),
            sig,
            model: None,
            instruction: None,
            demos: Vec::new(),
            tools: Vec::new(),
            tool_set: None,
            stop_tools: Vec::new(),
            max_turns: StopSpec::default().max_turns,
            until_parse: true,
            budget: NodeBudget::default(),
            context: ContextPolicy::default(),
            binds: Vec::new(),
        },
        name: None,
    }
}

/// A typed hole: sandboxed JS with a declared signature and capability set.
pub fn hole(name: &str, sig: SigId, js: &str, caps: &[&str]) -> NodeSpec {
    NodeSpec {
        kind: SpecKind::Hole {
            name: name.to_string(),
            sig,
            imp: HoleSpecImpl::Js(js.to_string()),
            caps: caps.iter().map(|c| c.to_string()).collect(),
            binds: Vec::new(),
        },
        name: None,
    }
}

/// An extern (host-backed) typed hole (RFC 0003 §4): a native fn bound by
/// leaf name from the runtime environment at load. `hash` is the stable
/// content hash of the host implementation — it travels in the artifact
/// (`extern "<hex>"`) as the integrity/replay fingerprint.
pub fn extern_hole(name: &str, sig: SigId, hash: u64, caps: &[&str]) -> NodeSpec {
    NodeSpec {
        kind: SpecKind::Hole {
            name: name.to_string(),
            sig,
            imp: HoleSpecImpl::Host(hash),
            caps: caps.iter().map(|c| c.to_string()).collect(),
            binds: Vec::new(),
        },
        name: None,
    }
}

/// Builder-side mirror of [`HoleImpl`].
#[derive(Clone, Debug)]
enum HoleSpecImpl {
    Js(String),
    Host(u64),
}

/// Sequential composition.
pub fn seq(body: impl IntoIterator<Item = NodeSpec>) -> NodeSpec {
    NodeSpec {
        kind: SpecKind::Seq {
            body: body.into_iter().collect(),
            out: Vec::new(),
        },
        name: None,
    }
}

/// Concurrent branches, joined all-success / fail-fast.
pub fn fork(branches: impl IntoIterator<Item = NodeSpec>) -> NodeSpec {
    NodeSpec {
        kind: SpecKind::Fork {
            branches: branches.into_iter().collect(),
            join: Vec::new(),
        },
        name: None,
    }
}

/// Enum-discriminated branching.
pub fn route(on: Port) -> NodeSpec {
    NodeSpec {
        kind: SpecKind::Route {
            on,
            arms: Vec::new(),
            default: None,
        },
        name: None,
    }
}

/// Bounded retry of a child.
pub fn retry(child: NodeSpec, max_attempts: u32) -> NodeSpec {
    NodeSpec {
        kind: SpecKind::Retry {
            child: Box::new(child),
            max_attempts: NonZeroU32::new(max_attempts).expect("max_attempts must be > 0"),
            backoff_ms: 0,
            feedback: false,
        },
        name: None,
    }
}

/// Judge-gated refinement of a child.
pub fn refine(child: NodeSpec, judge: NodeSpec, feedback_field: &str) -> NodeSpec {
    NodeSpec {
        kind: SpecKind::Refine {
            child: Box::new(child),
            judge: Box::new(judge),
            threshold: 1.0,
            max_rounds: NonZeroU32::new(2).unwrap(),
            feedback_field: feedback_field.to_string(),
        },
        name: None,
    }
}

/// Bounded loop.
pub fn loop_(body: NodeSpec, max_iters: u32) -> NodeSpec {
    NodeSpec {
        kind: SpecKind::Loop {
            body: Box::new(body),
            max_iters: NonZeroU32::new(max_iters).expect("max_iters must be > 0"),
            while_: None,
            carry: Vec::new(),
            out: Vec::new(),
        },
        name: None,
    }
}

// ---------------------------------------------------------------------------
// ProgramBuilder
// ---------------------------------------------------------------------------

struct ToolSpec {
    name: String,
    desc: String,
    sig: SigId,
    caps: CapSet,
    js: Option<String>,
}

/// Builds a [`Program`]. Declarations (caps, models, sigs, tools) accumulate;
/// [`main`](ProgramBuilder::main) lowers the node tree, materializes param
/// slots, seals the hash, and validates.
pub struct ProgramBuilder {
    name: String,
    caps: CapSet,
    models: PrimaryMap<ModelId, ModelDef>,
    sigs: PrimaryMap<SigId, SignatureDef>,
    types: crate::typesys::TypeTable,
    tool_specs: Vec<ToolSpec>,
}

impl ProgramBuilder {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            caps: CapSet::new(),
            models: PrimaryMap::new(),
            sigs: PrimaryMap::new(),
            types: crate::typesys::TypeTable::default(),
            tool_specs: Vec::new(),
        }
    }

    /// Adds a capability to the program ceiling.
    pub fn cap(&mut self, cap: &str) -> &mut Self {
        self.caps.insert(cap);
        self
    }

    /// Declares a model (`@name`).
    pub fn model(&mut self, name: &str, config: LMConfig) -> ModelId {
        self.models.push(ModelDef {
            name: name.into(),
            config,
        })
    }

    /// Registers a value-level signature.
    pub fn sig(&mut self, def: SignatureDef) -> SigId {
        self.sigs.push(def)
    }

    /// Static → value lane: registers `S`'s [`SignatureDef`] and merges its
    /// reachable class/enum definitions into the program's type table.
    pub fn sig_of<S: Signature>(&mut self) -> SigId {
        let def = SignatureDef::of::<S>().clone();
        let types = SignatureDef::types_of::<S>();
        for (token, class) in &types.classes {
            self.types
                .classes
                .entry(token.clone())
                .or_insert_with(|| class.clone());
        }
        for (token, enm) in &types.enums {
            self.types
                .enums
                .entry(token.clone())
                .or_insert_with(|| enm.clone());
        }
        self.sigs.push(def)
    }

    /// Merges externally built class/enum definitions (for runtime-only
    /// signatures that reference them).
    pub fn add_types(&mut self, types: &crate::typesys::TypeTable) -> &mut Self {
        for (token, class) in &types.classes {
            self.types
                .classes
                .entry(token.clone())
                .or_insert_with(|| class.clone());
        }
        for (token, enm) in &types.enums {
            self.types
                .enums
                .entry(token.clone())
                .or_insert_with(|| enm.clone());
        }
        self
    }

    /// Declares an extern host tool: bound by name from the runtime
    /// environment at load. The description becomes a `ToolDesc` param.
    pub fn host_tool(&mut self, name: &str, desc: &str, sig: SigId, caps: &[&str]) -> ToolId {
        self.tool_specs.push(ToolSpec {
            name: name.to_string(),
            desc: desc.to_string(),
            sig,
            caps: caps.iter().copied().collect(),
            js: None,
        });
        ToolId::new(self.tool_specs.len() - 1)
    }

    /// Declares a sandboxed tool whose JS source travels in the artifact as a
    /// `Code` param.
    pub fn sandboxed_tool(
        &mut self,
        name: &str,
        desc: &str,
        sig: SigId,
        caps: &[&str],
        js: &str,
    ) -> ToolId {
        self.tool_specs.push(ToolSpec {
            name: name.to_string(),
            desc: desc.to_string(),
            sig,
            caps: caps.iter().copied().collect(),
            js: Some(js.to_string()),
        });
        ToolId::new(self.tool_specs.len() - 1)
    }

    /// Lowers the tree into a validated [`Program`] whose root `Seq` is
    /// `root` and whose external interface is `sig`. Never panics: dangling
    /// ports, type mismatches, cap violations, and structural errors are
    /// [`BuildError`]s.
    pub fn main(self, sig: SigId, root: NodeSpec) -> Result<Program, BuildError> {
        // Root is always a Seq in v1 ("main").
        let root = match root.kind {
            SpecKind::Seq { .. } => root,
            _ => return Err(BuildError::Invalid(ValidateError::RootNotSeq)),
        };

        let mut lower = Lowering {
            nodes: PrimaryMap::new(),
            params: PrimaryMap::new(),
            syms: Interner::default(),
            node_names: HashMap::new(),
            tools: PrimaryMap::new(),
            single_model: (self.models.len() == 1).then(|| ModelId::new(0)),
        };

        // Tools first: their param slots exist regardless of node references.
        for spec in &self.tool_specs {
            let name_sym = lower.syms.intern(&spec.name);
            let tool_id = ToolId::new(lower.tools.len());
            let desc = lower.params.push(ParamSlot {
                path: format!("tool.{}.desc", spec.name).into(),
                owner: ParamOwner::Tool(tool_id),
                kind: ParamKind::ToolDesc,
                default: ParamValue::ToolDesc {
                    text: spec.desc.clone(),
                },
            });
            let kind = match &spec.js {
                None => ToolKind::Host,
                Some(js) => ToolKind::Sandboxed {
                    code: lower.params.push(ParamSlot {
                        path: format!("tool.{}.code", spec.name).into(),
                        owner: ParamOwner::Tool(tool_id),
                        kind: ParamKind::Code,
                        default: ParamValue::code(CodeLang::Js, js.clone()),
                    }),
                },
            };
            lower.tools.push(ToolDef {
                name: name_sym,
                desc,
                sig: spec.sig,
                caps: spec.caps.clone(),
                kind,
            });
        }

        let mut sigs = self.sigs;
        let root_id = lower.lower(root, &mut sigs)?;

        let mut program = Program {
            meta: ProgramMeta {
                format: 1,
                name: self.name.into(),
                program_hash: 0,
                lineage: None,
            },
            nodes: lower.nodes,
            sigs,
            params: lower.params,
            models: self.models,
            tools: lower.tools,
            types: self.types,
            syms: lower.syms,
            caps: self.caps,
            root: root_id,
            sig,
            param_index: HashMap::new(),
        };
        program.rebuild_param_index().map_err(BuildError::Invalid)?;
        // Validate before sealing: the hash preimage is the canonical printed
        // text, and printing assumes structurally valid arenas.
        program.validate().map_err(BuildError::Invalid)?;
        program.seal();
        Ok(program)
    }
}

struct Lowering {
    nodes: PrimaryMap<NodeId, Node>,
    params: PrimaryMap<ParamId, ParamSlot>,
    syms: Interner,
    /// Step/leaf name → lowered node, for name-based `Out` ports. Only nodes
    /// lowered *before* the port's owner are present — a forward reference is
    /// a dangling port, exactly as validation requires.
    node_names: HashMap<String, NodeId>,
    tools: PrimaryMap<ToolId, ToolDef>,
    single_model: Option<ModelId>,
}

impl Lowering {
    fn lower(
        &mut self,
        spec: NodeSpec,
        sigs: &mut PrimaryMap<SigId, SignatureDef>,
    ) -> Result<NodeId, BuildError> {
        let step_name = spec.step_name().map(str::to_string);
        let node = match spec.kind {
            SpecKind::Predict {
                name,
                sig,
                cot,
                model,
                instruction,
                demos,
                render,
                binds,
            } => {
                let sig = if cot {
                    let augmented = sigs[sig].augmented_with(&[cot_reasoning_field()]);
                    sigs.push(augmented)
                } else {
                    sig
                };
                let name_sym = self.syms.intern(&name);
                let node_id = NodeId::new(self.nodes.len());
                let model = model
                    .or(self.single_model)
                    .ok_or_else(|| BuildError::MissingModel { at: name.clone() })?;
                let instruction_text =
                    instruction.unwrap_or_else(|| sigs[sig].instruction.to_string());
                let instruction = self.leaf_param(
                    &name,
                    "instruction",
                    node_id,
                    ParamKind::Instruction,
                    ParamValue::Instruction {
                        text: instruction_text,
                    },
                );
                let demos = self.leaf_param(
                    &name,
                    "demos",
                    node_id,
                    ParamKind::Demos,
                    ParamValue::Demos { rows: demos },
                );
                let model = self.leaf_param(
                    &name,
                    "model",
                    node_id,
                    ParamKind::ModelRef,
                    ParamValue::ModelRef { model },
                );
                let render = self.leaf_param(
                    &name,
                    "render",
                    node_id,
                    ParamKind::Render,
                    ParamValue::Render { mode: render },
                );
                let binding = self.lower_binds(binds)?;
                Node::Predict(PredictNode {
                    name: name_sym,
                    sig,
                    instruction,
                    demos,
                    model,
                    render,
                    binding,
                })
            }
            SpecKind::Agent {
                name,
                sig,
                model,
                instruction,
                demos,
                tools,
                tool_set,
                stop_tools,
                max_turns,
                until_parse,
                budget,
                context,
                binds,
            } => {
                let name_sym = self.syms.intern(&name);
                let node_id = NodeId::new(self.nodes.len());
                let model = model
                    .or(self.single_model)
                    .ok_or_else(|| BuildError::MissingModel { at: name.clone() })?;
                let instruction_text =
                    instruction.unwrap_or_else(|| sigs[sig].instruction.to_string());
                let instruction = self.leaf_param(
                    &name,
                    "instruction",
                    node_id,
                    ParamKind::Instruction,
                    ParamValue::Instruction {
                        text: instruction_text,
                    },
                );
                let demos = self.leaf_param(
                    &name,
                    "demos",
                    node_id,
                    ParamKind::Demos,
                    ParamValue::Demos { rows: demos },
                );
                let model = self.leaf_param(
                    &name,
                    "model",
                    node_id,
                    ParamKind::ModelRef,
                    ParamValue::ModelRef { model },
                );
                let context_policy = self.leaf_param(
                    &name,
                    "context",
                    node_id,
                    ParamKind::ContextPolicy,
                    ParamValue::ContextPolicy { policy: context },
                );
                // The gene defaults to the full declared table: absent
                // selection = every declared tool, so pre-ToolSet programs
                // print, hash, and run unchanged.
                let tool_set = self.leaf_param(
                    &name,
                    "tool_set",
                    node_id,
                    ParamKind::ToolSet,
                    ParamValue::ToolSet {
                        tools: tool_set.unwrap_or_else(|| tools.clone()),
                    },
                );
                let binding = self.lower_binds(binds)?;
                Node::AgentLoop(AgentLoopNode {
                    name: name_sym,
                    sig,
                    instruction,
                    demos,
                    model,
                    tools: tools.into_boxed_slice(),
                    tool_set,
                    context_policy,
                    stop: StopSpec {
                        max_turns,
                        stop_tools: stop_tools.into_boxed_slice(),
                        until_parse,
                    },
                    budget,
                    binding,
                })
            }
            SpecKind::Hole {
                name,
                sig,
                imp,
                caps,
                binds,
            } => {
                let name_sym = self.syms.intern(&name);
                let node_id = NodeId::new(self.nodes.len());
                let imp = match imp {
                    HoleSpecImpl::Js(code) => HoleImpl::Sandboxed {
                        code: self.leaf_param(
                            &name,
                            "code",
                            node_id,
                            ParamKind::Code,
                            ParamValue::code(CodeLang::Js, code),
                        ),
                    },
                    HoleSpecImpl::Host(hash) => HoleImpl::Host { hash },
                };
                let binding = self.lower_binds(binds)?;
                Node::Hole(HoleNode {
                    name: name_sym,
                    sig,
                    imp,
                    caps: caps.iter().map(String::as_str).collect(),
                    binding,
                })
            }
            SpecKind::Seq { body, out } => {
                let mut ids = Vec::with_capacity(body.len());
                for child in body {
                    ids.push(self.lower(child, sigs)?);
                }
                let out = self.lower_binds(out)?;
                Node::Seq(SeqNode {
                    body: ids.into_boxed_slice(),
                    out,
                })
            }
            SpecKind::Fork { branches, join } => {
                let mut ids = Vec::with_capacity(branches.len());
                for branch in branches {
                    ids.push(self.lower(branch, sigs)?);
                }
                let join = self.lower_binds(join)?;
                Node::ForkJoin(ForkJoinNode {
                    branches: ids.into_boxed_slice(),
                    join,
                })
            }
            SpecKind::Route { on, arms, default } => {
                let on = self.lower_port(on)?;
                let mut lowered = Vec::with_capacity(arms.len());
                for (variant, arm) in arms {
                    let variant = self.syms.intern(&variant);
                    let arm = self.lower(arm, sigs)?;
                    lowered.push((variant, arm));
                }
                let default = match default {
                    Some(node) => Some(self.lower(*node, sigs)?),
                    None => None,
                };
                Node::Route(RouteNode {
                    on,
                    arms: lowered.into_boxed_slice(),
                    default,
                })
            }
            SpecKind::Retry {
                child,
                max_attempts,
                backoff_ms,
                feedback,
            } => {
                let child = self.lower(*child, sigs)?;
                Node::Retry(RetryNode {
                    child,
                    max_attempts,
                    backoff_ms,
                    feedback,
                })
            }
            SpecKind::Refine {
                child,
                judge,
                threshold,
                max_rounds,
                feedback_field,
            } => {
                let child = self.lower(*child, sigs)?;
                let judge = self.lower(*judge, sigs)?;
                let feedback_field = self.syms.intern(&feedback_field);
                Node::Refine(RefineNode {
                    child,
                    judge,
                    threshold,
                    max_rounds,
                    feedback_field,
                })
            }
            SpecKind::Loop {
                body,
                max_iters,
                while_,
                carry,
                out,
            } => {
                let body = self.lower(*body, sigs)?;
                let while_ = match while_ {
                    Some(port) => Some(self.lower_port(port)?),
                    None => None,
                };
                let carry = self.lower_binds(carry)?;
                let out = self.lower_binds(out)?;
                Node::Loop(LoopNode {
                    body,
                    max_iters,
                    while_,
                    carry,
                    out,
                })
            }
        };
        let id = self.nodes.push(node);
        if let Some(name) = step_name
            && self.node_names.insert(name.clone(), id).is_some()
        {
            return Err(BuildError::DuplicateStepName { name });
        }
        Ok(id)
    }

    fn leaf_param(
        &mut self,
        leaf: &str,
        slot: &str,
        node: NodeId,
        kind: ParamKind,
        default: ParamValue,
    ) -> ParamId {
        self.params.push(ParamSlot {
            path: format!("{leaf}.{slot}").into(),
            owner: ParamOwner::Node(node),
            kind,
            default,
        })
    }

    fn lower_binds(&mut self, binds: Vec<(String, Port)>) -> Result<Box<[Binding]>, BuildError> {
        binds
            .into_iter()
            .map(|(dst, port)| {
                Ok(Binding {
                    dst: self.syms.intern(&dst),
                    src: self.lower_port(port)?,
                })
            })
            .collect::<Result<Vec<_>, BuildError>>()
            .map(Vec::into_boxed_slice)
    }

    fn lower_port(&mut self, port: Port) -> Result<PortRef, BuildError> {
        Ok(match port {
            Port::Input(field) => PortRef::Input(self.syms.intern(&field)),
            Port::Carried(field) => PortRef::Carried(self.syms.intern(&field)),
            Port::Lit(value) => PortRef::Lit(value),
            Port::Out { node, field } => {
                let id = self
                    .node_names
                    .get(&node)
                    .copied()
                    .ok_or(BuildError::UnknownNode { name: node })?;
                PortRef::Out {
                    node: id,
                    field: self.syms.intern(&field),
                }
            }
        })
    }
}
