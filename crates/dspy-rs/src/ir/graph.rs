//! The IR graph core (RFC 0002 §2): entity ids, the [`Interner`], the closed
//! [`Node`] enum, field-level [`Binding`]/[`PortRef`] dataflow, and [`Program`]
//! — arenas over value-level signatures.
//!
//! Nodes form a tree (single parent, single use); fan-in happens through field
//! references, never shared nodes. Leaf nodes (`Predict`, `AgentLoop`, `Hole`)
//! carry a mandatory, program-unique name — the trace component name and the
//! `ParamPath` prefix. Containers are anonymous.
//!
//! # Serialization
//!
//! The `.dsrs` text format (RFC 0002 §4) is the eventual wire form; until it
//! lands (stage IR-5), [`Program`] serializes through a canonical JSON
//! projection ([`serde::Serialize`]/[`Deserialize`]). Deserialization is a
//! *load*: the interner and `ParamPath` index are rebuilt, the program hash is
//! recomputed, and [`Program::validate`] runs — a file that does not validate
//! does not load.

use std::collections::HashMap;

use cranelift_entity::{PrimaryMap, entity_impl};
use serde::{Deserialize, Serialize};

use crate::LMConfig;
use crate::ir::params::{ParamId, ParamSlot, Slot};
use crate::ir::sig::SignatureDef;
use crate::ir::validate::ValidateError;
use crate::typesys::TypeTable;

// ---------------------------------------------------------------------------
// Entity ids
// ---------------------------------------------------------------------------

/// Interned string: node names, field references in bindings, tool names.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Sym(u32);
entity_impl!(Sym, "sym");

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct NodeId(u32);
entity_impl!(NodeId, "n");

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SigId(u32);
entity_impl!(SigId, "sig");

/// Model reference id. Distinct from the trace format's per-trace
/// `trace::ModelId` — this one indexes `Program::models`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ModelId(u32);
entity_impl!(ModelId, "m");

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ToolId(u32);
entity_impl!(ToolId, "t");

// ---------------------------------------------------------------------------
// Interner
// ---------------------------------------------------------------------------

/// Graph-side string interner. Signatures speak strings ([`SignatureDef`] is
/// constructible with zero context); the *graph* speaks [`Sym`]s.
#[derive(Clone, Debug, Default)]
pub struct Interner {
    strings: PrimaryMap<Sym, Box<str>>,
    index: HashMap<Box<str>, Sym>,
}

impl Interner {
    pub fn intern(&mut self, s: &str) -> Sym {
        if let Some(&sym) = self.index.get(s) {
            return sym;
        }
        let sym = self.strings.push(s.into());
        self.index.insert(s.into(), sym);
        sym
    }

    pub fn get(&self, sym: Sym) -> &str {
        &self.strings[sym]
    }

    pub fn lookup(&self, s: &str) -> Option<Sym> {
        self.index.get(s).copied()
    }

    pub fn len(&self) -> usize {
        self.strings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.strings.len() == 0
    }

    /// All interned strings in id order — the serde projection.
    pub(crate) fn as_slice(&self) -> Vec<Box<str>> {
        self.strings.values().cloned().collect()
    }

    /// Rebuilds an interner from a serialized string table, preserving ids.
    /// Duplicate strings are rejected: they would silently remap symbols.
    pub(crate) fn from_slice(strings: Vec<Box<str>>) -> Result<Self, ValidateError> {
        let mut interner = Interner::default();
        for s in strings {
            if interner.index.contains_key(&s) {
                return Err(ValidateError::DuplicateInternedString {
                    string: s.to_string(),
                });
            }
            let sym = interner.strings.push(s.clone());
            interner.index.insert(s, sym);
        }
        Ok(interner)
    }
}

// ---------------------------------------------------------------------------
// Capabilities
// ---------------------------------------------------------------------------

/// Capability names: namespaced, colon-separated (`"net:search"`, `"fs:read"`).
/// `BTreeSet`: set ops happen at load only, never on the hot path.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CapSet(pub std::collections::BTreeSet<Box<str>>);

impl CapSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, cap: &str) {
        self.0.insert(cap.into());
    }

    pub fn contains(&self, cap: &str) -> bool {
        self.0.contains(cap)
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn is_subset(&self, of: &CapSet) -> bool {
        self.0.iter().all(|cap| of.0.contains(cap))
    }

    /// Capabilities in `self` that `of` does not grant.
    pub fn missing_from(&self, of: &CapSet) -> Vec<String> {
        self.0
            .iter()
            .filter(|cap| !of.0.contains(*cap))
            .map(|cap| cap.to_string())
            .collect()
    }

    pub fn iter(&self) -> impl Iterator<Item = &str> {
        self.0.iter().map(|cap| &**cap)
    }
}

impl<'a> FromIterator<&'a str> for CapSet {
    fn from_iter<T: IntoIterator<Item = &'a str>>(iter: T) -> Self {
        Self(iter.into_iter().map(Into::into).collect())
    }
}

// ---------------------------------------------------------------------------
// Models and tools
// ---------------------------------------------------------------------------

/// `LMConfig` is already serde with `api_key` `#[serde(skip)]` — reused
/// verbatim: model entries carry provider/URL/sampling only, never secrets.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelDef {
    /// The `@ref` name (`"fast"`).
    pub name: Box<str>,
    pub config: LMConfig,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolDef {
    pub name: Sym,
    /// The description IS a `ParamSlot` (`ToolDesc`) — tool docs are
    /// first-class genes (`"tool.<name>.desc"`).
    pub desc: ParamId,
    /// Declared interface, same shape as a signature.
    pub sig: SigId,
    /// ⊆ `program.caps`, checked at build/load.
    pub caps: CapSet,
    pub kind: ToolKind,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolKind {
    /// Bound by name from `RuntimeEnv` at load ("extern" — the loading host
    /// must supply a `rig::tool::ToolDyn`). Program is portable, binding isn't.
    Host,
    /// Sandboxed JS carried in the artifact; the source is a `Code`
    /// `ParamSlot` (`"tool.<name>.code"`) — tool *implementations* are
    /// optimizable.
    Sandboxed { code: ParamId },
}

// ---------------------------------------------------------------------------
// Dataflow
// ---------------------------------------------------------------------------

/// One field-level wire: `dst` input field (or exported name) fed from `src`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Binding {
    /// Input field of the owning node / exported field name.
    pub dst: Sym,
    pub src: PortRef,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PortRef {
    /// `$.field` — enclosing scope's input (program input at root; loop
    /// bodies see carry values here too).
    Input(Sym),
    /// `name.field` — output field of an earlier node in scope.
    Out { node: NodeId, field: Sym },
    /// `^field` — previous iteration's carried value (Loop bodies only).
    Carried(Sym),
    /// JSON literal.
    Lit(serde_json::Value),
}

// ---------------------------------------------------------------------------
// Nodes
// ---------------------------------------------------------------------------

/// The closed node vocabulary. `Predict` carries no tools — tool use is
/// [`AgentLoop`]; `cot` is signature sugar, not a node kind.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Node {
    Predict(PredictNode),
    AgentLoop(AgentLoopNode),
    Seq(SeqNode),
    ForkJoin(ForkJoinNode),
    Route(RouteNode),
    Retry(RetryNode),
    Refine(RefineNode),
    Loop(LoopNode),
    Hole(HoleNode),
}

/// One LM call. No tools. `cot` in the surface syntax lowers to a Predict over
/// `sig.augmented_with([reasoning])`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PredictNode {
    pub name: Sym,
    pub sig: SigId,
    /// `ParamKind::Instruction`.
    pub instruction: ParamId,
    /// `ParamKind::Demos`.
    pub demos: ParamId,
    /// `ParamKind::ModelRef`.
    pub model: ParamId,
    pub binding: Box<[Binding]>,
}

/// The LLM+tool loop as the first-class unit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AgentLoopNode {
    pub name: Sym,
    pub sig: SigId,
    pub instruction: ParamId,
    pub demos: ParamId,
    pub model: ParamId,
    pub tools: Box<[ToolId]>,
    /// `ParamKind::ContextPolicy`.
    pub context_policy: ParamId,
    pub stop: StopSpec,
    pub budget: NodeBudget,
    pub binding: Box<[Binding]>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StopSpec {
    /// Mandatory and bounded — the IR has no unbounded loops, anywhere.
    pub max_turns: std::num::NonZeroU32,
    /// Calling one of these tools ends the loop; its args become the raw
    /// final output (the "submit answer" pattern).
    pub stop_tools: Box<[ToolId]>,
    /// Stop when an assistant turn parses as sig outputs (default true).
    pub until_parse: bool,
}

impl Default for StopSpec {
    fn default() -> Self {
        Self {
            max_turns: std::num::NonZeroU32::new(8).unwrap(),
            stop_tools: Box::new([]),
            until_parse: true,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct NodeBudget {
    pub max_lm_calls: Option<u32>,
    pub max_tokens: Option<u64>,
    pub deadline_ms: Option<u64>,
    #[serde(default)]
    pub on_exhausted: BudgetPolicy,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BudgetPolicy {
    /// `RunError::Budget` — default.
    #[default]
    Fail,
    /// One forced final round-trip without tools ("wrap up now"), then parse.
    Finalize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SeqNode {
    pub body: Box<[NodeId]>,
    /// The exported fields of this scope: dst field name -> port.
    pub out: Box<[Binding]>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ForkJoinNode {
    /// Branches run concurrently; all-success or fail-fast (§3.4).
    pub branches: Box<[NodeId]>,
    pub join: Box<[Binding]>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RouteNode {
    /// Must resolve to an Enum-typed (or Literal-union) field.
    pub on: PortRef,
    /// (variant name, arm). Arms must export identical field name/type sets;
    /// that set is the RouteNode's output interface.
    pub arms: Box<[(Sym, NodeId)]>,
    /// Required unless arms cover the enum.
    pub default: Option<NodeId>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RetryNode {
    pub child: NodeId,
    pub max_attempts: std::num::NonZeroU32,
    pub backoff_ms: u32,
    /// On Parse failure, append the parse error as a corrective user turn on
    /// the retry (leaf children only; ignored for containers).
    pub feedback: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RefineNode {
    pub child: NodeId,
    /// A Predict or Hole whose sig outputs at least
    /// `{score: float, feedback: string}`.
    pub judge: NodeId,
    pub threshold: f64,
    pub max_rounds: std::num::NonZeroU32,
    /// Child input field that receives judge feedback on re-run.
    pub feedback_field: Sym,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LoopNode {
    pub body: NodeId,
    pub max_iters: std::num::NonZeroU32,
    /// Bool-typed output port of body; loop continues while true.
    /// `None` = run exactly `max_iters`.
    #[serde(rename = "while")]
    pub while_: Option<PortRef>,
    /// Next-iteration inputs from this iteration's outputs (`^name` ports).
    pub carry: Box<[Binding]>,
    pub out: Box<[Binding]>,
}

/// LACUNA-style typed hole: opaque-but-typed sandboxed code. The optimizer
/// sees a signature and a Code gene; the type system sees a normal node.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HoleNode {
    pub name: Sym,
    pub sig: SigId,
    /// `ParamKind::Code`.
    pub code: ParamId,
    /// ⊆ `program.caps`.
    pub caps: CapSet,
    pub binding: Box<[Binding]>,
}

impl Node {
    /// The leaf name, if this is a leaf node.
    pub fn leaf_name(&self) -> Option<Sym> {
        match self {
            Node::Predict(n) => Some(n.name),
            Node::AgentLoop(n) => Some(n.name),
            Node::Hole(n) => Some(n.name),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Program
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProgramMeta {
    /// `dsrs 1` pragma; this RFC = 1.
    pub format: u32,
    pub name: Box<str>,
    /// Stable hash over the canonical serialized program minus this field and
    /// the lineage block. Overlays, traces, and state artifacts reference it.
    pub program_hash: u64,
    pub lineage: Option<Lineage>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Lineage {
    /// `"gepa-0.3"`.
    pub optimizer: Box<str>,
    /// `"hotpotqa-train@b3:9f2c…"`.
    pub trainset: Box<str>,
    /// `"412 rollouts / $18.40"`.
    pub budget: Box<str>,
    /// Parent `program_hash`, hex.
    pub parent: Option<Box<str>>,
    pub date: Box<str>,
}

/// A loaded IR program: arenas + interner + capability ceiling.
///
/// Owns its signature arena and type table outright — nothing constructed at
/// runtime touches a global cache or leaks (RFC 0002 §1.2 dynamic lane).
#[derive(Clone, Debug)]
pub struct Program {
    pub meta: ProgramMeta,
    pub nodes: PrimaryMap<NodeId, Node>,
    pub sigs: PrimaryMap<SigId, SignatureDef>,
    pub params: PrimaryMap<ParamId, ParamSlot>,
    pub models: PrimaryMap<ModelId, ModelDef>,
    pub tools: PrimaryMap<ToolId, ToolDef>,
    pub types: TypeTable,
    pub syms: Interner,
    /// The program's capability ceiling.
    pub caps: CapSet,
    /// Always a `Seq` in v1 ("main").
    pub root: NodeId,
    /// The program's external interface.
    pub sig: SigId,
    /// `ParamPath` -> `ParamId`, resolved once at build/load; everything after
    /// speaks ids.
    pub(crate) param_index: HashMap<Box<str>, ParamId>,
}

impl Program {
    /// Resolves a canonical `ParamPath` (`"<leaf>.<slot>"`) to its id.
    pub fn param_id(&self, path: &str) -> Option<ParamId> {
        self.param_index.get(path).copied()
    }

    /// The canonical path of a slot.
    pub fn param_path(&self, id: ParamId) -> &str {
        &self.params[id].path
    }

    /// The optimizer contract, item (1): enumerable typed genes.
    pub fn slots(
        &self,
        kind: crate::ir::params::ParamKind,
    ) -> impl Iterator<Item = (ParamId, &ParamSlot)> {
        self.params
            .iter()
            .filter(move |(_, slot)| slot.kind == kind)
    }

    /// Kind-checked typed slot handle: `None` when the path is unknown or the
    /// slot has a different kind.
    pub fn slot_of<K: crate::ir::params::KindTag>(&self, path: &str) -> Option<Slot<K>> {
        let id = self.param_id(path)?;
        (self.params[id].kind == K::KIND).then(|| Slot::new(id))
    }

    /// The leaf name of a node, if it is a leaf.
    pub fn leaf_name(&self, id: NodeId) -> Option<&str> {
        self.nodes[id].leaf_name().map(|sym| self.syms.get(sym))
    }

    /// Rebuilds the `ParamPath` index (deserialization / builder finish).
    pub(crate) fn rebuild_param_index(&mut self) -> Result<(), ValidateError> {
        self.param_index.clear();
        for (id, slot) in self.params.iter() {
            if self
                .param_index
                .insert(slot.path.clone(), id)
                .is_some()
            {
                return Err(ValidateError::DuplicateParamPath {
                    path: slot.path.to_string(),
                });
            }
        }
        Ok(())
    }

    /// The canonical content hash: stable hash over the canonical JSON
    /// projection minus `program_hash` and `lineage`.
    ///
    /// Until the `.dsrs` text format lands (IR-5), the preimage is the
    /// canonical JSON serialization rather than the printed text.
    pub fn compute_hash(&self) -> u64 {
        let mut data = ProgramData::from_program(self);
        data.meta.program_hash = 0;
        data.meta.lineage = None;
        crate::optimizer::engine::canonical_hash(&data)
    }

    /// Stamps `meta.program_hash` from the current content.
    pub(crate) fn seal(&mut self) {
        self.meta.program_hash = self.compute_hash();
    }
}

// ---------------------------------------------------------------------------
// Serde projection
// ---------------------------------------------------------------------------

/// The serde mirror of [`Program`]: same arenas, interner flattened to its
/// string table, no derived index. Deserializing a `Program` goes through
/// this + [`Program::try_from`], which rebuilds indexes, recomputes the hash,
/// and validates.
#[derive(Serialize, Deserialize)]
pub(crate) struct ProgramData {
    pub meta: ProgramMeta,
    pub nodes: PrimaryMap<NodeId, Node>,
    pub sigs: PrimaryMap<SigId, SignatureDef>,
    pub params: PrimaryMap<ParamId, ParamSlot>,
    pub models: PrimaryMap<ModelId, ModelDef>,
    pub tools: PrimaryMap<ToolId, ToolDef>,
    pub types: TypeTable,
    pub syms: Vec<Box<str>>,
    pub caps: CapSet,
    pub root: NodeId,
    pub sig: SigId,
}

impl ProgramData {
    fn from_program(p: &Program) -> Self {
        Self {
            meta: p.meta.clone(),
            nodes: p.nodes.clone(),
            sigs: p.sigs.clone(),
            params: p.params.clone(),
            models: p.models.clone(),
            tools: p.tools.clone(),
            types: p.types.clone(),
            syms: p.syms.as_slice(),
            caps: p.caps.clone(),
            root: p.root,
            sig: p.sig,
        }
    }
}

impl Serialize for Program {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        ProgramData::from_program(self).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Program {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let data = ProgramData::deserialize(deserializer)?;
        Program::try_from(data).map_err(serde::de::Error::custom)
    }
}

impl TryFrom<ProgramData> for Program {
    type Error = ValidateError;

    /// The load path: rebuild the interner and param index, recompute the
    /// content hash, and validate. Parsing constructs data; nothing here
    /// executes code.
    fn try_from(data: ProgramData) -> Result<Self, ValidateError> {
        let syms = Interner::from_slice(data.syms)?;
        let mut program = Program {
            meta: data.meta,
            nodes: data.nodes,
            sigs: data.sigs,
            params: data.params,
            models: data.models,
            tools: data.tools,
            types: data.types,
            syms,
            caps: data.caps,
            root: data.root,
            sig: data.sig,
            param_index: HashMap::new(),
        };
        program.rebuild_param_index()?;
        program.seal();
        program.validate()?;
        Ok(program)
    }
}
