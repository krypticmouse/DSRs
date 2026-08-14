//! Parameters (RFC 0002 §2.4): every mutable thing is a named, addressable
//! slot; a candidate is an [`Overlay`] read through at render time, never a
//! mutation.
//!
//! Canonical `ParamPath`s: `"<leaf>.instruction"`, `"<leaf>.demos"`,
//! `"<leaf>.model"`, `"<leaf>.context"`, `"<leaf>.code"`,
//! `"tool.<name>.desc"`, `"tool.<name>.code"`. Paths are the serde boundary;
//! everything after load speaks [`ParamId`]s.

use std::collections::BTreeMap;
use std::marker::PhantomData;

use cranelift_entity::{SecondaryMap, entity_impl};
use serde::{Deserialize, Serialize};

use crate::ir::graph::{ModelId, NodeId, Program, ToolId};
use crate::trace::JsonMap;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ParamId(u32);
entity_impl!(ParamId, "p");

/// One optimizable slot: canonical path, owner, kind, and the incumbent value.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParamSlot {
    /// Canonical human/serde form: `"<leaf>.<slot>"`.
    pub path: Box<str>,
    pub owner: ParamOwner,
    pub kind: ParamKind,
    /// Current value, materialized at build (instruction slots copy the sig
    /// instruction; demos default empty; model slots hold the declared @ref).
    /// Optimizers read this as the incumbent gene.
    pub default: ParamValue,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParamOwner {
    Node(NodeId),
    Tool(ToolId),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParamKind {
    Instruction,
    Demos,
    ToolDesc,
    ModelRef,
    ContextPolicy,
    Code,
}

impl ParamKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Instruction => "instruction",
            Self::Demos => "demos",
            Self::ToolDesc => "tool_desc",
            Self::ModelRef => "model_ref",
            Self::ContextPolicy => "context_policy",
            Self::Code => "code",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "k", rename_all = "snake_case")]
pub enum ParamValue {
    Instruction {
        text: String,
    },
    Demos {
        rows: Vec<DemoRow>,
    },
    ToolDesc {
        text: String,
    },
    ModelRef {
        model: ModelId,
    },
    ContextPolicy {
        policy: ContextPolicy,
    },
    Code {
        lang: CodeLang,
        source: String,
        /// Stable hash of `source`.
        hash: u64,
    },
}

impl ParamValue {
    /// Builds a `Code` value, computing the source hash.
    pub fn code(lang: CodeLang, source: impl Into<String>) -> Self {
        let source = source.into();
        let hash = code_hash(&source);
        Self::Code { lang, source, hash }
    }

    pub fn kind(&self) -> ParamKind {
        match self {
            Self::Instruction { .. } => ParamKind::Instruction,
            Self::Demos { .. } => ParamKind::Demos,
            Self::ToolDesc { .. } => ParamKind::ToolDesc,
            Self::ModelRef { .. } => ParamKind::ModelRef,
            Self::ContextPolicy { .. } => ParamKind::ContextPolicy,
            Self::Code { .. } => ParamKind::Code,
        }
    }
}

/// Stable content hash of a code gene's source.
pub fn code_hash(source: &str) -> u64 {
    use std::hash::Hasher as _;
    let mut hasher = crate::utils::hash::StableHasher::new();
    hasher.write(source.as_bytes());
    hasher.finish()
}

/// One few-shot demonstration row: input and output field maps.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DemoRow {
    pub input: JsonMap,
    pub output: JsonMap,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CodeLang {
    /// QuickJS-sandboxed JavaScript (dsrs-tools ToolSource contract).
    Js,
    // wasm reserved
}

/// The open-lane optimizable slot (vision §6.6). Minimal v1; additive later.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ContextPolicy {
    pub max_history_turns: Option<u32>,
    pub tool_result_max_bytes: Option<u32>,
    /// Free-text playbook injected after the instruction (ACE/Dynamic
    /// Cheatsheet pattern) — reflective optimizers write here.
    pub playbook: Option<String>,
}

// ---------------------------------------------------------------------------
// Typed slot handles
// ---------------------------------------------------------------------------

/// Typed slot handle: optimizer-side mutation mistakes are compile errors.
pub struct Slot<K: KindTag> {
    pub id: ParamId,
    _k: PhantomData<K>,
}

impl<K: KindTag> Slot<K> {
    pub(crate) fn new(id: ParamId) -> Self {
        Self {
            id,
            _k: PhantomData,
        }
    }
}

impl<K: KindTag> Clone for Slot<K> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<K: KindTag> Copy for Slot<K> {}

pub enum Instruction {}
pub enum Demos {}
pub enum ToolDesc {}
pub enum ModelRefK {}
pub enum ContextK {}
pub enum CodeK {}

pub trait KindTag {
    const KIND: ParamKind;
}
impl KindTag for Instruction {
    const KIND: ParamKind = ParamKind::Instruction;
}
impl KindTag for Demos {
    const KIND: ParamKind = ParamKind::Demos;
}
impl KindTag for ToolDesc {
    const KIND: ParamKind = ParamKind::ToolDesc;
}
impl KindTag for ModelRefK {
    const KIND: ParamKind = ParamKind::ModelRef;
}
impl KindTag for ContextK {
    const KIND: ParamKind = ParamKind::ContextPolicy;
}
impl KindTag for CodeK {
    const KIND: ParamKind = ParamKind::Code;
}

// ---------------------------------------------------------------------------
// Overlay
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum OverlayError {
    #[error("overlay minted against program {expected:016x}, applied to {got:016x}")]
    BaseMismatch { expected: u64, got: u64 },
    #[error("param `{path}` has kind {expected:?}, got a {got:?} value")]
    KindMismatch {
        path: String,
        expected: ParamKind,
        got: ParamKind,
    },
    #[error("unknown param path `{path}`")]
    UnknownPath { path: String },
    /// A flat demo row (fx/`ModuleState` form) carries a field the owning
    /// leaf's signature does not declare, so it cannot be split into a
    /// [`DemoRow`]'s input/output maps.
    #[error("demo row for `{path}` has field `{field}` not present in the leaf signature")]
    DemoField { path: String, field: String },
}

/// A candidate = dense data overlay over a fixed skeleton. Clone is a vec
/// clone. Never applied by mutation in the dynamic lane — the interpreter
/// reads through it at render time, which is what makes N candidates
/// evaluable concurrently over one `Arc<Program>`.
#[derive(Clone, Debug, Default)]
pub struct Overlay {
    /// `program_hash` this overlay was minted against; apply/serde paths
    /// verify it. Prevents the stale-candidate-on-new-skeleton bug class.
    pub base: u64,
    values: SecondaryMap<ParamId, Option<ParamValue>>,
}

impl Overlay {
    pub fn new(p: &Program) -> Self {
        Self {
            base: p.meta.program_hash,
            values: SecondaryMap::new(),
        }
    }

    /// Kind-checked set: writing Demos into an Instruction slot is an error.
    pub fn set(&mut self, p: &Program, id: ParamId, v: ParamValue) -> Result<(), OverlayError> {
        if self.base != p.meta.program_hash {
            return Err(OverlayError::BaseMismatch {
                expected: self.base,
                got: p.meta.program_hash,
            });
        }
        let slot = &p.params[id];
        if slot.kind != v.kind() {
            return Err(OverlayError::KindMismatch {
                path: slot.path.to_string(),
                expected: slot.kind,
                got: v.kind(),
            });
        }
        self.values[id] = Some(v);
        Ok(())
    }

    pub fn set_instruction(&mut self, s: Slot<Instruction>, text: impl Into<String>) {
        self.values[s.id] = Some(ParamValue::Instruction { text: text.into() });
    }

    pub fn set_demos(&mut self, s: Slot<Demos>, rows: Vec<DemoRow>) {
        self.values[s.id] = Some(ParamValue::Demos { rows });
    }

    pub fn set_code(&mut self, s: Slot<CodeK>, source: String) {
        self.values[s.id] = Some(ParamValue::code(CodeLang::Js, source));
    }

    pub fn get(&self, id: ParamId) -> Option<&ParamValue> {
        self.values[id].as_ref()
    }

    /// Effective value: overlay else slot default.
    pub fn resolve<'a>(&'a self, p: &'a Program, id: ParamId) -> &'a ParamValue {
        match self.get(id) {
            Some(value) => value,
            None => &p.params[id].default,
        }
    }

    /// Set entries in id order.
    pub fn entries(&self) -> impl Iterator<Item = (ParamId, &ParamValue)> {
        self.values
            .iter()
            .filter_map(|(id, v)| v.as_ref().map(|v| (id, v)))
    }

    pub fn is_empty(&self) -> bool {
        self.entries().next().is_none()
    }

    /// Stable hash over `(base ++ set entries in id order)` — this is
    /// `TraceMeta.candidate_hash` and the rollout-cache key.
    pub fn hash(&self) -> u64 {
        use std::hash::Hasher as _;
        let mut hasher = crate::utils::hash::StableHasher::new();
        hasher.write(&self.base.to_le_bytes());
        for (id, value) in self.entries() {
            hasher.write(&(cranelift_entity::EntityRef::index(id) as u64).to_le_bytes());
            hasher.write(&crate::optimizer::engine::canonical_hash(value).to_le_bytes());
        }
        hasher.finish()
    }

    /// Serde boundary: path-keyed form (`"<path>": ParamValue`).
    pub fn to_named(&self, p: &Program) -> BTreeMap<String, ParamValue> {
        self.entries()
            .map(|(id, value)| (p.param_path(id).to_string(), value.clone()))
            .collect()
    }

    /// Rebuilds an overlay from the path-keyed form, verifying paths and
    /// kinds against `p`.
    pub fn from_named(p: &Program, m: BTreeMap<String, ParamValue>) -> Result<Self, OverlayError> {
        let mut overlay = Self::new(p);
        for (path, value) in m {
            let id = p
                .param_id(&path)
                .ok_or(OverlayError::UnknownPath { path: path.clone() })?;
            overlay.set(p, id, value)?;
        }
        Ok(overlay)
    }
}
