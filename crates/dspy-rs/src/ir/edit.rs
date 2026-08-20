//! The graph-edit calculus: the *structural* mutation half of the IR.
//!
//! [`Overlay`](crate::ir::params::Overlay) mutates parameter **values** over a
//! fixed skeleton; [`Edit`] mutates the skeleton itself. Edits are plain serde
//! values — inspectable, diffable, replayable — and are only ever applied
//! through [`Program::edited`], which is pure: clone the arenas, apply the
//! edits in order, re-run the same load-time validation the builder and loader
//! use, and seal a **new** content hash. A program value is never mutated in
//! place, so every hash-bound artifact (overlays, traces, caches) minted
//! against the parent stays coherent.
//!
//! # Design decisions
//!
//! - **NodeIds are positional handles against the parent.** Within one
//!   `edited()` batch, ids stay stable (swaps happen in place, removals only
//!   detach); dead nodes/sigs/params are garbage-collected once at the end.
//!   Ids in the child may therefore differ from the parent — re-locate leaves
//!   by name ([`Program::leaf_id`]) and params by `ParamPath`.
//! - **Signatures are copy-on-write.** [`Edit::AugmentSig`] never touches the
//!   leaf's current [`SignatureDef`] (other nodes may share it); it pushes an
//!   augmented copy. When the prepended field is exactly the `cot` reasoning
//!   field on a `Predict`, the copy keeps the base name so the canonical
//!   printer re-sugars it as `cot <Sig>`; otherwise it gets a fresh unique
//!   name (`<Sig>_<field>`) because two same-named `sig` blocks cannot print.
//! - **Batch validation.** Edits are validated as a *sequence*: intermediate
//!   states may be inconsistent (e.g. remove a producer, then its consumer);
//!   only the final program must pass `validate()`. Apply-time errors
//!   ([`ApplyError`]) cover what is checkable locally (stale ids, wrong node
//!   kind, unknown tools, capability ceilings); everything data-flow shaped is
//!   deliberately left to `validate.rs` so the edit layer and the loader can
//!   never disagree.
//! - **Garbage collection preserves identity.** `edited(&[])` returns a
//!   program with the parent's hash (only lineage differs, and lineage is
//!   outside the hash preimage). Signatures that were already unreferenced in
//!   the parent are kept; only *newly* orphaned ones are collected. Orphaned
//!   param slots never reach the canonical text, so they are always dropped.
//! - **Lineage.** `edited()` stamps `lineage.parent` with the parent's
//!   `program_hash` exactly like `bake()`; the other provenance fields are
//!   left empty for the optimizer to fill (an edit is not an optimization run
//!   record).
//! - **[`migrate_overlay`]** carries value-level progress across a structural
//!   edit. An entry survives when the child has a slot at the same `ParamPath`
//!   and kind whose owning leaf/tool still has a *carrying* signature: inputs
//!   identical (names + types, in order) and every parent output present in
//!   the child's outputs (name + type). Outputs may widen — that is what lets
//!   instruction and demos survive [`Edit::AugmentSig`] (demo rows still map
//!   onto the base fields; the new field is simply absent from the row).
//!   `ModelRef` entries are re-minted by model *name*, not ordinal.

use std::collections::{HashMap, HashSet};
use std::num::NonZeroU32;

use cranelift_entity::{EntityRef, PrimaryMap};
use serde::{Deserialize, Serialize};

use crate::ir::builder::cot_reasoning_field;
use crate::ir::graph::{
    AgentLoopNode, HoleImpl, Lineage, Node, NodeBudget, NodeId, PortRef, PredictNode, Program,
    RetryNode, SigId, StopSpec, ToolId, ToolKind,
};
use crate::ir::params::{
    ContextPolicy, Overlay, ParamId, ParamKind, ParamOwner, ParamSlot, ParamValue,
};
use crate::ir::sig::{FieldDef, SignatureDef};
use crate::ir::validate::ValidateError;

// ---------------------------------------------------------------------------
// Edits
// ---------------------------------------------------------------------------

/// One structural edit. Serde values: an optimizer's proposal is data, not
/// code — it can be logged, replayed against the same parent, and diffed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "edit", rename_all = "snake_case")]
pub enum Edit {
    /// Prepend an output field to a `Predict`/`AgentLoop` leaf's signature —
    /// the CoT move (mirrors [`SignatureDef::augmented_with`]). Copy-on-write:
    /// a new [`SigId`] is created; nodes sharing the old signature keep it.
    AugmentSig { leaf: NodeId, prepend: FieldDef },
    /// Swap a leaf's kind: `Predict` → `AgentLoop` (with the given tools ⊆
    /// `program.tools`, stop spec and budget) or `AgentLoop` → `Predict`.
    /// Name, signature, bindings, and the instruction/demos/model param slots
    /// are preserved; the `AgentLoop` direction mints a `<leaf>.context`
    /// slot, the `Predict` direction drops it.
    SwapLeaf { leaf: NodeId, to: SwapTarget },
    /// Wrap an existing node in a [`RetryNode`], rewiring the parent
    /// reference and redirecting downstream `Out` ports to the wrapper (the
    /// wrapper, not the child, is what later siblings can see).
    WrapRetry {
        node: NodeId,
        max_attempts: NonZeroU32,
        backoff_ms: u32,
        feedback: bool,
    },
    /// Remove a node from its parent `Seq` body (subtree and its params are
    /// garbage-collected). If a later binding still references its outputs,
    /// `validate()` rejects the batch with its own error.
    Remove { node: NodeId },
    /// Declare an existing program tool on an agent leaf.
    AddTool { agent: NodeId, tool: ToolId },
    /// Undeclare a tool from an agent leaf (also removed from `stop_tools`).
    RemoveTool { agent: NodeId, tool: ToolId },
    /// Replace an agent leaf's [`StopSpec`].
    SetStop { agent: NodeId, stop: StopSpec },
    /// Set the leaf's instruction slot *default* (a bake-like change without
    /// an overlay) — for structural optimizers that also seed text.
    SetInstructionDefault { leaf: NodeId, text: String },
}

/// Target kind of [`Edit::SwapLeaf`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "to", rename_all = "snake_case")]
pub enum SwapTarget {
    Agent {
        tools: Vec<ToolId>,
        #[serde(default)]
        stop: StopSpec,
        #[serde(default)]
        budget: NodeBudget,
    },
    Predict,
}

/// A lightweight, serializable descriptor of an edit kind admissible at a
/// node — the menu [`Program::legal_edits`] returns, suitable for prompting
/// an LLM proposer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EditKind {
    AugmentSig,
    SwapToAgent,
    SwapToPredict,
    WrapRetry,
    Remove,
    AddTool { tool: ToolId },
    RemoveTool { tool: ToolId },
    SetStop,
    SetInstructionDefault,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Why [`Program::edited`] refused.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum EditError {
    /// Edit `index` could not be applied to the (partially edited) program.
    /// Carries the offending edit (boxed: errors stay word-sized on the Ok
    /// path).
    #[error("edit #{index} ({edit:?}) failed: {reason}")]
    Apply {
        index: usize,
        edit: Box<Edit>,
        reason: ApplyError,
    },
    /// Every edit applied, but the resulting program failed the load-time
    /// rules — the error is `validate.rs`'s own.
    #[error("edited program failed validation: {0}")]
    Invalid(#[from] ValidateError),
}

/// A locally-checkable application failure.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ApplyError {
    #[error("no node {node} in this program (stale NodeId)")]
    StaleNode { node: NodeId },
    #[error("{node} is a `{got}` node, expected {expected}")]
    WrongKind {
        node: NodeId,
        expected: &'static str,
        got: &'static str,
    },
    #[error("signature of {node} already has a field `{field}`")]
    DuplicateField { node: NodeId, field: String },
    #[error("no tool {tool} in this program (stale ToolId)")]
    UnknownTool { tool: ToolId },
    #[error("tool `{name}` caps exceed the program ceiling: missing {missing:?}")]
    ToolCapsExceedProgram { name: String, missing: Vec<String> },
    #[error("tool `{name}` is already declared on {agent}")]
    ToolAlreadyDeclared { agent: NodeId, name: String },
    #[error("tool `{name}` is not declared on {agent}")]
    ToolNotDeclared { agent: NodeId, name: String },
    #[error("{node} is not a step of a `Seq` (only Seq steps can be removed)")]
    NotInSeq { node: NodeId },
    #[error("{node} has no parent to rewire (detached by an earlier edit?)")]
    Unparented { node: NodeId },
}

// ---------------------------------------------------------------------------
// Program surface
// ---------------------------------------------------------------------------

impl Program {
    /// Applies `edits` in order to a clone of `self` and returns the sealed,
    /// validated result. `self` is never mutated. The child gets a **new**
    /// content hash and `lineage.parent` set to `self`'s hash (like
    /// [`Program::bake`]); overlays minted against `self` must be re-minted
    /// (see [`migrate_overlay`]).
    pub fn edited(&self, edits: &[Edit]) -> Result<Program, EditError> {
        let mut work = self.clone();
        for (index, edit) in edits.iter().enumerate() {
            apply(&mut work, edit).map_err(|reason| EditError::Apply {
                index,
                edit: Box::new(edit.clone()),
                reason,
            })?;
        }

        // `Remove` only detaches; dead subtrees are still in the arena here.
        // Validate the un-collected graph first so a downstream reference to
        // a removed node surfaces as validate.rs's own error (NodeNotVisible
        // et al.) rather than a remap failure. If the *only* complaint is
        // unreachable nodes, collection is exactly the fix.
        let reachable = reachable_nodes(&work);
        let node_map: HashMap<NodeId, NodeId> = if reachable.len() == work.nodes.len() {
            work.nodes.keys().map(|id| (id, id)).collect()
        } else {
            if let Err(err) = work.validate()
                && !matches!(err, ValidateError::UnreachableNodes { .. })
            {
                return Err(EditError::Invalid(err));
            }
            gc_nodes(&mut work, &reachable)
        };

        gc_sigs(&mut work, self);
        gc_params(&mut work, &node_map);

        work.rebuild_param_index().map_err(EditError::Invalid)?;
        // Validate before sealing: the hash preimage is the canonical printed
        // text, and printing assumes structurally valid arenas.
        work.validate().map_err(EditError::Invalid)?;
        work.meta.lineage = Some(Lineage {
            parent: Some(format!("{:016x}", self.meta.program_hash).into()),
            ..Lineage::default()
        });
        work.seal();
        Ok(work)
    }

    /// The menu of edit kinds structurally admissible at `at`: leaf-only
    /// moves for leaves (split by `Predict`/`AgentLoop`), per-tool add/remove
    /// entries for agents, `WrapRetry` for any non-root node that is not a
    /// `Refine` judge (judges must stay bare leaves), `Remove` for `Seq`
    /// steps. Purely structural — data-flow legality (e.g. whether a removal
    /// orphans a downstream binding) is still `validate()`'s call. A stale id
    /// yields an empty menu.
    pub fn legal_edits(&self, at: NodeId) -> Vec<EditKind> {
        let Some(node) = self.nodes.get(at) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        match node {
            Node::Predict(_) => {
                out.push(EditKind::AugmentSig);
                out.push(EditKind::SetInstructionDefault);
                out.push(EditKind::SwapToAgent);
            }
            Node::AgentLoop(n) => {
                out.push(EditKind::AugmentSig);
                out.push(EditKind::SetInstructionDefault);
                out.push(EditKind::SwapToPredict);
                out.push(EditKind::SetStop);
                for (tool, _) in self.tools.iter() {
                    if n.tools.contains(&tool) {
                        out.push(EditKind::RemoveTool { tool });
                    } else {
                        out.push(EditKind::AddTool { tool });
                    }
                }
            }
            _ => {}
        }
        let parent = self.parent_of(at);
        let is_judge = matches!(
            parent.map(|p| &self.nodes[p]),
            Some(Node::Refine(r)) if r.judge == at
        );
        if at != self.root && !is_judge {
            out.push(EditKind::WrapRetry);
        }
        if matches!(parent.map(|p| &self.nodes[p]), Some(Node::Seq(_))) {
            out.push(EditKind::Remove);
        }
        out
    }

    /// The node id of the leaf named `name`, if any. Leaf names are
    /// program-unique and survive edits, which makes them the stable way to
    /// re-locate a node across [`Program::edited`].
    pub fn leaf_id(&self, name: &str) -> Option<NodeId> {
        self.nodes.iter().find_map(|(id, node)| {
            node.leaf_name()
                .is_some_and(|sym| self.syms.get(sym) == name)
                .then_some(id)
        })
    }

    /// The structural parent of `at` (`None` for the root or a stale id).
    fn parent_of(&self, at: NodeId) -> Option<NodeId> {
        self.nodes
            .iter()
            .find_map(|(id, node)| structural_children(node).contains(&at).then_some(id))
    }
}

// ---------------------------------------------------------------------------
// Overlay migration
// ---------------------------------------------------------------------------

/// Carries tuned values across a structural edit: for every entry in
/// `overlay` (minted against `parent`), re-mint it against `child` when the
/// child has a slot at the same `ParamPath` and kind whose owning leaf/tool
/// signature still *carries* the parent's — inputs identical, parent outputs
/// a subset of the child's (so [`Edit::AugmentSig`] keeps instruction and
/// demos alive; see the module docs). Entries that no longer fit are dropped.
/// The result is based on `child`'s hash. A base-mismatched `overlay` yields
/// an empty result rather than indexing with foreign ids.
pub fn migrate_overlay(parent: &Program, overlay: &Overlay, child: &Program) -> Overlay {
    let mut out = Overlay::new(child);
    if overlay.base != parent.meta.program_hash {
        return out;
    }
    for (id, value) in overlay.entries() {
        let path = parent.param_path(id);
        let Some(child_id) = child.param_id(path) else {
            continue;
        };
        if child.params[child_id].kind != value.kind() {
            continue;
        }
        let (Some(psig), Some(csig)) = (
            owner_sig(parent, parent.params[id].owner),
            owner_sig(child, child.params[child_id].owner),
        ) else {
            continue;
        };
        if !sig_carries(&parent.sigs[psig], &child.sigs[csig]) {
            continue;
        }
        let value = match value {
            // Model refs are ordinals into `models`; re-mint by name.
            ParamValue::ModelRef { model } => {
                let Some(def) = parent.models.get(*model) else {
                    continue;
                };
                let Some((child_model, _)) = child.models.iter().find(|(_, m)| m.name == def.name)
                else {
                    continue;
                };
                ParamValue::ModelRef { model: child_model }
            }
            // Tool sets carry ordinals into `tools`; re-mint each by name
            // and keep the intersection with what the child's agent still
            // declares — partial survival is the point of migration. A
            // selection with no survivors no longer fits and is dropped.
            ParamValue::ToolSet { tools } => {
                let declared: &[ToolId] = match child.params[child_id].owner {
                    ParamOwner::Node(node) => match &child.nodes[node] {
                        Node::AgentLoop(n) => &n.tools,
                        _ => continue,
                    },
                    ParamOwner::Tool(_) => continue,
                };
                let mut migrated: Vec<ToolId> = Vec::new();
                for t in tools {
                    let Some(def) = parent.tools.get(*t) else {
                        continue;
                    };
                    let name = parent.syms.get(def.name);
                    let Some((child_tool, _)) = child
                        .tools
                        .iter()
                        .find(|(_, d)| child.syms.get(d.name) == name)
                    else {
                        continue;
                    };
                    if declared.contains(&child_tool) && !migrated.contains(&child_tool) {
                        migrated.push(child_tool);
                    }
                }
                if migrated.is_empty() && !tools.is_empty() {
                    continue;
                }
                ParamValue::ToolSet { tools: migrated }
            }
            other => other.clone(),
        };
        // Kind was checked above; set cannot fail, but stay total.
        let _ = out.set(child, child_id, value);
    }
    out
}

/// `parent` signature values still make sense on `child`: inputs identical
/// (names + types, in order), every parent output present among the child's
/// outputs (name + type). Docs/constraints/render are shape-irrelevant.
fn sig_carries(parent: &SignatureDef, child: &SignatureDef) -> bool {
    parent.inputs.len() == child.inputs.len()
        && parent
            .inputs
            .iter()
            .zip(child.inputs.iter())
            .all(|(a, b)| a.name == b.name && a.ty == b.ty)
        && parent.outputs.iter().all(|f| {
            child
                .outputs
                .iter()
                .any(|g| g.name == f.name && g.ty == f.ty)
        })
}

/// The signature of a slot's owning leaf or tool (`None` for a stale owner
/// or a non-leaf node).
fn owner_sig(p: &Program, owner: ParamOwner) -> Option<SigId> {
    match owner {
        ParamOwner::Node(id) => match p.nodes.get(id)? {
            Node::Predict(n) => Some(n.sig),
            Node::AgentLoop(n) => Some(n.sig),
            Node::Hole(n) => Some(n.sig),
            _ => None,
        },
        ParamOwner::Tool(id) => p.tools.get(id).map(|t| t.sig),
    }
}

// ---------------------------------------------------------------------------
// Edit application
// ---------------------------------------------------------------------------

fn apply(work: &mut Program, edit: &Edit) -> Result<(), ApplyError> {
    match edit {
        Edit::AugmentSig { leaf, prepend } => augment_sig(work, *leaf, prepend),
        Edit::SwapLeaf { leaf, to } => swap_leaf(work, *leaf, to),
        Edit::WrapRetry {
            node,
            max_attempts,
            backoff_ms,
            feedback,
        } => wrap_retry(work, *node, *max_attempts, *backoff_ms, *feedback),
        Edit::Remove { node } => remove(work, *node),
        Edit::AddTool { agent, tool } => add_tool(work, *agent, *tool),
        Edit::RemoveTool { agent, tool } => remove_tool(work, *agent, *tool),
        Edit::SetStop { agent, stop } => {
            agent_mut(work, *agent)?.stop = stop.clone();
            Ok(())
        }
        Edit::SetInstructionDefault { leaf, text } => set_instruction_default(work, *leaf, text),
    }
}

fn node_checked(work: &Program, id: NodeId) -> Result<&Node, ApplyError> {
    work.nodes.get(id).ok_or(ApplyError::StaleNode { node: id })
}

fn agent_mut(work: &mut Program, id: NodeId) -> Result<&mut AgentLoopNode, ApplyError> {
    match node_checked(work, id)? {
        Node::AgentLoop(_) => {}
        other => {
            return Err(ApplyError::WrongKind {
                node: id,
                expected: "an `agent` leaf",
                got: kind_label(other),
            });
        }
    }
    match &mut work.nodes[id] {
        Node::AgentLoop(n) => Ok(n),
        _ => unreachable!("checked above"),
    }
}

fn kind_label(node: &Node) -> &'static str {
    match node {
        Node::Predict(_) => "predict",
        Node::AgentLoop(_) => "agent",
        Node::Hole(_) => "hole",
        Node::Seq(_) => "seq",
        Node::ForkJoin(_) => "fork",
        Node::Route(_) => "route",
        Node::Retry(_) => "retry",
        Node::Refine(_) => "refine",
        Node::Loop(_) => "loop",
    }
}

fn augment_sig(work: &mut Program, leaf: NodeId, prepend: &FieldDef) -> Result<(), ApplyError> {
    let (sig_id, is_predict) = match node_checked(work, leaf)? {
        Node::Predict(n) => (n.sig, true),
        Node::AgentLoop(n) => (n.sig, false),
        other => {
            return Err(ApplyError::WrongKind {
                node: leaf,
                expected: "a `predict` or `agent` leaf",
                got: kind_label(other),
            });
        }
    };
    let base = work.sigs[sig_id].clone();
    if base
        .inputs
        .iter()
        .chain(base.outputs.iter())
        .any(|f| f.name == prepend.name)
    {
        return Err(ApplyError::DuplicateField {
            node: leaf,
            field: prepend.name.to_string(),
        });
    }
    let mut augmented = base.augmented_with(std::slice::from_ref(prepend));
    // The exact `cot` reasoning field on a Predict keeps the base name: the
    // canonical printer re-sugars it as `cot <Sig>` against the base, so the
    // shared name never prints twice. Any other augmentation is a new
    // declaration and needs its own name.
    if !(is_predict && *prepend == cot_reasoning_field()) {
        augmented.name = unique_sig_name(work, &base.name, &prepend.name);
    }
    let new_sig = work.sigs.push(augmented);
    match &mut work.nodes[leaf] {
        Node::Predict(n) => n.sig = new_sig,
        Node::AgentLoop(n) => n.sig = new_sig,
        _ => unreachable!("leaf kind checked above"),
    }
    Ok(())
}

/// `<base>_<field>`, uniquified against every name the text format resolves
/// in or near the signature namespace (sigs, tools, class/enum tokens).
fn unique_sig_name(p: &Program, base: &str, field: &str) -> Box<str> {
    let mut taken: HashSet<String> = p.sigs.values().map(|s| s.name.to_string()).collect();
    taken.extend(p.tools.values().map(|t| p.syms.get(t.name).to_string()));
    taken.extend(p.types.classes.keys().cloned());
    taken.extend(p.types.enums.keys().cloned());
    let stem = format!("{base}_{field}");
    if !taken.contains(&stem) {
        return stem.into();
    }
    let mut i = 2usize;
    loop {
        let candidate = format!("{stem}{i}");
        if !taken.contains(&candidate) {
            return candidate.into();
        }
        i += 1;
    }
}

fn swap_leaf(work: &mut Program, leaf: NodeId, to: &SwapTarget) -> Result<(), ApplyError> {
    let node = node_checked(work, leaf)?.clone();
    match (node, to) {
        (
            Node::Predict(n),
            SwapTarget::Agent {
                tools,
                stop,
                budget,
            },
        ) => {
            let mut declared: Vec<ToolId> = Vec::new();
            for &tool in tools {
                let def = tool_checked(work, tool)?;
                let name = work.syms.get(def.name).to_string();
                if !def.caps.is_subset(&work.caps) {
                    return Err(ApplyError::ToolCapsExceedProgram {
                        name,
                        missing: def.caps.missing_from(&work.caps),
                    });
                }
                if declared.contains(&tool) {
                    return Err(ApplyError::ToolAlreadyDeclared { agent: leaf, name });
                }
                declared.push(tool);
            }
            let leaf_name = work.syms.get(n.name).to_string();
            // Same slot-creation convention as the builder: `<leaf>.context`
            // and `<leaf>.tool_set` (default = the full declared table).
            let context_policy = work.params.push(ParamSlot {
                path: format!("{leaf_name}.context").into(),
                owner: ParamOwner::Node(leaf),
                kind: ParamKind::ContextPolicy,
                default: ParamValue::ContextPolicy {
                    policy: ContextPolicy::default(),
                },
            });
            let tool_set = work.params.push(ParamSlot {
                path: format!("{leaf_name}.tool_set").into(),
                owner: ParamOwner::Node(leaf),
                kind: ParamKind::ToolSet,
                default: ParamValue::ToolSet {
                    tools: declared.clone(),
                },
            });
            work.nodes[leaf] = Node::AgentLoop(AgentLoopNode {
                name: n.name,
                sig: n.sig,
                instruction: n.instruction,
                demos: n.demos,
                model: n.model,
                tools: declared.into_boxed_slice(),
                tool_set,
                context_policy,
                stop: stop.clone(),
                budget: budget.clone(),
                binding: n.binding,
            });
            Ok(())
        }
        (Node::AgentLoop(n), SwapTarget::Predict) => {
            // The context and tool_set slots are orphaned here and collected
            // in `edited()`.
            work.nodes[leaf] = Node::Predict(PredictNode {
                name: n.name,
                sig: n.sig,
                instruction: n.instruction,
                demos: n.demos,
                model: n.model,
                binding: n.binding,
            });
            Ok(())
        }
        (other, SwapTarget::Agent { .. }) => Err(ApplyError::WrongKind {
            node: leaf,
            expected: "a `predict` leaf",
            got: kind_label(&other),
        }),
        (other, SwapTarget::Predict) => Err(ApplyError::WrongKind {
            node: leaf,
            expected: "an `agent` leaf",
            got: kind_label(&other),
        }),
    }
}

fn wrap_retry(
    work: &mut Program,
    node: NodeId,
    max_attempts: NonZeroU32,
    backoff_ms: u32,
    feedback: bool,
) -> Result<(), ApplyError> {
    node_checked(work, node)?;
    let attached = work.root == node
        || work
            .nodes
            .values()
            .any(|n| structural_children(n).contains(&node));
    if !attached {
        return Err(ApplyError::Unparented { node });
    }
    let retry = work.nodes.push(Node::Retry(RetryNode {
        child: node,
        max_attempts,
        backoff_ms,
        feedback,
    }));
    // Rewire the single structural parent (nodes form a tree) — or the root
    // slot itself, in which case validate() rejects with RootNotSeq.
    if work.root == node {
        work.root = retry;
    } else {
        let ids: Vec<NodeId> = work.nodes.keys().collect();
        for id in ids {
            if id != retry && replace_child(&mut work.nodes[id], node, retry) {
                break;
            }
        }
    }
    // Downstream dataflow must reference the wrapper: scope visibility is
    // sibling-level, and the retry is the sibling now. (Nothing inside the
    // wrapped subtree can reference the subtree's own root, so a global
    // redirect is safe.)
    for (id, n) in work.nodes.iter_mut() {
        if id != retry {
            redirect_out_ports(n, node, retry);
        }
    }
    Ok(())
}

fn remove(work: &mut Program, node: NodeId) -> Result<(), ApplyError> {
    node_checked(work, node)?;
    let ids: Vec<NodeId> = work.nodes.keys().collect();
    for id in ids {
        if let Node::Seq(seq) = &mut work.nodes[id]
            && seq.body.contains(&node)
        {
            let mut body = seq.body.to_vec();
            body.retain(|&child| child != node);
            seq.body = body.into_boxed_slice();
            return Ok(());
        }
    }
    Err(ApplyError::NotInSeq { node })
}

fn add_tool(work: &mut Program, agent: NodeId, tool: ToolId) -> Result<(), ApplyError> {
    let def = tool_checked(work, tool)?;
    let name = work.syms.get(def.name).to_string();
    if !def.caps.is_subset(&work.caps) {
        return Err(ApplyError::ToolCapsExceedProgram {
            name,
            missing: def.caps.missing_from(&work.caps),
        });
    }
    let n = agent_mut(work, agent)?;
    if n.tools.contains(&tool) {
        return Err(ApplyError::ToolAlreadyDeclared { agent, name });
    }
    let mut tools = n.tools.to_vec();
    tools.push(tool);
    n.tools = tools.into_boxed_slice();
    // Declaring a tool makes it live: the tool_set gene's default tracks the
    // declaration (a baked subset grows by exactly the tool just declared).
    let tool_set = n.tool_set;
    if let ParamValue::ToolSet { tools } = &mut work.params[tool_set].default
        && !tools.contains(&tool)
    {
        tools.push(tool);
    }
    Ok(())
}

fn remove_tool(work: &mut Program, agent: NodeId, tool: ToolId) -> Result<(), ApplyError> {
    let def = tool_checked(work, tool)?;
    let name = work.syms.get(def.name).to_string();
    let n = agent_mut(work, agent)?;
    if !n.tools.contains(&tool) {
        return Err(ApplyError::ToolNotDeclared { agent, name });
    }
    n.tools = n.tools.iter().copied().filter(|t| *t != tool).collect();
    n.stop.stop_tools = n
        .stop
        .stop_tools
        .iter()
        .copied()
        .filter(|t| *t != tool)
        .collect();
    // The tool_set gene's alphabet shrank; drop the tool from the default
    // selection too (validation would otherwise refuse the child).
    let tool_set = n.tool_set;
    if let ParamValue::ToolSet { tools } = &mut work.params[tool_set].default {
        tools.retain(|t| *t != tool);
    }
    Ok(())
}

fn set_instruction_default(work: &mut Program, leaf: NodeId, text: &str) -> Result<(), ApplyError> {
    let param = match node_checked(work, leaf)? {
        Node::Predict(n) => n.instruction,
        Node::AgentLoop(n) => n.instruction,
        other => {
            return Err(ApplyError::WrongKind {
                node: leaf,
                expected: "a `predict` or `agent` leaf",
                got: kind_label(other),
            });
        }
    };
    work.params[param].default = ParamValue::Instruction {
        text: text.to_string(),
    };
    Ok(())
}

fn tool_checked(work: &Program, tool: ToolId) -> Result<&crate::ir::graph::ToolDef, ApplyError> {
    work.tools.get(tool).ok_or(ApplyError::UnknownTool { tool })
}

// ---------------------------------------------------------------------------
// Tree plumbing
// ---------------------------------------------------------------------------

/// The structural children of a node — the same set `validate()` walks.
fn structural_children(node: &Node) -> Vec<NodeId> {
    match node {
        Node::Predict(_) | Node::AgentLoop(_) | Node::Hole(_) => Vec::new(),
        Node::Seq(n) => n.body.to_vec(),
        Node::ForkJoin(n) => n.branches.to_vec(),
        Node::Route(n) => n
            .arms
            .iter()
            .map(|(_, arm)| *arm)
            .chain(n.default)
            .collect(),
        Node::Retry(n) => vec![n.child],
        Node::Refine(n) => vec![n.child, n.judge],
        Node::Loop(n) => vec![n.body],
    }
}

/// Replaces `from` with `to` in a structural child position. Returns whether
/// a replacement happened.
fn replace_child(node: &mut Node, from: NodeId, to: NodeId) -> bool {
    let slot_in = |slots: &mut [NodeId]| {
        for slot in slots {
            if *slot == from {
                *slot = to;
                return true;
            }
        }
        false
    };
    match node {
        Node::Predict(_) | Node::AgentLoop(_) | Node::Hole(_) => false,
        Node::Seq(n) => slot_in(&mut n.body),
        Node::ForkJoin(n) => slot_in(&mut n.branches),
        Node::Route(n) => {
            for (_, arm) in n.arms.iter_mut() {
                if *arm == from {
                    *arm = to;
                    return true;
                }
            }
            if n.default == Some(from) {
                n.default = Some(to);
                return true;
            }
            false
        }
        Node::Retry(n) => {
            if n.child == from {
                n.child = to;
                true
            } else {
                false
            }
        }
        Node::Refine(n) => {
            if n.child == from {
                n.child = to;
                true
            } else if n.judge == from {
                n.judge = to;
                true
            } else {
                false
            }
        }
        Node::Loop(n) => {
            if n.body == from {
                n.body = to;
                true
            } else {
                false
            }
        }
    }
}

fn redirect_out_ports(node: &mut Node, from: NodeId, to: NodeId) {
    let redirect = |port: &mut PortRef| {
        if let PortRef::Out { node, .. } = port
            && *node == from
        {
            *node = to;
        }
    };
    for_each_port(node, redirect);
}

fn for_each_port(node: &mut Node, mut f: impl FnMut(&mut PortRef)) {
    match node {
        Node::Predict(n) => n.binding.iter_mut().for_each(|b| f(&mut b.src)),
        Node::AgentLoop(n) => n.binding.iter_mut().for_each(|b| f(&mut b.src)),
        Node::Hole(n) => n.binding.iter_mut().for_each(|b| f(&mut b.src)),
        Node::Seq(n) => n.out.iter_mut().for_each(|b| f(&mut b.src)),
        Node::ForkJoin(n) => n.join.iter_mut().for_each(|b| f(&mut b.src)),
        Node::Route(n) => f(&mut n.on),
        Node::Retry(_) | Node::Refine(_) => {}
        Node::Loop(n) => {
            if let Some(port) = &mut n.while_ {
                f(port);
            }
            n.carry.iter_mut().for_each(|b| f(&mut b.src));
            n.out.iter_mut().for_each(|b| f(&mut b.src));
        }
    }
}

// ---------------------------------------------------------------------------
// Garbage collection
// ---------------------------------------------------------------------------

fn reachable_nodes(p: &Program) -> HashSet<NodeId> {
    let mut seen = HashSet::new();
    let mut stack = vec![p.root];
    while let Some(id) = stack.pop() {
        if seen.insert(id) {
            stack.extend(structural_children(&p.nodes[id]));
        }
    }
    seen
}

/// Drops unreachable nodes, remapping ids everywhere they occur. Only called
/// after `validate()` has confirmed the reachable subgraph is self-contained,
/// so every surviving reference remaps.
fn gc_nodes(work: &mut Program, reachable: &HashSet<NodeId>) -> HashMap<NodeId, NodeId> {
    let mut map = HashMap::new();
    let mut nodes: PrimaryMap<NodeId, Node> = PrimaryMap::new();
    for (id, node) in work.nodes.iter() {
        if reachable.contains(&id) {
            map.insert(id, nodes.push(node.clone()));
        }
    }
    for node in nodes.values_mut() {
        remap_children(node, &map);
        for_each_port(node, |port| {
            if let PortRef::Out { node, .. } = port {
                *node = map[node];
            }
        });
    }
    work.nodes = nodes;
    work.root = map[&work.root];
    map
}

fn remap_children(node: &mut Node, map: &HashMap<NodeId, NodeId>) {
    match node {
        Node::Predict(_) | Node::AgentLoop(_) | Node::Hole(_) => {}
        Node::Seq(n) => n.body.iter_mut().for_each(|c| *c = map[c]),
        Node::ForkJoin(n) => n.branches.iter_mut().for_each(|c| *c = map[c]),
        Node::Route(n) => {
            n.arms.iter_mut().for_each(|(_, arm)| *arm = map[arm]);
            if let Some(default) = &mut n.default {
                *default = map[default];
            }
        }
        Node::Retry(n) => n.child = map[&n.child],
        Node::Refine(n) => {
            n.child = map[&n.child];
            n.judge = map[&n.judge];
        }
        Node::Loop(n) => n.body = map[&n.body],
    }
}

/// Signatures a program actually uses: the program interface, every leaf and
/// tool signature, and the `cot` base of every sugar-detected Predict (the
/// printer needs the base in the arena to re-sugar).
fn referenced_sigs(p: &Program) -> HashSet<SigId> {
    let mut set = HashSet::new();
    set.insert(p.sig);
    for node in p.nodes.values() {
        match node {
            Node::Predict(n) => {
                set.insert(n.sig);
                if let Some(base) = cot_base_of(p, n) {
                    set.insert(base);
                }
            }
            Node::AgentLoop(n) => {
                set.insert(n.sig);
            }
            Node::Hole(n) => {
                set.insert(n.sig);
            }
            _ => {}
        }
    }
    for tool in p.tools.values() {
        set.insert(tool.sig);
    }
    set
}

/// Mirrors the canonical printer's `cot` detection (print.rs): the node's
/// signature is `base.augmented_with([reasoning])` for some *other* arena
/// signature with identical name/instruction/inputs.
fn cot_base_of(p: &Program, n: &PredictNode) -> Option<SigId> {
    let sig = &p.sigs[n.sig];
    let first = sig.outputs.first()?;
    if *first != cot_reasoning_field() {
        return None;
    }
    p.sigs.iter().find_map(|(id, base)| {
        (id != n.sig
            && base.name == sig.name
            && base.instruction == sig.instruction
            && base.inputs == sig.inputs
            && *base.outputs == sig.outputs[1..])
            .then_some(id)
    })
}

/// Collects *newly* orphaned signatures. Sigs that were already unreferenced
/// in the parent stay (they print in both, keeping `edited(&[])` a hash
/// no-op); sigs orphaned by this batch (replaced by `AugmentSig`, or owned by
/// removed leaves) are dropped so same-named `sig` blocks never print twice.
fn gc_sigs(work: &mut Program, parent: &Program) {
    let used = referenced_sigs(work);
    if used.len() == work.sigs.len() {
        return;
    }
    let parent_used = referenced_sigs(parent);
    let parent_len = parent.sigs.len();
    let retained: Vec<SigId> = work
        .sigs
        .keys()
        .filter(|id| used.contains(id) || (id.index() < parent_len && !parent_used.contains(id)))
        .collect();
    if retained.len() == work.sigs.len() {
        return;
    }
    let mut map = HashMap::new();
    let mut sigs: PrimaryMap<SigId, SignatureDef> = PrimaryMap::new();
    for id in retained {
        map.insert(id, sigs.push(work.sigs[id].clone()));
    }
    work.sigs = sigs;
    work.sig = map[&work.sig];
    for node in work.nodes.values_mut() {
        match node {
            Node::Predict(n) => n.sig = map[&n.sig],
            Node::AgentLoop(n) => n.sig = map[&n.sig],
            Node::Hole(n) => n.sig = map[&n.sig],
            _ => {}
        }
    }
    for tool in work.tools.values_mut() {
        tool.sig = map[&tool.sig];
    }
}

/// Drops param slots no node or tool references (orphaned by `Remove` and by
/// `AgentLoop` → `Predict` swaps), remapping surviving [`ParamId`]s and their
/// owners. Orphans never reach the canonical text, so this never moves the
/// hash; it does keep `ParamPath`s collision-free across swap sequences.
fn gc_params(work: &mut Program, node_map: &HashMap<NodeId, NodeId>) {
    let mut used: HashSet<ParamId> = HashSet::new();
    for node in work.nodes.values() {
        match node {
            Node::Predict(n) => used.extend([n.instruction, n.demos, n.model]),
            Node::AgentLoop(n) => {
                used.extend([
                    n.instruction,
                    n.demos,
                    n.model,
                    n.tool_set,
                    n.context_policy,
                ]);
            }
            Node::Hole(n) => {
                if let HoleImpl::Sandboxed { code } = n.imp {
                    used.insert(code);
                }
            }
            _ => {}
        }
    }
    for tool in work.tools.values() {
        used.insert(tool.desc);
        if let ToolKind::Sandboxed { code } = tool.kind {
            used.insert(code);
        }
    }
    let identity_nodes = node_map.iter().all(|(from, to)| from == to);
    if used.len() == work.params.len() && identity_nodes {
        return;
    }
    let mut map = HashMap::new();
    let mut params: PrimaryMap<ParamId, ParamSlot> = PrimaryMap::new();
    for (id, slot) in work.params.iter() {
        if !used.contains(&id) {
            continue;
        }
        let mut slot = slot.clone();
        if let ParamOwner::Node(owner) = slot.owner {
            slot.owner = ParamOwner::Node(node_map[&owner]);
        }
        map.insert(id, params.push(slot));
    }
    work.params = params;
    for node in work.nodes.values_mut() {
        match node {
            Node::Predict(n) => {
                n.instruction = map[&n.instruction];
                n.demos = map[&n.demos];
                n.model = map[&n.model];
            }
            Node::AgentLoop(n) => {
                n.instruction = map[&n.instruction];
                n.demos = map[&n.demos];
                n.model = map[&n.model];
                n.tool_set = map[&n.tool_set];
                n.context_policy = map[&n.context_policy];
            }
            Node::Hole(n) => {
                if let HoleImpl::Sandboxed { code } = &mut n.imp {
                    *code = map[code];
                }
            }
            _ => {}
        }
    }
    for tool in work.tools.values_mut() {
        tool.desc = map[&tool.desc];
        if let ToolKind::Sandboxed { code } = &mut tool.kind {
            *code = map[code];
        }
    }
}
