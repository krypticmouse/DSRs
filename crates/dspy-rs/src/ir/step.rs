//! RFC 0003 stage M-2: step metadata.
//!
//! The `#[predict]`, `#[cot]`, `#[agent]`, and `#[tool]` macros emit a
//! `__dsrs_step()` / `__dsrs_tool()` item inside their generated module
//! returning the types here — the metadata the `#[module]` frontend (M-3)
//! consumes to lower an ordinary Rust fn body into a [`Program`]
//! (crate::ir::Program). Semantic resolution stays with rustc: a `#[module]`
//! body calling a fn that was not declared with a step macro fails at the
//! call site with a missing-item error.

use std::sync::Arc;

use crate::ir::graph::NodeBudget;
use crate::ir::params::ContextPolicy;
use crate::ir::sig::SignatureDef;
use crate::typesys::TypeTable;

/// One step declaration — what a `#[predict]`/`#[cot]`/`#[agent]` fn *is*,
/// as data.
pub struct StepDef {
    /// The fn name — the fx params slot and the default leaf-name basis.
    pub name: &'static str,
    pub kind: StepKind,
    /// The base signature (un-augmented for `cot`; lowering augments).
    pub sig: &'static SignatureDef,
    /// Class/enum definitions reachable from the signature.
    pub types: &'static TypeTable,
    /// `@name` model ref from the attribute, leading `@` stripped.
    /// `None` = the module's default model.
    pub model: Option<&'static str>,
    /// Agent options; `Some` iff `kind == StepKind::Agent`.
    pub agent: Option<AgentStepOpts>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StepKind {
    Predict,
    Cot,
    Agent,
}

/// `#[agent(...)]` attribute options, mirroring the `AgentLoopNode` surface.
pub struct AgentStepOpts {
    /// Resolved from the `tools(...)` attribute list, in declaration order.
    pub tools: Vec<ToolStepDef>,
    /// Names of tools (⊆ `tools`) whose call ends the loop.
    pub stop_tools: Vec<&'static str>,
    /// `None` = the IR default ([`StopSpec::default`](crate::ir::StopSpec)).
    pub max_turns: Option<u32>,
    pub until_parse: Option<bool>,
    pub budget: NodeBudget,
    pub context: ContextPolicy,
}

/// One `#[tool]` declaration: metadata plus the host implementation.
pub struct ToolStepDef {
    pub name: &'static str,
    /// Doc comment — the `ToolDesc` gene's default text.
    pub desc: &'static str,
    pub caps: &'static [&'static str],
    /// Declared interface: params → inputs, return type → one output field
    /// named after the fn.
    pub sig: &'static SignatureDef,
    pub types: &'static TypeTable,
    /// The host implementation, ready to bind into a
    /// [`RuntimeEnv`](crate::ir::RuntimeEnv) (`ToolKind::Host`).
    pub dyn_tool: Arc<dyn rig::tool::ToolDyn>,
}

/// One hole-ized expression in a `#[module]` body (RFC 0003 §6) — the
/// opacity report entry.
#[derive(Clone, Copy, Debug)]
pub struct HoleReport {
    /// The leaf name (the `let` binding).
    pub name: &'static str,
    /// `"host"` (native, extern-bound) or `"js"` (sandboxed).
    pub kind: &'static str,
    /// The source expression, verbatim.
    pub excerpt: &'static str,
    /// Why it could not be lowered to the closed node set.
    pub reason: &'static str,
}
