//! The intermediate representation (RFC 0002).
//!
//! - **IR-1, value-level signatures** — [`SignatureDef`] is a signature as an
//!   owned runtime value: constructible without macros via
//!   [`SignatureDef::build`], bridged from `#[derive(Signature)]` types via
//!   [`SignatureDef::of`], serde-derivable for the program artifact. The type
//!   model is [`typesys::FieldType`](crate::typesys::FieldType) unchanged;
//!   class/enum definitions live in a program-owned [`TypeTable`].
//! - **IR-2, the graph** (`ir` feature, default-on) — [`Program`]: entity
//!   arenas over the closed [`Node`] enum, field-level [`Binding`] dataflow,
//!   [`ParamSlot`] parameters addressed by `ParamPath`, [`Overlay`]
//!   candidates, capability ceilings, and load-time validation.
//! - **IR-3, the interpreter** (`ir` feature) — [`Interpreter`]: async
//!   evaluation of a loaded program with overlay read-through at render time,
//!   RFC 0001 trace spans (component = leaf name), budget metering, and
//!   sandboxed [`Hole`](Node::Hole) execution via `dsrs-tools`.
//! - The `.dsrs` text format is stage IR-5 (not yet built); until then
//!   programs serialize through a canonical JSON projection.

pub mod sig;

pub use crate::typesys::{ClassDef, EnumDef, EnumValueDef, FieldType, TypeTable};
pub use sig::{
    ConstraintDef, FieldDef, RenderSpec, SigError, SigMismatch, SignatureBuilder, SignatureDef,
};

#[cfg(feature = "ir")]
pub mod builder;
#[cfg(feature = "ir")]
pub mod graph;
#[cfg(feature = "ir")]
pub mod params;
#[cfg(feature = "ir")]
pub mod validate;

#[cfg(feature = "ir")]
pub use builder::{
    AsNodeName, BuildError, NodeSpec, Port, ProgramBuilder, agent, carried, cot, fork, hole,
    input, lit, loop_, out, predict, refine, retry, route, seq,
};
#[cfg(feature = "ir")]
pub use graph::{
    AgentLoopNode, Binding, BudgetPolicy, CapSet, ForkJoinNode, HoleNode, Interner, Lineage,
    LoopNode, ModelDef, ModelId, Node, NodeBudget, NodeId, PortRef, PredictNode, Program,
    ProgramMeta, RefineNode, RetryNode, RouteNode, SeqNode, SigId, StopSpec, Sym, ToolDef, ToolId,
    ToolKind,
};
#[cfg(feature = "ir")]
pub use params::{
    CodeK, CodeLang, ContextK, ContextPolicy, DemoRow, Demos, Instruction, KindTag, ModelRefK,
    Overlay, OverlayError, ParamId, ParamKind, ParamOwner, ParamSlot, ParamValue, Slot, ToolDesc,
    code_hash,
};
#[cfg(feature = "ir")]
pub use validate::ValidateError;
