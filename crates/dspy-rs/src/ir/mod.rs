//! The intermediate representation (RFC 0002).
//!
//! - **IR-1, value-level signatures** — [`SignatureDef`] is a signature as an
//!   owned runtime value: constructible without macros via
//!   [`SignatureDef::build`], bridged from `#[derive(Signature)]` types via
//!   [`SignatureDef::of`], serde-derivable for the program artifact. The type
//!   model is [`typesys::FieldType`](crate::typesys::FieldType) unchanged;
//!   class/enum definitions live in a program-owned [`TypeTable`].
//! - **IR-2, the graph** — [`Program`]: entity
//!   arenas over the closed [`Node`] enum, field-level [`Binding`] dataflow,
//!   [`ParamSlot`] parameters addressed by `ParamPath`, [`Overlay`]
//!   candidates, capability ceilings, and load-time validation.
//! - **IR-3, the interpreter** — [`Interpreter`]: async
//!   evaluation of a loaded program with overlay read-through at render time,
//!   RFC 0001 trace spans (component = leaf name), budget metering, and
//!   sandboxed [`Hole`](Node::Hole) execution via `dsrs-tools`.
//! - **IR-5, the `.dsrs` text format** — the wire form of a
//!   program: [`Program::from_dsrs`]/[`Program::to_dsrs`] parse and
//!   canonically print RFC 0002 §4 text, and the canonical text (minus
//!   lineage) is the [`Program::compute_hash`] preimage. A canonical JSON
//!   projection ([`serde::Serialize`]) remains for embedding programs in
//!   JSON documents; both forms agree on the hash.

pub mod sig;

pub use crate::typesys::{ClassDef, EnumDef, EnumValueDef, FieldType, TypeTable};
pub use sig::{
    ConstraintDef, FieldDef, RenderSpec, SigError, SigMismatch, SignatureBuilder, SignatureDef,
};

pub mod bridge;
pub mod builder;
pub mod edit;
pub mod graph;
pub mod interp;
pub mod module_build;
pub mod params;
pub mod step;
pub mod text;
pub mod validate;

pub use bridge::{current_overlay, with_ambient_overlay, with_overlay};
pub use builder::{
    AsNodeName, BuildError, NodeSpec, Port, ProgramBuilder, agent, carried, cot, extern_hole, fork,
    hole, input, lit, loop_, out, predict, refine, retry, route, seq,
};
pub use edit::{ApplyError, Edit, EditError, EditKind, SwapTarget, migrate_overlay};
pub use graph::{
    AgentLoopNode, BakeError, Binding, BudgetPolicy, CapSet, ForkJoinNode, HoleImpl, HoleNode,
    Interner, Lineage, LoopNode, ModelDef, ModelId, Node, NodeBudget, NodeId, PortRef, PredictNode,
    Program, ProgramMeta, RefineNode, RetryNode, RouteNode, SeqNode, SigId, StopSpec, Sym, ToolDef,
    ToolId, ToolKind,
};
pub use interp::{
    Budget, BudgetMeter, ConversationTurn, Exhausted, HostHoleFn, Interpreter, LeafOutcome,
    LoadError, RunError, RunOutput, RuntimeEnv, ToolSuspension, input_schema_of,
};
pub use module_build::{
    ModuleBuildError, ModuleSpec, ModuleStep, ModuleStepKind, PortSpec, build_module_program,
    default_lm, unbound_model_config,
};
pub use params::{
    CodeK, CodeLang, ContextK, ContextPolicy, DemoRow, Demos, Instruction, KindTag, ModelRefK,
    Overlay, OverlayError, ParamId, ParamKind, ParamOwner, ParamSlot, ParamValue, Slot, ToolDesc,
    ToolSetK, code_hash,
};
pub use step::{AgentStepOpts, HoleReport, StepDef, StepKind, ToolStepDef};
pub use text::{DsrsFileError, ParseError};
pub use validate::ValidateError;
