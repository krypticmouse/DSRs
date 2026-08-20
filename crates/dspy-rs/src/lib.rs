//! Typed prompt engineering and LM program optimization.
//!
//! DSRs is a Rust port of [DSPy](https://github.com/stanfordnlp/dspy): you declare what
//! you want the LM to produce (a [`Signature`]), pick a prompting strategy (a [`Module`]
//! like [`Predict`] or [`ChainOfThought`]), and let an [`Optimizer`] tune the program's
//! instructions and demos on your training data. The type system enforces correctness
//! at every layer — field types, strategy swaps, and augmentation composition are all
//! compile-time checked.
//!
//! # The mental model
//!
//! Three concepts, three layers:
//!
//! | Layer | Concept | Key types | Who |
//! |-------|---------|-----------|-----|
//! | **Signatures** | "Given these inputs, produce these outputs" | [`Signature`], `#[derive(Signature)]` | Everyone |
//! | **Modules** | Prompting strategies that implement a signature | [`Module`], [`Predict`], [`ChainOfThought`] | Everyone |
//! | **Optimization** | Auto-tuning instructions and demos | [`Optimizer`], [`COPRO`], [`GEPA`], [`MIPROv2`] | When you need better results |
//!
//! A [`Predict`] is the leaf — the only thing that actually calls the LM. Every other
//! module ([`ChainOfThought`], custom pipelines) delegates to one or more `Predict` leaves.
//! Modules name their leaves explicitly via [`Predictors`] (see the `predictors!` macro);
//! optimizers tune those leaves' instructions and few-shot demos by injecting candidates
//! ambiently per call, installing only the winner.
//!
//! # Quick start
//!
//! The recommended import is [`prelude`] — the curated core surface. (The
//! crate root also glob re-exports everything for backwards compatibility.)
//!
//! ```no_run
//! use dspy_rs::prelude::*;
//!
//! #[derive(Signature, Clone, Debug)]
//! /// Answer questions accurately and concisely.
//! struct QA {
//!     #[input] question: String,
//!     #[output] answer: String,
//! }
//!
//! # async fn example() -> Result<(), PredictError> {
//! // 1. Configure the LM
//! let lm = LM::builder()
//!     .model("openai:gpt-4o-mini".to_string())
//!     .build()
//!     .await
//!     .unwrap();
//! dspy_rs::configure(lm);
//!
//! // 2. Pick a strategy
//! let cot = ChainOfThought::<QA>::new();
//!
//! // 3. Call it
//! let result = cot.call(QAInput { question: "What is 2+2?".into() }).await?;
//! println!("{}", result.reasoning);  // chain-of-thought text
//! println!("{}", result.answer);     // the actual answer, via Deref
//! # Ok(())
//! # }
//! ```
//!
//! `ChainOfThought<QA>` returns [`Predicted<WithReasoning<QAOutput>>`](Predicted), not
//! `Predicted<QAOutput>`. You access `.reasoning` directly and `.answer` through auto-deref
//! ([`WithReasoning<O>`] derefs to `O`). This pattern holds for all augmentations — the
//! compiler tells you what changed when you swap strategies.
//!
//! # What doesn't work (yet)
//!
//! - **No structural optimization.** The [`ir`] layer ships a dynamic program
//!   graph ([`ir::Program`]) with an interpreter, an edit calculus
//!   ([`ir::Edit`]), and the `.dsrs` text format, but the shipped optimizers
//!   only tune instructions and demos — none of them rewrite graph structure
//!   yet.
//! - **No `BestOfN`, `Refine`, or other advanced modules** beyond
//!   [`ChainOfThought`]. Agentic tool loops live in the IR instead
//!   (`AgentLoopNode` via the `#[agent]` macro); the module trait and
//!   augmentation system could host the rest, but nobody's built them.
//! - **`CallMetadata` is not extensible.** Modules can't attach custom metadata (e.g.
//!   "which attempt won in BestOfN"). This should probably be a trait with associated
//!   types, but it isn't.
//! - **Leaf discovery is explicit.** Optimizable [`Predict`] leaves are whatever a
//!   module declares in its [`Predictors`] impl — there is no reflection walker.
//!   A leaf you forget to declare simply isn't optimized or persisted.
//!
//! # Crate organization
//!
//! - [`adapter`] — Prompt formatting and LM response parsing ([`ChatAdapter`])
//! - [`core`] — [`Module`] trait, [`Signature`] trait, [`SignatureSchema`], error types,
//!   LM client, [`Predicted`] and [`CallMetadata`]
//! - [`predictors`] — [`Predict`] (the leaf module) and typed [`Demo`]
//! - [`modules`] — [`ChainOfThought`] and augmentation types
//! - [`evaluate`] — [`TypedMetric`] trait, [`evaluate_trainset`], scoring utilities
//! - [`optimizer`] — [`Optimizer`] trait, [`COPRO`], [`GEPA`], [`MIPROv2`]
//! - [`ir`] — dynamic program graph, interpreter, and the `.dsrs` text format
//! - [`data`] — [`DataLoader`] for JSON/CSV/Parquet/HuggingFace datasets
//! - [`trace`] — Execution trace capture (spans per `Predict` call, JSONL serialization)
//! - [`utils`] — Response caching

extern crate self as dspy_rs;

pub mod adapter;
pub mod augmentation;
pub mod core;
pub mod data;
pub mod evaluate;
pub mod fx;
pub mod ir;
pub mod modules;
pub mod optimizer;
pub mod predictors;
pub mod trace;
pub mod utils;

pub use adapter::chat::*;
pub use augmentation::*;
pub use core::*;
pub use data::dataloader::*;
pub use data::utils::*;
pub use evaluate::*;
pub use modules::*;
pub use optimizer::*;
pub use predictors::*;
// The unified trace format (RFC 0001).
pub use trace::{
    CompId, Eval, JsonMap, ModelEntry, ModelId, PrefixEntry, PrefixId, ReplayError, ReplayMode,
    ReplayReport, Span, SpanError, SpanErrorKind, SpanEvent, SpanGuard, SpanId, SpanOutcome,
    SpanRequest, Trace, TraceMeta, TraceOutcome, begin_span, capture, capture_with_meta,
    is_capturing, is_replaying, replay,
};
pub use utils::*;

// Code Mode (vision report §5.5): tools as a sandboxed JS API. See
// `ToolSet::code_mode` for the module lane and `RuntimeEnv` for the IR lane.
pub use dsrs_tools::{Capability, CodeModeTool, RUN_JS_TOOL_NAME, SandboxConfig};

pub mod typesys;
pub use dsrs_macros::*;
pub use facet::{Facet, Shape};
pub use typesys::{Constraint, ConstraintLevel, FieldType, Flag, OutputSchema, Schema};

/// The curated core surface — the recommended import for DSRs programs.
///
/// ```no_run
/// use dspy_rs::prelude::*;
/// ```
///
/// Covers the basic path end to end: declare a [`Signature`], pick a module
/// ([`Predict`], [`ChainOfThought`]), [`configure`] an [`LM`], load data,
/// evaluate with a [`TypedMetric`], optimize with any [`Optimizer`], and
/// capture/replay traces. The crate root's glob re-exports remain for the
/// long tail (adapters, schema internals, fx, the full [`ir`] surface).
pub mod prelude {
    // Signatures: the trait and the derive share a name across namespaces.
    pub use crate::core::signature::Signature;
    pub use dsrs_macros::{Example, Signature};

    // Modules and predictors.
    pub use crate::core::{
        CallMetadata, Module, ModuleState, PredictError, PredictState, Predicted, Predictors,
    };
    pub use crate::modules::{ChainOfThought, WithReasoning};
    pub use crate::predictors::{Demo, Predict};
    // The `predictors!` leaf-declaration macro (and, same name, the module).
    pub use crate::predictors;

    // LM configuration.
    pub use crate::core::lm::{LM, LMConfig};
    pub use crate::core::settings::configure;

    // Data loading.
    pub use crate::data::dataloader::{DataLoader, TypedLoadOptions};

    // Evaluation.
    pub use crate::evaluate::{
        TypedMetric, average_score, evaluate_trainset, evaluate_trainset_with_concurrency,
    };
    pub use crate::trace::Eval;

    // Optimization: the trait, the five strategies, and the engine surface.
    pub use crate::optimizer::{
        BootstrapFewShot, COPRO, Candidate, Engine, GEPA, MIPROv2, OptimizeTarget, Optimizer,
        SIMBA,
    };

    // Trace capture and replay entry points.
    pub use crate::trace::{
        ReplayMode, ReplayReport, Trace, capture, capture_with_meta, is_capturing, is_replaying,
        replay,
    };

    // The IR: programs as data.
    pub use crate::ir::{Edit, Interpreter, Overlay, Program};

    // Telemetry sugar every quickstart uses.
    pub use crate::utils::init_tracing;
}

/// Pre-built signature for use in doc examples. Not part of the public API.
#[doc(hidden)]
pub mod doctest {
    #[derive(crate::Signature, Clone, Debug)]
    /// Answer questions accurately and concisely.
    pub struct QA {
        #[input]
        pub question: String,
        #[output]
        pub answer: String,
    }
}

#[doc(hidden)]
pub mod __macro_support {
    pub use anyhow;
    pub use facet;
    pub use indexmap;
    pub use rig;
    pub use serde;
    pub use serde_json;
    pub use tokio;
}


