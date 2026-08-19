//! Execution trace capture (RFC 0001).
//!
//! The unified trace format records one [`Span`] per [`Predict`](crate::Predict)
//! invocation into a [`Trace`] while a [`capture()`] scope is active:
//!
//! ```ignore
//! let (result, trace) = dspy_rs::trace::capture(|| module.call(input)).await;
//! for span in trace.for_component("drafter") {
//!     println!("call {}: {:?}", span.seq, span.output);
//! }
//! ```
//!
//! Spans are addressed by the same names the params system uses (fx slot names,
//! facet dotted paths), so traces join back to optimizable parameters without
//! any pointer bookkeeping. A tool-looping `Predict` stays one span; the loop's
//! provider round-trips and tool executions are its ordered [`SpanEvent`]s.
//!
//! Capture is scoped to the current tokio task — spawned subtasks do not
//! inherit it, and nested scopes are exclusive (innermost wins). With no scope
//! active, the cost is one task-local probe per `Predict` call.
//!
//! Traces serialize to JSONL via [`Trace::to_jsonl`]/[`Trace::from_jsonl`].
//!
//! A recorded trace doubles as a set of canned LM responses: [`replay()`]
//! serves `Predict` calls from it — strictly (fixtures, zero API calls) or
//! until divergence (counterfactual replay of mutated candidates).

pub mod capture;
pub mod export;
pub mod replay;
pub mod serialize;
pub mod span;

pub use capture::*;
pub use export::*;
pub use replay::{ReplayError, ReplayMode, ReplayReport, is_replaying, replay};
pub use serialize::TRACE_FORMAT_VERSION;
pub use span::*;
