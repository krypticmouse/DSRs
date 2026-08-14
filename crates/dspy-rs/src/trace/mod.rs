//! Execution trace capture (RFC 0001) and the legacy execution graph.
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
//! Capture is scoped to the current tokio task — spawned subtasks do not
//! inherit it, and nested scopes are exclusive (innermost wins). With no scope
//! active, the cost is one task-local probe per `Predict` call.
//!
//! The legacy `trace()`/[`Graph`] API is deprecated by `capture()` and will be
//! removed; both record during the transition.

pub mod capture;
pub mod context;
pub mod dag;
pub mod serialize;
pub mod span;
pub mod value;

pub use capture::*;
pub use context::*;
pub use dag::*;
pub use serialize::TRACE_FORMAT_VERSION;
pub use span::*;
pub use value::*;
