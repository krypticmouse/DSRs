//! Execution graph recording for debugging and inspection.
//!
//! Wrap a module call in [`trace()`] to capture a DAG of every [`Predict`](crate::Predict)
//! invocation, with inputs and outputs at each node. The trace is scoped — only calls
//! within the closure are recorded. The resulting [`Graph`] can be inspected or replayed
//! via the [`Executor`].
//!
//! ```ignore
//! let (result, graph) = dspy_rs::trace::trace(|| module.call(input)).await;
//! println!("{} nodes recorded", graph.nodes.len());
//! ```
//!
//! This is a debugging tool, not a performance tool. The `Mutex<Graph>` inside the
//! trace scope adds synchronization overhead. Don't trace in production hot paths.

pub mod context;
pub mod dag;
pub mod executor;
pub mod value;

pub use context::*;
pub use dag::*;
pub use executor::*;
pub use value::*;
