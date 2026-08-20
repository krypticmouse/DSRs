//! Evaluation and metrics for measuring module performance.
//!
//! The evaluation loop is simple: run the module on each training example under a
//! trace [`capture`](crate::trace::capture) scope, score the result with a
//! [`TypedMetric`], collect [`Eval`]s. Optimizers use this internally, but you
//! can also call [`evaluate_trainset`] directly to benchmark your module before
//! and after optimization.
//!
//! Two kinds of metrics:
//! - **Score-only** — return [`Eval::score()`] with an `f64`. Enough for
//!   [`COPRO`](crate::COPRO) and [`MIPROv2`](crate::MIPROv2).
//! - **Score + feedback** — return [`Eval::with_feedback()`]. Required by
//!   [`GEPA`](crate::GEPA), which uses the textual feedback to guide
//!   evolutionary search.
//!
//! Metrics also receive the rollout's execution [`Trace`](crate::trace::Trace)
//! and can inspect intermediate steps per component
//! (`trace.for_component("retriever")`).

pub mod evaluator;

pub use evaluator::*;
