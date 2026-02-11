//! Evaluation and metrics for measuring module performance.
//!
//! The evaluation loop is simple: run the module on each training example, score the
//! result with a [`TypedMetric`], collect [`MetricOutcome`]s. Optimizers use this
//! internally, but you can also call [`evaluate_trainset`] directly to benchmark
//! your module before and after optimization.
//!
//! Two kinds of metrics:
//! - **Score-only** — return [`MetricOutcome::score()`] with a `f32`. Enough for
//!   [`COPRO`](crate::COPRO) and [`MIPROv2`](crate::MIPROv2).
//! - **Score + feedback** — return [`MetricOutcome::with_feedback()`] with a
//!   [`FeedbackMetric`]. Required by [`GEPA`](crate::GEPA), which uses the textual
//!   feedback to guide evolutionary search.

pub mod evaluator;
pub mod feedback;
pub mod feedback_helpers;
pub mod metrics;

pub use evaluator::*;
pub use feedback::*;
pub use feedback_helpers::*;
