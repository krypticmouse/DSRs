//! Exports: projections of the trace format onto external training and
//! observability conventions (RFC 0001 §4f/§4g).
//!
//! Everything here is a pure serialization-side projection — no new capture
//! machinery, no external dependencies.

pub mod rl;

pub use rl::{RlRollout, RlTransition};
