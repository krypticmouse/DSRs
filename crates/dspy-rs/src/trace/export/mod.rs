//! Exports: projections of the trace format onto external training and
//! observability conventions (RFC 0001 §4f/§4g).
//!
//! Everything here is a pure serialization-side projection — no new capture
//! machinery, no external dependencies.

pub mod otel;
pub mod rl;

pub use otel::{OtelEvent, OtelKeyValue, OtelSpan, OtelStatus, OtelValue};
pub use rl::{RlRollout, RlTransition};
