//! LM response caching.
//!
//! The [`ResponseCache`] provides a hybrid memory + disk cache backed by
//! [foyer](https://docs.rs/foyer). Execution history lives independently in
//! [`core::history`](crate::core::history).
//!
//! Caching is per-LM-instance and keyed on the full prompt content. Cache entries
//! are not shared across LM instances.

pub mod cache;
pub mod hash;
pub mod telemetry;

pub use cache::{CacheEntry, CacheKey, ResponseCache};
pub use telemetry::{TelemetryInitError, init_tracing, truncate};
