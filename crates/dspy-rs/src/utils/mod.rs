//! LM response caching.
//!
//! The [`ResponseCache`] provides a hybrid memory + disk cache backed by
//! [foyer](https://docs.rs/foyer). It also maintains a sliding window of recent
//! entries for [`LM::inspect_history`](crate::LM::inspect_history).
//!
//! Caching is per-LM-instance and keyed on the full prompt content. Cache entries
//! are not shared across LM instances.

pub mod cache;
pub mod hash;
pub mod telemetry;

pub use cache::{CacheEntry, CacheKey, ResponseCache};
pub use telemetry::{TelemetryInitError, init_tracing, truncate};
