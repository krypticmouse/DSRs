//! LM response caching.
//!
//! With the `cache` feature, `ResponseCache` provides a hybrid memory + disk cache backed by
//! [foyer](https://docs.rs/foyer). Execution history lives independently in
//! [`core::history`](crate::core::history).
//!
//! Caching is per-LM-instance and keyed on the full prompt content. Cache entries
//! are not shared across LM instances.

#[cfg(feature = "cache")]
pub mod cache;
pub mod hash;
pub mod telemetry;

#[cfg(feature = "cache")]
pub use cache::{CacheEntry, CacheError, CacheKey, ResponseCache};
pub use telemetry::{TelemetryInitError, init_tracing, truncate};
