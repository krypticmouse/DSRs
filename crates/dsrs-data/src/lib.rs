//! Data loading and runtime row types.
//!
//! Typed ingestion is now first-class:
//!
//! - [`DataLoader`] provides `load_*` methods that return
//!   [`Example<S>`](dsrs_core::Example) directly.
//! - Typed examples flow directly into evaluation and optimizer APIs.
//!
//! The untyped row type (`RawExample`) remains for internal runtime/tracing/cache bridges.

#[cfg(any(
    feature = "csv",
    feature = "json",
    feature = "parquet",
    feature = "hf-hub"
))]
pub mod dataloader;
pub mod example {
    pub use dsrs_core::RawExample as Example;
}
pub mod prediction {
    pub use dsrs_core::Prediction;
}
pub mod serialize;
pub mod utils;

#[cfg(any(
    feature = "csv",
    feature = "json",
    feature = "parquet",
    feature = "hf-hub"
))]
pub use dataloader::*;
pub use example::*;
pub use prediction::*;
pub use serialize::*;
pub use utils::*;

pub type RawExample = dsrs_core::RawExample;
