//! Data loading and runtime row types.
//!
//! Typed ingestion is now first-class:
//!
//! - [`DataLoader`] provides `load_*` methods that return
//!   [`Example<S>`](crate::predictors::Example) directly.
//! - Typed examples flow directly into evaluation and optimizer APIs.
//!
//! The untyped row type (`RawExample`) remains for internal runtime/tracing/cache bridges.

pub mod dataloader;
pub mod example;
pub mod prediction;
pub mod serialize;
pub mod utils;

pub use dataloader::*;
pub use example::*;
pub use prediction::*;
pub use serialize::*;
pub use utils::*;

pub type RawExample = example::Example;
