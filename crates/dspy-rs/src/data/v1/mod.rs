//! Data loading.
//!
//! Typed ingestion is first-class:
//!
//! - [`DataLoader`] provides `load_*` methods that return
//!   [`Example<S>`](crate::predictors::Example) directly.
//! - Typed examples flow directly into evaluation and optimizer APIs.
//!
//! There is no untyped row type: custom mappers work with [`RowRecord`]
//! (`serde_json`-valued source rows) at the load boundary, and demo rows
//! travel as flat JSON objects (see [`crate::PredictState`]).

pub mod dataloader;
pub mod utils;

pub use dataloader::*;
pub use utils::*;
