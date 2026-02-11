//! Data loading and example types.
//!
//! Two example types serve different layers:
//!
//! - **[`RawExample`]** (aliased from `example::Example`) — untyped key-value pairs with
//!   explicit `input_keys`/`output_keys`. Used by the data loaders, the optimizer's
//!   dynamic predictor bridge, and serialization. This is the wire format for examples.
//!
//! - **[`Example<S>`](crate::predictors::Example)** (in `predictors`) — typed input/output
//!   pair anchored to a [`Signature`](crate::Signature). Used by [`Predict`](crate::Predict)
//!   for demos and by [`TypedMetric`](crate::TypedMetric) for evaluation. This is what
//!   users work with.
//!
//! [`DataLoader`] reads JSON, CSV, Parquet, and HuggingFace datasets into `Vec<RawExample>`.
//! To use with typed modules, convert via the signature's schema.

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
