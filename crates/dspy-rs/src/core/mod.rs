//! The foundational abstractions everything else is built on.
//!
//! A [`Signature`] declares what you want the LM to do — input fields, output fields,
//! and an instruction. [`SignatureSchema`] is the Facet-derived metadata for those fields,
//! cached once per type and shared by the adapter and optimizer. [`Module`] is the trait
//! every prompting strategy implements — it's deliberately narrow (`forward` takes an
//! input, returns a predicted output) so that strategies are interchangeable.
//!
//! [`Predicted`] wraps a typed output with [`CallMetadata`] (raw response text, token
//! usage, per-field parse results). The error hierarchy — [`PredictError`], [`ParseError`],
//! [`LmError`] — distinguishes LM failures from parse failures so callers can handle
//! retries differently. [`LM`] is the language model client itself.
//!
//! Optimizer leaf discovery is explicit: modules declare their [`Predict`](crate::Predict)
//! leaves by name through the [`Predictors`] trait (usually one `predictors!` line),
//! and optimizers read them through the object-safe [`PredictorInfo`] view.
//!
//! Most users import these through the crate root (`use dspy_rs::*`). Module authors
//! who need fine-grained prompt control also use [`SignatureSchema`] and the adapter
//! building blocks directly.

mod errors;
pub mod example;
pub mod lm;
pub mod module;
mod predicted;
mod schema;
pub mod settings;
pub mod signature;
mod state;

pub use errors::{ConversionError, ErrorClass, JsonishError, LmError, ParseError, PredictError};
pub use example::{ToInput, ToOutput};
pub use state::{ModuleState, PredictState};
pub use lm::*;
pub use module::*;
pub use predicted::{CallMetadata, ConstraintResult, FieldMeta, Predicted};
pub use schema::{FieldMetadataSpec, FieldPath, FieldSchema, InputRenderSpec, SignatureSchema};
pub use settings::*;
pub use signature::*;
