//! The foundational abstractions everything else is built on.
//!
//! A [`Signature`] declares what you want the LM to do — input fields, output fields,
//! and an instruction. [`SignatureSchema`] is the Facet-derived metadata for those fields,
//! cached once per type and shared by the adapter and optimizer. [`Module`] is the trait
//! every prompting strategy implements — it's deliberately narrow (`forward` takes an
//! input, returns a predicted output) so that strategies are interchangeable.
//!
//! [`Predicted`] wraps a typed output with [`CallMetadata`] (raw response text, token
//! usage, per-field parse results) and [`Chat`] (the conversation history from the LM
//! call). The error hierarchy — [`PredictError`], [`ParseError`],
//! [`LmError`] — distinguishes LM failures from parse failures so callers can handle
//! retries differently. [`LM`] is the language model client itself.
//!
//! Optimizer leaf discovery is internal (`visit_named_predictors_mut`) and currently
//! traverses struct fields plus `Option`, `Vec`, `HashMap<String, _>`, and `Box`.
//! `Rc`/`Arc` wrappers that contain `Predict` leaves are rejected with explicit
//! container errors.
//!
//! Most users import these through the crate root (`use dspy_rs::*`). Module authors
//! who need fine-grained prompt control also use [`SignatureSchema`] and the adapter
//! building blocks directly.

pub(crate) mod dyn_predictor;
mod errors;
pub mod lm;
pub mod module;
mod module_ext;
mod predicted;
mod schema;
pub mod settings;
pub mod signature;
pub mod specials;

pub(crate) use dyn_predictor::*;
pub use errors::{ConversionError, ErrorClass, JsonishError, LmError, ParseError, PredictError};
pub use lm::*;
pub use module::*;
pub use module_ext::*;
pub use predicted::{CallMetadata, ConstraintResult, FieldMeta, Predicted};
pub use schema::{FieldMetadataSpec, FieldPath, FieldSchema, InputRenderSpec, SignatureSchema};
pub use settings::*;
pub use signature::*;
pub use specials::*;
