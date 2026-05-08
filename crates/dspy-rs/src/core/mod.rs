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
//! Optimizer leaf discovery is internal (`visit_named_predictors_mut`) and currently
//! traverses struct fields plus `Option`, `Vec`, `HashMap<String, _>`, and `Box`.
//! `Rc`/`Arc` wrappers that contain `Predict` leaves are rejected with explicit
//! container errors.
//!
//! Most users import these through the crate root (`use dspy_rs::*`). Module authors
//! who need fine-grained prompt control also use [`SignatureSchema`] and the adapter
//! building blocks directly.

pub mod lm;
pub mod settings;

pub use dsrs_core::{
    Augmentation, Augmented, BamlConvertError, BamlType, BamlValue, CallMetadata, Constraint,
    ConstraintKind, ConstraintLevel, ConstraintResult, ConstraintSpec, ConversionError,
    DynPredictor, ErrorClass, Facet, FieldMeta, FieldMetadataSpec, FieldPath, FieldSchema, Flag,
    InputRenderSpec, JsonishError, LmError, LmUsage, Module, ModuleExt, NamedParametersError,
    OutputFormatContent, ParseError, PredictError, PredictState, Predicted, Prediction,
    RawExample, RenderOptions, ResponseCheck, Shape, Signature, SignatureSchema, StreamingMode,
    TrackedValue, TypeIR, visit_named_predictors_mut,
};
pub use lm::*;
pub use settings::*;
