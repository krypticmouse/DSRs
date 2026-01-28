#![cfg(feature = "rlm")]

pub mod config;
pub mod command;
pub mod adapter;
pub mod exec;
pub mod error;
mod history;
pub mod prompt;
pub mod submit;
pub mod tools;
mod rlm;

pub use adapter::RlmAdapter;
pub use config::{ConstraintSummary, RlmConfig, RlmResult};
pub use command::Command;
pub use exec::execute_repl_code;
pub use error::RlmError;
pub use prompt::{
    ACTION_INSTRUCTIONS_TEMPLATE, format_baml_shape, generate_output_schema_description,
    generate_typed_preamble,
};
pub use tools::LlmTools;
pub use rlm::{Rlm, RlmBuilder};

/// Backwards-compatible alias for `Rlm`.
#[deprecated(since = "0.8.0", note = "Use Rlm instead")]
pub type TypedRlm<S> = Rlm<S>;
