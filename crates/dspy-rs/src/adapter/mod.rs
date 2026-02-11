//! Prompt formatting and LM response parsing.
//!
//! The adapter turns a [`SignatureSchema`](crate::SignatureSchema) into prompts and parses
//! LM responses back into typed values. All prompts use the `[[ ## field_name ## ]]`
//! delimiter protocol — input fields, output fields, and the `[[ ## completed ## ]]`
//! marker that signals the end of the response.
//!
//! Most users never touch this — [`Predict`](crate::Predict) calls the adapter internally.
//! Module authors who need fine-grained control over prompt construction use the
//! building blocks directly: [`build_system`](ChatAdapter::build_system),
//! [`format_input`](ChatAdapter::format_input),
//! [`parse_output`](ChatAdapter::parse_output).

pub mod chat;

pub use chat::*;

/// Marker trait for configurable adapters.
///
/// Typed call paths currently use `ChatAdapter` directly, while global settings keep
/// an adapter instance to preserve public configuration shape.
pub trait Adapter: Send + Sync + 'static {}
