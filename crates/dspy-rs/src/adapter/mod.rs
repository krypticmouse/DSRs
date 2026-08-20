//! Prompt formatting and LM response parsing.
//!
//! The adapter turns a [`SignatureDef`](crate::ir::SignatureDef) into prompts and parses
//! LM responses back into value-level maps. All prompts use the `[[ ## field_name ## ]]`
//! delimiter protocol — input fields, output fields, and the `[[ ## completed ## ]]`
//! marker that signals the end of the response.
//!
//! Most users never touch this — [`Predict`](crate::Predict) renders and parses through
//! the adapter via the IR interpreter. Module authors who need fine-grained control over
//! prompt construction use the building blocks directly:
//! [`build_system_def`](ChatAdapter::build_system_def),
//! [`format_input_def`](ChatAdapter::format_input_def),
//! [`format_output_def`](ChatAdapter::format_output_def),
//! [`parse_output_def`](ChatAdapter::parse_output_def).

pub mod chat;

pub use chat::*;
