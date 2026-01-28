//! Core traits for RLM type description and introspection.
//!
//! This crate provides the foundational traits and types for describing
//! Rust types to language models. It is used by `rlm-derive` to generate
//! implementations automatically.
//!
//! # Overview
//!
//! The main components are:
//!
//! - [`RlmDescribe`] - Core trait for type description
//! - [`RlmTypeInfo`] - Compile-time type metadata
//! - [`RlmFieldDesc`] - Field description struct
//! - [`RlmPropertyDesc`] - Computed property description struct
//!
//! # Example
//!
//! ```ignore
//! use rlm_core::{RlmDescribe, RlmFieldDesc};
//!
//! struct User {
//!     name: String,
//!     age: u32,
//! }
//!
//! impl RlmDescribe for User {
//!     fn type_name() -> &'static str { "User" }
//!
//!     fn fields() -> Vec<RlmFieldDesc> {
//!         vec![
//!             RlmFieldDesc::new("name", "String").with_desc("User's name"),
//!             RlmFieldDesc::new("age", "u32").with_desc("User's age"),
//!         ]
//!     }
//!
//!     fn describe_value(&self) -> String {
//!         format!("User(name={}, age={})", self.name, self.age)
//!     }
//! }
//! ```

pub mod describe;
pub mod variable;
#[cfg(feature = "pyo3")]
pub mod input;

// Re-export main types at crate root for convenience
pub use describe::{RlmDescribe, RlmFieldDesc, RlmPropertyDesc, RlmTypeInfo};
pub use variable::RlmVariable;
#[cfg(feature = "pyo3")]
pub use input::RlmInputFields;
