//! Data loading and runtime row types, versioned under [`v1`].

pub mod v1;

// Re-export items and submodules so both `crate::data::RawExample` and
// pre-v1 module paths like `crate::data::example::Example` keep resolving.
pub use v1::*;
