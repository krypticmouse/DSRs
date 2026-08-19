//! Data loading, versioned under [`v1`].

pub mod v1;

// Re-export items and submodules so both `crate::data::DataLoader` and
// versioned paths like `crate::data::v1::dataloader::DataLoader` keep resolving.
pub use v1::*;
