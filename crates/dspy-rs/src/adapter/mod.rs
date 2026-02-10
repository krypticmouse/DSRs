pub mod chat;

pub use chat::*;

/// Marker trait for configurable adapters.
///
/// Typed call paths currently use `ChatAdapter` directly, while global settings keep
/// an adapter instance to preserve public configuration shape.
pub trait Adapter: Send + Sync + 'static {}
