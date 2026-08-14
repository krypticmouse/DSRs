//! Explicit host capabilities: the only doorway out of the sandbox.
//!
//! A fresh QuickJS runtime has no filesystem, no network, no environment, no
//! clock beyond `Date`, and no module loader. Anything a tool needs from the
//! host must be injected as a [`Capability`] — an async Rust function exposed
//! to JavaScript as a global. This is how existing DSRs tools become a JS API
//! (the Code Mode pattern): wrap each `rig::tool::ToolDyn` as a capability and
//! generated code can call it directly.

use std::fmt;
use std::future::Future;
use std::sync::Arc;

use futures::future::BoxFuture;
use serde_json::Value;

use crate::error::RegisterError;

/// Boxed async host function: JSON in, JSON out, `Err(String)` surfaces to the
/// sandbox as a JavaScript exception.
pub type CapabilityHandler =
    Arc<dyn Fn(Value) -> BoxFuture<'static, Result<Value, String>> + Send + Sync>;

/// An async Rust function injected into the sandbox as a global JS function.
///
/// From JavaScript the capability looks synchronous — `const rows = query({q:
/// "..."})` — the executor bridges the call onto the host's Tokio runtime and
/// blocks the sandbox thread until it resolves. Capability calls are host
/// code: the sandbox deadline cannot interrupt them mid-flight (it re-arms as
/// soon as control returns to JS), so handlers should enforce their own
/// timeouts.
#[derive(Clone)]
pub struct Capability {
    name: String,
    description: String,
    handler: CapabilityHandler,
}

impl Capability {
    /// Create a capability from an async closure.
    ///
    /// ```no_run
    /// # use dsrs_tools::Capability;
    /// let double = Capability::new("double", "double a number", |args| async move {
    ///     let n = args["n"].as_f64().ok_or("expected {n: number}")?;
    ///     Ok(serde_json::json!(n * 2.0))
    /// });
    /// ```
    pub fn new<F, Fut>(name: impl Into<String>, description: impl Into<String>, f: F) -> Self
    where
        F: Fn(Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Value, String>> + Send + 'static,
    {
        Self {
            name: name.into(),
            description: description.into(),
            handler: Arc::new(move |args| Box::pin(f(args))),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn description(&self) -> &str {
        &self.description
    }

    pub(crate) fn handler(&self) -> CapabilityHandler {
        Arc::clone(&self.handler)
    }

    /// Capability names become JS globals, so they must be valid identifiers
    /// and must not collide with the runtime's reserved `__dsrs_*` namespace.
    pub(crate) fn validate_name(name: &str) -> Result<(), RegisterError> {
        let invalid = |reason: &str| RegisterError::InvalidCapability {
            name: name.to_string(),
            reason: reason.to_string(),
        };
        let mut chars = name.chars();
        let Some(first) = chars.next() else {
            return Err(invalid("name is empty"));
        };
        if !(first.is_ascii_alphabetic() || first == '_' || first == '$') {
            return Err(invalid("must start with a letter, `_` or `$`"));
        }
        if !chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '$') {
            return Err(invalid(
                "may only contain ASCII letters, digits, `_` and `$`",
            ));
        }
        if name.starts_with("__dsrs") {
            return Err(invalid("the `__dsrs` prefix is reserved by the runtime"));
        }
        Ok(())
    }
}

impl fmt::Debug for Capability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Capability")
            .field("name", &self.name)
            .field("description", &self.description)
            .finish_non_exhaustive()
    }
}
