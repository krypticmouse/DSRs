//! Typed, serializable errors for tool registration and execution.
//!
//! Both error enums serialize to tagged JSON (`kind` / `stage`) so an LLM in a
//! tool-authoring loop can pattern-match on the failure and repair its own
//! output (LATM-style: regenerate source on `compile`, fix the test on
//! `self_test`, resize its program on `memory_exceeded`, ...).

use serde::Serialize;

/// Error raised while executing an already-registered tool.
#[derive(Debug, Clone, Serialize, thiserror::Error)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExecError {
    /// No tool with this name is registered.
    #[error("tool `{name}` is not registered")]
    NotFound { name: String },

    /// The tool ran past its wall-clock deadline and was killed by the
    /// interrupt handler.
    #[error("tool `{name}` exceeded its {deadline_ms}ms deadline and was killed")]
    Timeout { name: String, deadline_ms: u64 },

    /// The tool exceeded the sandbox memory limit and was killed.
    #[error("tool `{name}` exceeded its memory limit of {limit_bytes} bytes")]
    MemoryExceeded { name: String, limit_bytes: usize },

    /// The tool's JavaScript threw an uncaught exception.
    #[error("tool `{name}` threw: {message}")]
    Js { name: String, message: String },

    /// The provided arguments were rejected before the sandbox was entered.
    #[error("invalid arguments for tool `{name}`: {reason}")]
    InvalidArgs { name: String, reason: String },

    /// An injected host capability returned an error (surfaced to JS as an
    /// exception; if the tool lets it propagate it is reported here).
    #[error("capability `{capability}` failed while running tool `{name}`: {message}")]
    Capability {
        name: String,
        capability: String,
        message: String,
    },

    /// The tool returned a promise that never settled (Tier 1 has no event
    /// loop: only microtasks run, so a promise must resolve without timers or
    /// external IO).
    #[error("tool `{name}` returned a promise that never settled (no event loop in the sandbox)")]
    PendingPromise { name: String },

    /// The executor itself failed (thread pool, serialization, ...). Not an
    /// LLM-repairable error.
    #[error("executor internal error: {message}")]
    Internal { message: String },
}

impl ExecError {
    /// Serialize to a JSON string an LLM can act on. Falls back to `Display`.
    pub fn to_llm_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| self.to_string())
    }
}

/// Error raised while validating/registering a tool ([LATM] lifecycle).
///
/// The `stage` tag tells the calling loop which artifact to regenerate.
///
/// [LATM]: https://arxiv.org/abs/2305.17126
#[derive(Debug, Clone, Serialize, thiserror::Error)]
#[serde(tag = "stage", rename_all = "snake_case")]
pub enum RegisterError {
    /// Tool name is empty, too long, or contains characters outside
    /// `[A-Za-z0-9_-]`.
    #[error("invalid tool name `{name}`: {reason}")]
    InvalidName { name: String, reason: String },

    /// A tool with this name is already registered.
    #[error("tool `{name}` is already registered")]
    Duplicate { name: String },

    /// The params JSON schema is structurally invalid.
    #[error("invalid params schema: {reason}")]
    InvalidSchema { reason: String },

    /// The capability name is not a valid JS identifier or is reserved.
    #[error("invalid capability name `{name}`: {reason}")]
    InvalidCapability { name: String, reason: String },

    /// The JavaScript source failed to parse/compile.
    #[error("source failed to compile: {message}")]
    Compile { message: String },

    /// The source compiled but did not evaluate to a function.
    #[error("source must evaluate to a function, got `{evaluated_type}`")]
    NotAFunction { evaluated_type: String },

    /// The self-test ran and failed (threw, or completed with `false`).
    #[error("self-test failed: {message}")]
    SelfTest { message: String },

    /// The sandbox itself failed while validating (timeout/memory during
    /// module evaluation or self-test).
    #[error("validation run failed: {0}")]
    Execution(#[from] ExecError),
}

impl RegisterError {
    /// Serialize to a JSON string an LLM can act on. Falls back to `Display`.
    pub fn to_llm_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| self.to_string())
    }
}
