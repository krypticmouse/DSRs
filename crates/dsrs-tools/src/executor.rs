//! The [`Executor`] trait: the escape hatch between DSRs and *how* a tool
//! actually runs.
//!
//! Tier 1 ships an in-process QuickJS implementation
//! ([`QuickJsExecutor`](crate::QuickJsExecutor)). The trait is deliberately
//! narrow — register, validate, execute, enumerate — so the same contract can
//! later be backed by:
//!
//! - **Tier 2**: Wasmtime components (pooling allocator, epoch interruption),
//! - subprocess executors hardened with seccomp/landlock,
//! - microVM executors (Hyperlight), or
//! - remote execution services.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{ExecError, RegisterError};
use crate::source::ToolSource;

/// A single tool call: name plus JSON arguments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInvocation {
    pub name: String,
    pub args: Value,
}

impl ToolInvocation {
    pub fn new(name: impl Into<String>, args: Value) -> Self {
        Self {
            name: name.into(),
            args,
        }
    }
}

/// Metadata for a tool that survived the full validation lifecycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisteredTool {
    pub name: String,
    pub description: String,
    /// JSON Schema of the arguments, as declared in the [`ToolSource`].
    pub parameters: Value,
    /// Hex BLAKE3 hash of the JavaScript source; the bytecode-cache key.
    pub source_hash: String,
    /// Whether the tool passed an explicit self-test (`false` means no
    /// self-test was provided; failing tools are never registered).
    pub self_tested: bool,
}

/// An engine that can validate, register, and execute tool invocations.
///
/// Implementations must be `Send + Sync`: DSRs optimizers evaluate candidate
/// programs concurrently, and tool execution sits on that hot path.
#[async_trait]
pub trait Executor: Send + Sync {
    /// Cheap synchronous structural validation (name, schema shape). Does not
    /// touch a sandbox; [`register`](Self::register) runs the full pipeline.
    fn validate(&self, source: &ToolSource) -> Result<(), RegisterError>;

    /// Run the full validate-then-register lifecycle:
    /// parse/compile -> schema validation -> sandboxed self-test -> register.
    ///
    /// Only sources that pass every stage become executable. Errors are typed
    /// per stage so a generating LLM can repair the specific artifact.
    async fn register(&self, source: ToolSource) -> Result<RegisteredTool, RegisterError>;

    /// Execute a registered tool with JSON args, returning its JSON result.
    async fn execute(&self, invocation: ToolInvocation) -> Result<Value, ExecError>;

    /// Metadata for one registered tool, if present.
    fn tool(&self, name: &str) -> Option<RegisteredTool>;

    /// Metadata for every registered tool.
    fn tools(&self) -> Vec<RegisteredTool>;

    /// Remove a tool. Returns `true` if it was registered.
    fn deregister(&self, name: &str) -> bool;
}
