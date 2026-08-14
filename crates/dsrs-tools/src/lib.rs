//! # dsrs-tools: sandboxed tool execution for DSRs
//!
//! DSRs' tool runtime follows a **two-tier** design (see
//! `docs/v1-vision-report.md` §4.2/§5.5):
//!
//! - **Tier 1 (this crate, implemented): ephemeral tools.** LLM-generated
//!   JavaScript runs in an in-process QuickJS (quickjs-ng) sandbox with a
//!   per-call memory limit, an interrupt-driven wall-clock deadline, and *no
//!   ambient authority* — no filesystem, network, environment, or module
//!   loader. Host access happens only through explicitly injected
//!   [`Capability`] functions, which is also how existing DSRs tools become a
//!   JS API (the "Code Mode" pattern). The full runtime lifecycle costs on the
//!   order of 100µs, so every call gets a fresh, disposable sandbox.
//! - **Tier 2 (future): graduated tools.** Tools that prove useful graduate to
//!   Wasmtime components (pooled instantiation, epoch interruption, typed WIT
//!   interfaces). The [`Executor`] trait is the seam: subprocess, microVM, and
//!   remote executors slot in behind the same contract.
//!
//! ## Lifecycle (LATM: validate, then register)
//!
//! A [`ToolSource`] only becomes callable after passing every stage:
//!
//! 1. **shape** — name and params-schema structural checks,
//! 2. **compile** — the source must parse (bytecode is cached by BLAKE3
//!    content hash),
//! 3. **instantiate** — the module must evaluate to a function,
//! 4. **self-test** — if provided, the test must pass *inside the sandbox*.
//!
//! Failures are typed ([`RegisterError`]/[`ExecError`]) and serialize to
//! tagged JSON so a generating LLM can repair the failing artifact.
//!
//! Registered tools implement [`rig::tool::ToolDyn`], so they drop into every
//! DSRs surface that already accepts tools.
//!
//! ## Example
//!
//! ```no_run
//! use dsrs_tools::{Executor, QuickJsExecutor, ToolInvocation, ToolSource};
//! use serde_json::json;
//!
//! # async fn demo() -> Result<(), Box<dyn std::error::Error>> {
//! let executor = QuickJsExecutor::new();
//! let source = ToolSource::new(
//!     "add",
//!     "Add two numbers",
//!     json!({
//!         "type": "object",
//!         "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
//!         "required": ["x", "y"]
//!     }),
//!     "(args) => args.x + args.y",
//! )
//! .with_self_test("if (tool({x: 2, y: 3}) !== 5) throw new Error('bad math')");
//!
//! executor.register(source).await?;
//! let sum = executor
//!     .execute(ToolInvocation::new("add", json!({"x": 40, "y": 2})))
//!     .await?;
//! assert_eq!(sum, json!(42));
//! # Ok(())
//! # }
//! ```

mod capability;
mod error;
mod executor;
mod quickjs;
mod rig_tool;
mod source;

pub use capability::{Capability, CapabilityHandler};
pub use error::{ExecError, RegisterError};
pub use executor::{Executor, RegisteredTool, ToolInvocation};
pub use quickjs::{CacheStats, QuickJsExecutor, QuickJsExecutorBuilder, SandboxConfig};
pub use rig_tool::SandboxTool;
pub use source::ToolSource;

/// Re-export of the rig tool traits that sandbox tools plug into, so
/// downstream users don't need a direct (version-matched) rig dependency.
pub use rig::tool::{ToolDyn, ToolError as RigToolError};
