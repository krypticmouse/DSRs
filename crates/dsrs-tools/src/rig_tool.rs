//! Bridge from sandboxed tools to [`rig::tool::ToolDyn`], the trait DSRs
//! already threads through `Predict`, `ChainOfThought`, and `ReAct`. A
//! graduated ephemeral tool is indistinguishable from a hand-written one.

use std::sync::Arc;

use rig::completion::ToolDefinition;
use rig::tool::{ToolDyn, ToolError};
use rig::wasm_compat::WasmBoxedFuture;
use serde_json::Value;

use crate::executor::{Executor, RegisteredTool, ToolInvocation};

/// A validated sandbox tool exposed as a [`rig::tool::ToolDyn`].
///
/// Obtain one via [`QuickJsExecutor::register_rig`](crate::QuickJsExecutor::register_rig)
/// or [`QuickJsExecutor::rig_tool`](crate::QuickJsExecutor::rig_tool); every
/// call round-trips through the owning [`Executor`], so limits, capabilities,
/// and the bytecode cache all apply.
pub struct SandboxTool {
    executor: Arc<dyn Executor>,
    meta: RegisteredTool,
}

impl SandboxTool {
    pub fn new(executor: Arc<dyn Executor>, meta: RegisteredTool) -> Self {
        Self { executor, meta }
    }

    pub fn meta(&self) -> &RegisteredTool {
        &self.meta
    }
}

impl ToolDyn for SandboxTool {
    fn name(&self) -> String {
        self.meta.name.clone()
    }

    fn definition<'a>(&'a self, _prompt: String) -> WasmBoxedFuture<'a, ToolDefinition> {
        Box::pin(async move {
            ToolDefinition {
                name: self.meta.name.clone(),
                description: self.meta.description.clone(),
                parameters: self.meta.parameters.clone(),
            }
        })
    }

    fn call<'a>(&'a self, args: String) -> WasmBoxedFuture<'a, Result<String, ToolError>> {
        Box::pin(async move {
            let args: Value = if args.trim().is_empty() {
                Value::Object(serde_json::Map::new())
            } else {
                serde_json::from_str(&args).map_err(ToolError::JsonError)?
            };
            match self
                .executor
                .execute(ToolInvocation::new(self.meta.name.clone(), args))
                .await
            {
                Ok(value) => Ok(value.to_string()),
                // Surface the structured error JSON so the model sees the
                // typed failure, not just prose.
                Err(err) => Err(ToolError::ToolCallError(err.to_llm_json().into())),
            }
        })
    }
}
