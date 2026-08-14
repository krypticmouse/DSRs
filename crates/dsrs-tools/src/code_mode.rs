//! The Code Mode tool: N tools collapsed into one `run_js` surface.
//!
//! Instead of advertising every tool schema and paying one round-trip per
//! call, the model sees a single `run_js` tool whose description lists the
//! available tool APIs; it writes JavaScript that calls them as plain global
//! functions and composes their results in one execution (the Cloudflare
//! Code Mode / CodeAct pattern — see `docs/v1-vision-report.md` §4.2).
//!
//! [`CodeModeTool`] is itself a [`rig::tool::ToolDyn`], so it drops into
//! every DSRs surface that accepts tools: hand `Predict` a tool set of just
//! this one tool and Code Mode is on.

use std::collections::HashMap;
use std::sync::Arc;

use rig::completion::ToolDefinition;
use rig::tool::{ToolDyn, ToolError};
use rig::wasm_compat::WasmBoxedFuture;
use serde_json::{Value, json};

use crate::capability::{Capability, js_identifier};
use crate::error::{ExecError, RegisterError};
use crate::quickjs::{SandboxConfig, run_script};

/// The name the Code Mode tool is advertised under.
pub const RUN_JS_TOOL_NAME: &str = "run_js";

/// The argument schema of the `run_js` tool: one `code` string.
pub fn run_js_parameters() -> Value {
    json!({
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "JavaScript source. Runs as an async function body: `return` a JSON-serializable value as the result."
            }
        },
        "required": ["code"]
    })
}

/// One entry of the JS API surface presented to the model.
#[derive(Debug, Clone)]
pub struct ToolApi {
    /// The JS global name the tool is callable under ([`js_identifier`]).
    pub js_name: String,
    pub description: String,
    /// JSON Schema of the tool's arguments object.
    pub parameters: Value,
}

/// Generates the default `run_js` description: the execution contract plus a
/// token-compact listing of the injected tool APIs with their argument
/// schemas.
///
/// Deliberately a plain `&[ToolApi] -> String` function: in the IR this
/// description is an optimizable `ToolDesc` parameter, and this is its
/// default value.
pub fn code_mode_description(apis: &[ToolApi]) -> String {
    let mut out = String::from(
        "Run JavaScript to call tools and compose their results in one step. \
         `code` is the body of an async function: `return` a JSON-serializable \
         value as the final result. Call the APIs below as plain global \
         functions, each taking one arguments object and returning the tool \
         result; a failed call throws a catchable Error. No filesystem, \
         network, imports, or other globals.",
    );
    if apis.is_empty() {
        return out;
    }
    out.push_str("\n\nAPIs:\n");
    for api in apis {
        out.push_str(&format!(
            "- {}(args): {} args: {}\n",
            api.js_name,
            ensure_period(&api.description),
            compact_args(&api.parameters)
        ));
    }
    out.push_str(&format!(
        "\nExample:\nconst r = {}({{...}});\nreturn r;",
        apis[0].js_name
    ));
    out
}

fn ensure_period(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return "(no description).".to_string();
    }
    if trimmed.ends_with(['.', '!', '?']) {
        trimmed.to_string()
    } else {
        format!("{trimmed}.")
    }
}

/// Token-compact rendering of an arguments object schema:
/// `{query: string, limit?: integer}` — `?` marks optional properties.
/// Non-object or property-less schemas render as `object` (free-form).
fn compact_args(schema: &Value) -> String {
    match compact_object(schema, 0) {
        Some(rendered) => rendered,
        None => "object".to_string(),
    }
}

fn compact_object(schema: &Value, depth: usize) -> Option<String> {
    if depth > 2 {
        return None;
    }
    let obj = schema.as_object()?;
    let props = obj.get("properties")?.as_object()?;
    let required: Vec<&str> = obj
        .get("required")
        .and_then(Value::as_array)
        .map(|reqs| reqs.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let fields: Vec<String> = props
        .iter()
        .map(|(name, prop)| {
            let marker = if required.contains(&name.as_str()) {
                ""
            } else {
                "?"
            };
            format!("{name}{marker}: {}", compact_type(prop, depth + 1))
        })
        .collect();
    Some(format!("{{{}}}", fields.join(", ")))
}

fn compact_type(schema: &Value, depth: usize) -> String {
    let Some(obj) = schema.as_object() else {
        return "any".to_string();
    };
    if let Some(c) = obj.get("const") {
        return c.to_string();
    }
    if let Some(values) = obj.get("enum").and_then(Value::as_array) {
        return values
            .iter()
            .map(|v| v.as_str().map(str::to_owned).unwrap_or_else(|| v.to_string()))
            .collect::<Vec<_>>()
            .join("|");
    }
    if let Some(any_of) = obj
        .get("anyOf")
        .or_else(|| obj.get("oneOf"))
        .and_then(Value::as_array)
    {
        return any_of
            .iter()
            .map(|v| compact_type(v, depth))
            .collect::<Vec<_>>()
            .join("|");
    }
    match obj.get("type").and_then(Value::as_str) {
        Some("array") => match obj.get("items") {
            Some(items) => format!("{}[]", compact_type(items, depth)),
            None => "any[]".to_string(),
        },
        Some("object") => compact_object(schema, depth).unwrap_or_else(|| "object".to_string()),
        Some(other) => other.to_string(),
        None => "any".to_string(),
    }
}

/// A single rig tool (`run_js`) that executes model-written JavaScript in the
/// sandbox with a fixed set of DSRs tools injected as JS globals.
///
/// # Error contract
///
/// Model-repairable failures — script errors, capability/tool failures,
/// deadline and memory kills, bad arguments — are returned as `Ok` with the
/// typed error serialized to JSON ([`ExecError::to_llm_json`]), so an outer
/// tool loop feeds them back to the model instead of aborting. Only
/// host-internal faults ([`ExecError::Internal`]) surface as `Err`.
pub struct CodeModeTool {
    description: String,
    capabilities: Vec<Capability>,
    config: SandboxConfig,
}

impl CodeModeTool {
    /// Wraps `tools` as sandbox capabilities and builds the `run_js` tool.
    ///
    /// Each tool's definition is fetched once here; names are mangled to JS
    /// identifiers per [`js_identifier`], and two tools mangling to the same
    /// identifier are refused ([`RegisterError::InvalidCapability`]). The
    /// description is generated by [`code_mode_description`]; override it
    /// with [`with_description`](Self::with_description).
    pub async fn new(
        tools: Vec<Arc<dyn ToolDyn>>,
        config: SandboxConfig,
    ) -> Result<Self, RegisterError> {
        let mut seen: HashMap<String, String> = HashMap::with_capacity(tools.len());
        let mut apis = Vec::with_capacity(tools.len());
        let mut capabilities = Vec::with_capacity(tools.len());
        for tool in &tools {
            let definition = tool.definition(String::new()).await;
            let js_name = js_identifier(&definition.name);
            if let Some(previous) = seen.insert(js_name.clone(), definition.name.clone()) {
                return Err(RegisterError::InvalidCapability {
                    name: js_name.clone(),
                    reason: format!(
                        "tool names `{previous}` and `{}` both mangle to the JS identifier `{js_name}`",
                        definition.name
                    ),
                });
            }
            apis.push(ToolApi {
                js_name: js_name.clone(),
                description: definition.description.clone(),
                parameters: definition.parameters.clone(),
            });
            capabilities.push(Capability::wrap_tool(
                js_name,
                definition.description,
                definition.name,
                Arc::clone(tool),
            ));
        }
        Ok(Self {
            description: code_mode_description(&apis),
            capabilities,
            config,
        })
    }

    /// Replace the auto-generated description (the optimizable-`ToolDesc`
    /// seam: an optimizer proposes description variants, the tool surface
    /// stays fixed).
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    pub fn description(&self) -> &str {
        &self.description
    }

    pub fn config(&self) -> SandboxConfig {
        self.config
    }
}

impl std::fmt::Debug for CodeModeTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CodeModeTool")
            .field("config", &self.config)
            .field(
                "capabilities",
                &self
                    .capabilities
                    .iter()
                    .map(Capability::name)
                    .collect::<Vec<_>>(),
            )
            .finish_non_exhaustive()
    }
}

impl ToolDyn for CodeModeTool {
    fn name(&self) -> String {
        RUN_JS_TOOL_NAME.to_string()
    }

    fn definition<'a>(&'a self, _prompt: String) -> WasmBoxedFuture<'a, ToolDefinition> {
        Box::pin(async move {
            ToolDefinition {
                name: RUN_JS_TOOL_NAME.to_string(),
                description: self.description.clone(),
                parameters: run_js_parameters(),
            }
        })
    }

    fn call<'a>(&'a self, args: String) -> WasmBoxedFuture<'a, Result<String, ToolError>> {
        Box::pin(async move {
            let invalid = |message: String| {
                Ok(json!({"kind": "invalid_args", "name": RUN_JS_TOOL_NAME, "reason": message})
                    .to_string())
            };
            let args: Value = match serde_json::from_str(&args) {
                Ok(value) => value,
                Err(err) => return invalid(format!("arguments must be a JSON object: {err}")),
            };
            let Some(code) = args.get("code").and_then(Value::as_str) else {
                return invalid("missing required string argument `code`".to_string());
            };
            match run_script(code, self.capabilities.clone(), self.config).await {
                Ok(value) => Ok(value.to_string()),
                Err(err @ ExecError::Internal { .. }) => {
                    Err(ToolError::ToolCallError(err.to_llm_json().into()))
                }
                // Model-repairable: hand the typed error back as the result.
                Err(err) => Ok(err.to_llm_json()),
            }
        })
    }
}
