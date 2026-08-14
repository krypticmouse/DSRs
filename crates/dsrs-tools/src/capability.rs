//! Explicit host capabilities: the only doorway out of the sandbox.
//!
//! A fresh QuickJS runtime has no filesystem, no network, no environment, no
//! clock beyond `Date`, and no module loader. Anything a tool needs from the
//! host must be injected as a [`Capability`] — an async Rust function exposed
//! to JavaScript as a global. This is how existing DSRs tools become a JS API
//! (the Code Mode pattern): wrap each `rig::tool::ToolDyn` as a capability and
//! generated code can call it directly.

use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::sync::Arc;

use futures::future::BoxFuture;
use rig::tool::ToolDyn;
use serde_json::Value;

use crate::error::RegisterError;

/// Mangle an arbitrary tool name into a valid JavaScript identifier.
///
/// The mangling rule, in order:
///
/// 1. every character outside `[A-Za-z0-9_$]` becomes `_`
///    (`my-tool.v2` → `my_tool_v2`),
/// 2. if the first character is a digit, a `_` is prepended
///    (`2fast` → `_2fast`),
/// 3. an empty name becomes `_tool`,
/// 4. a result starting with the runtime-reserved `__dsrs` prefix gets one
///    more leading `_` (`__dsrs_x` → `___dsrs_x`).
///
/// The result always passes capability-name validation. The mapping is not
/// injective — distinct tool names can mangle to the same identifier —
/// so batch wrappers ([`Capability::from_toolset`]) error on collision at
/// registration time rather than silently shadowing a tool.
pub fn js_identifier(name: &str) -> String {
    let mut out: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '$' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if out.is_empty() {
        return "_tool".to_string();
    }
    if out.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        out.insert(0, '_');
    }
    if out.starts_with("__dsrs") {
        out.insert(0, '_');
    }
    out
}

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

    /// Wrap an existing DSRs tool (any [`rig::tool::ToolDyn`]) as a sandbox
    /// capability — the Code Mode seam: generated JavaScript calls the tool as
    /// a plain global function.
    ///
    /// The capability name and description come from the tool's definition,
    /// fetched once here at wrap time (the fetch is async in rig). The name is
    /// mangled into a JS identifier per [`js_identifier`]. From JS, a call is
    /// `result = tool_name(argsObject)` — the args object is serialized to
    /// JSON, handed to [`ToolDyn::call`], and the result string is parsed back
    /// to JSON (or returned as a plain string if it isn't valid JSON). A tool
    /// error becomes a catchable JS exception whose message names the original
    /// tool: ``tool `<name>` failed: <error>``.
    pub async fn from_tool(tool: Arc<dyn ToolDyn>) -> Self {
        let definition = tool.definition(String::new()).await;
        Self::wrap_tool(
            js_identifier(&definition.name),
            definition.description,
            definition.name,
            tool,
        )
    }

    /// [`from_tool`](Self::from_tool) for a whole tool set, erroring if two
    /// tool names mangle to the same JS identifier (see [`js_identifier`]:
    /// the mapping is not injective, and a silent shadow would misroute
    /// calls).
    pub async fn from_toolset(tools: &[Arc<dyn ToolDyn>]) -> Result<Vec<Self>, RegisterError> {
        let mut seen: HashMap<String, String> = HashMap::with_capacity(tools.len());
        let mut capabilities = Vec::with_capacity(tools.len());
        for tool in tools {
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
            capabilities.push(Self::wrap_tool(
                js_name,
                definition.description,
                definition.name,
                Arc::clone(tool),
            ));
        }
        Ok(capabilities)
    }

    /// Lower-level [`from_tool`](Self::from_tool): the caller supplies the JS
    /// name and description instead of fetching the tool's own definition —
    /// used when descriptions are resolved elsewhere (the IR treats tool
    /// descriptions as optimizable parameters). `tool_name` is the original
    /// (unmangled) name used for error attribution.
    pub fn wrap_tool(
        js_name: impl Into<String>,
        description: impl Into<String>,
        tool_name: impl Into<String>,
        tool: Arc<dyn ToolDyn>,
    ) -> Self {
        let tool_name = tool_name.into();
        Self::new(js_name, description, move |args: Value| {
            let tool = Arc::clone(&tool);
            let tool_name = tool_name.clone();
            async move {
                let result = tool
                    .call(args.to_string())
                    .await
                    .map_err(|err| format!("tool `{tool_name}` failed: {err}"))?;
                // Tool results are strings by contract; most DSRs tools return
                // JSON. Give JS the parsed value when possible so results
                // compose (`r.field`), falling back to the raw string.
                Ok(serde_json::from_str(&result).unwrap_or(Value::String(result)))
            }
        })
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
