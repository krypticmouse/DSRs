//! Code Mode: existing DSRs tools (rig `ToolDyn`) injected into the sandbox
//! as a JS API — wrapping, name mangling, error attribution.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use dsrs_tools::{
    Capability, ExecError, Executor, QuickJsExecutor, RegisterError, ToolDyn, ToolInvocation,
    ToolSource, js_identifier,
};
use rig::completion::ToolDefinition;
use serde_json::json;

// ---------------------------------------------------------------- canned rig tools

#[derive(Debug)]
struct CannedError(String);
impl std::fmt::Display for CannedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for CannedError {}

/// A canned tool with a mangling-worthy name that counts its invocations.
#[derive(Clone, Default)]
struct WeatherTool {
    calls: Arc<AtomicUsize>,
}

impl rig::tool::Tool for WeatherTool {
    const NAME: &'static str = "get-weather.v2";
    type Error = CannedError;
    type Args = serde_json::Value;
    type Output = serde_json::Value;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Get current weather for a city".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let city = args["city"].as_str().unwrap_or("?");
        Ok(json!({"city": city, "temp_c": 21}))
    }
}

/// A canned tool that always fails.
struct FailTool;

impl rig::tool::Tool for FailTool {
    const NAME: &'static str = "fail-tool";
    type Error = CannedError;
    type Args = serde_json::Value;
    type Output = serde_json::Value;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Always fails".to_string(),
            parameters: json!({"type": "object"}),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        Err(CannedError("upstream service unavailable".to_string()))
    }
}

/// A tool that returns a non-JSON string result.
struct PlainTextTool;

impl rig::tool::Tool for PlainTextTool {
    const NAME: &'static str = "motd";
    type Error = CannedError;
    type Args = serde_json::Value;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Message of the day".to_string(),
            parameters: json!({"type": "object"}),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok("plain text, not JSON".to_string())
    }
}

/// Builds a tool whose definition reports an arbitrary name (for mangling and
/// collision tests without one struct per name).
struct NamedTool(&'static str);

impl ToolDyn for NamedTool {
    fn name(&self) -> String {
        self.0.to_string()
    }

    fn definition(&self, _prompt: String) -> rig::wasm_compat::WasmBoxedFuture<'_, ToolDefinition> {
        Box::pin(async move {
            ToolDefinition {
                name: self.0.to_string(),
                description: format!("named tool {}", self.0),
                parameters: json!({"type": "object"}),
            }
        })
    }

    fn call(
        &self,
        _args: String,
    ) -> rig::wasm_compat::WasmBoxedFuture<'_, Result<String, rig::tool::ToolError>> {
        Box::pin(async move { Ok(json!({"from": self.0}).to_string()) })
    }
}

fn schemaless(name: &str, js: &str) -> ToolSource {
    ToolSource::new(name, "test tool", json!({"type": "object"}), js)
}

// ---------------------------------------------------------------- Capability::from_tool

#[tokio::test]
async fn wrapped_tool_round_trip() {
    let weather = WeatherTool::default();
    let calls = Arc::clone(&weather.calls);
    let capability = Capability::from_tool(Arc::new(weather)).await;
    // Name mangled, description straight from the tool definition.
    assert_eq!(capability.name(), "get_weather_v2");
    assert_eq!(capability.description(), "Get current weather for a city");

    let executor = QuickJsExecutor::builder()
        .capability(capability)
        .build()
        .expect("build");
    executor
        .register(schemaless(
            "fetch_temp",
            "(args) => get_weather_v2({city: args.city}).temp_c",
        ))
        .await
        .expect("register");
    let result = executor
        .execute(ToolInvocation::new("fetch_temp", json!({"city": "Paris"})))
        .await
        .expect("execute");
    assert_eq!(result, json!(21));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn wrapped_tool_non_json_result_becomes_string() {
    let capability = Capability::from_tool(Arc::new(PlainTextTool)).await;
    let executor = QuickJsExecutor::builder()
        .capability(capability)
        .build()
        .expect("build");
    executor
        .register(schemaless("greet", "(args) => motd({}) + '!'"))
        .await
        .expect("register");
    let result = executor
        .execute(ToolInvocation::new("greet", json!({})))
        .await
        .expect("execute");
    assert_eq!(result, json!("plain text, not JSON!"));
}

#[tokio::test]
async fn tool_error_is_catchable_in_js_with_attribution() {
    let capability = Capability::from_tool(Arc::new(FailTool)).await;
    let executor = QuickJsExecutor::builder()
        .capability(capability)
        .build()
        .expect("build");
    executor
        .register(schemaless(
            "recovers",
            r#"(args) => { try { fail_tool({}); return "no-throw"; } catch (e) { return "caught: " + String(e.message ?? e); } }"#,
        ))
        .await
        .expect("register");
    let result = executor
        .execute(ToolInvocation::new("recovers", json!({})))
        .await
        .expect("execute");
    let text = result.as_str().expect("string result");
    assert!(text.starts_with("caught: "), "exception was catchable: {text}");
    assert!(
        text.contains("tool `fail-tool` failed"),
        "attributed to the original tool name: {text}"
    );
    assert!(text.contains("upstream service unavailable"), "{text}");
}

#[tokio::test]
async fn uncaught_tool_error_is_typed_and_attributed() {
    let capability = Capability::from_tool(Arc::new(FailTool)).await;
    let executor = QuickJsExecutor::builder()
        .capability(capability)
        .build()
        .expect("build");
    executor
        .register(schemaless("propagates", "(args) => fail_tool({})"))
        .await
        .expect("register");
    let err = executor
        .execute(ToolInvocation::new("propagates", json!({})))
        .await
        .expect_err("tool error should propagate");
    match err {
        ExecError::Capability {
            name,
            capability,
            message,
        } => {
            assert_eq!(name, "propagates");
            assert_eq!(capability, "fail_tool");
            assert!(message.contains("tool `fail-tool` failed"), "{message}");
        }
        other => panic!("expected Capability error, got {other:?}"),
    }
}

// ---------------------------------------------------------------- name mangling

#[test]
fn js_identifier_mangling_rule() {
    assert_eq!(js_identifier("get_weather"), "get_weather");
    assert_eq!(js_identifier("my-tool.v2"), "my_tool_v2");
    assert_eq!(js_identifier("2fast"), "_2fast");
    assert_eq!(js_identifier(""), "_tool");
    assert_eq!(js_identifier("__dsrs_evil"), "___dsrs_evil");
    assert_eq!(js_identifier("emoji🔥name"), "emoji_name");
    assert_eq!(js_identifier("$ok"), "$ok");
}

#[tokio::test]
async fn mangled_reserved_prefix_is_injectable() {
    // `__dsrs_x` would be rejected as a capability name; mangling adds one
    // more `_`, which must pass validation.
    let capability = Capability::from_tool(Arc::new(NamedTool("__dsrs_evil"))).await;
    assert_eq!(capability.name(), "___dsrs_evil");
    let executor = QuickJsExecutor::new();
    executor.add_capability(capability).expect("injectable");
}

#[tokio::test]
async fn from_toolset_wraps_all_and_detects_collisions() {
    let tools: Vec<Arc<dyn ToolDyn>> =
        vec![Arc::new(NamedTool("alpha-x")), Arc::new(NamedTool("beta"))];
    let capabilities = Capability::from_toolset(&tools).await.expect("no collision");
    let names: Vec<&str> = capabilities.iter().map(|c| c.name()).collect();
    assert_eq!(names, vec!["alpha_x", "beta"]);

    // `a-b` and `a.b` both mangle to `a_b` — registration must refuse.
    let colliding: Vec<Arc<dyn ToolDyn>> =
        vec![Arc::new(NamedTool("a-b")), Arc::new(NamedTool("a.b"))];
    let err = Capability::from_toolset(&colliding)
        .await
        .expect_err("collision");
    match err {
        RegisterError::InvalidCapability { name, reason } => {
            assert_eq!(name, "a_b");
            assert!(reason.contains("a-b") && reason.contains("a.b"), "{reason}");
        }
        other => panic!("expected InvalidCapability, got {other:?}"),
    }
}
