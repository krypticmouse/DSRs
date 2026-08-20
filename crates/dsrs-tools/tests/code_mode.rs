//! Code Mode: existing DSRs tools (rig `ToolDyn`) injected into the sandbox
//! as a JS API — wrapping, name mangling, error attribution.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use std::time::{Duration, Instant};

use dsrs_tools::{
    Capability, CodeModeTool, ExecError, Executor, QuickJsExecutor, RUN_JS_TOOL_NAME,
    RegisterError, SandboxConfig, ToolApi, ToolDyn, ToolInvocation, ToolSource,
    code_mode_description, js_identifier, run_script,
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
    // Reserved globals and words are suffixed rather than shadowed.
    assert_eq!(js_identifier("JSON"), "JSON_tool");
    assert_eq!(js_identifier("Object"), "Object_tool");
    assert_eq!(js_identifier("class"), "class_tool");
    assert_eq!(js_identifier("undefined"), "undefined_tool");
    // Only exact collisions are mangled.
    assert_eq!(js_identifier("JSONish"), "JSONish");
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

// ---------------------------------------------------------------- run_script

#[tokio::test]
async fn run_script_returns_and_chains_capabilities() {
    let weather = WeatherTool::default();
    let calls = Arc::clone(&weather.calls);
    let capability = Capability::from_tool(Arc::new(weather)).await;
    let result = run_script(
        "const a = get_weather_v2({city: 'Paris'});\n\
         const b = get_weather_v2({city: a.city});\n\
         return {temps: [a.temp_c, b.temp_c]};",
        vec![capability],
        SandboxConfig::default(),
    )
    .await
    .expect("script");
    assert_eq!(result, json!({"temps": [21, 21]}));
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn run_script_rejects_capability_name_injection() {
    // `run_script` installs caller-supplied capabilities without going
    // through `add_capability`; a hostile name must still be refused before
    // it can reach the sandbox bootstrap.
    let evil = Capability::new(
        "x; globalThis.leak = 1; //",
        "injection attempt",
        |_| async move { Ok(json!(null)) },
    );
    let err = run_script("return typeof leak;", vec![evil], SandboxConfig::default())
        .await
        .expect_err("must reject the capability");
    match err {
        ExecError::Internal { message } => {
            assert!(message.contains("capability"), "{message}");
        }
        other => panic!("expected Internal (host misconfiguration), got {other:?}"),
    }

    // Reserved names are refused on the same path.
    let shadow = Capability::new("JSON", "shadows JSON", |_| async move { Ok(json!(null)) });
    let err = run_script("return 1;", vec![shadow], SandboxConfig::default())
        .await
        .expect_err("must reject the capability");
    assert!(matches!(err, ExecError::Internal { .. }), "{err:?}");
}

#[tokio::test]
async fn json_named_tool_is_mangled_and_does_not_break_other_capabilities() {
    // A tool named `JSON` must not shadow `globalThis.JSON` (every capability
    // shim depends on it); it is installed as `JSON_tool` instead, and other
    // capabilities keep working.
    let tools: Vec<Arc<dyn ToolDyn>> = vec![Arc::new(NamedTool("JSON")), Arc::new(NamedTool("beta"))];
    let tool = CodeModeTool::new(tools, SandboxConfig::default())
        .await
        .expect("build");
    let definition = tool.definition(String::new()).await;
    assert!(definition.description.contains("JSON_tool(args)"));

    let args = json!({
        "code": "const a = JSON_tool({});\n\
                 const b = beta({});\n\
                 return {a: a.from, b: b.from, json_intact: typeof JSON.stringify === 'function'};"
    });
    let result = tool.call(args.to_string()).await.expect("call");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&result).unwrap(),
        json!({"a": "JSON", "b": "beta", "json_intact": true})
    );
}

#[tokio::test]
async fn run_script_syntax_error_is_repairable() {
    let err = run_script("return {", Vec::new(), SandboxConfig::default())
        .await
        .expect_err("syntax error");
    match err {
        ExecError::Js { name, message } => {
            assert_eq!(name, RUN_JS_TOOL_NAME);
            assert!(!message.is_empty());
        }
        other => panic!("expected Js error, got {other:?}"),
    }
}

#[tokio::test]
async fn run_script_await_resolves_on_microtasks() {
    let result = run_script(
        "const v = await Promise.resolve(7); return v * 6;",
        Vec::new(),
        SandboxConfig::default(),
    )
    .await
    .expect("script");
    assert_eq!(result, json!(42));
}

// ---------------------------------------------------------------- CodeModeTool

fn tight_config() -> SandboxConfig {
    SandboxConfig {
        deadline: Duration::from_millis(150),
        ..SandboxConfig::default()
    }
}

#[tokio::test]
async fn code_mode_tool_chains_two_tools_in_one_call() {
    let weather = WeatherTool::default();
    let weather_calls = Arc::clone(&weather.calls);
    let tools: Vec<Arc<dyn ToolDyn>> = vec![Arc::new(weather), Arc::new(NamedTool("beta"))];
    let tool = CodeModeTool::new(tools, SandboxConfig::default())
        .await
        .expect("build");

    assert_eq!(ToolDyn::name(&tool), RUN_JS_TOOL_NAME);
    let definition = tool.definition(String::new()).await;
    assert_eq!(definition.name, RUN_JS_TOOL_NAME);
    // The auto-generated description advertises both JS APIs with schemas.
    assert!(definition.description.contains("get_weather_v2(args)"));
    assert!(definition.description.contains("beta(args)"));
    assert!(definition.description.contains("{city: string}"));
    assert_eq!(definition.parameters["required"], json!(["code"]));

    // One run_js call, two tool calls chained in JS — the token-economy win.
    let args = json!({
        "code": "const w = get_weather_v2({city: 'Tokyo'});\n\
                 const b = beta({});\n\
                 return {temp: w.temp_c, from: b.from};"
    });
    let result = tool.call(args.to_string()).await.expect("call");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&result).unwrap(),
        json!({"temp": 21, "from": "beta"})
    );
    assert_eq!(weather_calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn code_mode_tool_rejects_colliding_names() {
    let tools: Vec<Arc<dyn ToolDyn>> =
        vec![Arc::new(NamedTool("a-b")), Arc::new(NamedTool("a.b"))];
    let err = CodeModeTool::new(tools, SandboxConfig::default())
        .await
        .expect_err("collision");
    assert!(matches!(err, RegisterError::InvalidCapability { .. }));
}

#[tokio::test]
async fn code_mode_deadline_kills_runaway_script() {
    let tool = CodeModeTool::new(vec![Arc::new(NamedTool("beta"))], tight_config())
        .await
        .expect("build");
    let started = Instant::now();
    let result = tool
        .call(json!({"code": "while (true) {}"}).to_string())
        .await
        .expect("repairable errors come back as Ok results");
    let elapsed = started.elapsed();
    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    assert_eq!(parsed["kind"], "timeout", "typed timeout error: {result}");
    assert_eq!(parsed["name"], RUN_JS_TOOL_NAME);
    assert!(
        elapsed < Duration::from_secs(5),
        "deadline enforced, not hung: {elapsed:?}"
    );
}

/// A tool whose call never finishes within any sane deadline.
struct StallTool;

impl rig::tool::Tool for StallTool {
    const NAME: &'static str = "stall";
    type Error = CannedError;
    type Args = serde_json::Value;
    type Output = serde_json::Value;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Never returns".to_string(),
            parameters: json!({"type": "object"}),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        tokio::time::sleep(Duration::from_secs(3600)).await;
        Ok(json!(null))
    }
}

#[tokio::test]
async fn code_mode_deadline_bounds_capability_calls() {
    // In Code Mode every tool call is a capability call; the wall-clock
    // deadline must bound the host future too, not just JS execution.
    let tool = CodeModeTool::new(vec![Arc::new(StallTool) as Arc<dyn ToolDyn>], tight_config())
        .await
        .expect("build");
    let started = Instant::now();
    let result = tool
        .call(json!({"code": "return stall({});"}).to_string())
        .await
        .expect("repairable errors come back as Ok results");
    let elapsed = started.elapsed();
    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    assert_eq!(parsed["kind"], "timeout", "typed timeout error: {result}");
    assert_eq!(parsed["name"], RUN_JS_TOOL_NAME);
    assert!(
        elapsed < Duration::from_secs(5),
        "capability call was not bounded by the deadline: {elapsed:?}"
    );
}

#[tokio::test]
async fn code_mode_script_error_is_repairable_result() {
    let tool = CodeModeTool::new(vec![Arc::new(FailTool) as Arc<dyn ToolDyn>], tight_config())
        .await
        .expect("build");
    // Uncaught tool failure: comes back as a typed capability error result.
    let result = tool
        .call(json!({"code": "return fail_tool({});"}).to_string())
        .await
        .expect("repairable");
    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    assert_eq!(parsed["kind"], "capability");
    assert_eq!(parsed["capability"], "fail_tool");

    // Missing `code` argument: also repairable.
    let result = tool.call(json!({"script": "x"}).to_string()).await.expect("repairable");
    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    assert_eq!(parsed["kind"], "invalid_args");
}

#[tokio::test]
async fn code_mode_description_is_overridable() {
    let tool = CodeModeTool::new(vec![Arc::new(NamedTool("beta"))], SandboxConfig::default())
        .await
        .expect("build")
        .with_description("custom optimized description");
    let definition = tool.definition(String::new()).await;
    assert_eq!(definition.description, "custom optimized description");
}

// ---------------------------------------------------------------- description generation

#[test]
fn description_generation_is_token_compact() {
    let apis = vec![
        ToolApi {
            js_name: "get_weather".to_string(),
            description: "Get current weather".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "units": {"type": "string", "enum": ["c", "f"]}
                },
                "required": ["city"]
            }),
        },
        ToolApi {
            js_name: "search".to_string(),
            description: "Search the web.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                    "tags": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["query"]
            }),
        },
        ToolApi {
            js_name: "freeform".to_string(),
            description: String::new(),
            parameters: json!({"type": "object"}),
        },
    ];
    let description = code_mode_description(&apis);
    assert!(description.contains(
        "- get_weather(args): Get current weather. args: {city: string, units?: c|f}"
    ));
    assert!(description.contains(
        "- search(args): Search the web. args: {query: string, limit?: integer, tags?: string[]}"
    ));
    assert!(description.contains("- freeform(args): (no description). args: object"));
    // The contract and one example are present.
    assert!(description.contains("`return`"));
    assert!(description.contains("const r = get_weather({...});"));
}

#[test]
fn description_with_no_apis_is_contract_only() {
    let description = code_mode_description(&[]);
    assert!(!description.contains("APIs:"));
    assert!(description.contains("Run JavaScript"));
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
