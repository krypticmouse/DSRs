//! Code Mode, module lane: `ToolSet::code_mode` collapses a tool set into one
//! sandboxed `run_js` tool, usable in the LM tool loop today. A canned LM
//! emits a `run_js` call whose script chains two tool calls in a single
//! execution — the token-economy win over N JSON tool calls.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use dspy_rs::{
    Chat, LM, LMClient, Message, RUN_JS_TOOL_NAME, SandboxConfig, SpanEvent, TestCompletionModel,
    ToolLoopMode, ToolSet,
};
use rig::completion::{AssistantContent, ToolDefinition};
use rig::message::{Text, ToolCall, ToolFunction};
use serde_json::json;

// ---------------------------------------------------------------- fixtures

#[derive(Debug)]
struct CannedError(String);
impl std::fmt::Display for CannedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for CannedError {}

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
        Ok(json!({"city": args["city"], "temp_c": 21}))
    }
}

#[derive(Clone, Default)]
struct CounterTool {
    calls: Arc<AtomicUsize>,
}

impl rig::tool::Tool for CounterTool {
    const NAME: &'static str = "counter";
    type Error = CannedError;
    type Args = serde_json::Value;
    type Output = serde_json::Value;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Increment and read a counter".to_string(),
            parameters: json!({"type": "object"}),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        Ok(json!({"n": n}))
    }
}

fn text(content: impl Into<String>) -> AssistantContent {
    AssistantContent::Text(Text {
        text: content.into(),
    })
}

fn tool_call(name: &str, args: serde_json::Value) -> AssistantContent {
    AssistantContent::ToolCall(ToolCall::new(
        format!("tc-{name}"),
        ToolFunction {
            name: name.to_string(),
            arguments: args,
        },
    ))
}

async fn canned_lm(responses: Vec<AssistantContent>) -> (LM, TestCompletionModel) {
    let client = TestCompletionModel::new(responses);
    let lm = temp_env::async_with_vars(
        [("OPENAI_API_KEY", Some("test"))],
        LM::builder()
            .model("openai:gpt-4o-mini".to_string())
            .build(),
    )
    .await
    .unwrap()
    .with_client(LMClient::Test(client.clone()))
    .await
    .unwrap();
    (lm, client)
}

// ---------------------------------------------------------------- tests

#[tokio::test]
async fn code_mode_toolset_chains_two_tools_in_one_run_js_call() {
    let weather = WeatherTool::default();
    let counter = CounterTool::default();
    let weather_calls = Arc::clone(&weather.calls);
    let counter_calls = Arc::clone(&counter.calls);

    let toolset = ToolSet::code_mode(
        vec![Arc::new(weather), Arc::new(counter)],
        SandboxConfig::default(),
    )
    .await
    .expect("build code-mode toolset");

    // The model sees exactly one tool: run_js, whose description lists both
    // JS APIs with their schemas.
    assert_eq!(toolset.definitions().len(), 1);
    let definition = &toolset.definitions()[0];
    assert_eq!(definition.name, RUN_JS_TOOL_NAME);
    assert!(definition.description.contains("get_weather_v2(args)"));
    assert!(definition.description.contains("counter(args)"));
    assert!(definition.description.contains("{city: string}"));

    let (lm, client) = canned_lm(vec![
        tool_call(
            RUN_JS_TOOL_NAME,
            json!({"code": "const w = get_weather_v2({city: 'Tokyo'});\n\
                            const c = counter({});\n\
                            return {temp: w.temp_c, n: c.n};"}),
        ),
        text("Tokyo is at 21C; counter is 1."),
    ])
    .await;

    let chat = Chat::new(vec![Message::user("Weather in Tokyo, and bump the counter.")]);
    let response = lm
        .call_with_toolset(chat, &toolset, ToolLoopMode::Auto)
        .await
        .expect("tool loop");

    // Both tools executed, inside ONE run_js execution.
    assert_eq!(weather_calls.load(Ordering::SeqCst), 1);
    assert_eq!(counter_calls.load(Ordering::SeqCst), 1);
    assert_eq!(response.tool_calls.len(), 1);
    assert_eq!(response.tool_calls[0].function.name, RUN_JS_TOOL_NAME);

    // The composed result went back to the model as the tool result.
    let composed: serde_json::Value = serde_json::from_str(&response.tool_executions[0]).unwrap();
    assert_eq!(composed, json!({"temp": 21, "n": 1}));
    assert_eq!(response.output.content(), "Tokyo is at 21C; counter is 1.");

    // The loop recorded one ToolRun event for run_js.
    let tool_runs: Vec<&String> = response
        .events
        .iter()
        .filter_map(|event| match event {
            SpanEvent::ToolRun { name, .. } => Some(name),
            _ => None,
        })
        .collect();
    assert_eq!(tool_runs, vec![RUN_JS_TOOL_NAME]);

    // Every provider round-trip advertised only run_js.
    let last = client.last_request().unwrap();
    let advertised: Vec<&str> = last.tools.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(advertised, vec![RUN_JS_TOOL_NAME]);
}

#[tokio::test]
async fn code_mode_script_errors_feed_back_to_the_model() {
    let counter = CounterTool::default();
    let counter_calls = Arc::clone(&counter.calls);
    let toolset = ToolSet::code_mode(vec![Arc::new(counter)], SandboxConfig::default())
        .await
        .expect("build");

    // Round 1: broken script. Round 2: repaired. Round 3: final text. The
    // loop must survive the script error (it is a tool RESULT, not a failure).
    let (lm, _client) = canned_lm(vec![
        tool_call(RUN_JS_TOOL_NAME, json!({"code": "return {"})),
        tool_call(RUN_JS_TOOL_NAME, json!({"code": "return counter({}).n;"})),
        text("Counter is 1."),
    ])
    .await;

    let chat = Chat::new(vec![Message::user("Bump the counter.")]);
    let response = lm
        .call_with_toolset(chat, &toolset, ToolLoopMode::Auto)
        .await
        .expect("loop survives the script error");

    assert_eq!(response.tool_executions.len(), 2);
    let first: serde_json::Value = serde_json::from_str(&response.tool_executions[0]).unwrap();
    assert_eq!(first["kind"], "js", "typed repairable error: {first}");
    assert_eq!(response.tool_executions[1], "1");
    assert_eq!(counter_calls.load(Ordering::SeqCst), 1);
    assert_eq!(response.output.content(), "Counter is 1.");
}
