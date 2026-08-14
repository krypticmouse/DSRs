//! Code Mode, IR lane: a `RuntimeEnv` binding option makes every `AgentLoop`
//! present its non-stop tools as one sandboxed `run_js` tool. Same canned
//! flow as the module lane, run through the interpreter — spans record the
//! `ToolRun`, stop tools stay individual, collisions refuse the load.
#![cfg(all(feature = "ir", feature = "code-mode"))]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use dspy_rs::ir::{
    self, Budget, FieldType as T, Interpreter, LoadError, Program, ProgramBuilder, RuntimeEnv,
    SignatureDef,
};
use dspy_rs::trace::{SpanEvent, capture};
use dspy_rs::{LM, LMClient, LMConfig, RUN_JS_TOOL_NAME, SandboxConfig, TestCompletionModel};
use rig::completion::{AssistantContent, ToolDefinition};
use rig::message::{Text, ToolCall, ToolFunction};
use serde_json::json;

// ---------------------------------------------------------------- fixtures

fn fields(pairs: &[(&str, &str)]) -> String {
    let mut out = String::new();
    for (name, value) in pairs {
        out.push_str(&format!("[[ ## {name} ## ]]\n{value}\n\n"));
    }
    out.push_str("[[ ## completed ## ]]\n");
    out
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

async fn canned_lm(responses: Vec<AssistantContent>) -> (Arc<LM>, TestCompletionModel) {
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
    (Arc::new(lm), client)
}

fn config() -> LMConfig {
    LMConfig {
        model: "openai:gpt-4o-mini".to_string(),
        ..LMConfig::default()
    }
}

fn obj(pairs: &[(&str, serde_json::Value)]) -> serde_json::Map<String, serde_json::Value> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.clone()))
        .collect()
}

#[derive(Debug)]
struct CannedError;
impl std::fmt::Display for CannedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "canned error")
    }
}
impl std::error::Error for CannedError {}

#[derive(Clone, Default)]
struct WeatherTool {
    calls: Arc<AtomicUsize>,
}

impl rig::tool::Tool for WeatherTool {
    // The IR declares the interface; the dash exercises name mangling.
    const NAME: &'static str = "get-weather";
    type Error = CannedError;
    type Args = serde_json::Value;
    type Output = serde_json::Value;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "host-side definition (ignored: the IR declares the interface)"
                .to_string(),
            parameters: json!({"type": "object", "additionalProperties": true}),
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
            description: "host-side definition (ignored)".to_string(),
            parameters: json!({"type": "object", "additionalProperties": true}),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        Ok(json!({"n": n}))
    }
}

/// question → agent("researcher") with two host tools → answer.
fn two_tool_program(tool_a: &str, tool_b: &str) -> Program {
    let mut b = ProgramBuilder::new("code-mode-agent");
    b.cap("net:tools");
    b.model("m", config());
    let main_sig = b.sig(
        SignatureDef::build("Main")
            .input("question", T::String)
            .output("answer", T::String)
            .finish()
            .unwrap(),
    );
    let research = b.sig(
        SignatureDef::build("Research")
            .instruction("Research and answer.")
            .input("question", T::String)
            .output("answer", T::String)
            .finish()
            .unwrap(),
    );
    let weather_sig = b.sig(
        SignatureDef::build("Weather")
            .input("city", T::String)
            .output("report", T::String)
            .finish()
            .unwrap(),
    );
    let counter_sig = b.sig(
        SignatureDef::build("Counter")
            .input("amount", T::Optional(Box::new(T::Int)))
            .output("n", T::Int)
            .finish()
            .unwrap(),
    );
    let weather = b.host_tool(tool_a, "Weather by city", weather_sig, &["net:tools"]);
    let counter = b.host_tool(tool_b, "Bump and read the counter", counter_sig, &["net:tools"]);
    let researcher = ir::agent("researcher", research)
        .bind("question", ir::input("question"))
        .tools([weather, counter])
        .max_turns(4);
    b.main(
        main_sig,
        ir::seq([researcher]).out("answer", ir::out("researcher", "answer")),
    )
    .unwrap()
}

// ---------------------------------------------------------------- tests

#[tokio::test]
async fn agent_loop_code_mode_chains_two_tools_in_one_run_js() {
    let weather = WeatherTool::default();
    let counter = CounterTool::default();
    let weather_calls = Arc::clone(&weather.calls);
    let counter_calls = Arc::clone(&counter.calls);

    let (lm, client) = canned_lm(vec![
        tool_call(
            RUN_JS_TOOL_NAME,
            json!({"code": "const w = get_weather({city: 'Tokyo'});\n\
                            const c = counter({});\n\
                            return {temp: w.temp_c, n: c.n};"}),
        ),
        text(fields(&[("answer", "21C, counter at 1")])),
    ])
    .await;
    let env = RuntimeEnv::new()
        .bind_model("m", lm)
        .bind_host_tool("get-weather", Arc::new(weather))
        .bind_host_tool("counter", Arc::new(counter))
        .grant("net:tools")
        .with_code_mode(SandboxConfig::default());
    let interp = Interpreter::load(two_tool_program("get-weather", "counter"), env)
        .await
        .unwrap();

    let (result, trace) = capture(|| {
        interp.run(
            obj(&[("question", json!("weather in Tokyo?"))]),
            None,
            Budget::unlimited(),
        )
    })
    .await;
    assert_eq!(result.unwrap()["answer"], "21C, counter at 1");

    // Both host tools ran inside ONE run_js execution.
    assert_eq!(weather_calls.load(Ordering::SeqCst), 1);
    assert_eq!(counter_calls.load(Ordering::SeqCst), 1);

    // One span; its events record the run_js ToolRun with the composed result.
    assert_eq!(trace.spans.len(), 1);
    let span = &trace.spans[0];
    let kinds: Vec<&str> = span
        .events
        .iter()
        .map(|event| match event {
            SpanEvent::Exchange { .. } => "exchange",
            SpanEvent::ToolRun { .. } => "tool_run",
            _ => "other",
        })
        .collect();
    assert_eq!(kinds, vec!["exchange", "tool_run", "exchange"]);
    match &span.events[1] {
        SpanEvent::ToolRun {
            name,
            result,
            error,
            ..
        } => {
            assert_eq!(name, RUN_JS_TOOL_NAME);
            assert!(error.is_none());
            let composed: serde_json::Value = serde_json::from_str(result).unwrap();
            assert_eq!(composed, json!({"temp": 21, "n": 1}));
        }
        other => panic!("expected ToolRun, got {other:?}"),
    }

    // The model saw exactly one tool: run_js, whose generated description
    // lists both JS APIs with the IR-declared (ToolDesc-gene) descriptions
    // and schemas projected from the declared signatures.
    let last = client.last_request().unwrap();
    let advertised: Vec<&str> = last.tools.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(advertised, vec![RUN_JS_TOOL_NAME]);
    let description = &last.tools[0].description;
    assert!(description.contains("get_weather(args)"), "{description}");
    assert!(description.contains("counter(args)"), "{description}");
    assert!(description.contains("Weather by city"), "{description}");
    assert!(description.contains("Bump and read the counter"), "{description}");
    assert!(description.contains("{city: string}"), "{description}");
}

#[tokio::test]
async fn stop_tools_stay_individual_next_to_run_js() {
    let mut b = ProgramBuilder::new("stoppable");
    b.cap("net:tools");
    b.model("m", config());
    let main_sig = b.sig(
        SignatureDef::build("Main")
            .input("question", T::String)
            .output("answer", T::String)
            .finish()
            .unwrap(),
    );
    let research = b.sig(
        SignatureDef::build("Research")
            .input("question", T::String)
            .output("answer", T::String)
            .finish()
            .unwrap(),
    );
    let counter_sig = b.sig(
        SignatureDef::build("Counter")
            .input("amount", T::Optional(Box::new(T::Int)))
            .output("n", T::Int)
            .finish()
            .unwrap(),
    );
    let submit_sig = b.sig(
        SignatureDef::build("Submit")
            .input("answer", T::String)
            .output("ok", T::Bool)
            .finish()
            .unwrap(),
    );
    let counter = b.host_tool("counter", "Bump the counter", counter_sig, &["net:tools"]);
    let submit = b.host_tool("submit", "Submit the final answer", submit_sig, &["net:tools"]);
    let researcher = ir::agent("researcher", research)
        .bind("question", ir::input("question"))
        .tools([counter, submit])
        .stop_tools([submit])
        .max_turns(4);
    let program = b
        .main(
            main_sig,
            ir::seq([researcher]).out("answer", ir::out("researcher", "answer")),
        )
        .unwrap();

    let counter_tool = CounterTool::default();
    let counter_calls = Arc::clone(&counter_tool.calls);
    let (lm, client) = canned_lm(vec![
        tool_call(RUN_JS_TOOL_NAME, json!({"code": "return counter({}).n;"})),
        tool_call("submit", json!({"answer": "counter says 1"})),
    ])
    .await;
    let env = RuntimeEnv::new()
        .bind_model("m", lm)
        .bind_host_tool("counter", Arc::new(counter_tool))
        // Stop tools end the loop before execution; a bound ToolDyn is still
        // required at load.
        .bind_host_tool("submit", Arc::new(CounterTool::default()))
        .grant("net:tools")
        .with_code_mode(SandboxConfig::default());
    let interp = Interpreter::load(program, env).await.unwrap();

    let result = interp
        .run(
            obj(&[("question", json!("bump it"))]),
            None,
            Budget::unlimited(),
        )
        .await
        .unwrap();
    // The stop tool's args became the raw final output.
    assert_eq!(result["answer"], "counter says 1");
    assert_eq!(counter_calls.load(Ordering::SeqCst), 1);

    // The advertised surface was run_js + the individual stop tool.
    let last = client.last_request().unwrap();
    let advertised: Vec<&str> = last.tools.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(advertised, vec![RUN_JS_TOOL_NAME, "submit"]);
    // The stop tool is not part of the JS API.
    assert!(!last.tools[0].description.contains("submit(args)"));
}

#[tokio::test]
async fn sandboxed_tools_join_the_js_api() {
    let mut b = ProgramBuilder::new("mixed");
    b.cap("net:tools");
    b.model("m", config());
    let main_sig = b.sig(
        SignatureDef::build("Main")
            .input("question", T::String)
            .output("answer", T::String)
            .finish()
            .unwrap(),
    );
    let research = b.sig(
        SignatureDef::build("Research")
            .input("question", T::String)
            .output("answer", T::String)
            .finish()
            .unwrap(),
    );
    let weather_sig = b.sig(
        SignatureDef::build("Weather")
            .input("city", T::String)
            .output("report", T::String)
            .finish()
            .unwrap(),
    );
    let double_sig = b.sig(
        SignatureDef::build("Double")
            .input("n", T::Int)
            .output("doubled", T::Int)
            .finish()
            .unwrap(),
    );
    let weather = b.host_tool("get-weather", "Weather by city", weather_sig, &["net:tools"]);
    let double = b.sandboxed_tool("double", "Double a number", double_sig, &[], "(args) => args.n * 2");
    let researcher = ir::agent("researcher", research)
        .bind("question", ir::input("question"))
        .tools([weather, double])
        .max_turns(4);
    let program = b
        .main(
            main_sig,
            ir::seq([researcher]).out("answer", ir::out("researcher", "answer")),
        )
        .unwrap();

    let weather_tool = WeatherTool::default();
    let weather_calls = Arc::clone(&weather_tool.calls);
    let (lm, _client) = canned_lm(vec![
        tool_call(
            RUN_JS_TOOL_NAME,
            json!({"code": "const w = get_weather({city: 'Oslo'});\n\
                            const d = double({n: w.temp_c});\n\
                            return d;"}),
        ),
        text(fields(&[("answer", "doubled to 42")])),
    ])
    .await;
    let env = RuntimeEnv::new()
        .bind_model("m", lm)
        .bind_host_tool("get-weather", Arc::new(weather_tool))
        .grant("net:tools")
        .with_sandbox(Arc::new(dsrs_tools::QuickJsExecutor::new()))
        .with_code_mode(SandboxConfig::default());
    let interp = Interpreter::load(program, env).await.unwrap();

    let (result, trace) = capture(|| {
        interp.run(
            obj(&[("question", json!("double Oslo's temperature"))]),
            None,
            Budget::unlimited(),
        )
    })
    .await;
    assert_eq!(result.unwrap()["answer"], "doubled to 42");
    assert_eq!(weather_calls.load(Ordering::SeqCst), 1);
    // The chained host→sandboxed composition produced 21 * 2.
    let run = trace.spans[0]
        .events
        .iter()
        .find_map(|event| match event {
            SpanEvent::ToolRun { name, result, .. } if name == RUN_JS_TOOL_NAME => Some(result),
            _ => None,
        })
        .expect("run_js ToolRun recorded");
    assert_eq!(run.as_str(), "42");
}

#[tokio::test]
async fn script_failure_is_conversational_and_recorded() {
    let weather = WeatherTool::default();
    let counter = CounterTool::default();
    let (lm, _client) = canned_lm(vec![
        tool_call(RUN_JS_TOOL_NAME, json!({"code": "return {"})),
        text(fields(&[("answer", "gave up on js")])),
    ])
    .await;
    let env = RuntimeEnv::new()
        .bind_model("m", lm)
        .bind_host_tool("get-weather", Arc::new(weather))
        .bind_host_tool("counter", Arc::new(counter))
        .grant("net:tools")
        .with_code_mode(SandboxConfig::default());
    let interp = Interpreter::load(two_tool_program("get-weather", "counter"), env)
        .await
        .unwrap();

    let (result, trace) = capture(|| {
        interp.run(
            obj(&[("question", json!("hm"))]),
            None,
            Budget::unlimited(),
        )
    })
    .await;
    // The syntax error went back to the model as a tool result; the loop and
    // the run both completed.
    assert_eq!(result.unwrap()["answer"], "gave up on js");
    match trace.spans[0]
        .events
        .iter()
        .find(|event| matches!(event, SpanEvent::ToolRun { .. }))
        .expect("ToolRun recorded")
    {
        SpanEvent::ToolRun { name, result, error, .. } => {
            assert_eq!(name, RUN_JS_TOOL_NAME);
            assert!(error.is_some());
            assert!(result.contains("\"kind\":\"js\""), "{result}");
        }
        _ => unreachable!(),
    }
}

#[tokio::test]
async fn js_identifier_collision_refuses_the_load() {
    // `counter-x` and `counter.x` both mangle to `counter_x`.
    let (lm, _client) = canned_lm(vec![]).await;
    let env = RuntimeEnv::new()
        .bind_model("m", lm)
        .bind_host_tool("counter-x", Arc::new(CounterTool::default()))
        .bind_host_tool("counter.x", Arc::new(CounterTool::default()))
        .grant("net:tools")
        .with_code_mode(SandboxConfig::default());
    let err = Interpreter::load(two_tool_program("counter-x", "counter.x"), env)
        .await
        .unwrap_err();
    match err {
        LoadError::Register { at, source } => {
            assert_eq!(at, "researcher");
            assert!(source.to_string().contains("counter_x"), "{source}");
        }
        other => panic!("expected Register load error, got {other:?}"),
    }
}

#[tokio::test]
async fn without_code_mode_binding_tools_stay_individual() {
    // Same program, same env minus with_code_mode: the surface is unchanged.
    let weather = WeatherTool::default();
    let counter = CounterTool::default();
    let (lm, client) = canned_lm(vec![text(fields(&[("answer", "direct")]))]).await;
    let env = RuntimeEnv::new()
        .bind_model("m", lm)
        .bind_host_tool("get-weather", Arc::new(weather))
        .bind_host_tool("counter", Arc::new(counter))
        .grant("net:tools");
    let interp = Interpreter::load(two_tool_program("get-weather", "counter"), env)
        .await
        .unwrap();
    let result = interp
        .run(
            obj(&[("question", json!("plain"))]),
            None,
            Budget::unlimited(),
        )
        .await
        .unwrap();
    assert_eq!(result["answer"], "direct");
    let last = client.last_request().unwrap();
    let advertised: Vec<&str> = last.tools.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(advertised, vec!["get-weather", "counter"]);
}
