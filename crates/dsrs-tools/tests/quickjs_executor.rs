//! Integration tests for the Tier-1 QuickJS executor: lifecycle gating,
//! resource-limit kills, ambient-authority denial, capability injection, the
//! bytecode cache, and the rig `ToolDyn` bridge.

use std::sync::Arc;
use std::time::{Duration, Instant};

use dsrs_tools::{
    Capability, ExecError, Executor, QuickJsExecutor, RegisterError, ToolInvocation, ToolSource,
};
use serde_json::{Value, json};

fn add_tool() -> ToolSource {
    ToolSource::new(
        "add",
        "Add two numbers",
        json!({
            "type": "object",
            "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
            "required": ["x", "y"]
        }),
        "(args) => args.x + args.y",
    )
}

fn schemaless(name: &str, js: &str) -> ToolSource {
    ToolSource::new(name, "test tool", json!({"type": "object"}), js)
}

// ---------------------------------------------------------------- happy path

#[tokio::test]
async fn happy_path_register_and_execute() {
    let executor = QuickJsExecutor::new();
    let meta = executor.register(add_tool()).await.expect("register");
    assert_eq!(meta.name, "add");
    assert!(!meta.self_tested);
    assert_eq!(meta.source_hash.len(), 64);

    let result = executor
        .execute(ToolInvocation::new("add", json!({"x": 40, "y": 2})))
        .await
        .expect("execute");
    assert_eq!(result, json!(42));
}

#[tokio::test]
async fn tool_returning_object_and_unicode() {
    let executor = QuickJsExecutor::new();
    executor
        .register(schemaless(
            "shape",
            r#"(args) => ({sum: args.a + args.b, label: `Σ ${args.a}+${args.b}`, list: [1, 2, 3]})"#,
        ))
        .await
        .expect("register");
    let result = executor
        .execute(ToolInvocation::new("shape", json!({"a": 1, "b": 2})))
        .await
        .expect("execute");
    assert_eq!(
        result,
        json!({"sum": 3, "label": "Σ 1+2", "list": [1, 2, 3]})
    );
}

#[tokio::test]
async fn iife_with_helpers_and_trailing_semicolon() {
    let executor = QuickJsExecutor::new();
    executor
        .register(schemaless(
            "clamp",
            r#"(() => {
                const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
                return (args) => clamp(args.v, 0, 10);
            })();"#,
        ))
        .await
        .expect("register");
    let result = executor
        .execute(ToolInvocation::new("clamp", json!({"v": 99})))
        .await
        .expect("execute");
    assert_eq!(result, json!(10));
}

#[tokio::test]
async fn async_tool_resolving_on_microtasks() {
    let executor = QuickJsExecutor::new();
    executor
        .register(schemaless(
            "async_double",
            "async (args) => (await Promise.resolve(args.n)) * 2",
        ))
        .await
        .expect("register");
    let result = executor
        .execute(ToolInvocation::new("async_double", json!({"n": 21})))
        .await
        .expect("execute");
    assert_eq!(result, json!(42));
}

#[tokio::test]
async fn never_settling_promise_is_typed() {
    let executor = QuickJsExecutor::new();
    executor
        .register(schemaless("hang", "(args) => new Promise(() => {})"))
        .await
        .expect("register");
    let err = executor
        .execute(ToolInvocation::new("hang", json!({})))
        .await
        .expect_err("must fail");
    assert!(matches!(err, ExecError::PendingPromise { .. }), "{err:?}");
}

#[tokio::test]
async fn undefined_result_maps_to_null() {
    let executor = QuickJsExecutor::new();
    executor
        .register(schemaless("noop", "(args) => undefined"))
        .await
        .expect("register");
    let result = executor
        .execute(ToolInvocation::new("noop", json!({})))
        .await
        .expect("execute");
    assert_eq!(result, Value::Null);
}

// ------------------------------------------------------------ resource kills

#[tokio::test]
async fn runaway_loop_is_killed_by_deadline() {
    let executor = QuickJsExecutor::builder()
        .deadline(Duration::from_millis(100))
        .build()
        .expect("build");
    executor
        .register(schemaless("spin", "(args) => { while (true) {} }"))
        .await
        .expect("register");

    let start = Instant::now();
    let err = executor
        .execute(ToolInvocation::new("spin", json!({})))
        .await
        .expect_err("must be killed");
    let elapsed = start.elapsed();

    match &err {
        ExecError::Timeout { name, deadline_ms } => {
            assert_eq!(name, "spin");
            assert_eq!(*deadline_ms, 100);
        }
        other => panic!("expected Timeout, got {other:?}"),
    }
    assert!(
        elapsed < Duration::from_secs(5),
        "kill took {elapsed:?}, interrupt handler did not fire promptly"
    );
    // The typed error serializes with its tag for LLM repair loops.
    assert!(err.to_llm_json().contains("\"kind\":\"timeout\""));
}

#[tokio::test]
async fn unbounded_allocation_is_killed_by_memory_limit() {
    let executor = QuickJsExecutor::builder()
        .memory_limit(8 * 1024 * 1024)
        .deadline(Duration::from_secs(10))
        .build()
        .expect("build");
    executor
        .register(schemaless(
            "hog",
            "(args) => { const a = []; while (true) { a.push(new Array(65536).fill(1)); } }",
        ))
        .await
        .expect("register");

    let err = executor
        .execute(ToolInvocation::new("hog", json!({})))
        .await
        .expect_err("must be killed");
    match &err {
        ExecError::MemoryExceeded { name, limit_bytes } => {
            assert_eq!(name, "hog");
            assert_eq!(*limit_bytes, 8 * 1024 * 1024);
        }
        other => panic!("expected MemoryExceeded, got {other:?}"),
    }
    assert!(err.to_llm_json().contains("\"kind\":\"memory_exceeded\""));
}

// --------------------------------------------------- ambient-authority denial

#[tokio::test]
async fn sandbox_has_no_ambient_authority() {
    let executor = QuickJsExecutor::new();
    executor
        .register(schemaless(
            "probe",
            r#"(args) => ({
                fetch: typeof fetch,
                xhr: typeof XMLHttpRequest,
                websocket: typeof WebSocket,
                require: typeof require,
                process: typeof process,
                std: typeof std,
                os: typeof os,
                settimeout: typeof setTimeout,
                read_file: typeof readFile,
            })"#,
        ))
        .await
        .expect("register");
    let result = executor
        .execute(ToolInvocation::new("probe", json!({})))
        .await
        .expect("execute");
    let map = result.as_object().expect("object");
    for (surface, ty) in map {
        assert_eq!(
            ty,
            &json!("undefined"),
            "ambient authority leak: `{surface}` is {ty}"
        );
    }
}

#[tokio::test]
async fn dynamic_import_of_std_modules_fails() {
    let executor = QuickJsExecutor::new();
    executor
        .register(schemaless(
            "escape",
            r#"async (args) => {
                try {
                    const m = await import("qjs:std");
                    return {escaped: true, module: typeof m};
                } catch (e) {
                    return {escaped: false, error: String(e)};
                }
            }"#,
        ))
        .await
        .expect("register");
    let result = executor
        .execute(ToolInvocation::new("escape", json!({})))
        .await
        .expect("execute");
    assert_eq!(result["escaped"], json!(false), "std escape: {result}");
}

// ------------------------------------------------------- capability injection

#[tokio::test(flavor = "multi_thread")]
async fn capability_round_trip() {
    let executor = QuickJsExecutor::builder()
        .capability(Capability::new(
            "double",
            "double a number on the host",
            |args| async move {
                let n = args["n"].as_f64().ok_or("expected {n: number}")?;
                Ok(json!(n * 2.0))
            },
        ))
        .build()
        .expect("build");
    assert_eq!(executor.capability_names(), vec!["double".to_string()]);

    executor
        .register(schemaless(
            "quadruple",
            "(args) => double({n: double({n: args.n}).valueOf()})",
        ))
        .await
        .expect("register");
    let result = executor
        .execute(ToolInvocation::new("quadruple", json!({"n": 10})))
        .await
        .expect("execute");
    assert_eq!(result.as_f64(), Some(40.0), "{result:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn capability_doing_real_async_work() {
    let executor = QuickJsExecutor::builder()
        .capability(Capability::new(
            "slow_echo",
            "sleep then echo",
            |args| async move {
                tokio::time::sleep(Duration::from_millis(25)).await;
                Ok(args)
            },
        ))
        .build()
        .expect("build");
    executor
        .register(schemaless("relay", "(args) => slow_echo({msg: args.msg})"))
        .await
        .expect("register");
    let result = executor
        .execute(ToolInvocation::new("relay", json!({"msg": "hi"})))
        .await
        .expect("execute");
    assert_eq!(result, json!({"msg": "hi"}));
}

#[tokio::test(flavor = "multi_thread")]
async fn capability_error_is_typed_and_attributed() {
    let executor = QuickJsExecutor::builder()
        .capability(Capability::new("flaky", "always fails", |_| async move {
            Err("upstream exploded".to_string())
        }))
        .build()
        .expect("build");
    executor
        .register(schemaless("caller", "(args) => flaky({})"))
        .await
        .expect("register");
    let err = executor
        .execute(ToolInvocation::new("caller", json!({})))
        .await
        .expect_err("must fail");
    match &err {
        ExecError::Capability {
            name,
            capability,
            message,
        } => {
            assert_eq!(name, "caller");
            assert_eq!(capability, "flaky");
            assert!(message.contains("upstream exploded"), "{message}");
        }
        other => panic!("expected Capability error, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn tool_can_catch_capability_errors() {
    let executor = QuickJsExecutor::builder()
        .capability(Capability::new("flaky", "always fails", |_| async move {
            Err("boom".to_string())
        }))
        .build()
        .expect("build");
    executor
        .register(schemaless(
            "resilient",
            r#"(args) => { try { return flaky({}); } catch (e) { return "recovered"; } }"#,
        ))
        .await
        .expect("register");
    let result = executor
        .execute(ToolInvocation::new("resilient", json!({})))
        .await
        .expect("execute");
    assert_eq!(result, json!("recovered"));
}

#[test]
fn reserved_and_invalid_capability_names_are_rejected() {
    for bad in [
        "__dsrs_cap_x",
        "has space",
        "1starts_with_digit",
        "",
        // Injection attempt: must never reach the sandbox bootstrap.
        "x; globalThis.leak = 1; //",
        // Reserved globals/words: shadowing them breaks the runtime's shims.
        "JSON",
        "Object",
        "Promise",
        "globalThis",
        "undefined",
        "class",
        "eval",
    ] {
        let err = QuickJsExecutor::builder()
            .capability(Capability::new(
                bad,
                "bad",
                |_| async move { Ok(json!(null)) },
            ))
            .build()
            .expect_err("must reject");
        assert!(
            matches!(err, RegisterError::InvalidCapability { .. }),
            "{bad:?} -> {err:?}"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn capability_call_is_bounded_by_the_deadline() {
    // The interrupt handler cannot fire while host code runs; the executor
    // must bound the capability call itself with the remaining deadline.
    let executor = QuickJsExecutor::builder()
        .deadline(Duration::from_millis(100))
        .capability(Capability::new(
            "stall",
            "never returns within the deadline",
            |_| async move {
                tokio::time::sleep(Duration::from_secs(3600)).await;
                Ok(json!(null))
            },
        ))
        .build()
        .expect("build");
    executor
        .register(schemaless("stuck", "(args) => stall({})"))
        .await
        .expect("register");

    let start = Instant::now();
    let err = executor
        .execute(ToolInvocation::new("stuck", json!({})))
        .await
        .expect_err("must time out");
    let elapsed = start.elapsed();

    match &err {
        ExecError::Timeout { name, deadline_ms } => {
            assert_eq!(name, "stuck");
            assert_eq!(*deadline_ms, 100);
        }
        other => panic!("expected Timeout, got {other:?}"),
    }
    assert!(
        elapsed < Duration::from_secs(5),
        "capability call was not bounded: {elapsed:?}"
    );
}

#[tokio::test]
async fn capabilities_work_on_a_current_thread_runtime() {
    // Capability calls are bridged via spawn + channel (not
    // `Handle::block_on`), so a current-thread runtime must work too.
    let executor = QuickJsExecutor::builder()
        .capability(Capability::new("double", "double a number", |args| async move {
            let n = args["n"].as_f64().ok_or("expected {n: number}")?;
            Ok(json!(n * 2.0))
        }))
        .build()
        .expect("build");
    executor
        .register(schemaless("via_cap", "(args) => double({n: args.n})"))
        .await
        .expect("register");
    let result = executor
        .execute(ToolInvocation::new("via_cap", json!({"n": 21})))
        .await
        .expect("execute");
    assert_eq!(result.as_f64(), Some(42.0), "{result:?}");
}

// ----------------------------------------------------- validate-then-register

#[tokio::test]
async fn failing_self_test_blocks_registration() {
    let executor = QuickJsExecutor::new();
    let source = add_tool()
        .with_self_test("if (tool({x: 1, y: 1}) !== 3) throw new Error('math is broken today')");
    let err = executor.register(source).await.expect_err("must fail");
    match &err {
        RegisterError::SelfTest { message } => {
            assert!(message.contains("math is broken today"), "{message}");
        }
        other => panic!("expected SelfTest, got {other:?}"),
    }
    // The gate held: nothing was registered.
    assert!(executor.tool("add").is_none());
    assert!(executor.tools().is_empty());
    let exec_err = executor
        .execute(ToolInvocation::new("add", json!({"x": 1, "y": 1})))
        .await
        .expect_err("not registered");
    assert!(matches!(exec_err, ExecError::NotFound { .. }));
    assert!(err.to_llm_json().contains("\"stage\":\"self_test\""));
}

#[tokio::test]
async fn self_test_returning_false_blocks_registration() {
    let executor = QuickJsExecutor::new();
    let source = add_tool().with_self_test("tool({x: 1, y: 1}) === 3");
    let err = executor.register(source).await.expect_err("must fail");
    assert!(matches!(err, RegisterError::SelfTest { .. }), "{err:?}");
    assert!(executor.tool("add").is_none());
}

#[tokio::test]
async fn passing_self_test_registers() {
    let executor = QuickJsExecutor::new();
    let source =
        add_tool().with_self_test("if (tool({x: 2, y: 3}) !== 5) throw new Error('bad math')");
    let meta = executor.register(source).await.expect("register");
    assert!(meta.self_tested);
    assert_eq!(
        executor
            .execute(ToolInvocation::new("add", json!({"x": 2, "y": 3})))
            .await
            .expect("execute"),
        json!(5)
    );
}

#[tokio::test]
async fn syntax_error_fails_at_compile_stage() {
    let executor = QuickJsExecutor::new();
    let err = executor
        .register(schemaless("broken", "(args => {"))
        .await
        .expect_err("must fail");
    match &err {
        RegisterError::Compile { message } => assert!(!message.is_empty()),
        other => panic!("expected Compile, got {other:?}"),
    }
    assert!(executor.tool("broken").is_none());
    assert!(err.to_llm_json().contains("\"stage\":\"compile\""));
}

#[tokio::test]
async fn non_function_source_is_rejected() {
    let executor = QuickJsExecutor::new();
    let err = executor
        .register(schemaless("not_fn", "40 + 2"))
        .await
        .expect_err("must fail");
    match &err {
        RegisterError::NotAFunction { evaluated_type } => assert_eq!(evaluated_type, "number"),
        other => panic!("expected NotAFunction, got {other:?}"),
    }
}

#[tokio::test]
async fn malformed_schema_is_rejected() {
    let executor = QuickJsExecutor::new();
    let mut source = add_tool();
    source.params = json!({"type": "object", "properties": {"x": {}}, "required": ["missing"]});
    let err = executor.register(source).await.expect_err("must fail");
    assert!(
        matches!(err, RegisterError::InvalidSchema { .. }),
        "{err:?}"
    );

    let mut source = add_tool();
    source.params = json!("not a schema");
    let err = executor.register(source).await.expect_err("must fail");
    assert!(
        matches!(err, RegisterError::InvalidSchema { .. }),
        "{err:?}"
    );
}

#[tokio::test]
async fn invalid_names_and_duplicates_are_rejected() {
    let executor = QuickJsExecutor::new();
    let mut source = add_tool();
    source.name = "no spaces!".to_string();
    let err = executor.register(source).await.expect_err("must fail");
    assert!(matches!(err, RegisterError::InvalidName { .. }), "{err:?}");

    executor.register(add_tool()).await.expect("register");
    let err = executor.register(add_tool()).await.expect_err("duplicate");
    assert!(matches!(err, RegisterError::Duplicate { .. }), "{err:?}");
}

#[tokio::test]
async fn missing_required_args_fail_before_sandbox() {
    let executor = QuickJsExecutor::new();
    executor.register(add_tool()).await.expect("register");
    let err = executor
        .execute(ToolInvocation::new("add", json!({"x": 1})))
        .await
        .expect_err("must fail");
    match &err {
        ExecError::InvalidArgs { reason, .. } => assert!(reason.contains("`y`"), "{reason}"),
        other => panic!("expected InvalidArgs, got {other:?}"),
    }

    let err = executor
        .execute(ToolInvocation::new("add", json!([1, 2])))
        .await
        .expect_err("must fail");
    assert!(matches!(err, ExecError::InvalidArgs { .. }), "{err:?}");
}

#[tokio::test]
async fn js_exceptions_are_reported_with_message() {
    let executor = QuickJsExecutor::new();
    executor
        .register(schemaless(
            "thrower",
            r#"(args) => { throw new Error("deliberate failure: " + args.why); }"#,
        ))
        .await
        .expect("register");
    let err = executor
        .execute(ToolInvocation::new("thrower", json!({"why": "testing"})))
        .await
        .expect_err("must fail");
    match &err {
        ExecError::Js { name, message } => {
            assert_eq!(name, "thrower");
            assert!(message.contains("deliberate failure: testing"), "{message}");
        }
        other => panic!("expected Js, got {other:?}"),
    }
}

// ------------------------------------------------------------- bytecode cache

#[tokio::test]
async fn identical_sources_share_cached_bytecode() {
    let executor = QuickJsExecutor::new();
    let js = "(args) => args.x * 3";
    executor
        .register(schemaless("triple_a", js))
        .await
        .expect("register a");
    let stats = executor.cache_stats();
    assert_eq!((stats.entries, stats.hits, stats.misses), (1, 0, 1));

    executor
        .register(schemaless("triple_b", js))
        .await
        .expect("register b");
    let stats = executor.cache_stats();
    assert_eq!((stats.entries, stats.hits, stats.misses), (1, 1, 1));

    // Same hash reported for both; both actually run.
    assert_eq!(
        executor.tool("triple_a").unwrap().source_hash,
        executor.tool("triple_b").unwrap().source_hash
    );
    for name in ["triple_a", "triple_b"] {
        assert_eq!(
            executor
                .execute(ToolInvocation::new(name, json!({"x": 4})))
                .await
                .expect("execute"),
            json!(12)
        );
    }

    // A different source misses the cache.
    executor
        .register(schemaless("triple_c", "(args) => 3 * args.x"))
        .await
        .expect("register c");
    let stats = executor.cache_stats();
    assert_eq!((stats.entries, stats.misses), (2, 2));
}

#[tokio::test]
async fn deregister_evicts_bytecode_unless_shared() {
    let executor = QuickJsExecutor::new();
    let js = "(args) => args.x * 3";
    executor
        .register(schemaless("triple_a", js))
        .await
        .expect("register a");
    executor
        .register(schemaless("triple_b", js))
        .await
        .expect("register b");
    // Two tools, one shared cache entry.
    assert_eq!(executor.cache_stats().entries, 1);

    // Still referenced by triple_b: the entry must survive.
    assert!(executor.deregister("triple_a"));
    assert_eq!(executor.cache_stats().entries, 1);
    assert_eq!(
        executor
            .execute(ToolInvocation::new("triple_b", json!({"x": 2})))
            .await
            .expect("execute"),
        json!(6)
    );

    // Last reference gone: the bytecode is evicted with it.
    assert!(executor.deregister("triple_b"));
    assert_eq!(executor.cache_stats().entries, 0);
    assert!(!executor.deregister("triple_b"), "already gone");
}

#[tokio::test]
async fn bytecode_cache_is_bounded() {
    // The cache is capped (cap-and-clear); an optimizer generating many
    // candidate sources must not grow it without bound, and registered tools
    // must keep working after eviction (they hold their own bytecode Arc).
    let executor = QuickJsExecutor::new();
    let count = 140; // > the 128-entry cap
    for i in 0..count {
        executor
            .register(schemaless(
                &format!("cand_{i}"),
                &format!("(args) => args.x + {i}"),
            ))
            .await
            .expect("register");
    }
    let stats = executor.cache_stats();
    assert!(
        stats.entries <= 128,
        "cache exceeded its bound: {} entries",
        stats.entries
    );
    // Tools registered before the clear still execute.
    for i in [0, count - 1] {
        assert_eq!(
            executor
                .execute(ToolInvocation::new(format!("cand_{i}"), json!({"x": 1})))
                .await
                .expect("execute"),
            json!(1 + i)
        );
    }
}

#[tokio::test]
async fn executing_many_times_never_recompiles() {
    let executor = QuickJsExecutor::new();
    executor.register(add_tool()).await.expect("register");
    let before = executor.cache_stats();
    for i in 0..10 {
        executor
            .execute(ToolInvocation::new("add", json!({"x": i, "y": i})))
            .await
            .expect("execute");
    }
    let after = executor.cache_stats();
    // Execution replays cached bytecode: no new compiles (misses) at all.
    assert_eq!(before.misses, after.misses);
    assert_eq!(before.entries, after.entries);
}

// ----------------------------------------------------------- rig ToolDyn bridge

#[tokio::test]
async fn registered_tool_works_as_rig_tooldyn() {
    let executor = Arc::new(QuickJsExecutor::new());
    let tool = executor
        .register_rig(add_tool().with_self_test("tool({x: 1, y: 2}) === 3"))
        .await
        .expect("register");

    assert_eq!(tool.name(), "add");
    let definition = tool.definition(String::new()).await;
    assert_eq!(definition.name, "add");
    assert_eq!(definition.description, "Add two numbers");
    assert_eq!(definition.parameters["required"], json!(["x", "y"]));

    let output = tool
        .call(r#"{"x": 20, "y": 22}"#.to_string())
        .await
        .expect("call");
    assert_eq!(output, "42");

    // Errors surface as structured JSON the model can parse.
    let err = tool
        .call(r#"{"x": 1}"#.to_string())
        .await
        .expect_err("missing arg");
    assert!(err.to_string().contains("invalid_args"), "{err}");

    // `rig_tool` hands out additional handles to the same registered tool.
    let again = executor.rig_tool("add").expect("registered");
    assert_eq!(again.name(), "add");
    assert!(executor.rig_tool("nope").is_none());
}

#[tokio::test]
async fn rig_tool_coerces_to_plain_tooldyn_arc() {
    // dspy-rs surfaces take `Arc<dyn ToolDyn>`; make sure our handle coerces.
    let executor = Arc::new(QuickJsExecutor::new());
    let tool = executor.register_rig(add_tool()).await.expect("register");
    let plain: Arc<dyn dsrs_tools::ToolDyn> = tool;
    assert_eq!(plain.name(), "add");
}

// -------------------------------------------------------------- sync escape

#[tokio::test(flavor = "multi_thread")]
async fn execute_blocking_matches_async_path() {
    let executor = Arc::new(QuickJsExecutor::new());
    executor.register(add_tool()).await.expect("register");
    let executor2 = Arc::clone(&executor);
    let result = tokio::task::spawn_blocking(move || {
        executor2.execute_blocking(ToolInvocation::new("add", json!({"x": 2, "y": 2})))
    })
    .await
    .expect("join")
    .expect("execute");
    assert_eq!(result, json!(4));
}
