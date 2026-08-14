//! Latency microbench for the Tier-1 QuickJS tool runtime.
//!
//! Run with optimizations or the numbers are meaningless:
//!
//! ```sh
//! cargo run -p dsrs-tools --example bench --release
//! ```
//!
//! Measures, in microseconds:
//! 1. raw sandbox lifecycle (runtime + context create, eval `1+1`, teardown),
//! 2. cold tool registration (compile + instantiate + self-test),
//! 3. warm registration of identical source (bytecode-cache hit),
//! 4. cached tool call, sync path (`execute_blocking`: fresh sandbox + cached
//!    bytecode load + call + teardown, no thread hop),
//! 5. cached tool call, async path (`execute`: adds the Tokio blocking-pool
//!    round trip),
//! 6. cached call with a capability round trip (JS -> async Rust -> JS).

use std::sync::Arc;
use std::time::{Duration, Instant};

use dsrs_tools::{Capability, Executor, QuickJsExecutor, ToolInvocation, ToolSource};
use serde_json::json;

fn stats(label: &str, samples: &mut [u128]) {
    samples.sort_unstable();
    let n = samples.len();
    let avg = samples.iter().sum::<u128>() as f64 / n as f64 / 1000.0;
    let p = |q: f64| samples[((n as f64 * q) as usize).min(n - 1)] as f64 / 1000.0;
    println!(
        "{label:<42} n={n:<5} min={:8.1}µs  p50={:8.1}µs  avg={avg:8.1}µs  p99={:8.1}µs  max={:8.1}µs",
        samples[0] as f64 / 1000.0,
        p(0.50),
        p(0.99),
        samples[n - 1] as f64 / 1000.0,
    );
}

fn time<R>(f: impl FnOnce() -> R) -> (u128, R) {
    let start = Instant::now();
    let out = f();
    (start.elapsed().as_nanos(), out)
}

const N: usize = 2000;

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    println!(
        "dsrs-tools Tier-1 bench ({} mode)\n",
        if cfg!(debug_assertions) {
            "DEBUG -- rerun with --release"
        } else {
            "release"
        }
    );

    // 1. Raw sandbox lifecycle: create runtime+context, eval, teardown.
    let mut samples = Vec::with_capacity(N);
    for _ in 0..N {
        let (ns, out) = time(|| {
            let runtime = rquickjs::Runtime::new().unwrap();
            runtime.set_memory_limit(32 * 1024 * 1024);
            let context = rquickjs::Context::full(&runtime).unwrap();
            context.with(|ctx| ctx.eval::<i32, _>("1+1").unwrap())
        });
        assert_eq!(out, 2);
        samples.push(ns);
    }
    stats("sandbox lifecycle (create/eval/teardown)", &mut samples);

    // 2. Cold registration: unique source each time -> full compile pipeline.
    let executor = Arc::new(QuickJsExecutor::new());
    let mut samples = Vec::with_capacity(500);
    for i in 0..500 {
        let source = ToolSource::new(
            format!("cold_{i}"),
            "bench tool",
            json!({"type": "object"}),
            format!("(args) => args.x + {i}"),
        )
        .with_self_test(format!("tool({{x: 0}}) === {i}"));
        let start = Instant::now();
        executor.register(source).await.expect("register");
        samples.push(start.elapsed().as_nanos());
    }
    stats("cold register (compile+instantiate+test)", &mut samples);

    // 3. Warm registration: same source, new name -> bytecode cache hit.
    let shared_js = "(args) => args.x * 2";
    executor
        .register(ToolSource::new(
            "warm_seed",
            "bench tool",
            json!({"type": "object"}),
            shared_js,
        ))
        .await
        .expect("register");
    let mut samples = Vec::with_capacity(500);
    for i in 0..500 {
        let source = ToolSource::new(
            format!("warm_{i}"),
            "bench tool",
            json!({"type": "object"}),
            shared_js,
        );
        let start = Instant::now();
        executor.register(source).await.expect("register");
        samples.push(start.elapsed().as_nanos());
    }
    stats("warm register (bytecode-cache hit)", &mut samples);
    let cache = executor.cache_stats();
    println!(
        "{:<42} entries={} hits={} misses={}",
        "  cache after warm registrations", cache.entries, cache.hits, cache.misses
    );

    // 4. Cached call, sync path: full sandbox per call, no thread hop.
    executor
        .register(ToolSource::new(
            "add",
            "add",
            json!({"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}}, "required": ["x", "y"]}),
            "(args) => args.x + args.y",
        ))
        .await
        .expect("register");
    let sync_executor = Arc::clone(&executor);
    let mut samples = tokio::task::spawn_blocking(move || {
        let mut samples = Vec::with_capacity(N);
        for i in 0..N {
            let invocation = ToolInvocation::new("add", json!({"x": i, "y": 1}));
            let (ns, out) = time(|| sync_executor.execute_blocking(invocation).unwrap());
            assert_eq!(out, json!(i + 1));
            samples.push(ns);
        }
        samples
    })
    .await
    .expect("join");
    stats("cached call, sync (execute_blocking)", &mut samples);

    // 5. Cached call, async path: adds the blocking-pool round trip.
    let mut samples = Vec::with_capacity(N);
    for i in 0..N {
        let invocation = ToolInvocation::new("add", json!({"x": i, "y": 2}));
        let start = Instant::now();
        let out = executor.execute(invocation).await.expect("execute");
        samples.push(start.elapsed().as_nanos());
        assert_eq!(out, json!(i + 2));
    }
    stats("cached call, async (execute)", &mut samples);

    // 6. Cached call crossing the capability bridge on every invocation.
    let cap_executor = Arc::new(
        QuickJsExecutor::builder()
            .deadline(Duration::from_secs(1))
            .capability(Capability::new("echo", "echo", |args| async move {
                Ok(args)
            }))
            .build()
            .expect("build"),
    );
    cap_executor
        .register(ToolSource::new(
            "relay",
            "relay through host",
            json!({"type": "object"}),
            "(args) => echo({v: args.v}).v",
        ))
        .await
        .expect("register");
    let mut samples = Vec::with_capacity(N);
    for i in 0..N {
        let invocation = ToolInvocation::new("relay", json!({"v": i}));
        let start = Instant::now();
        let out = cap_executor.execute(invocation).await.expect("execute");
        samples.push(start.elapsed().as_nanos());
        assert_eq!(out, json!(i));
    }
    stats("cached call + capability round trip", &mut samples);
}
