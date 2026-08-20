//! Hot-path microbenchmark with allocation counting.
//!
//! Measures the framework overhead per LM call (prompt build, dispatch, parse)
//! using the in-process test client, so provider latency is excluded and what
//! remains is pure DSRs CPU + allocator cost.
//!
//! Run with: `cargo run --release --example 97-perf-microbench`

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use dspy_rs::{
    Chat, ChatAdapter, Demo, LM, LMClient, Message, Predict, Signature, SignatureSchema,
    TestCompletionModel, configure, fx,
};
use rig::completion::{AssistantContent, ToolDefinition};
use rig::message::Text;
use rig::tool::Tool;

struct CountingAlloc;

static ALLOCS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

#[derive(Signature, Clone, Debug)]
/// Answer the question using the context. Be concise and accurate.
struct BenchQA {
    #[input]
    question: String,

    #[input]
    context: String,

    #[output]
    answer: String,

    #[output]
    confidence: f32,
}

#[derive(Signature, Clone, Debug)]
/// Rate the sentiment of the text.
struct BenchChecked {
    #[input]
    text: String,

    #[output]
    #[check("this|length > 0", label = "non_empty")]
    label: String,

    #[output]
    #[check("this >= 0.0 and this <= 1.0", label = "valid_confidence")]
    confidence: f32,
}

#[derive(Clone)]
struct NoopTool;

#[derive(Debug)]
struct NoopToolError;

impl std::fmt::Display for NoopToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "noop tool error")
    }
}

impl std::error::Error for NoopToolError {}

impl Tool for NoopTool {
    const NAME: &'static str = "noop";
    type Error = NoopToolError;
    type Args = serde_json::Value;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "does nothing, quickly".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "arg": { "type": "string" } },
                "additionalProperties": false
            }),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok("ok".to_string())
    }
}

fn checked_response_text() -> String {
    "[[ ## label ## ]]\npositive\n\n[[ ## confidence ## ]]\n0.85\n\n[[ ## completed ## ]]\n"
        .to_string()
}

fn response_text() -> String {
    "[[ ## answer ## ]]\nParis is the capital of France.\n\n[[ ## confidence ## ]]\n0.95\n\n[[ ## completed ## ]]\n".to_string()
}

fn assistant_content() -> AssistantContent {
    AssistantContent::Text(Text {
        text: response_text(),
    })
}

fn bench_input() -> BenchQAInput {
    BenchQAInput {
        question: "What is the capital of France?".to_string(),
        context: "France is a country in Western Europe. Its capital and largest city is Paris, known for the Eiffel Tower and the Louvre.".to_string(),
    }
}

fn demo(idx: usize) -> Demo<BenchQA> {
    Demo::new(
        BenchQAInput {
            question: format!("Demo question {idx}?"),
            context: format!("Demo context {idx} with enough text to look like a real retrieval chunk for the benchmark."),
        },
        BenchQAOutput {
            answer: format!("Demo answer {idx}."),
            confidence: 0.9,
        },
    )
}

struct Snapshot {
    allocs: u64,
    bytes: u64,
    start: Instant,
}

fn snap() -> Snapshot {
    Snapshot {
        allocs: ALLOCS.load(Ordering::Relaxed),
        bytes: BYTES.load(Ordering::Relaxed),
        start: Instant::now(),
    }
}

fn report(name: &str, iters: u64, s: Snapshot) {
    let elapsed = s.start.elapsed();
    let allocs = ALLOCS.load(Ordering::Relaxed) - s.allocs;
    let bytes = BYTES.load(Ordering::Relaxed) - s.bytes;
    println!(
        "{name:<44} {:>10.0} ns/op {:>9.1} allocs/op {:>10.0} B/op",
        elapsed.as_nanos() as f64 / iters as f64,
        allocs as f64 / iters as f64,
        bytes as f64 / iters as f64,
    );
}

async fn make_lm(responses: u64, cache: bool) -> LM {
    let client =
        TestCompletionModel::new((0..responses).map(|_| assistant_content()));
    temp_env::async_with_vars(
        [("OPENAI_API_KEY", Some("bench"))],
        LM::builder()
            .model("openai:gpt-4o-mini".to_string())
            .cache(cache)
            .build(),
    )
    .await
    .unwrap()
    .with_client(LMClient::Test(client))
    .await
    .unwrap()
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    println!(
        "{:<44} {:>13} {:>15} {:>12}",
        "phase", "time", "allocs", "bytes"
    );

    // --- 1. Schema lookup: derive fast path vs global map ------------------
    let iters = 2_000_000u64;
    let s = snap();
    for _ in 0..iters {
        std::hint::black_box(<BenchQA as Signature>::schema());
    }
    report("schema(): per-type OnceLock fast path", iters, s);

    let s = snap();
    for _ in 0..iters {
        std::hint::black_box(SignatureSchema::of::<BenchQA>());
    }
    report("schema(): global RwLock map path", iters, s);

    // --- 2. Prompt build (system + 2 demos + user) --------------------------
    let predict = Predict::<BenchQA>::builder()
        .demo(demo(1))
        .demo(demo(2))
        .build();
    let input = bench_input();

    let iters = 100_000u64;
    let s = snap();
    for _ in 0..iters {
        std::hint::black_box(predict.build_chat(&input).unwrap());
    }
    report("build_chat (system + 2 demos + user)", iters, s);

    // --- 3. Parse (typed output extraction) ---------------------------------
    let assistant = Message::assistant(&response_text());
    let adapter = ChatAdapter;
    let iters = 100_000u64;
    let s = snap();
    for _ in 0..iters {
        std::hint::black_box(
            adapter
                .parse_output_def(
                    dspy_rs::ir::SignatureDef::of::<BenchQA>(),
                    dspy_rs::ir::SignatureDef::types_of::<BenchQA>(),
                    &assistant,
                )
                .unwrap(),
        );
    }
    report("parse_output_def (2 fields)", iters, s);

    // --- 4. Full forward with test client (no demos) ------------------------
    let iters = 50_000u64;
    let lm = make_lm(iters, false).await;
    let plain = Predict::<BenchQA>::builder().lm(lm).build();
    let s = snap();
    for _ in 0..iters {
        std::hint::black_box(plain.call(bench_input()).await.unwrap());
    }
    report("forward end-to-end (0 demos, test LM)", iters, s);

    // --- 5. Full forward with 2 demos ----------------------------------------
    let iters = 50_000u64;
    let lm = make_lm(iters, false).await;
    let demoed = Predict::<BenchQA>::builder()
        .lm(lm)
        .demo(demo(1))
        .demo(demo(2))
        .build();
    let s = snap();
    for _ in 0..iters {
        std::hint::black_box(demoed.call(bench_input()).await.unwrap());
    }
    report("forward end-to-end (2 demos, test LM)", iters, s);

    // --- 6. Cached LM call (cache hit path) ----------------------------------
    let lm = make_lm(1, true).await;
    let chat = Chat::new(vec![
        Message::system("Answer the question."),
        Message::user("What is the capital of France?"),
    ]);
    // Prime the cache.
    lm.call(chat.clone(), vec![]).await.unwrap();
    let iters = 20_000u64;
    let s = snap();
    for _ in 0..iters {
        std::hint::black_box(lm.call(chat.clone(), vec![]).await.unwrap());
    }
    report("LM::call cache HIT (foyer + key build)", iters, s);

    // --- 7. LM::call alone (no cache, minimal chat) --------------------------
    let iters = 50_000u64;
    let lm = make_lm(iters, false).await;
    let s = snap();
    for _ in 0..iters {
        std::hint::black_box(lm.call(chat.clone(), vec![]).await.unwrap());
    }
    report("LM::call no cache (dispatch + history)", iters, s);

    // --- 8. Parse with #[check] constraints (2 fields, 2 checks) -------------
    let checked_assistant = Message::assistant(&checked_response_text());
    let iters = 100_000u64;
    let s = snap();
    for _ in 0..iters {
        std::hint::black_box(
            adapter
                .parse_output_def(
                    dspy_rs::ir::SignatureDef::of::<BenchChecked>(),
                    dspy_rs::ir::SignatureDef::types_of::<BenchChecked>(),
                    &checked_assistant,
                )
                .unwrap(),
        );
    }
    report("parse_output_def (2 checks)", iters, s);

    // --- 9. Forward with 1 tool attached (never called) -----------------------
    let iters = 50_000u64;
    let lm = make_lm(iters, false).await;
    let tooled = Predict::<BenchQA>::builder().lm(lm).add_tool(NoopTool).build();
    let s = snap();
    for _ in 0..iters {
        std::hint::black_box(tooled.call(bench_input()).await.unwrap());
    }
    report("forward end-to-end (1 tool, unused)", iters, s);

    // --- 10. fx::predict vs struct (same signature, global LM) ---------------
    let iters = 50_000u64;
    configure(make_lm(iters, false).await);
    let s = snap();
    for _ in 0..iters {
        std::hint::black_box(
            fx::predict::<BenchQA>("bench_fx", bench_input()).await.unwrap(),
        );
    }
    report("fx::predict (0 demos, default params)", iters, s);

    // --- 11. fx::predict under a with_params scope ----------------------------
    let iters = 50_000u64;
    configure(make_lm(iters, false).await);
    let mut params = fx::Params::new();
    params.set_instruction("bench_fx", "Answer concisely with high confidence.");
    let s = snap();
    fx::with_params(params, async {
        for _ in 0..iters {
            std::hint::black_box(
                fx::predict::<BenchQA>("bench_fx", bench_input()).await.unwrap(),
            );
        }
    })
    .await;
    report("fx::predict (with_params override)", iters, s);

    // --- 12. Struct Predict through the same global-LM path -------------------
    let iters = 50_000u64;
    configure(make_lm(iters, false).await);
    let global_predict = Predict::<BenchQA>::new();
    let s = snap();
    for _ in 0..iters {
        std::hint::black_box(global_predict.call(bench_input()).await.unwrap());
    }
    report("struct Predict (0 demos, global LM)", iters, s);
}
