//! Deep/heavy harness benchmark: framework cost on realistic pipeline shapes —
//! long chains, wide fan-outs, layered DAGs, demo- and context-heavy prompts,
//! deep tool loops, and concurrent evaluation of deep pipelines. All LM calls
//! hit the in-process test client, so numbers are pure framework overhead.
//!
//! Run with: `cargo run --release --example 99-deep-harness-bench`

use std::time::Instant;

use anyhow::Result;
use dspy_rs::{
    CallMetadata, Example, LM, LMClient, MetricOutcome, Module, Predict, PredictError, Predicted,
    Signature, TestCompletionModel, TypedMetric, evaluate_trainset_with_concurrency,
};
use rig::completion::{AssistantContent, ToolDefinition};
use rig::message::Text;
use rig::tool::Tool;

// --- Signatures -------------------------------------------------------------

#[derive(Signature, Clone, Debug)]
/// Transform the text one step.
struct Step {
    #[input]
    text: String,

    #[output]
    result: String,
}

#[derive(Signature, Clone, Debug)]
/// Produce all sixteen analysis fields.
struct Wide16 {
    #[input]
    text: String,

    #[output]
    f01: String,
    #[output]
    f02: String,
    #[output]
    f03: String,
    #[output]
    f04: String,
    #[output]
    f05: String,
    #[output]
    f06: String,
    #[output]
    f07: String,
    #[output]
    f08: String,
    #[output]
    f09: String,
    #[output]
    f10: String,
    #[output]
    f11: String,
    #[output]
    f12: String,
    #[output]
    f13: String,
    #[output]
    f14: String,
    #[output]
    f15: String,
    #[output]
    f16: String,
}

// --- Test LM plumbing -------------------------------------------------------

fn step_response() -> String {
    "[[ ## result ## ]]\nProcessed step output with a plausible amount of text in it.\n\n[[ ## completed ## ]]\n".to_string()
}

fn wide_response() -> String {
    let mut out = String::new();
    for i in 1..=16 {
        out.push_str(&format!("[[ ## f{i:02} ## ]]\nvalue for field {i}\n\n"));
    }
    out.push_str("[[ ## completed ## ]]\n");
    out
}

fn text_content(text: &str) -> AssistantContent {
    AssistantContent::Text(Text { text: text.into() })
}

async fn test_lm(responses: Vec<AssistantContent>, max_tool_iterations: u32) -> LM {
    let client = TestCompletionModel::new(responses);
    temp_env::async_with_vars(
        [("OPENAI_API_KEY", Some("bench"))],
        LM::builder()
            .model("openai:gpt-4o-mini".to_string())
            .max_tool_iterations(max_tool_iterations)
            .build(),
    )
    .await
    .unwrap()
    .with_client(LMClient::Test(client))
    .await
    .unwrap()
}

// --- Pipeline shapes --------------------------------------------------------

/// N Predict stages in strict sequence, each consuming the previous output.
struct Chain {
    stages: Vec<Predict<Step>>,
}

impl Chain {
    fn new(depth: usize, lm: &LM) -> Self {
        Self {
            stages: (0..depth)
                .map(|_| Predict::<Step>::builder().lm(lm.clone()).build())
                .collect(),
        }
    }
}

impl Module for Chain {
    type Input = StepInput;
    type Output = StepOutput;

    async fn forward(&self, input: StepInput) -> Result<Predicted<StepOutput>, PredictError> {
        let mut text = input.text;
        let mut last = None;
        for stage in &self.stages {
            let predicted = stage.call(StepInput { text }).await?;
            text = predicted.result.clone();
            last = Some(predicted);
        }
        Ok(last.expect("chain has at least one stage"))
    }
}

/// N branches executed concurrently, then one aggregation stage.
struct FanOut {
    branches: Vec<Predict<Step>>,
    aggregate: Predict<Step>,
}

impl FanOut {
    fn new(width: usize, lm: &LM) -> Self {
        Self {
            branches: (0..width)
                .map(|_| Predict::<Step>::builder().lm(lm.clone()).build())
                .collect(),
            aggregate: Predict::<Step>::builder().lm(lm.clone()).build(),
        }
    }
}

impl Module for FanOut {
    type Input = StepInput;
    type Output = StepOutput;

    async fn forward(&self, input: StepInput) -> Result<Predicted<StepOutput>, PredictError> {
        let branch_outputs = futures::future::try_join_all(
            self.branches
                .iter()
                .map(|branch| branch.call(input.clone())),
        )
        .await?;
        let combined = branch_outputs
            .iter()
            .map(|predicted| predicted.result.as_str())
            .collect::<Vec<_>>()
            .join(" | ");
        self.aggregate.call(StepInput { text: combined }).await
    }
}

/// `layers` fan-outs in sequence — a layered DAG (depth x width Predicts + one
/// aggregate per layer).
struct LayeredDag {
    layers: Vec<FanOut>,
}

impl Module for LayeredDag {
    type Input = StepInput;
    type Output = StepOutput;

    async fn forward(&self, input: StepInput) -> Result<Predicted<StepOutput>, PredictError> {
        let mut text = input.text;
        let mut last = None;
        for layer in &self.layers {
            let predicted = layer.forward(StepInput { text }).await?;
            text = predicted.result.clone();
            last = Some(predicted);
        }
        Ok(last.expect("dag has at least one layer"))
    }
}

struct Exact;

impl TypedMetric<Step, Chain> for Exact {
    async fn evaluate(
        &self,
        _example: &Example<Step>,
        prediction: &Predicted<StepOutput>,
    ) -> Result<MetricOutcome> {
        Ok(MetricOutcome::score(
            (!prediction.result.is_empty()) as u8 as f32,
        ))
    }
}

// --- Tool-loop depth --------------------------------------------------------

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
        Ok("tool result payload".to_string())
    }
}

fn tool_call_response() -> AssistantContent {
    AssistantContent::tool_call(
        "call_1".to_string(),
        "noop".to_string(),
        serde_json::json!({"arg": "value"}),
    )
}

// --- Bench harness ----------------------------------------------------------

fn report(name: &str, iters: u64, calls_per_op: u64, start: Instant) {
    let elapsed = start.elapsed();
    let per_op = elapsed.as_nanos() as f64 / iters as f64;
    println!(
        "{name:<46} {:>10.1} us/op {:>8.2} us/LM-call",
        per_op / 1e3,
        per_op / 1e3 / calls_per_op as f64,
    );
}

fn input(text: &str) -> StepInput {
    StepInput {
        text: text.to_string(),
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    println!("{:<46} {:>13} {:>15}", "shape", "per run", "per LM call");

    // --- 1-2. Sequential chains ------------------------------------------
    for (depth, iters) in [(10usize, 2000u64), (50, 400)] {
        let lm = test_lm(
            (0..iters * depth as u64).map(|_| text_content(&step_response())).collect(),
            10,
        )
        .await;
        let chain = Chain::new(depth, &lm);
        let start = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(chain.forward(input("start")).await?);
        }
        report(&format!("chain depth {depth}"), iters, depth as u64, start);
    }

    // --- 3. Fan-out 16 + aggregate ----------------------------------------
    let iters = 1000u64;
    let lm = test_lm(
        (0..iters * 17).map(|_| text_content(&step_response())).collect(),
        10,
    )
    .await;
    let fanout = FanOut::new(16, &lm);
    let start = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(fanout.forward(input("start")).await?);
    }
    report("fan-out 16 branches + aggregate", iters, 17, start);

    // --- 4. Layered DAG 4x8 ------------------------------------------------
    let iters = 500u64;
    let lm = test_lm(
        (0..iters * 36).map(|_| text_content(&step_response())).collect(),
        10,
    )
    .await;
    let dag = LayeredDag {
        layers: (0..4).map(|_| FanOut::new(8, &lm)).collect(),
    };
    let start = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(dag.forward(input("start")).await?);
    }
    report("layered DAG 4 layers x 8 wide", iters, 36, start);

    // --- 5-6. Demo-heavy prompts -------------------------------------------
    for (demo_count, iters) in [(16usize, 5000u64), (64, 2000)] {
        let lm = test_lm(
            (0..iters).map(|_| text_content(&step_response())).collect(),
            10,
        )
        .await;
        let mut builder = Predict::<Step>::builder().lm(lm.clone());
        for i in 0..demo_count {
            builder = builder.demo(Example::new(
                StepInput {
                    text: format!("Demo input {i} with a realistic sentence of content."),
                },
                StepOutput {
                    result: format!("Demo output {i} with a realistic sentence of content."),
                },
            ));
        }
        let predict = builder.build();
        let start = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(predict.forward(input("run")).await?);
        }
        report(&format!("forward with {demo_count} demos"), iters, 1, start);
    }

    // --- 7. Wide signature (16 output fields) ------------------------------
    let iters = 5000u64;
    let lm = test_lm(
        (0..iters).map(|_| text_content(&wide_response())).collect(),
        10,
    )
    .await;
    let wide = Predict::<Wide16>::builder().lm(lm.clone()).build();
    let start = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(
            wide.forward(Wide16Input {
                text: "Analyze this input across all dimensions.".to_string(),
            })
            .await?,
        );
    }
    report("forward with 16 output fields", iters, 1, start);

    // --- 8. Context-heavy prompt (50KB input) ------------------------------
    let iters = 2000u64;
    let big_context = "The quick brown fox jumps over the lazy dog. ".repeat(1150); // ~50KB
    let lm = test_lm(
        (0..iters).map(|_| text_content(&step_response())).collect(),
        10,
    )
    .await;
    let predict = Predict::<Step>::builder().lm(lm.clone()).build();
    let start = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(predict.forward(input(&big_context)).await?);
    }
    report("forward with 50KB context", iters, 1, start);

    // --- 9. Deep tool loop (16 tool iterations) -----------------------------
    let iters = 200u64;
    let mut responses = Vec::with_capacity((iters * 17) as usize);
    for _ in 0..iters {
        for _ in 0..16 {
            responses.push(tool_call_response());
        }
        responses.push(text_content(&step_response()));
    }
    let lm = test_lm(responses, 32).await;
    let predict = Predict::<Step>::builder().lm(lm.clone()).add_tool(NoopTool).build();
    let start = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(predict.forward(input("use the tool repeatedly")).await?);
    }
    report("tool loop, 16 tool iterations", iters, 17, start);

    // --- 10. Concurrent eval of a deep pipeline ----------------------------
    let examples: Vec<Example<Step>> = (0..64)
        .map(|i| {
            Example::new(
                StepInput {
                    text: i.to_string(),
                },
                StepOutput {
                    result: String::new(),
                },
            )
        })
        .collect();
    let lm = test_lm(
        (0..64 * 10 * 2).map(|_| text_content(&step_response())).collect(),
        10,
    )
    .await;
    let chain = Chain::new(10, &lm);
    let start = Instant::now();
    let outcomes = evaluate_trainset_with_concurrency(&chain, &examples, &Exact, 16).await?;
    let wall = start.elapsed();
    println!(
        "{:<46} {:>10.1} ms wall ({} pipelines, 640 LM calls)",
        "eval 64 x depth-10 chain @ 16 concurrency",
        wall.as_secs_f64() * 1e3,
        outcomes.len(),
    );

    Ok(())
}
