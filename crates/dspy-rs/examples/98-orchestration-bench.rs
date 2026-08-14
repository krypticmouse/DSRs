//! Orchestration-efficiency benchmark: evaluation throughput around simulated
//! provider latency. Counterpart of the cross-framework bench scripts — each
//! "LM call" is a 20ms sleep, so wall time measures pure scheduling overhead.
//!
//! Run with: `cargo run --release --example 98-orchestration-bench`

use std::time::{Duration, Instant};

use anyhow::Result;
use dspy_rs::{
    CallMetadata, Example, Eval, Module, Predicted, PredictError, Signature, TypedMetric,
    evaluate_trainset_with_concurrency,
};

#[derive(Signature, Clone, Debug)]
/// Echo the question.
struct EchoQA {
    #[input]
    question: String,

    #[output]
    answer: String,
}

/// Simulates a provider round-trip: 20ms of latency, then an echo response.
struct SleepEcho;

impl Module for SleepEcho {
    type Input = EchoQAInput;
    type Output = EchoQAOutput;

    async fn forward(&self, input: EchoQAInput) -> Result<Predicted<EchoQAOutput>, PredictError> {
        tokio::time::sleep(Duration::from_millis(20)).await;
        Ok(Predicted::new(
            EchoQAOutput {
                answer: input.question,
            },
            CallMetadata::default(),
        ))
    }
}

struct Exact;

impl TypedMetric<EchoQA, SleepEcho> for Exact {
    async fn evaluate(
        &self,
        example: &Example<EchoQA>,
        prediction: &Predicted<EchoQAOutput>,
        _trace: Option<&dspy_rs::Trace>,
    ) -> Result<Eval> {
        Ok(Eval::score(
            (prediction.answer == example.output.answer) as u8 as f64,
        ))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    const N: usize = 256;
    const CONCURRENCY: usize = 16;
    const LATENCY_MS: f64 = 20.0;

    let trainset: Vec<Example<EchoQA>> = (0..N)
        .map(|i| {
            Example::new(
                EchoQAInput {
                    question: i.to_string(),
                },
                EchoQAOutput {
                    answer: i.to_string(),
                },
            )
        })
        .collect();

    // Warmup.
    evaluate_trainset_with_concurrency(&SleepEcho, &trainset[..32], &Exact, CONCURRENCY).await?;

    let start = Instant::now();
    let outcomes = evaluate_trainset_with_concurrency(&SleepEcho, &trainset, &Exact, CONCURRENCY)
        .await?;
    let wall = start.elapsed();

    let score: f64 = outcomes.iter().map(|o| o.score).sum::<f64>() / outcomes.len() as f64;
    let ideal_ms = (N as f64 / CONCURRENCY as f64) * LATENCY_MS;
    println!(
        "eval {N} x {LATENCY_MS}ms @ {CONCURRENCY} concurrency   {:10.1} ms wall (ideal {ideal_ms:.0} ms, overhead {:.1} ms, score {score})",
        wall.as_secs_f64() * 1e3,
        wall.as_secs_f64() * 1e3 - ideal_ms,
    );
    Ok(())
}
