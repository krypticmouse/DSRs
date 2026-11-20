use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use anyhow::Result;
use dspy_rs::{
    CallMetadata, Eval, Module, PredictError, Predicted, Signature, Trace, TypedMetric,
    evaluate_trainset_with_concurrency, trace::SpanId,
};

#[derive(Signature, Clone, Debug)]
struct Probe {
    #[input]
    index: u32,
    #[output]
    number: u32,
    #[output]
    payload: Vec<String>,
}

struct LocalModule(Vec<String>);

impl Module for LocalModule {
    type Input = ProbeInput;
    type Output = ProbeOutput;

    async fn forward(&self, input: ProbeInput) -> Result<Predicted<ProbeOutput>, PredictError> {
        Ok(Predicted::new(
            ProbeOutput {
                number: input.index,
                payload: self.0.clone(),
            },
            CallMetadata::default(),
        ))
    }
}

type Row = (ProbeInput, ProbeOutput);

struct Metric {
    payload_bytes: usize,
    span_calls: AtomicUsize,
}

impl TypedMetric<Row, LocalModule> for Metric {
    async fn evaluate(
        &self,
        example: &Row,
        output: &Predicted<ProbeOutput>,
        trace: Option<&Trace>,
    ) -> Result<Eval> {
        assert!(trace.expect("metric receives a trace").outcome.is_none());
        assert_eq!(example.1.number, output.number);
        assert_eq!(
            output.payload.iter().map(String::len).sum::<usize>(),
            self.payload_bytes
        );
        Ok(Eval::score(score(example.0.index as usize)))
    }

    async fn evaluate_spans(
        &self,
        example: &Row,
        output: &Predicted<ProbeOutput>,
        trace: &Trace,
    ) -> Result<Vec<(SpanId, Eval)>> {
        assert!(trace.outcome.is_none());
        assert_eq!(example.0.index, output.number);
        self.span_calls.fetch_add(1, Ordering::Relaxed);
        Ok(Vec::new())
    }
}

fn score(index: usize) -> f64 {
    (index % 16) as f64 / 16.0
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    anyhow::ensure!(
        std::env::args().nth(1).as_deref() == Some("score-only"),
        "expected score-only workload"
    );
    const ROWS: usize = 512;
    const REPEATS: usize = 8;
    const CONCURRENCY: usize = 8;

    let data: Vec<Row> = (0..ROWS as u32)
        .map(|index| {
            (
                ProbeInput { index },
                ProbeOutput {
                    number: index,
                    payload: Vec::new(),
                },
            )
        })
        .collect();

    for payload_bytes in [64, 4096, 65536] {
        let module = LocalModule(vec!["x".repeat(payload_bytes / 8); 8]);
        let metric = Metric {
            payload_bytes,
            span_calls: AtomicUsize::new(0),
        };
        let mut elapsed = Duration::ZERO;
        for _ in 0..REPEATS {
            let started = Instant::now();
            let results =
                evaluate_trainset_with_concurrency(&module, &data, &metric, CONCURRENCY).await?;
            elapsed += started.elapsed();
            assert_eq!(results.len(), ROWS);
            for (index, item) in results.iter().enumerate() {
                assert_eq!(item.score, score(index));
            }
        }
        assert_eq!(metric.span_calls.load(Ordering::Relaxed), ROWS * REPEATS);
        println!("score-only,{payload_bytes},{}", elapsed.as_micros());
    }
    Ok(())
}
