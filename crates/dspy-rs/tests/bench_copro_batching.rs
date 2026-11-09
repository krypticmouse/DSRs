//! Controlled latency workload for COPRO candidate batching.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use anyhow::Result;
use dspy_rs::{
    COPRO, CallMetadata, Eval, LM, LMClient, LMConfig, Module, Predict, PredictError, Predicted,
    PredictorInfo, Signature, TestCompletionModel, TypedMetric,
};
use rig::completion::AssistantContent;

#[derive(Signature, Clone, Debug)]
struct BenchSig {
    #[input]
    question: String,
    #[output]
    answer: String,
}

struct SlowModule {
    predictor: Predict<BenchSig>,
    calls: Arc<AtomicUsize>,
}

dspy_rs::predictors!(SlowModule { predictor });

impl Module for SlowModule {
    type Input = BenchSigInput;
    type Output = BenchSigOutput;

    async fn forward(&self, input: BenchSigInput) -> Result<Predicted<Self::Output>, PredictError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        tokio::time::sleep(Duration::from_millis(25)).await;
        Ok(Predicted::new(
            BenchSigOutput {
                answer: input.question,
            },
            CallMetadata::default(),
        ))
    }
}

struct ConstantMetric;

impl TypedMetric<(BenchSigInput, BenchSigOutput), SlowModule> for ConstantMetric {
    async fn evaluate(
        &self,
        _: &(BenchSigInput, BenchSigOutput),
        _: &Predicted<BenchSigOutput>,
        _: Option<&dspy_rs::Trace>,
    ) -> Result<Eval> {
        Ok(Eval::score(1.0))
    }
}

async fn run(batch_size: usize, examples: usize) -> Result<(u128, usize, String)> {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut module = SlowModule {
        predictor: Predict::builder().instruction("seed").build(),
        calls: Arc::clone(&calls),
    };
    let trainset = (0..examples)
        .map(|idx| {
            let question = idx.to_string();
            (
                BenchSigInput {
                    question: question.clone(),
                },
                BenchSigOutput { answer: question },
            )
        })
        .collect::<Vec<_>>();
    let proposal = TestCompletionModel::new([AssistantContent::text(
        "[[ ## instructions ## ]]\n[\"one\",\"two\",\"three\",\"four\"]\n[[ ## completed ## ]]",
    )]);
    let lm = LM::from_config(LMConfig {
        base_url: Some("http://localhost/v1".into()),
        ..LMConfig::default()
    })
    .await?
    .with_client(LMClient::Test(proposal))
    .await?;
    let optimizer = COPRO::builder()
        .breadth(5)
        .depth(1)
        .eval_concurrency(8)
        .candidate_batch_size(batch_size)
        .prompt_model(lm)
        .build();
    let start = Instant::now();
    optimizer
        .compile_module(&mut module, &trainset, &ConstantMetric)
        .await?;
    Ok((
        start.elapsed().as_nanos(),
        calls.load(Ordering::Relaxed),
        PredictorInfo::instruction(&module.predictor).to_string(),
    ))
}

#[tokio::test]
async fn batching_keeps_rollout_count_and_winner() -> Result<()> {
    let sequential = run(1, 1).await?;
    let batched = run(5, 1).await?;
    assert_eq!((sequential.1, sequential.2), (batched.1, batched.2));
    Ok(())
}

#[tokio::test]
#[ignore = "controlled performance run"]
async fn copro_candidate_batch_latency() -> Result<()> {
    for examples in [1, 8] {
        let _ = run(1, examples).await?;
        let _ = run(5, examples).await?;
        for (sample, batch_size) in [1, 5, 5, 1, 1, 5, 5, 1].into_iter().enumerate() {
            let (elapsed_ns, calls, instruction) = run(batch_size, examples).await?;
            assert_eq!(calls, 5 * examples);
            assert_eq!(instruction, "seed");
            println!(
                "sample={sample} examples={examples} batch={batch_size} elapsed_ns={elapsed_ns} calls={calls}"
            );
        }
    }
    Ok(())
}
