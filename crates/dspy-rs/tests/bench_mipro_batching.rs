//! Controlled latency workload for MIPRO trial batching.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use anyhow::Result;
use dspy_rs::{
    CallMetadata, Eval, LM, LMClient, LMConfig, MIPROv2, Module, Predict, PredictError, Predicted,
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
    first: Predict<BenchSig>,
    second: Predict<BenchSig>,
    calls: Arc<AtomicUsize>,
}

dspy_rs::predictors!(SlowModule { first, second });

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

struct CandidateHashMetric;

impl TypedMetric<(BenchSigInput, BenchSigOutput), SlowModule> for CandidateHashMetric {
    async fn evaluate(
        &self,
        _: &(BenchSigInput, BenchSigOutput),
        _: &Predicted<BenchSigOutput>,
        trace: Option<&dspy_rs::Trace>,
    ) -> Result<Eval> {
        let hash = trace
            .and_then(|trace| trace.meta.candidate_hash)
            .expect("optimizer rollouts record the candidate hash");
        Ok(Eval::score((hash % 1_000_003) as f64 / 1_000_003.0))
    }
}

async fn run(batch_size: usize, examples: usize) -> Result<(u128, usize, (String, String))> {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut module = SlowModule {
        first: Predict::builder().instruction("seed-first").build(),
        second: Predict::builder().instruction("seed-second").build(),
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
    let proposals = (0..2).map(|_| {
        AssistantContent::text(
            "[[ ## instructions ## ]]\n[\"one\",\"two\",\"three\",\"four\"]\n[[ ## completed ## ]]",
        )
    });
    let lm = LM::from_config(LMConfig {
        base_url: Some("http://localhost/v1".into()),
        ..LMConfig::default()
    })
    .await?
    .with_client(LMClient::Test(TestCompletionModel::new(proposals)))
    .await?;
    let optimizer = MIPROv2::builder()
        .num_candidates(5)
        .num_trials(20)
        .full_eval_steps(5)
        .minibatch_size(examples)
        .seed(42)
        .eval_concurrency(8)
        .candidate_batch_size(batch_size)
        .prompt_model(lm)
        .build();
    let start = Instant::now();
    optimizer
        .compile_module(&mut module, &trainset, &CandidateHashMetric)
        .await?;
    Ok((
        start.elapsed().as_nanos(),
        calls.load(Ordering::Relaxed),
        (
            PredictorInfo::instruction(&module.first).to_string(),
            PredictorInfo::instruction(&module.second).to_string(),
        ),
    ))
}

#[tokio::test]
async fn batching_keeps_rollout_count_and_winner() -> Result<()> {
    for examples in [1, 8] {
        let sequential = run(1, examples).await?;
        let batched = run(5, examples).await?;
        assert_eq!((sequential.1, sequential.2), (batched.1, batched.2));
    }
    Ok(())
}

#[tokio::test]
#[ignore = "controlled performance run"]
async fn mipro_trial_batch_latency() -> Result<()> {
    for examples in [1, 8] {
        let reference = run(1, examples).await?;
        let batched = run(5, examples).await?;
        assert_eq!((reference.1, &reference.2), (batched.1, &batched.2));
        for (sample, batch_size) in [1, 5, 5, 1, 1, 5, 5, 1].into_iter().enumerate() {
            let (elapsed_ns, calls, instructions) = run(batch_size, examples).await?;
            assert_eq!((calls, &instructions), (reference.1, &reference.2));
            println!(
                "sample={sample} examples={examples} batch={batch_size} elapsed_ns={elapsed_ns} calls={calls}"
            );
        }
    }
    Ok(())
}
