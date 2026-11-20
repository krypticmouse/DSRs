use anyhow::{Result, anyhow};
use dspy_rs::{
    CallMetadata, Eval, Module, PredictError, Predicted, Signature, TypedMetric, average_score,
    evaluate_trainset, evaluate_trainset_with_concurrency,
};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::Notify;

#[derive(Signature, Clone, Debug)]
struct EvalSig {
    #[input]
    prompt: String,

    #[output]
    answer: String,
}

struct EchoModule;

impl Module for EchoModule {
    type Input = EvalSigInput;
    type Output = EvalSigOutput;

    async fn forward(&self, input: EvalSigInput) -> Result<Predicted<EvalSigOutput>, PredictError> {
        Ok(Predicted::new(
            EvalSigOutput {
                answer: input.prompt,
            },
            CallMetadata::default(),
        ))
    }
}

struct RecordingMetric {
    seen_answers: Arc<Mutex<Vec<String>>>,
}

impl TypedMetric<(EvalSigInput, EvalSigOutput), EchoModule> for RecordingMetric {
    async fn evaluate(
        &self,
        example: &(EvalSigInput, EvalSigOutput),
        prediction: &Predicted<<EchoModule as Module>::Output>,
        _trace: Option<&dspy_rs::Trace>,
    ) -> Result<Eval> {
        self.seen_answers
            .lock()
            .expect("metric lock should not be poisoned")
            .push(prediction.answer.clone());

        let score = (prediction.answer == example.1.answer) as u8 as f64;
        Ok(Eval::score(score))
    }
}

struct FailingMetric;

impl TypedMetric<(EvalSigInput, EvalSigOutput), EchoModule> for FailingMetric {
    async fn evaluate(
        &self,
        _example: &(EvalSigInput, EvalSigOutput),
        _prediction: &Predicted<<EchoModule as Module>::Output>,
        _trace: Option<&dspy_rs::Trace>,
    ) -> Result<Eval> {
        Err(anyhow!("typed metric failure"))
    }
}

fn trainset() -> Vec<(EvalSigInput, EvalSigOutput)> {
    vec![
        (
            EvalSigInput {
                prompt: "one".to_string(),
            },
            EvalSigOutput {
                answer: "one".to_string(),
            },
        ),
        (
            EvalSigInput {
                prompt: "two".to_string(),
            },
            EvalSigOutput {
                answer: "two".to_string(),
            },
        ),
    ]
}

#[tokio::test]
async fn evaluate_trainset_runs_typed_rows_and_metric() {
    let seen_answers = Arc::new(Mutex::new(Vec::new()));
    let metric = RecordingMetric {
        seen_answers: Arc::clone(&seen_answers),
    };

    let outcomes = evaluate_trainset(&EchoModule, &trainset(), &metric)
        .await
        .expect("typed evaluate_trainset should succeed");

    assert_eq!(outcomes.len(), 2);
    assert_eq!(average_score(&outcomes), 1.0);

    let seen = seen_answers
        .lock()
        .expect("metric lock should not be poisoned");
    assert_eq!(seen.as_slice(), ["one", "two"]);
}

#[tokio::test]
async fn evaluate_trainset_propagates_typed_metric_errors() {
    let err = evaluate_trainset(&EchoModule, &trainset(), &FailingMetric)
        .await
        .expect_err("typed metric errors should propagate");

    assert!(err.to_string().contains("typed metric failure"));
}

struct GatedMetric {
    release_on: &'static str,
    release: Notify,
    fail: bool,
}

impl TypedMetric<(EvalSigInput, EvalSigOutput), EchoModule> for GatedMetric {
    async fn evaluate(
        &self,
        _example: &(EvalSigInput, EvalSigOutput),
        prediction: &Predicted<EvalSigOutput>,
        trace: Option<&dspy_rs::Trace>,
    ) -> Result<Eval> {
        assert!(trace.is_some());
        if prediction.answer == "one" {
            self.release.notified().await;
        }
        if prediction.answer == self.release_on {
            self.release.notify_one();
        }
        if self.fail {
            assert_ne!(prediction.answer, "three", "a failure stops further admission");
            anyhow::bail!("{}", prediction.answer);
        }
        Ok(Eval::with_feedback(1.0, prediction.answer.clone()))
    }
}

fn three_examples() -> Vec<(EvalSigInput, EvalSigOutput)> {
    let mut rows = trainset();
    rows.push((
        EvalSigInput {
            prompt: "three".into(),
        },
        EvalSigOutput {
            answer: "three".into(),
        },
    ));
    rows
}

#[tokio::test]
async fn evaluator_refills_slots_and_preserves_trainset_order() {
    let metric = GatedMetric {
        release_on: "three",
        release: Notify::new(),
        fail: false,
    };
    let evals = tokio::time::timeout(
        Duration::from_secs(5),
        evaluate_trainset_with_concurrency(&EchoModule, &three_examples(), &metric, 2),
    )
    .await
    .expect("the third example must start while the first is still waiting")
    .unwrap();
    let feedback: Vec<_> = evals.iter().map(|eval| eval.feedback.as_deref()).collect();
    assert_eq!(feedback, [Some("one"), Some("two"), Some("three")]);
}

#[tokio::test]
async fn evaluator_returns_the_earliest_failed_row() {
    let metric = GatedMetric {
        release_on: "two",
        release: Notify::new(),
        fail: true,
    };
    let error = evaluate_trainset_with_concurrency(&EchoModule, &three_examples(), &metric, 2)
        .await
        .unwrap_err();
    assert_eq!(error.to_string(), "one");
}
