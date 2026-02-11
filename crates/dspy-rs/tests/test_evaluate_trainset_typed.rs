use anyhow::{Result, anyhow};
use dspy_rs::{
    CallMetadata, Example, MetricOutcome, Module, PredictError, Predicted, Signature, TypedMetric,
    average_score, evaluate_trainset,
};
use std::sync::{Arc, Mutex};

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

impl TypedMetric<EvalSig, EchoModule> for RecordingMetric {
    async fn evaluate(
        &self,
        example: &Example<EvalSig>,
        prediction: &Predicted<<EchoModule as Module>::Output>,
    ) -> Result<MetricOutcome> {
        self.seen_answers
            .lock()
            .expect("metric lock should not be poisoned")
            .push(prediction.answer.clone());

        let score = (prediction.answer == example.output.answer) as u8 as f32;
        Ok(MetricOutcome::score(score))
    }
}

struct FailingMetric;

impl TypedMetric<EvalSig, EchoModule> for FailingMetric {
    async fn evaluate(
        &self,
        _example: &Example<EvalSig>,
        _prediction: &Predicted<<EchoModule as Module>::Output>,
    ) -> Result<MetricOutcome> {
        Err(anyhow!("typed metric failure"))
    }
}

fn trainset() -> Vec<Example<EvalSig>> {
    vec![
        Example::new(
            EvalSigInput {
                prompt: "one".to_string(),
            },
            EvalSigOutput {
                answer: "one".to_string(),
            },
        ),
        Example::new(
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

    let outcomes = evaluate_trainset::<EvalSig, _, _>(&EchoModule, &trainset(), &metric)
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
    let err = evaluate_trainset::<EvalSig, _, _>(&EchoModule, &trainset(), &FailingMetric)
        .await
        .expect_err("typed metric errors should propagate");

    assert!(err.to_string().contains("typed metric failure"));
}
