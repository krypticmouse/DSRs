use anyhow::{Result, anyhow};
use dspy_rs::{
    COPRO, CallMetadata, Example, MIPROv2, MetricOutcome, Module, Optimizer, Predict, PredictError,
    Predicted, Signature, TypedMetric,
};
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

#[derive(Signature, Clone, Debug)]
struct OptimizerSig {
    #[input]
    prompt: String,

    #[output]
    answer: String,
}

#[derive(facet::Facet)]
#[facet(crate = facet)]
struct InstructionEchoModule {
    predictor: Predict<OptimizerSig>,
}

impl Module for InstructionEchoModule {
    type Input = OptimizerSigInput;
    type Output = OptimizerSigOutput;

    async fn forward(
        &self,
        input: OptimizerSigInput,
    ) -> Result<Predicted<OptimizerSigOutput>, PredictError> {
        let _ = &self.predictor;
        Ok(Predicted::new(
            OptimizerSigOutput {
                answer: input.prompt,
            },
            CallMetadata::default(),
        ))
    }
}

struct RecordingMetric {
    seen_answers: Arc<Mutex<Vec<String>>>,
}

impl TypedMetric<OptimizerSig, InstructionEchoModule> for RecordingMetric {
    async fn evaluate(
        &self,
        example: &Example<OptimizerSig>,
        prediction: &Predicted<OptimizerSigOutput>,
    ) -> Result<MetricOutcome> {
        self.seen_answers
            .lock()
            .expect("metric lock should not be poisoned")
            .push(prediction.answer.clone());

        let score = (prediction.answer == example.input.prompt) as u8 as f32;
        Ok(MetricOutcome::score(score))
    }
}

struct FailingMetric;

impl TypedMetric<OptimizerSig, InstructionEchoModule> for FailingMetric {
    async fn evaluate(
        &self,
        _example: &Example<OptimizerSig>,
        _prediction: &Predicted<OptimizerSigOutput>,
    ) -> Result<MetricOutcome> {
        Err(anyhow!("metric failure"))
    }
}

fn trainset() -> Vec<Example<OptimizerSig>> {
    vec![
        Example::new(
            OptimizerSigInput {
                prompt: "one".to_string(),
            },
            OptimizerSigOutput {
                answer: "one".to_string(),
            },
        ),
        Example::new(
            OptimizerSigInput {
                prompt: "two".to_string(),
            },
            OptimizerSigOutput {
                answer: "two".to_string(),
            },
        ),
    ]
}

#[tokio::test]
async fn copro_compile_uses_typed_metric_predictions() {
    let seen_answers = Arc::new(Mutex::new(Vec::new()));
    let metric = RecordingMetric {
        seen_answers: Arc::clone(&seen_answers),
    };

    let mut module = InstructionEchoModule {
        predictor: Predict::<OptimizerSig>::builder()
            .instruction("seed")
            .build(),
    };

    let optimizer = COPRO::builder().breadth(3).depth(1).build();
    optimizer
        .compile::<OptimizerSig, _, _>(&mut module, trainset(), &metric)
        .await
        .expect("COPRO compile should succeed on typed metric");

    let seen = seen_answers
        .lock()
        .expect("metric lock should not be poisoned");
    assert!(!seen.is_empty(), "metric should receive typed predictions");
    let expected_prompts = HashSet::from(["one".to_string(), "two".to_string()]);
    assert!(seen.iter().all(|answer| expected_prompts.contains(answer)));
    assert!(seen.iter().any(|answer| answer == "one"));
    assert!(seen.iter().any(|answer| answer == "two"));
}

#[tokio::test]
async fn mipro_compile_uses_typed_metric_predictions() {
    let seen_answers = Arc::new(Mutex::new(Vec::new()));
    let metric = RecordingMetric {
        seen_answers: Arc::clone(&seen_answers),
    };

    let mut module = InstructionEchoModule {
        predictor: Predict::<OptimizerSig>::builder()
            .instruction("seed")
            .build(),
    };

    let optimizer = MIPROv2::builder()
        .num_candidates(4)
        .num_trials(2)
        .minibatch_size(2)
        .build();

    optimizer
        .compile::<OptimizerSig, _, _>(&mut module, trainset(), &metric)
        .await
        .expect("MIPRO compile should succeed on typed metric");

    let seen = seen_answers
        .lock()
        .expect("metric lock should not be poisoned");
    assert!(!seen.is_empty(), "metric should receive typed predictions");
    let expected_prompts = HashSet::from(["one".to_string(), "two".to_string()]);
    assert!(seen.iter().all(|answer| expected_prompts.contains(answer)));
    assert!(seen.iter().any(|answer| answer == "one"));
    assert!(seen.iter().any(|answer| answer == "two"));
}

#[tokio::test]
async fn copro_compile_propagates_metric_errors() {
    let mut module = InstructionEchoModule {
        predictor: Predict::<OptimizerSig>::builder()
            .instruction("seed")
            .build(),
    };
    let optimizer = COPRO::builder().breadth(3).depth(1).build();

    let err = optimizer
        .compile::<OptimizerSig, _, _>(&mut module, trainset(), &FailingMetric)
        .await
        .expect_err("COPRO should propagate typed metric errors");

    assert!(err.to_string().contains("metric failure"));
}

#[tokio::test]
async fn mipro_compile_propagates_metric_errors() {
    let mut module = InstructionEchoModule {
        predictor: Predict::<OptimizerSig>::builder()
            .instruction("seed")
            .build(),
    };
    let optimizer = MIPROv2::builder()
        .num_candidates(4)
        .num_trials(2)
        .minibatch_size(2)
        .build();

    let err = optimizer
        .compile::<OptimizerSig, _, _>(&mut module, trainset(), &FailingMetric)
        .await
        .expect_err("MIPRO should propagate typed metric errors");

    assert!(err.to_string().contains("metric failure"));
}
