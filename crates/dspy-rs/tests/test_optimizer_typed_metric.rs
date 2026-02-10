use anyhow::Result;
use dspy_rs::__macro_support::bamltype::facet;
use dspy_rs::{
    COPRO, CallMetadata, DynPredictor, Example, MIPROv2, MetricOutcome, Module, Optimizer,
    Predict, PredictError, Predicted, Signature, TypedMetric,
};
use serde_json::json;
use std::collections::HashMap;
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
        _input: OptimizerSigInput,
    ) -> Result<Predicted<OptimizerSigOutput>, PredictError> {
        let answer = <Predict<OptimizerSig> as DynPredictor>::instruction(&self.predictor);
        Ok(Predicted::new(
            OptimizerSigOutput { answer },
            CallMetadata::default(),
        ))
    }
}

struct RecordingMetric {
    seen_answers: Arc<Mutex<Vec<String>>>,
}

impl TypedMetric<InstructionEchoModule> for RecordingMetric {
    async fn evaluate(
        &self,
        _example: &Example,
        prediction: &Predicted<OptimizerSigOutput>,
    ) -> Result<MetricOutcome> {
        self.seen_answers
            .lock()
            .expect("metric lock should not be poisoned")
            .push(prediction.answer.clone());

        Ok(MetricOutcome::score(prediction.answer.len() as f32))
    }
}

fn trainset() -> Vec<Example> {
    vec![
        Example::new(
            HashMap::from([
                ("prompt".to_string(), json!("one")),
                ("answer".to_string(), json!("seed")),
            ]),
            vec!["prompt".to_string()],
            vec!["answer".to_string()],
        ),
        Example::new(
            HashMap::from([
                ("prompt".to_string(), json!("two")),
                ("answer".to_string(), json!("seed")),
            ]),
            vec!["prompt".to_string()],
            vec!["answer".to_string()],
        ),
    ]
}

fn trainset_with_invalid_input_keys() -> Vec<Example> {
    vec![Example::new(
        HashMap::from([
            ("prompt".to_string(), json!("one")),
            ("wrong_input".to_string(), json!("unused")),
            ("answer".to_string(), json!("seed")),
        ]),
        vec!["wrong_input".to_string()],
        vec!["answer".to_string()],
    )]
}

#[tokio::test]
async fn copro_compile_uses_typed_metric_predictions() {
    let seen_answers = Arc::new(Mutex::new(Vec::new()));
    let metric = RecordingMetric {
        seen_answers: Arc::clone(&seen_answers),
    };

    let mut module = InstructionEchoModule {
        predictor: Predict::<OptimizerSig>::builder().instruction("seed").build(),
    };

    let optimizer = COPRO::builder().breadth(3).depth(1).build();
    optimizer
        .compile(&mut module, trainset(), &metric)
        .await
        .expect("COPRO compile should succeed on typed metric");

    let seen = seen_answers
        .lock()
        .expect("metric lock should not be poisoned");
    assert!(!seen.is_empty(), "metric should receive typed predictions");
    assert!(seen.iter().all(|answer| !answer.is_empty()));
}

#[tokio::test]
async fn mipro_compile_uses_typed_metric_predictions() {
    let seen_answers = Arc::new(Mutex::new(Vec::new()));
    let metric = RecordingMetric {
        seen_answers: Arc::clone(&seen_answers),
    };

    let mut module = InstructionEchoModule {
        predictor: Predict::<OptimizerSig>::builder().instruction("seed").build(),
    };

    let optimizer = MIPROv2::builder()
        .num_candidates(4)
        .num_trials(2)
        .minibatch_size(2)
        .build();

    optimizer
        .compile(&mut module, trainset(), &metric)
        .await
        .expect("MIPRO compile should succeed on typed metric");

    let seen = seen_answers
        .lock()
        .expect("metric lock should not be poisoned");
    assert!(!seen.is_empty(), "metric should receive typed predictions");
    assert!(seen.iter().all(|answer| !answer.is_empty()));
}

#[tokio::test]
async fn copro_compile_respects_example_input_keys_for_typed_conversion() {
    let metric = RecordingMetric {
        seen_answers: Arc::new(Mutex::new(Vec::new())),
    };
    let mut module = InstructionEchoModule {
        predictor: Predict::<OptimizerSig>::builder().instruction("seed").build(),
    };

    let optimizer = COPRO::builder().breadth(3).depth(1).build();
    let err = optimizer
        .compile(&mut module, trainset_with_invalid_input_keys(), &metric)
        .await
        .expect_err("compile should fail when input_keys omits required typed fields");

    assert!(
        err.to_string().contains("prompt"),
        "error should mention missing required field: {err}"
    );
}

#[tokio::test]
async fn mipro_compile_respects_example_input_keys_for_typed_conversion() {
    let metric = RecordingMetric {
        seen_answers: Arc::new(Mutex::new(Vec::new())),
    };
    let mut module = InstructionEchoModule {
        predictor: Predict::<OptimizerSig>::builder().instruction("seed").build(),
    };

    let optimizer = MIPROv2::builder()
        .num_candidates(4)
        .num_trials(2)
        .minibatch_size(2)
        .build();

    let err = optimizer
        .compile(&mut module, trainset_with_invalid_input_keys(), &metric)
        .await
        .expect_err("compile should fail when input_keys omits required typed fields");

    assert!(
        err.to_string().contains("prompt"),
        "error should mention missing required field: {err}"
    );
}
