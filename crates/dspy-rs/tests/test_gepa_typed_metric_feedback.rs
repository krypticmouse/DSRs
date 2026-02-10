use anyhow::Result;
use dspy_rs::__macro_support::bamltype::facet;
use dspy_rs::{
    CallMetadata, DynPredictor, Example, FeedbackMetric, GEPA, MetricOutcome, Module, Optimizer,
    Predict, PredictError, Predicted, Signature, TypedMetric,
};
use serde_json::json;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
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

struct FeedbackMetricImpl;

impl TypedMetric<InstructionEchoModule> for FeedbackMetricImpl {
    async fn evaluate(
        &self,
        _example: &Example,
        prediction: &Predicted<OptimizerSigOutput>,
    ) -> Result<MetricOutcome> {
        let score = prediction.answer.len() as f32;
        Ok(MetricOutcome::with_feedback(
            score,
            FeedbackMetric::new(score, format!("answer={}", prediction.answer)),
        ))
    }
}

struct ScoreOnlyMetric;

impl TypedMetric<InstructionEchoModule> for ScoreOnlyMetric {
    async fn evaluate(
        &self,
        _example: &Example,
        prediction: &Predicted<OptimizerSigOutput>,
    ) -> Result<MetricOutcome> {
        Ok(MetricOutcome::score(prediction.answer.len() as f32))
    }
}

struct PartialFeedbackMetric;

impl TypedMetric<InstructionEchoModule> for PartialFeedbackMetric {
    async fn evaluate(
        &self,
        example: &Example,
        prediction: &Predicted<OptimizerSigOutput>,
    ) -> Result<MetricOutcome> {
        let score = prediction.answer.len() as f32;
        let prompt = example
            .data
            .get("prompt")
            .and_then(|value| value.as_str())
            .unwrap_or_default();

        if prompt == "one" {
            Ok(MetricOutcome::with_feedback(
                score,
                FeedbackMetric::new(score, "only first example has feedback"),
            ))
        } else {
            Ok(MetricOutcome::score(score))
        }
    }
}

struct FeedbackThenScoreMetric {
    feedback_calls: usize,
    calls: AtomicUsize,
}

impl FeedbackThenScoreMetric {
    fn new(feedback_calls: usize) -> Self {
        Self {
            feedback_calls,
            calls: AtomicUsize::new(0),
        }
    }
}

impl TypedMetric<InstructionEchoModule> for FeedbackThenScoreMetric {
    async fn evaluate(
        &self,
        _example: &Example,
        prediction: &Predicted<OptimizerSigOutput>,
    ) -> Result<MetricOutcome> {
        let call_index = self.calls.fetch_add(1, Ordering::SeqCst);
        let score = prediction.answer.len() as f32;
        if call_index < self.feedback_calls {
            Ok(MetricOutcome::with_feedback(
                score,
                FeedbackMetric::new(score, format!("call={call_index}: feedback")),
            ))
        } else {
            Ok(MetricOutcome::score(score))
        }
    }
}

struct RecordingFeedbackMetric {
    seen_prompts: Arc<Mutex<Vec<String>>>,
}

impl TypedMetric<InstructionEchoModule> for RecordingFeedbackMetric {
    async fn evaluate(
        &self,
        example: &Example,
        prediction: &Predicted<OptimizerSigOutput>,
    ) -> Result<MetricOutcome> {
        let prompt = example
            .data
            .get("prompt")
            .and_then(|value| value.as_str())
            .unwrap_or_default()
            .to_string();
        self.seen_prompts
            .lock()
            .expect("metric lock should not be poisoned")
            .push(prompt.clone());

        let score = if prompt == "val-only" {
            prediction.answer.len() as f32 + 100.0
        } else {
            prediction.answer.len() as f32
        };
        Ok(MetricOutcome::with_feedback(
            score,
            FeedbackMetric::new(score, format!("prompt={prompt}")),
        ))
    }
}

fn trainset() -> Vec<Example> {
    vec![
        Example::new(
            HashMap::from([("prompt".to_string(), json!("one"))]),
            vec!["prompt".to_string()],
            vec![],
        ),
        Example::new(
            HashMap::from([("prompt".to_string(), json!("two"))]),
            vec!["prompt".to_string()],
            vec![],
        ),
    ]
}

fn trainset_with_invalid_input_keys() -> Vec<Example> {
    vec![Example::new(
        HashMap::from([
            ("prompt".to_string(), json!("one")),
            ("wrong_input".to_string(), json!("unused")),
        ]),
        vec!["wrong_input".to_string()],
        vec![],
    )]
}

fn valset_for_gepa() -> Vec<Example> {
    vec![Example::new(
        HashMap::from([("prompt".to_string(), json!("val-only"))]),
        vec!["prompt".to_string()],
        vec![],
    )]
}

#[tokio::test]
async fn gepa_compile_succeeds_when_feedback_present() {
    let metric = FeedbackMetricImpl;
    let mut module = InstructionEchoModule {
        predictor: Predict::<OptimizerSig>::builder().instruction("seed").build(),
    };

    let optimizer = GEPA::builder()
        .num_iterations(2)
        .minibatch_size(2)
        .track_stats(true)
        .build();

    let result = optimizer
        .compile(&mut module, trainset(), &metric)
        .await
        .expect("GEPA compile should succeed when feedback is present");

    assert!(result.total_rollouts > 0);
    assert_eq!(result.best_candidate.module_name, "predictor");
}

#[tokio::test]
async fn gepa_compile_fails_without_feedback() {
    let metric = ScoreOnlyMetric;
    let mut module = InstructionEchoModule {
        predictor: Predict::<OptimizerSig>::builder().instruction("seed").build(),
    };

    let optimizer = GEPA::builder()
        .num_iterations(1)
        .minibatch_size(2)
        .build();

    let err = optimizer
        .compile(&mut module, trainset(), &metric)
        .await
        .expect_err("GEPA should reject score-only metrics");

    assert!(err
        .to_string()
        .contains("GEPA requires feedback for every evaluated example"));
}

#[tokio::test]
async fn gepa_compile_fails_when_feedback_is_partial() {
    let metric = PartialFeedbackMetric;
    let mut module = InstructionEchoModule {
        predictor: Predict::<OptimizerSig>::builder().instruction("seed").build(),
    };

    let optimizer = GEPA::builder()
        .num_iterations(1)
        .minibatch_size(2)
        .build();

    let err = optimizer
        .compile(&mut module, trainset(), &metric)
        .await
        .expect_err("GEPA should reject partially-populated feedback outcomes");

    let message = err.to_string();
    assert!(message.contains("GEPA requires feedback for every evaluated example"));
    assert!(message.contains("module=`predictor`"));
}

#[tokio::test]
async fn gepa_compile_respects_example_input_keys_for_typed_conversion() {
    let metric = FeedbackMetricImpl;
    let mut module = InstructionEchoModule {
        predictor: Predict::<OptimizerSig>::builder().instruction("seed").build(),
    };

    let optimizer = GEPA::builder()
        .num_iterations(1)
        .minibatch_size(1)
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

#[tokio::test]
async fn gepa_compile_fails_when_feedback_disappears_during_generation() {
    // Trainset has two examples and one predictor:
    // calls 0-1: initial frontier seeding
    // calls 2-3: parent minibatch in generation 0
    // call 4+: child eval in generation 1 should fail GEPA feedback gate.
    let metric = FeedbackThenScoreMetric::new(4);
    let mut module = InstructionEchoModule {
        predictor: Predict::<OptimizerSig>::builder().instruction("seed").build(),
    };

    let optimizer = GEPA::builder()
        .num_iterations(1)
        .minibatch_size(2)
        .track_stats(true)
        .build();

    let err = optimizer
        .compile(&mut module, trainset(), &metric)
        .await
        .expect_err("GEPA should fail once feedback becomes unavailable mid-loop");

    let message = err.to_string();
    assert!(message.contains("GEPA requires feedback for every evaluated example"));
    assert!(message.contains("generation=1"), "expected generation marker: {message}");
}

#[tokio::test]
async fn gepa_compile_uses_valset_and_tracks_best_outputs_when_enabled() {
    let seen_prompts = Arc::new(Mutex::new(Vec::new()));
    let metric = RecordingFeedbackMetric {
        seen_prompts: Arc::clone(&seen_prompts),
    };
    let mut module = InstructionEchoModule {
        predictor: Predict::<OptimizerSig>::builder().instruction("seed").build(),
    };
    let valset = valset_for_gepa();

    let optimizer = GEPA::builder()
        .num_iterations(0)
        .minibatch_size(1)
        .track_best_outputs(true)
        .valset(valset.clone())
        .build();

    let result = optimizer
        .compile(&mut module, trainset(), &metric)
        .await
        .expect("GEPA compile should succeed with a dedicated valset");

    let seen = seen_prompts
        .lock()
        .expect("metric lock should not be poisoned")
        .clone();
    assert_eq!(seen, vec!["val-only".to_string()]);
    assert_eq!(result.highest_score_achieved_per_val_task.len(), valset.len());
    assert_eq!(
        result
            .best_outputs_valset
            .as_ref()
            .expect("best outputs should be captured when tracking is enabled")
            .len(),
        valset.len()
    );
}
