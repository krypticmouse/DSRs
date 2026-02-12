use anyhow::Result;
use dspy_rs::{
    CallMetadata, Example, FeedbackMetric, GEPA, MetricOutcome, Module, Optimizer, Predict,
    PredictError, Predicted, Signature, TypedMetric,
};
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

struct FeedbackMetricImpl;

impl TypedMetric<OptimizerSig, InstructionEchoModule> for FeedbackMetricImpl {
    async fn evaluate(
        &self,
        _example: &Example<OptimizerSig>,
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

impl TypedMetric<OptimizerSig, InstructionEchoModule> for ScoreOnlyMetric {
    async fn evaluate(
        &self,
        _example: &Example<OptimizerSig>,
        prediction: &Predicted<OptimizerSigOutput>,
    ) -> Result<MetricOutcome> {
        Ok(MetricOutcome::score(prediction.answer.len() as f32))
    }
}

struct PartialFeedbackMetric;

impl TypedMetric<OptimizerSig, InstructionEchoModule> for PartialFeedbackMetric {
    async fn evaluate(
        &self,
        example: &Example<OptimizerSig>,
        prediction: &Predicted<OptimizerSigOutput>,
    ) -> Result<MetricOutcome> {
        let score = prediction.answer.len() as f32;

        if example.input.prompt == "one" {
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

impl TypedMetric<OptimizerSig, InstructionEchoModule> for FeedbackThenScoreMetric {
    async fn evaluate(
        &self,
        _example: &Example<OptimizerSig>,
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

impl TypedMetric<OptimizerSig, InstructionEchoModule> for RecordingFeedbackMetric {
    async fn evaluate(
        &self,
        example: &Example<OptimizerSig>,
        prediction: &Predicted<OptimizerSigOutput>,
    ) -> Result<MetricOutcome> {
        let prompt = example.input.prompt.clone();
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

fn valset_for_gepa() -> Vec<Example<OptimizerSig>> {
    vec![Example::new(
        OptimizerSigInput {
            prompt: "val-only".to_string(),
        },
        OptimizerSigOutput {
            answer: "val-only".to_string(),
        },
    )]
}

#[tokio::test]
async fn gepa_compile_succeeds_when_feedback_present() {
    let metric = FeedbackMetricImpl;
    let mut module = InstructionEchoModule {
        predictor: Predict::<OptimizerSig>::builder()
            .instruction("seed")
            .build(),
    };

    let optimizer = GEPA::builder()
        .num_iterations(2)
        .minibatch_size(2)
        .track_stats(true)
        .build();

    let result = optimizer
        .compile::<OptimizerSig, _, _>(&mut module, trainset(), &metric)
        .await
        .expect("GEPA compile should succeed when feedback is present");

    assert!(result.total_rollouts > 0);
    assert_eq!(result.best_candidate.module_name, "predictor");
}

#[tokio::test]
async fn gepa_compile_fails_without_feedback() {
    let metric = ScoreOnlyMetric;
    let mut module = InstructionEchoModule {
        predictor: Predict::<OptimizerSig>::builder()
            .instruction("seed")
            .build(),
    };

    let optimizer = GEPA::builder().num_iterations(1).minibatch_size(2).build();

    let err = optimizer
        .compile::<OptimizerSig, _, _>(&mut module, trainset(), &metric)
        .await
        .expect_err("GEPA should reject score-only metrics");

    assert!(
        err.to_string()
            .contains("GEPA requires feedback for every evaluated example")
    );
}

#[tokio::test]
async fn gepa_compile_fails_when_feedback_is_partial() {
    let metric = PartialFeedbackMetric;
    let mut module = InstructionEchoModule {
        predictor: Predict::<OptimizerSig>::builder()
            .instruction("seed")
            .build(),
    };

    let optimizer = GEPA::builder().num_iterations(1).minibatch_size(2).build();

    let err = optimizer
        .compile::<OptimizerSig, _, _>(&mut module, trainset(), &metric)
        .await
        .expect_err("GEPA should reject partially-populated feedback outcomes");

    let message = err.to_string();
    assert!(message.contains("GEPA requires feedback for every evaluated example"));
    assert!(message.contains("module=`predictor`"));
}

#[tokio::test]
async fn gepa_compile_fails_when_feedback_disappears_during_generation() {
    // Trainset has two examples and one predictor:
    // calls 0-1: initial frontier seeding
    // calls 2-3: parent minibatch in generation 0
    // call 4+: child eval in generation 1 should fail GEPA feedback gate.
    let metric = FeedbackThenScoreMetric::new(4);
    let mut module = InstructionEchoModule {
        predictor: Predict::<OptimizerSig>::builder()
            .instruction("seed")
            .build(),
    };

    let optimizer = GEPA::builder()
        .num_iterations(1)
        .minibatch_size(2)
        .track_stats(true)
        .build();

    let err = optimizer
        .compile::<OptimizerSig, _, _>(&mut module, trainset(), &metric)
        .await
        .expect_err("GEPA should fail once feedback becomes unavailable mid-loop");

    let message = err.to_string();
    assert!(message.contains("GEPA requires feedback for every evaluated example"));
    assert!(
        message.contains("generation=1"),
        "expected generation marker: {message}"
    );
}

#[tokio::test]
async fn gepa_compile_with_valset_uses_valset_and_tracks_best_outputs_when_enabled() {
    let seen_prompts = Arc::new(Mutex::new(Vec::new()));
    let metric = RecordingFeedbackMetric {
        seen_prompts: Arc::clone(&seen_prompts),
    };
    let mut module = InstructionEchoModule {
        predictor: Predict::<OptimizerSig>::builder()
            .instruction("seed")
            .build(),
    };
    let valset = valset_for_gepa();

    let optimizer = GEPA::builder()
        .num_iterations(0)
        .minibatch_size(1)
        .track_best_outputs(true)
        .build();

    let result = optimizer
        .compile_with_valset::<OptimizerSig, _, _>(
            &mut module,
            trainset(),
            Some(valset.clone()),
            &metric,
        )
        .await
        .expect("GEPA compile should succeed with a dedicated valset");

    let seen = seen_prompts
        .lock()
        .expect("metric lock should not be poisoned")
        .clone();
    assert_eq!(seen, vec!["val-only".to_string()]);
    assert_eq!(
        result.highest_score_achieved_per_val_task.len(),
        valset.len()
    );
    assert!(
        result.highest_score_achieved_per_val_task[0] >= 100.0,
        "valset-only scoring should dominate, got {:?}",
        result.highest_score_achieved_per_val_task
    );

    let best_outputs = result
        .best_outputs_valset
        .as_ref()
        .expect("best outputs should be captured when tracking is enabled");
    assert_eq!(best_outputs.len(), valset.len());
    assert!(
        best_outputs[0].to_string().contains("val-only"),
        "best valset output should come from valset prompt, got {}",
        best_outputs[0]
    );
}

#[tokio::test]
async fn gepa_compile_respects_max_lm_calls_budget() {
    let metric = FeedbackMetricImpl;
    let mut module = InstructionEchoModule {
        predictor: Predict::<OptimizerSig>::builder()
            .instruction("seed")
            .build(),
    };

    let optimizer = GEPA::builder()
        .num_iterations(5)
        .minibatch_size(2)
        .max_lm_calls(2)
        .build();

    let result = optimizer
        .compile::<OptimizerSig, _, _>(&mut module, trainset(), &metric)
        .await
        .expect("GEPA compile should succeed under LM call budget");

    assert!(
        result.total_lm_calls <= 2,
        "LM call budget should be enforced, got {}",
        result.total_lm_calls
    );
}

#[tokio::test]
async fn gepa_compile_respects_max_rollouts_budget() {
    let metric = FeedbackMetricImpl;
    let mut module = InstructionEchoModule {
        predictor: Predict::<OptimizerSig>::builder()
            .instruction("seed")
            .build(),
    };

    let optimizer = GEPA::builder()
        .num_iterations(5)
        .minibatch_size(2)
        .max_rollouts(2)
        .build();

    let result = optimizer
        .compile::<OptimizerSig, _, _>(&mut module, trainset(), &metric)
        .await
        .expect("GEPA compile should succeed under rollout budget");

    assert!(
        result.total_rollouts <= 2,
        "rollout budget should be enforced, got {}",
        result.total_rollouts
    );
}

#[tokio::test]
async fn gepa_track_best_outputs_respects_lm_call_budget() {
    let metric = FeedbackMetricImpl;
    let mut module = InstructionEchoModule {
        predictor: Predict::<OptimizerSig>::builder()
            .instruction("seed")
            .build(),
    };

    let optimizer = GEPA::builder()
        .num_iterations(0)
        .minibatch_size(2)
        .track_best_outputs(true)
        .max_lm_calls(2)
        .build();

    let result = optimizer
        .compile::<OptimizerSig, _, _>(&mut module, trainset(), &metric)
        .await
        .expect("GEPA compile should respect LM call budget when tracking outputs");

    assert!(
        result.total_lm_calls <= 2,
        "LM call budget should be enforced, got {}",
        result.total_lm_calls
    );
    assert!(
        result.best_outputs_valset.is_none(),
        "best outputs should be skipped when budget does not allow extra eval calls"
    );
}
