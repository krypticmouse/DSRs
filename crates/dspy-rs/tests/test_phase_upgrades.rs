//! End-to-end coverage for the runtime + optimizer upgrades: trace input/edge
//! recording, module state persistence, GEPA LM reflection, MIPRO demo
//! bootstrapping, ordered concurrent evaluation, and the LM response cache.

use anyhow::Result;
use dspy_rs::{
    CallMetadata, Chat, Demo, Eval, GEPA, LM, LMClient, MIPROv2, Message, Module, ModuleState,
    Predict, PredictError, Predicted, Signature, TestCompletionModel, TypedMetric,
    evaluate_trainset,
};
use rig::completion::AssistantContent;
use rig::message::Text;

fn response_with_fields(fields: &[(&str, &str)]) -> String {
    let mut response = String::new();
    for (name, value) in fields {
        response.push_str(&format!("[[ ## {name} ## ]]\n{value}\n\n"));
    }
    response.push_str("[[ ## completed ## ]]\n");
    response
}

fn text_response(text: impl Into<String>) -> AssistantContent {
    AssistantContent::Text(Text { text: text.into() })
}

async fn make_test_lm(responses: Vec<String>) -> LM {
    let client = TestCompletionModel::new(responses.into_iter().map(text_response));
    temp_env::async_with_vars(
        [("OPENAI_API_KEY", Some("test"))],
        LM::builder().model("openai:gpt-4o-mini".to_string()).build(),
    )
    .await
    .unwrap()
    .with_client(LMClient::Test(client))
    .await
    .unwrap()
}

#[derive(Signature, Clone, Debug)]
/// Answer questions accurately.
struct QA {
    #[input]
    question: String,

    #[output]
    answer: String,
}

// --- Trace: inputs, edges, and component names ------------------------------

#[derive(facet::Facet)]
#[facet(crate = facet)]
struct TwoStep {
    first: Predict<QA>,
    second: Predict<QA>,
}

impl Module for TwoStep {
    type Input = QAInput;
    type Output = QAOutput;

    async fn forward(&self, input: QAInput) -> Result<Predicted<QAOutput>, PredictError> {
        let first = self.first.call(input).await?;
        self.second
            .call(QAInput {
                question: first.answer.clone(),
            })
            .await
    }
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn capture_records_inputs_edges_and_component_names() {
    let module = TwoStep {
        first: Predict::<QA>::builder()
            .named("first")
            .lm(make_test_lm(vec![response_with_fields(&[("answer", "intermediate")])]).await)
            .build(),
        second: Predict::<QA>::builder()
            .named("second")
            .lm(make_test_lm(vec![response_with_fields(&[("answer", "final")])]).await)
            .build(),
    };

    let (result, trace) = dspy_rs::trace::capture(|| {
        module.forward(QAInput {
            question: "start".to_string(),
        })
    })
    .await;
    let predicted = result.expect("captured pipeline should succeed");
    assert_eq!(predicted.answer, "final");

    assert_eq!(trace.spans.len(), 2, "one span per Predict call");

    let first = &trace.spans[0];
    let second = &trace.spans[1];

    // Inputs are recorded for both spans.
    assert_eq!(first.input.as_ref().expect("first input")["question"], "start");
    assert_eq!(
        second.input.as_ref().expect("second input")["question"],
        "intermediate"
    );

    // Outputs are recorded.
    assert_eq!(
        first.output.as_ref().expect("first output")["answer"],
        "intermediate"
    );
    assert_eq!(
        second.output.as_ref().expect("second output")["answer"],
        "final"
    );

    // Distinct Predict instances record under their own component names — the
    // same names the params system addresses.
    assert_eq!(trace.component_name(first.component), "first");
    assert_eq!(trace.component_name(second.component), "second");
    assert_eq!(trace.for_component("first").count(), 1);
    assert_eq!(trace.for_component("second").count(), 1);
}

// --- ModuleState: save / load round trip -----------------------------------

struct OneStep {
    predictor: Predict<QA>,
}

dspy_rs::predictors!(OneStep { predictor });

#[test]
fn module_state_round_trips_through_json() {
    let mut tuned = OneStep {
        predictor: Predict::<QA>::builder()
            .instruction("tuned instruction")
            .demo(Demo::new(
                QAInput {
                    question: "1+1?".to_string(),
                },
                QAOutput {
                    answer: "2".to_string(),
                },
            ))
            .build(),
    };

    let state = ModuleState::from_module(&mut tuned).expect("dump should succeed");
    let json = state.to_json().expect("serialize should succeed");
    let restored = ModuleState::from_json(&json).expect("deserialize should succeed");

    let mut fresh = OneStep {
        predictor: Predict::<QA>::new(),
    };
    restored.apply(&mut fresh).expect("apply should succeed");

    let round_trip = ModuleState::from_module(&mut fresh).expect("dump should succeed");
    let predictor_state = &round_trip.predictors["predictor"];
    assert_eq!(
        predictor_state.instruction_override.as_deref(),
        Some("tuned instruction")
    );
    assert_eq!(predictor_state.demos.len(), 1);
    assert_eq!(predictor_state.demos[0]["question"], "1+1?");
    assert_eq!(predictor_state.demos[0]["answer"], "2");
}

#[test]
fn module_state_apply_rejects_unknown_predictors() {
    let mut tuned = OneStep {
        predictor: Predict::<QA>::builder().instruction("tuned").build(),
    };
    let mut state = ModuleState::from_module(&mut tuned).expect("dump should succeed");

    // Simulate a saved state whose structure diverged from the module.
    let ghost = state.predictors["predictor"].clone();
    state.predictors.insert("ghost".to_string(), ghost);

    let err = state
        .apply(&mut tuned)
        .expect_err("unknown predictor paths should be rejected");
    assert!(err.to_string().contains("ghost"));
}

// --- Concurrent evaluation preserves order ---------------------------------

struct Echo;

impl Module for Echo {
    type Input = QAInput;
    type Output = QAOutput;

    async fn forward(&self, input: QAInput) -> Result<Predicted<QAOutput>, PredictError> {
        Ok(Predicted::new(
            QAOutput {
                answer: input.question,
            },
            CallMetadata::default(),
        ))
    }
}

struct IndexScore;

impl TypedMetric<(QAInput, QAOutput), Echo> for IndexScore {
    async fn evaluate(
        &self,
        _example: &(QAInput, QAOutput),
        prediction: &Predicted<QAOutput>,
        _trace: Option<&dspy_rs::Trace>,
    ) -> Result<Eval> {
        Ok(Eval::score(prediction.answer.parse::<f64>()?))
    }
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn concurrent_evaluation_preserves_trainset_order() {
    let trainset: Vec<(QAInput, QAOutput)> = (0..8)
        .map(|idx| {
            (
                QAInput {
                    question: idx.to_string(),
                },
                QAOutput {
                    answer: idx.to_string(),
                },
            )
        })
        .collect();

    let outcomes = evaluate_trainset(&Echo, &trainset, &IndexScore)
        .await
        .expect("evaluation should succeed");

    let scores: Vec<f64> = outcomes.iter().map(|outcome| outcome.score).collect();
    assert_eq!(scores, (0..8).map(|idx| idx as f64).collect::<Vec<_>>());
}

// --- GEPA: reflection through prompt_model ---------------------------------

struct FeedbackEcho;

impl TypedMetric<(QAInput, QAOutput), OneStepEcho> for FeedbackEcho {
    async fn evaluate(
        &self,
        example: &(QAInput, QAOutput),
        prediction: &Predicted<QAOutput>,
        _trace: Option<&dspy_rs::Trace>,
    ) -> Result<Eval> {
        let score = if prediction.answer == example.1.answer {
            1.0
        } else {
            0.0
        };
        Ok(Eval::with_feedback(score, "answer should match exactly"))
    }
}

struct OneStepEcho {
    predictor: Predict<QA>,
}

dspy_rs::predictors!(OneStepEcho { predictor });

impl Module for OneStepEcho {
    type Input = QAInput;
    type Output = QAOutput;

    async fn forward(&self, input: QAInput) -> Result<Predicted<QAOutput>, PredictError> {
        Ok(Predicted::new(
            QAOutput {
                answer: input.question,
            },
            CallMetadata::default(),
        ))
    }
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn gepa_uses_reflection_lm_to_rewrite_instructions() {
    let reflection_lm = make_test_lm(vec![response_with_fields(&[(
        "improved_instruction",
        "Be concise and cite evidence.",
    )])])
    .await;

    let optimizer = GEPA::builder()
        .num_iterations(1)
        .minibatch_size(1)
        .prompt_model(reflection_lm)
        .seed(7)
        .build();

    let mut module = OneStepEcho {
        predictor: Predict::<QA>::builder().instruction("seed").build(),
    };
    let trainset = vec![(
        QAInput {
            question: "echo".to_string(),
        },
        QAOutput {
            answer: "echo".to_string(),
        },
    )];
    // A distinct valset keeps the trainset minibatch out of the engine's
    // rollout cache, so every phase below is a fresh, countable rollout.
    let valset = vec![(
        QAInput {
            question: "echo-val".to_string(),
        },
        QAOutput {
            answer: "echo-val".to_string(),
        },
    )];

    let report = optimizer
        .compile_module_with_valset(&mut module, &trainset, Some(&valset), &FeedbackEcho)
        .await
        .expect("gepa compile should succeed");

    assert_eq!(report.all_candidates.len(), 1);
    assert_eq!(
        report.all_candidates[0].instruction,
        "Be concise and cite evidence.",
        "child instruction should come from the reflection LM, not concatenation"
    );
    // seed eval on valset (1) + parent minibatch on trainset (1)
    // + reflection (1) + child eval on valset (1)
    assert_eq!(report.total_lm_calls, 4);
}

struct FeedbackForPredict;

impl TypedMetric<(QAInput, QAOutput), OneStepPredict> for FeedbackForPredict {
    async fn evaluate(
        &self,
        example: &(QAInput, QAOutput),
        prediction: &Predicted<QAOutput>,
        _trace: Option<&dspy_rs::Trace>,
    ) -> Result<Eval> {
        let score = if prediction.answer == example.1.answer {
            1.0
        } else {
            0.0
        };
        Ok(Eval::with_feedback(score, "match the expected answer"))
    }
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn gepa_reflection_receives_component_subtrace() {
    // A module with a real Predict leaf, so evaluation rollouts carry spans.
    // 3 module calls: initial eval + parent minibatch + child eval.
    let responses = vec![response_with_fields(&[("answer", "4")]); 3];
    let mut module = OneStepPredict {
        predictor: Predict::<QA>::builder()
            .instruction("seed")
            .lm(make_test_lm(responses).await)
            .build(),
    };

    let reflection_client = TestCompletionModel::new([text_response(response_with_fields(&[(
        "improved_instruction",
        "Improved.",
    )]))]);
    let reflection_lm = temp_env::async_with_vars(
        [("OPENAI_API_KEY", Some("test"))],
        LM::builder().model("openai:gpt-4o-mini".to_string()).build(),
    )
    .await
    .unwrap()
    .with_client(LMClient::Test(reflection_client.clone()))
    .await
    .unwrap();

    let optimizer = GEPA::builder()
        .num_iterations(1)
        .minibatch_size(1)
        .prompt_model(reflection_lm)
        .seed(7)
        .build();

    let trainset = vec![(
        QAInput {
            question: "What is 2+2?".to_string(),
        },
        QAOutput {
            answer: "4".to_string(),
        },
    )];
    // A distinct valset keeps the parent's trainset minibatch out of the
    // engine's rollout cache, so its rollout carries a fresh trace for the
    // reflector to read.
    let valset = vec![(
        QAInput {
            question: "What is 3+1?".to_string(),
        },
        QAOutput {
            answer: "4".to_string(),
        },
    )];

    optimizer
        .compile_module_with_valset(&mut module, &trainset, Some(&valset), &FeedbackForPredict)
        .await
        .expect("gepa compile should succeed");

    // The reflection prompt's execution_feedback carries the mutated
    // component's per-invocation sub-trace, not just the metric string.
    let request = reflection_client
        .last_request()
        .expect("reflection LM should have been called");
    let rendered = format!("{:?}", request.chat_history);
    assert!(
        rendered.contains("call 0:"),
        "reflection input should include the component's invocation record: {rendered}"
    );
    assert!(
        rendered.contains("What is 2+2?"),
        "reflection input should include the span's recorded input"
    );
    assert!(
        rendered.contains("match the expected answer"),
        "reflection input should keep the metric feedback"
    );
}

// --- MIPRO: demo bootstrapping from traces ---------------------------------

struct ExactMatch;

impl TypedMetric<(QAInput, QAOutput), OneStepPredict> for ExactMatch {
    async fn evaluate(
        &self,
        example: &(QAInput, QAOutput),
        prediction: &Predicted<QAOutput>,
        _trace: Option<&dspy_rs::Trace>,
    ) -> Result<Eval> {
        let score = if prediction.answer == example.1.answer {
            1.0
        } else {
            0.0
        };
        Ok(Eval::score(score))
    }
}

struct OneStepPredict {
    predictor: Predict<QA>,
}

dspy_rs::predictors!(OneStepPredict { predictor });

impl Module for OneStepPredict {
    type Input = QAInput;
    type Output = QAOutput;

    async fn forward(&self, input: QAInput) -> Result<Predicted<QAOutput>, PredictError> {
        self.predictor.call(input).await
    }
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn mipro_bootstraps_demos_from_successful_traces() {
    // 1 traced call + 2 candidate evaluations x 1 minibatch example = 3 LM calls.
    let responses = vec![response_with_fields(&[("answer", "4")]); 3];
    let mut module = OneStepPredict {
        predictor: Predict::<QA>::builder()
            .lm(make_test_lm(responses).await)
            .build(),
    };

    let optimizer = MIPROv2::builder()
        .num_candidates(2)
        .num_trials(2)
        .minibatch_size(1)
        .min_demo_score(1.0)
        .seed(7)
        .build();

    let trainset = vec![(
        QAInput {
            question: "What is 2+2?".to_string(),
        },
        QAOutput {
            answer: "4".to_string(),
        },
    )];

    optimizer
        .compile_module(&mut module, &trainset, &ExactMatch)
        .await
        .expect("mipro compile should succeed");

    let state = ModuleState::from_module(&mut module).expect("dump should succeed");
    let predictor_state = &state.predictors["predictor"];
    assert_eq!(
        predictor_state.demos.len(),
        1,
        "the successful trace should be installed as a demo"
    );
    assert_eq!(predictor_state.demos[0]["question"], "What is 2+2?");
    assert_eq!(predictor_state.demos[0]["answer"], "4");
    assert!(
        predictor_state.instruction_override.is_some(),
        "instruction search should install a candidate"
    );
}

// --- LM response cache -----------------------------------------------------

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn lm_response_cache_serves_repeat_calls_without_provider() {
    // Only ONE response is queued: a second provider call would error with
    // "test response queue is empty", so a passing second call proves a cache hit.
    let client = TestCompletionModel::new([text_response("cached answer")]);
    let lm = temp_env::async_with_vars(
        [("OPENAI_API_KEY", Some("test"))],
        LM::builder()
            .model("openai:gpt-4o-mini".to_string())
            .cache(true)
            .build(),
    )
    .await
    .unwrap()
    .with_client(LMClient::Test(client))
    .await
    .unwrap();

    let chat = Chat::new(vec![Message::user("hello")]);

    let first = lm
        .call(chat.clone(), vec![])
        .await
        .expect("first call should hit the provider");
    let second = lm
        .call(chat, vec![])
        .await
        .expect("second call should be served from cache");

    assert_eq!(first.output.content(), "cached answer");
    assert_eq!(second.output.content(), "cached answer");
}
