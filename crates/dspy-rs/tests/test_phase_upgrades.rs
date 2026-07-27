//! End-to-end coverage for the runtime + optimizer upgrades: trace input/edge
//! recording, module state persistence, GEPA LM reflection, MIPRO demo
//! bootstrapping, ordered concurrent evaluation, and the LM response cache.

use anyhow::Result;
use dspy_rs::{
    CallMetadata, Chat, Example, GEPA, LM, LMClient, MIPROv2, Message, MetricOutcome, Module,
    ModuleState, Optimizer, Predict, PredictError, Predicted, Signature, TestCompletionModel,
    TypedMetric, evaluate_trainset,
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

// --- Trace: inputs, edges, and instance keys -------------------------------

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
async fn trace_records_inputs_edges_and_instance_keys() {
    let module = TwoStep {
        first: Predict::<QA>::builder()
            .lm(make_test_lm(vec![response_with_fields(&[("answer", "intermediate")])]).await)
            .build(),
        second: Predict::<QA>::builder()
            .lm(make_test_lm(vec![response_with_fields(&[("answer", "final")])]).await)
            .build(),
    };

    let (result, graph) = dspy_rs::trace::trace(|| {
        module.forward(QAInput {
            question: "start".to_string(),
        })
    })
    .await;
    let predicted = result.expect("traced pipeline should succeed");
    assert_eq!(predicted.answer, "final");

    assert_eq!(graph.nodes.len(), 2, "one node per Predict call");

    let first = &graph.nodes[0];
    let second = &graph.nodes[1];

    // Inputs are recorded for both nodes.
    let first_input = first.input_data.as_ref().expect("first node input recorded");
    assert_eq!(first_input.data["question"], "start");
    let second_input = second
        .input_data
        .as_ref()
        .expect("second node input recorded");
    assert_eq!(second_input.data["question"], "intermediate");

    // The second node is chained to the first.
    assert!(first.inputs.is_empty());
    assert_eq!(second.inputs, vec![first.id]);

    // Outputs are recorded.
    assert_eq!(
        first.output.as_ref().expect("first output").data["answer"],
        "intermediate"
    );
    assert_eq!(
        second.output.as_ref().expect("second output").data["answer"],
        "final"
    );

    // Instance keys identify distinct Predict instances.
    let key = |node: &dspy_rs::trace::Node| match &node.node_type {
        dspy_rs::trace::NodeType::Predict { instance_key, .. } => *instance_key,
        other => panic!("expected Predict node, got {other:?}"),
    };
    assert_ne!(key(first), 0);
    assert_ne!(key(first), key(second));
}

// --- ModuleState: save / load round trip -----------------------------------

#[derive(facet::Facet)]
#[facet(crate = facet)]
struct OneStep {
    predictor: Predict<QA>,
}

#[test]
fn module_state_round_trips_through_json() {
    let mut tuned = OneStep {
        predictor: Predict::<QA>::builder()
            .instruction("tuned instruction")
            .demo(Example::new(
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
    assert_eq!(predictor_state.demos[0].data["question"], "1+1?");
    assert_eq!(predictor_state.demos[0].data["answer"], "2");
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

impl TypedMetric<QA, Echo> for IndexScore {
    async fn evaluate(
        &self,
        _example: &Example<QA>,
        prediction: &Predicted<QAOutput>,
    ) -> Result<MetricOutcome> {
        Ok(MetricOutcome::score(prediction.answer.parse::<f32>()?))
    }
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn concurrent_evaluation_preserves_trainset_order() {
    let trainset: Vec<Example<QA>> = (0..8)
        .map(|idx| {
            Example::new(
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

    let scores: Vec<f32> = outcomes.iter().map(|outcome| outcome.score).collect();
    assert_eq!(scores, (0..8).map(|idx| idx as f32).collect::<Vec<_>>());
}

// --- GEPA: reflection through prompt_model ---------------------------------

struct FeedbackEcho;

impl TypedMetric<QA, OneStepEcho> for FeedbackEcho {
    async fn evaluate(
        &self,
        example: &Example<QA>,
        prediction: &Predicted<QAOutput>,
    ) -> Result<MetricOutcome> {
        let score = if prediction.answer == example.output.answer {
            1.0
        } else {
            0.0
        };
        Ok(MetricOutcome::with_feedback(
            score,
            dspy_rs::FeedbackMetric::new(score, "answer should match exactly"),
        ))
    }
}

#[derive(facet::Facet)]
#[facet(crate = facet)]
struct OneStepEcho {
    predictor: Predict<QA>,
}

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
    let trainset = vec![Example::new(
        QAInput {
            question: "echo".to_string(),
        },
        QAOutput {
            answer: "echo".to_string(),
        },
    )];

    let report = optimizer
        .compile(&mut module, trainset, &FeedbackEcho)
        .await
        .expect("gepa compile should succeed");

    assert_eq!(report.all_candidates.len(), 1);
    assert_eq!(
        report.all_candidates[0].instruction,
        "Be concise and cite evidence.",
        "child instruction should come from the reflection LM, not concatenation"
    );
    // seed eval (1) + parent minibatch (1) + reflection (1) + child eval (1)
    assert_eq!(report.total_lm_calls, 4);
}

// --- MIPRO: demo bootstrapping from traces ---------------------------------

struct ExactMatch;

impl TypedMetric<QA, OneStepPredict> for ExactMatch {
    async fn evaluate(
        &self,
        example: &Example<QA>,
        prediction: &Predicted<QAOutput>,
    ) -> Result<MetricOutcome> {
        let score = if prediction.answer == example.output.answer {
            1.0
        } else {
            0.0
        };
        Ok(MetricOutcome::score(score))
    }
}

#[derive(facet::Facet)]
#[facet(crate = facet)]
struct OneStepPredict {
    predictor: Predict<QA>,
}

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

    let trainset = vec![Example::new(
        QAInput {
            question: "What is 2+2?".to_string(),
        },
        QAOutput {
            answer: "4".to_string(),
        },
    )];

    optimizer
        .compile(&mut module, trainset, &ExactMatch)
        .await
        .expect("mipro compile should succeed");

    let state = ModuleState::from_module(&mut module).expect("dump should succeed");
    let predictor_state = &state.predictors["predictor"];
    assert_eq!(
        predictor_state.demos.len(),
        1,
        "the successful trace should be installed as a demo"
    );
    assert_eq!(predictor_state.demos[0].data["question"], "What is 2+2?");
    assert_eq!(predictor_state.demos[0].data["answer"], "4");
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
