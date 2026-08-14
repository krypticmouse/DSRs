//! SIMBA end-to-end on canned LM responses: minibatch introspective ascent
//! with both move types (append-demo, append-rule), gate rejection, and the
//! reflection LM canned through its own TestCompletionModel.

use anyhow::Result;
use dspy_rs::{
    Eval, Example, LM, LMClient, Module, ModuleState, Optimizer, Predict, PredictError, Predicted,
    SIMBA, Signature, SimbaMove, TestCompletionModel, TypedMetric,
};
use rig::completion::AssistantContent;
use rig::message::Text;

#[derive(Signature, Clone, Debug)]
/// Answer the prompt.
struct SimbaSig {
    #[input]
    prompt: String,

    #[output]
    answer: String,
}

#[derive(facet::Facet)]
#[facet(crate = facet)]
struct SimbaModule {
    predictor: Predict<SimbaSig>,
}

impl Module for SimbaModule {
    type Input = SimbaSigInput;
    type Output = SimbaSigOutput;

    async fn forward(
        &self,
        input: SimbaSigInput,
    ) -> Result<Predicted<SimbaSigOutput>, PredictError> {
        self.predictor.call(input).await
    }
}

struct ExactMatch;

impl TypedMetric<SimbaSig, SimbaModule> for ExactMatch {
    async fn evaluate(
        &self,
        example: &Example<SimbaSig>,
        prediction: &Predicted<SimbaSigOutput>,
        _trace: Option<&dspy_rs::Trace>,
    ) -> Result<Eval> {
        let score = (prediction.answer == example.output.answer) as u8 as f64;
        Ok(Eval::with_feedback(score, "exact-match"))
    }
}

fn answer_response(text: &str) -> AssistantContent {
    AssistantContent::Text(Text {
        text: format!("[[ ## answer ## ]]\n{text}\n\n[[ ## completed ## ]]\n"),
    })
}

fn rule_response(text: &str) -> AssistantContent {
    AssistantContent::Text(Text {
        text: format!("[[ ## rule ## ]]\n{text}\n\n[[ ## completed ## ]]\n"),
    })
}

async fn make_test_lm(client: TestCompletionModel) -> LM {
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

async fn simba_module(client: TestCompletionModel) -> SimbaModule {
    SimbaModule {
        predictor: Predict::<SimbaSig>::builder()
            .lm(make_test_lm(client).await)
            .build(),
    }
}

fn trainset(n: usize) -> Vec<Example<SimbaSig>> {
    (0..n)
        .map(|idx| {
            Example::new(
                SimbaSigInput {
                    prompt: idx.to_string(),
                },
                SimbaSigOutput {
                    answer: idx.to_string(),
                },
            )
        })
        .collect()
}

#[tokio::test]
async fn append_demo_move_is_harvested_gated_and_installed() {
    // Baseline: examples 0 and 2 correct, example 1 wrong (mean 2/3). The best
    // rollout (example 0) qualifies as a demo source, so step 0 proposes
    // append-demo. Child gate minibatch (= full set): everything correct.
    let client = TestCompletionModel::new([
        answer_response("0"),
        answer_response("wrong"),
        answer_response("2"),
        answer_response("0"),
        answer_response("1"),
        answer_response("2"),
    ]);
    let mut module = simba_module(client).await;

    let simba = SIMBA::builder()
        .max_steps(1)
        .minibatch_size(3)
        .min_demo_score(1.0)
        .eval_concurrency(1)
        .seed(0)
        .build();

    let report = simba
        .compile::<SimbaSig, _, _>(&mut module, trainset(3), &ExactMatch)
        .await
        .expect("SIMBA should succeed on canned responses");

    assert!((report.baseline_score - 2.0 / 3.0).abs() < 1e-9);
    assert!((report.final_score - 1.0).abs() < 1e-9);
    assert_eq!(report.accepted, 1);
    assert_eq!(report.rejected, 0);
    assert_eq!(report.steps.len(), 1);
    let step = &report.steps[0];
    assert_eq!(step.move_kind, SimbaMove::AppendDemo);
    assert!(step.accepted);
    assert!((step.parent_minibatch_score - 2.0 / 3.0).abs() < 1e-9);
    assert!((step.child_minibatch_score - 1.0).abs() < 1e-9);
    assert_eq!(step.full_score, Some(1.0));

    // Baseline pass (3) + gate minibatch (3); the promotion full pass is
    // served entirely from the cache (minibatch == full set).
    assert_eq!(report.spend.lm_calls, 6);
    assert_eq!(report.spend.metric_calls, 6);
    assert_eq!(report.spend.cache_hits, 3);

    // The accepted demo is installed permanently; it came from the best
    // rollout (example 0) via the trace name-join.
    let state = ModuleState::from_module(&mut module).unwrap();
    let predictor = &state.predictors["predictor"];
    assert_eq!(predictor.demos.len(), 1);
    assert_eq!(predictor.demos[0].data["prompt"].as_str(), Some("0"));
    assert_eq!(predictor.demos[0].data["answer"].as_str(), Some("0"));
    assert_eq!(predictor.instruction_override, None, "demo moves leave instructions alone");
}

#[tokio::test]
async fn append_rule_move_uses_the_reflection_lm() {
    // Baseline: everything wrong, so no rollout qualifies as a demo source and
    // step 0 falls back to append-rule via the canned reflection LM.
    let client = TestCompletionModel::new([
        answer_response("x"),
        answer_response("x"),
        answer_response("0"),
        answer_response("1"),
    ]);
    let mut module = simba_module(client).await;
    let reflection_client =
        TestCompletionModel::new([rule_response("Always echo the prompt digits exactly.")]);

    let simba = SIMBA::builder()
        .max_steps(1)
        .minibatch_size(2)
        .min_demo_score(1.0)
        .eval_concurrency(1)
        .seed(0)
        .prompt_model(make_test_lm(reflection_client).await)
        .build();

    let report = simba
        .compile::<SimbaSig, _, _>(&mut module, trainset(2), &ExactMatch)
        .await
        .unwrap();

    assert_eq!(report.baseline_score, 0.0);
    assert!((report.final_score - 1.0).abs() < 1e-9);
    assert_eq!(report.accepted, 1);
    assert_eq!(report.steps[0].move_kind, SimbaMove::AppendRule);
    assert!(report.steps[0].accepted);

    // Baseline (2) + gate minibatch (2) + one charged reflection call.
    assert_eq!(report.spend.lm_calls, 5);
    assert_eq!(report.spend.metric_calls, 4);

    // The distilled rule is appended to the predictor's instruction.
    let state = ModuleState::from_module(&mut module).unwrap();
    let instruction = state.predictors["predictor"]
        .instruction_override
        .as_deref()
        .expect("rule moves install an instruction override");
    assert!(instruction.contains("[SIMBA rule] Always echo the prompt digits exactly."));
    assert!(state.predictors["predictor"].demos.is_empty(), "rule moves leave demos alone");
}

#[tokio::test]
async fn append_rule_falls_back_to_metric_feedback_without_prompt_model() {
    let client = TestCompletionModel::new([
        answer_response("x"),
        answer_response("x"),
        answer_response("0"),
        answer_response("1"),
    ]);
    let mut module = simba_module(client).await;

    let simba = SIMBA::builder()
        .max_steps(1)
        .minibatch_size(2)
        .min_demo_score(1.0)
        .eval_concurrency(1)
        .seed(0)
        .build();

    let report = simba
        .compile::<SimbaSig, _, _>(&mut module, trainset(2), &ExactMatch)
        .await
        .unwrap();

    assert_eq!(report.steps[0].move_kind, SimbaMove::AppendRule);
    assert!(report.steps[0].accepted);
    // No reflection LM: only the 4 rollouts spend LM calls.
    assert_eq!(report.spend.lm_calls, 4);

    let state = ModuleState::from_module(&mut module).unwrap();
    let instruction = state.predictors["predictor"]
        .instruction_override
        .as_deref()
        .unwrap();
    // The worst rollout's metric feedback becomes the rule verbatim.
    assert!(instruction.contains("[SIMBA rule] exact-match"));
}

#[tokio::test]
async fn gate_rejection_leaves_the_module_untouched() {
    // Baseline: example 0 correct (mean 0.5). Step 0 proposes append-demo from
    // example 0's rollout, but the child scores 0 on the gate minibatch — the
    // move is rejected and the module keeps its baseline state.
    let client = TestCompletionModel::new([
        answer_response("0"),
        answer_response("wrong"),
        answer_response("wrong"),
        answer_response("wrong"),
    ]);
    let mut module = simba_module(client).await;

    let simba = SIMBA::builder()
        .max_steps(1)
        .minibatch_size(2)
        .min_demo_score(1.0)
        .eval_concurrency(1)
        .seed(0)
        .build();

    let report = simba
        .compile::<SimbaSig, _, _>(&mut module, trainset(2), &ExactMatch)
        .await
        .unwrap();

    assert!((report.baseline_score - 0.5).abs() < 1e-9);
    assert!((report.final_score - 0.5).abs() < 1e-9, "rejected moves never change the score");
    assert_eq!(report.accepted, 0);
    assert_eq!(report.rejected, 1);
    let step = &report.steps[0];
    assert_eq!(step.move_kind, SimbaMove::AppendDemo);
    assert!(!step.accepted);
    assert_eq!(step.full_score, None, "rejected moves never run the full set");
    assert_eq!(step.child_minibatch_score, 0.0);

    // Rejection ran only baseline (2) + gate minibatch (2).
    assert_eq!(report.spend.metric_calls, 4);

    let state = ModuleState::from_module(&mut module).unwrap();
    assert!(state.predictors["predictor"].demos.is_empty());
    assert_eq!(state.predictors["predictor"].instruction_override, None);
}

#[tokio::test]
async fn budget_stops_the_ascent_cleanly() {
    // Budget covers exactly the baseline pass; step 0's gate minibatch no
    // longer fits, so the run stops with zero steps and reports the baseline.
    let client = TestCompletionModel::new([answer_response("0"), answer_response("1")]);
    let mut module = simba_module(client).await;

    let simba = SIMBA::builder()
        .max_steps(3)
        .minibatch_size(2)
        .eval_concurrency(1)
        .seed(0)
        .max_metric_calls(3)
        .build();

    let report = simba
        .compile::<SimbaSig, _, _>(&mut module, trainset(2), &ExactMatch)
        .await
        .unwrap();

    assert!((report.baseline_score - 1.0).abs() < 1e-9);
    assert!(report.steps.is_empty(), "no step fit the remaining budget");
    assert_eq!(report.spend.metric_calls, 2, "only the baseline ran");
}
