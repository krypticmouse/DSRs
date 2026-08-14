//! BootstrapFewShot end-to-end on canned LM responses: teacher pass, demo
//! harvest via the trace name-join, engine-evaluated demo candidate, and
//! keep-if-better adoption.

use anyhow::Result;
use dspy_rs::{
    BootstrapFewShot, Eval, Example, LM, LMClient, Module, ModuleState, Optimizer, Predict,
    PredictError, Predicted, Signature, TestCompletionModel, TypedMetric,
};
use rig::completion::AssistantContent;
use rig::message::Text;

#[derive(Signature, Clone, Debug)]
/// Answer the prompt.
struct BootSig {
    #[input]
    prompt: String,

    #[output]
    answer: String,
}

#[derive(facet::Facet)]
#[facet(crate = facet)]
struct BootModule {
    predictor: Predict<BootSig>,
}

impl Module for BootModule {
    type Input = BootSigInput;
    type Output = BootSigOutput;

    async fn forward(&self, input: BootSigInput) -> Result<Predicted<BootSigOutput>, PredictError> {
        self.predictor.call(input).await
    }
}

struct ExactMatch;

impl TypedMetric<BootSig, BootModule> for ExactMatch {
    async fn evaluate(
        &self,
        example: &Example<BootSig>,
        prediction: &Predicted<BootSigOutput>,
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

async fn boot_module(client: TestCompletionModel) -> BootModule {
    let lm = temp_env::async_with_vars(
        [("OPENAI_API_KEY", Some("test"))],
        LM::builder().model("openai:gpt-4o-mini".to_string()).build(),
    )
    .await
    .unwrap()
    .with_client(LMClient::Test(client))
    .await
    .unwrap();
    BootModule {
        predictor: Predict::<BootSig>::builder().lm(lm).build(),
    }
}

fn trainset() -> Vec<Example<BootSig>> {
    (0..3)
        .map(|idx: usize| {
            Example::new(
                BootSigInput {
                    prompt: idx.to_string(),
                },
                BootSigOutput {
                    answer: idx.to_string(),
                },
            )
        })
        .collect()
}

#[tokio::test]
async fn bootstrap_harvests_demos_and_adopts_when_better() {
    // Teacher pass: examples 0 and 2 answered correctly, example 1 wrong.
    // Candidate pass (demos attached): everything correct.
    let client = TestCompletionModel::new([
        answer_response("0"),
        answer_response("wrong"),
        answer_response("2"),
        answer_response("0"),
        answer_response("1"),
        answer_response("2"),
    ]);
    let mut module = boot_module(client).await;

    let bootstrap = BootstrapFewShot::builder()
        .max_demos(4)
        .min_demo_score(1.0)
        .eval_concurrency(1)
        .build();

    let report = bootstrap
        .compile::<BootSig, _, _>(&mut module, trainset(), &ExactMatch)
        .await
        .expect("bootstrap should succeed on canned responses");

    assert!((report.baseline_score - 2.0 / 3.0).abs() < 1e-9);
    assert_eq!(report.candidate_score, Some(1.0));
    assert!(report.adopted);
    // Demos joined to the predictor by trace component name (the dotted path).
    assert_eq!(report.demos_per_predictor.get("predictor"), Some(&2));
    assert_eq!(report.spend.metric_calls, 6);
    assert_eq!(report.spend.lm_calls, 6);

    // The winning demos are installed on the module.
    let state = ModuleState::from_module(&mut module).unwrap();
    let predictor = &state.predictors["predictor"];
    assert_eq!(predictor.demos.len(), 2);
    assert_eq!(predictor.instruction_override, None, "instructions untouched");
    let demo_prompts: Vec<String> = predictor
        .demos
        .iter()
        .map(|demo| demo["prompt"].as_str().unwrap().to_string())
        .collect();
    assert!(demo_prompts.contains(&"0".to_string()));
    assert!(demo_prompts.contains(&"2".to_string()));
    assert!(!demo_prompts.contains(&"1".to_string()), "failed rollouts never become demos");
}

#[tokio::test]
async fn bootstrap_keeps_baseline_when_candidate_is_worse() {
    // Teacher pass: all correct. Candidate pass: all wrong — must not adopt.
    let client = TestCompletionModel::new([
        answer_response("0"),
        answer_response("1"),
        answer_response("2"),
        answer_response("x"),
        answer_response("x"),
        answer_response("x"),
    ]);
    let mut module = boot_module(client).await;

    let bootstrap = BootstrapFewShot::builder()
        .min_demo_score(1.0)
        .eval_concurrency(1)
        .build();

    let report = bootstrap
        .compile::<BootSig, _, _>(&mut module, trainset(), &ExactMatch)
        .await
        .unwrap();

    assert!((report.baseline_score - 1.0).abs() < 1e-9);
    assert_eq!(report.candidate_score, Some(0.0));
    assert!(!report.adopted);

    let state = ModuleState::from_module(&mut module).unwrap();
    assert!(
        state.predictors["predictor"].demos.is_empty(),
        "a losing candidate leaves the module untouched"
    );
}

#[tokio::test]
async fn bootstrap_without_qualifying_rollouts_skips_candidate_eval() {
    // Nothing scores >= 1.0, so no demos are harvested and no second pass runs
    // (the queue holds exactly the teacher-pass responses).
    let client = TestCompletionModel::new([
        answer_response("x"),
        answer_response("x"),
        answer_response("x"),
    ]);
    let mut module = boot_module(client).await;

    let bootstrap = BootstrapFewShot::builder()
        .min_demo_score(1.0)
        .eval_concurrency(1)
        .build();

    let report = bootstrap
        .compile::<BootSig, _, _>(&mut module, trainset(), &ExactMatch)
        .await
        .unwrap();

    assert_eq!(report.baseline_score, 0.0);
    assert_eq!(report.candidate_score, None);
    assert!(!report.adopted);
    assert!(report.demos_per_predictor.is_empty());
    assert_eq!(report.spend.metric_calls, 3, "only the teacher pass ran");
}

#[tokio::test]
async fn bootstrap_respects_max_demos() {
    let client = TestCompletionModel::new([
        answer_response("0"),
        answer_response("1"),
        answer_response("2"),
        answer_response("0"),
        answer_response("1"),
        answer_response("2"),
    ]);
    let mut module = boot_module(client).await;

    let bootstrap = BootstrapFewShot::builder()
        .max_demos(1)
        .min_demo_score(1.0)
        .eval_concurrency(1)
        .build();

    let report = bootstrap
        .compile::<BootSig, _, _>(&mut module, trainset(), &ExactMatch)
        .await
        .unwrap();

    assert_eq!(report.demos_per_predictor.get("predictor"), Some(&1));
}
