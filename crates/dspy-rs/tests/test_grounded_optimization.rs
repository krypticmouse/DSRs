use anyhow::Result;
use dspy_rs::*;
use rand::{Rng, SeedableRng, seq::SliceRandom};
use rig::completion::AssistantContent;

#[derive(Signature, Clone, Debug)]
struct Task {
    #[input]
    question: String,
    #[output]
    answer: String,
}
struct Program {
    predictor: Predict<Task>,
}
dspy_rs::predictors!(Program { predictor });
impl Module for Program {
    type Input = TaskInput;
    type Output = TaskOutput;
    async fn forward(&self, input: TaskInput) -> Result<Predicted<TaskOutput>, PredictError> {
        self.predictor.call(input).await
    }
}
struct Score;
impl TypedMetric<(TaskInput, TaskOutput), Program> for Score {
    async fn evaluate(
        &self,
        _: &(TaskInput, TaskOutput),
        pred: &Predicted<TaskOutput>,
        _: Option<&Trace>,
    ) -> Result<Eval> {
        Ok(Eval::score(if pred.answer == "good" { 1.0 } else { 0.0 }))
    }
}
async fn lm(responses: Vec<String>) -> (LM, TestCompletionModel) {
    let model = TestCompletionModel::new(responses.into_iter().map(AssistantContent::text));
    let config = LMConfig {
        base_url: Some("http://localhost/v1".into()),
        ..LMConfig::default()
    };
    let lm = LM::from_config(config)
        .await
        .unwrap()
        .with_client(LMClient::Test(model.clone()))
        .await
        .unwrap();
    (lm, model)
}
fn answer(text: &str) -> String {
    format!("[[ ## answer ## ]]\n{text}\n[[ ## completed ## ]]")
}
fn rows(question: &str) -> Vec<(TaskInput, TaskOutput)> {
    vec![(
        TaskInput {
            question: question.into(),
        },
        TaskOutput {
            answer: "good".into(),
        },
    )]
}

fn seed_for_demo_trial(proposed_instruction: bool, validation_len: usize) -> u64 {
    (0..10_000)
        .find(|seed| {
            let mut rng = rand::rngs::StdRng::seed_from_u64(*seed);
            let validation: Vec<_> = (0..validation_len).collect();
            let _ = validation.choose_multiple(&mut rng, 1).count();
            let instruction = rng.gen_range(0..2) == 1;
            let demos = rng.gen_range(0..3) == 2;
            instruction == proposed_instruction && demos
        })
        .expect("a seed should produce the requested joint choices")
}

#[tokio::test]
async fn mipro_keeps_baseline_and_excludes_validation_from_proposals_and_demos() -> Result<()> {
    let (task_lm, _) = lm(std::iter::once(answer("good"))
        .chain(std::iter::once(answer("good")))
        .chain(std::iter::repeat_n(answer("bad"), 40))
        .collect())
    .await;
    let (proposal_lm, proposal) = lm(vec![
        "[[ ## instructions ## ]]\n[\"Proposed replacement\"]\n[[ ## completed ## ]]".into(),
    ])
    .await;
    let mut program = Program {
        predictor: Predict::builder()
            .lm(task_lm)
            .instruction("Preserve original task")
            .build(),
    };
    let train = rows("TRAIN_ONLY");
    let val = rows("VALIDATION_SECRET");
    let mut target = OptimizeTarget::module_with_valset(&mut program, &train, Some(&val), &Score);
    let mut engine = Engine::new(EngineConfig::default());
    let optimizer = MIPROv2::builder()
        .prompt_model(proposal_lm)
        .num_candidates(2)
        .num_trials(12)
        .full_eval_steps(2)
        .minibatch_size(1)
        .seed(7)
        .build();
    optimizer.compile(&mut target, &mut engine).await?;
    for row in 0..engine.matrix().candidates() {
        if let Some(candidate) = engine.candidate(row) {
            let text = serde_json::to_string(candidate)?;
            assert!(!text.contains("VALIDATION_SECRET"));
        }
    }
    let request = format!("{:?}", proposal.last_request().unwrap());
    assert!(request.contains("TRAIN_ONLY"));
    assert!(request.contains("Preserve original task"));
    assert!(!request.contains("VALIDATION_SECRET"));
    drop(target);
    assert_eq!(
        PredictorInfo::instruction(&program.predictor),
        "Preserve original task"
    );
    assert!(PredictorInfo::demos_as_json(&program.predictor).is_empty());
    Ok(())
}

#[tokio::test]
async fn copro_uses_proposal_model_and_installs_improvement() -> Result<()> {
    let (task_lm, _) = lm(vec![answer("bad"), answer("good")]).await;
    let (proposal_lm, proposal) = lm(vec![
        "[[ ## instructions ## ]]\n[\"Improved task instruction\"]\n[[ ## completed ## ]]".into(),
    ])
    .await;
    let mut program = Program {
        predictor: Predict::builder()
            .lm(task_lm)
            .instruction("Original")
            .build(),
    };
    COPRO::builder()
        .prompt_model(proposal_lm)
        .init_temperature(0.8)
        .breadth(2)
        .depth(1)
        .build()
        .compile_module(&mut program, &rows("COPRO_TRAIN_MARKER"), &Score)
        .await?;
    assert_eq!(
        PredictorInfo::instruction(&program.predictor),
        "Improved task instruction"
    );
    assert!((proposal.last_request().unwrap().temperature.unwrap() - 0.8).abs() < 1e-6);
    assert!(format!("{:?}", proposal.last_request().unwrap()).contains("COPRO_TRAIN_MARKER"));
    Ok(())
}

#[tokio::test]
async fn proposal_failure_does_not_install_partial_state() {
    let (task_lm, _) = lm(vec![answer("good")]).await;
    let (proposal_lm, _) = lm(vec![
        "[[ ## instructions ## ]]\n[]\n[[ ## completed ## ]]".into(),
    ])
    .await;
    let mut program = Program {
        predictor: Predict::builder()
            .lm(task_lm)
            .instruction("Original")
            .build(),
    };
    let result = COPRO::builder()
        .prompt_model(proposal_lm)
        .breadth(2)
        .depth(1)
        .build()
        .compile_module(&mut program, &rows("train"), &Score)
        .await;
    assert!(result.is_err());
    assert_eq!(PredictorInfo::instruction(&program.predictor), "Original");
}

#[tokio::test]
async fn mipro_installs_bootstrapped_demos_on_improvement_in_final_partial_interval() -> Result<()>
{
    let (task_lm, _) = lm(vec![answer("good"), answer("bad"), answer("good")]).await;
    let (proposal_lm, _) = lm(vec![
        "[[ ## instructions ## ]]\n[\"Improved task instruction\"]\n[[ ## completed ## ]]".into(),
    ])
    .await;
    let mut program = Program {
        predictor: Predict::builder()
            .lm(task_lm)
            .instruction("Original")
            .build(),
    };
    let train = rows("demo_source");
    let val = rows("validation");
    let mut target = OptimizeTarget::module_with_valset(&mut program, &train, Some(&val), &Score);
    let mut engine = Engine::new(EngineConfig::default());
    MIPROv2::builder()
        .prompt_model(proposal_lm)
        .num_candidates(2)
        .num_trials(1)
        .full_eval_steps(5)
        .minibatch_size(1)
        .max_bootstrapped_demos(1)
        .min_demo_score(1.0)
        .eval_concurrency(1)
        .seed(seed_for_demo_trial(false, 1))
        .build()
        .compile(&mut target, &mut engine)
        .await?;
    drop(target);
    assert_eq!(PredictorInfo::demos_as_json(&program.predictor).len(), 1);
    Ok(())
}

#[tokio::test]
async fn mipro_rejects_minibatch_win_that_ties_on_full_validation() -> Result<()> {
    let (task_lm, _) = lm(vec![
        answer("good"),
        answer("good"),
        answer("bad"),
        answer("good"),
        answer("bad"),
    ])
    .await;
    let (proposal_lm, _) = lm(vec![
        "[[ ## instructions ## ]]\n[\"Candidate instruction\"]\n[[ ## completed ## ]]".into(),
    ])
    .await;
    let mut program = Program {
        predictor: Predict::builder()
            .lm(task_lm)
            .instruction("Original")
            .build(),
    };
    let train = rows("train");
    let val = vec![
        (
            TaskInput {
                question: "validation_a".into(),
            },
            TaskOutput {
                answer: "good".into(),
            },
        ),
        (
            TaskInput {
                question: "validation_b".into(),
            },
            TaskOutput {
                answer: "good".into(),
            },
        ),
    ];
    let mut target = OptimizeTarget::module_with_valset(&mut program, &train, Some(&val), &Score);
    let mut engine = Engine::new(EngineConfig::default());
    MIPROv2::builder()
        .prompt_model(proposal_lm)
        .num_candidates(2)
        .num_trials(1)
        .full_eval_steps(5)
        .minibatch_size(1)
        .max_bootstrapped_demos(1)
        .min_demo_score(1.0)
        .eval_concurrency(1)
        .seed(seed_for_demo_trial(true, 2))
        .build()
        .compile(&mut target, &mut engine)
        .await?;
    drop(target);
    assert_eq!(PredictorInfo::instruction(&program.predictor), "Original");
    Ok(())
}
