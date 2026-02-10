use anyhow::Result;
use dspy_rs::{Demo, DynPredictor, Example, Predict, PredictState, Signature};
use serde_json::json;
use std::collections::HashMap;

#[derive(Signature, Clone, Debug, PartialEq)]
struct QA {
    #[input]
    question: String,

    #[output]
    answer: String,
}

fn qa_demo(question: &str, answer: &str) -> Demo<QA> {
    Demo::new(
        QAInput {
            question: question.to_string(),
        },
        QAOutput {
            answer: answer.to_string(),
        },
    )
}

fn qa_example(question: &str, answer: &str) -> Example {
    Example::new(
        HashMap::from([
            ("question".to_string(), json!(question)),
            ("answer".to_string(), json!(answer)),
        ]),
        vec!["question".to_string()],
        vec!["answer".to_string()],
    )
}

#[test]
fn predict_builder_sets_initial_instruction() {
    let predictor = Predict::<QA>::builder().instruction("Be concise").build();
    assert_eq!(<Predict<QA> as DynPredictor>::instruction(&predictor), "Be concise");
}

#[test]
fn dyn_predictor_state_dump_load_roundtrip() -> Result<()> {
    let mut predictor = Predict::<QA>::builder()
        .instruction("Initial instruction")
        .demo(qa_demo("What is 2+2?", "4"))
        .build();

    let saved = <Predict<QA> as DynPredictor>::dump_state(&predictor);
    assert_eq!(saved.demos.len(), 1);
    assert_eq!(saved.instruction_override.as_deref(), Some("Initial instruction"));

    <Predict<QA> as DynPredictor>::load_state(
        &mut predictor,
        PredictState {
            demos: vec![qa_example("Capital of France?", "Paris")],
            instruction_override: Some("Loaded instruction".to_string()),
        },
    )?;

    assert_eq!(
        <Predict<QA> as DynPredictor>::instruction(&predictor),
        "Loaded instruction"
    );

    let demos = <Predict<QA> as DynPredictor>::demos_as_examples(&predictor);
    assert_eq!(demos.len(), 1);
    assert_eq!(demos[0].data.get("question"), Some(&json!("Capital of France?")));
    assert_eq!(demos[0].data.get("answer"), Some(&json!("Paris")));

    <Predict<QA> as DynPredictor>::load_state(&mut predictor, saved)?;
    assert_eq!(
        <Predict<QA> as DynPredictor>::instruction(&predictor),
        "Initial instruction"
    );

    Ok(())
}

#[test]
fn dyn_predictor_rejects_invalid_demo_shape() {
    let mut predictor = Predict::<QA>::new();
    let bad_demo = Example::new(
        HashMap::from([("wrong".to_string(), json!("field"))]),
        vec!["wrong".to_string()],
        vec![],
    );

    let result = <Predict<QA> as DynPredictor>::set_demos_from_examples(&mut predictor, vec![bad_demo]);
    assert!(result.is_err());
}
