use std::collections::HashMap;

use dspy_rs::__macro_support::bamltype::facet;
use dspy_rs::{ChainOfThought, Example, Predict, Signature, named_parameters};
use serde_json::json;

#[derive(Signature, Clone, Debug, PartialEq, facet::Facet)]
#[facet(crate = facet)]
struct QA {
    #[input]
    question: String,

    #[output]
    answer: String,
}

#[derive(facet::Facet)]
#[facet(crate = facet)]
struct MultiLeafInner {
    second: Predict<QA>,
    third: Predict<QA>,
}

#[derive(facet::Facet)]
#[facet(crate = facet)]
struct MultiLeafModule {
    first: Predict<QA>,
    nested: MultiLeafInner,
    fourth: Predict<QA>,
}

#[derive(facet::Facet)]
#[facet(crate = facet)]
struct StateRoundtripModule {
    predictor: Predict<QA>,
}

fn qa_demo(question: &str, answer: &str) -> Example {
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
fn named_parameters_chain_of_thought_exposes_predictor_and_mutates_state() {
    let mut module = ChainOfThought::<QA>::new();
    let mut params = named_parameters(&mut module).expect("walker should find predictor");

    assert_eq!(params.len(), 1);
    assert_eq!(params[0].0, "predictor");

    params[0]
        .1
        .set_instruction("Use short direct answers".to_string());
    assert_eq!(params[0].1.instruction(), "Use short direct answers");
    assert_eq!(params[0].1.demos_as_examples().len(), 0);

    drop(params);

    let mut roundtrip = named_parameters(&mut module).expect("walker should still succeed");
    let (_, predictor) = roundtrip
        .iter_mut()
        .find(|(name, _)| name == "predictor")
        .expect("predictor should still be discoverable");
    assert_eq!(predictor.instruction(), "Use short direct answers");
    assert_eq!(predictor.demos_as_examples().len(), 0);
}

#[test]
fn named_parameters_predict_dump_load_state_roundtrip() {
    let mut module = StateRoundtripModule {
        predictor: Predict::<QA>::new(),
    };

    let saved_state = {
        let mut params = named_parameters(&mut module).expect("walker should find predictor");
        let (_, predictor) = params
            .iter_mut()
            .find(|(name, _)| name == "predictor")
            .expect("predictor should exist");
        predictor.set_instruction("Use short direct answers".to_string());
        predictor
            .set_demos_from_examples(vec![qa_demo("What is 2 + 2?", "4")])
            .expect("demo setup should succeed");
        predictor.dump_state()
    };

    let mut params = named_parameters(&mut module).expect("walker should still find predictor");
    let (_, predictor) = params
        .iter_mut()
        .find(|(name, _)| name == "predictor")
        .expect("predictor should exist");
    predictor.set_instruction("temporary".to_string());
    predictor
        .set_demos_from_examples(Vec::new())
        .expect("demo reset should succeed");
    predictor
        .load_state(saved_state)
        .expect("state roundtrip should succeed");

    assert_eq!(predictor.instruction(), "Use short direct answers");
    let demos = predictor.demos_as_examples();
    assert_eq!(demos.len(), 1);
    assert_eq!(
        demos[0].data.get("question"),
        Some(&json!("What is 2 + 2?"))
    );
    assert_eq!(demos[0].data.get("answer"), Some(&json!("4")));
}

#[test]
fn named_parameters_multi_leaf_discovery_order_is_deterministic() {
    let mut module = MultiLeafModule {
        first: Predict::<QA>::new(),
        nested: MultiLeafInner {
            second: Predict::<QA>::new(),
            third: Predict::<QA>::new(),
        },
        fourth: Predict::<QA>::new(),
    };

    let expected = vec![
        "first".to_string(),
        "nested.second".to_string(),
        "nested.third".to_string(),
        "fourth".to_string(),
    ];

    for _ in 0..32 {
        let names = named_parameters(&mut module)
            .expect("walker should find all leaves")
            .into_iter()
            .map(|(name, _)| name)
            .collect::<Vec<_>>();
        assert_eq!(names, expected);
    }
}

#[test]
fn named_parameters_dump_load_is_idempotent_across_multiple_roundtrips() {
    let mut module = StateRoundtripModule {
        predictor: Predict::<QA>::new(),
    };

    let first_dump = {
        let mut params = named_parameters(&mut module).expect("walker should find predictor");
        let (_, predictor) = params
            .iter_mut()
            .find(|(name, _)| name == "predictor")
            .expect("predictor should exist");
        predictor.set_instruction("first-pass".to_string());
        predictor
            .set_demos_from_examples(vec![qa_demo("Q1", "A1"), qa_demo("Q2", "A2")])
            .expect("demo setup should succeed");
        predictor.dump_state()
    };

    let second_dump = {
        let mut params = named_parameters(&mut module).expect("walker should find predictor");
        let (_, predictor) = params
            .iter_mut()
            .find(|(name, _)| name == "predictor")
            .expect("predictor should exist");
        predictor
            .load_state(first_dump.clone())
            .expect("loading first state should succeed");
        predictor.dump_state()
    };

    assert_eq!(second_dump.instruction_override, first_dump.instruction_override);
    assert_eq!(second_dump.demos.len(), first_dump.demos.len());
    for (actual, expected) in second_dump.demos.iter().zip(first_dump.demos.iter()) {
        assert_eq!(actual.data, expected.data);
        assert_eq!(actual.input_keys, expected.input_keys);
        assert_eq!(actual.output_keys, expected.output_keys);
    }
}
