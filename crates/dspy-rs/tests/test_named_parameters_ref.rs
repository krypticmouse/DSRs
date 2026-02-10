use dspy_rs::__macro_support::bamltype::facet;
use dspy_rs::{
    ChainOfThought, ModuleExt, Predict, PredictError, ReAct, Signature, WithReasoning,
    named_parameters, named_parameters_ref,
};

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
struct Wrapper {
    first: Predict<QA>,
    cot: ChainOfThought<QA>,
}

#[derive(facet::Facet)]
#[facet(crate = facet)]
struct DeepWrapper {
    nested: Wrapper,
    extra: ChainOfThought<QA>,
}

fn drop_reasoning(output: WithReasoning<QAOutput>) -> QAOutput {
    output.inner
}

fn drop_reasoning_checked(output: WithReasoning<QAOutput>) -> Result<QAOutput, PredictError> {
    Ok(output.inner)
}

#[test]
fn named_parameters_ref_discovers_same_paths_as_named_parameters() {
    let mut module = Wrapper {
        first: Predict::<QA>::new(),
        cot: ChainOfThought::<QA>::new(),
    };

    let mut mutable = named_parameters(&mut module).expect("mutable walker should succeed");
    let mutable_paths = mutable
        .iter_mut()
        .map(|(path, _)| path.clone())
        .collect::<Vec<_>>();
    let mutable_first_instruction = mutable
        .iter_mut()
        .find(|(path, _)| path == "first")
        .expect("first predictor should be present")
        .1
        .instruction();

    let immutable = named_parameters_ref(&module).expect("immutable walker should succeed");
    let immutable_paths = immutable
        .iter()
        .map(|(path, _)| path.clone())
        .collect::<Vec<_>>();

    assert_eq!(immutable_paths, mutable_paths);

    let first = immutable
        .iter()
        .find(|(path, _)| path == "first")
        .expect("first predictor should be present");
    assert_eq!(first.1.instruction(), mutable_first_instruction);
}

#[test]
fn named_parameters_ref_reflects_mutations_from_named_parameters() {
    let mut module = Wrapper {
        first: Predict::<QA>::new(),
        cot: ChainOfThought::<QA>::new(),
    };

    {
        let mut mutable = named_parameters(&mut module).expect("mutable walker should succeed");
        for (path, predictor) in mutable.iter_mut() {
            predictor.set_instruction(format!("inst::{path}"));
        }
    }

    let immutable = named_parameters_ref(&module).expect("immutable walker should succeed");
    let collected = immutable
        .iter()
        .map(|(path, predictor)| (path.clone(), predictor.instruction()))
        .collect::<Vec<_>>();

    assert_eq!(
        collected,
        vec![
            ("first".to_string(), "inst::first".to_string()),
            ("cot.predictor".to_string(), "inst::cot.predictor".to_string()),
        ]
    );
}

#[test]
fn named_parameters_ref_preserves_canonical_paths_through_nested_wrappers() {
    let mut module = DeepWrapper {
        nested: Wrapper {
            first: Predict::<QA>::new(),
            cot: ChainOfThought::<QA>::new(),
        },
        extra: ChainOfThought::<QA>::new(),
    };

    {
        let mut mutable = named_parameters(&mut module).expect("mutable walker should succeed");
        let mutable_paths = mutable
            .iter()
            .map(|(path, _)| path.clone())
            .collect::<Vec<_>>();
        assert_eq!(
            mutable_paths,
            vec![
                "nested.first".to_string(),
                "nested.cot.predictor".to_string(),
                "extra.predictor".to_string(),
            ]
        );
        for (path, predictor) in mutable.iter_mut() {
            predictor.set_instruction(format!("nested::{path}"));
        }
    }

    let immutable = named_parameters_ref(&module).expect("shared walker should succeed");
    let immutable_collected = immutable
        .iter()
        .map(|(path, predictor)| (path.clone(), predictor.instruction()))
        .collect::<Vec<_>>();

    assert_eq!(
        immutable_collected,
        vec![
            (
                "nested.first".to_string(),
                "nested::nested.first".to_string(),
            ),
            (
                "nested.cot.predictor".to_string(),
                "nested::nested.cot.predictor".to_string(),
            ),
            (
                "extra.predictor".to_string(),
                "nested::extra.predictor".to_string(),
            ),
        ]
    );
}

#[test]
fn named_parameters_wrapper_paths_are_consistent_for_map_and_and_then() {
    let mut mapped = ChainOfThought::<QA>::new().map(
        drop_reasoning as fn(WithReasoning<QAOutput>) -> QAOutput,
    );
    let mapped_mut_paths = named_parameters(&mut mapped)
        .expect("mutable map traversal should succeed")
        .into_iter()
        .map(|(path, _)| path)
        .collect::<Vec<_>>();
    let mapped_ref_paths = named_parameters_ref(&mapped)
        .expect("shared map traversal should succeed")
        .into_iter()
        .map(|(path, _)| path)
        .collect::<Vec<_>>();
    assert_eq!(mapped_mut_paths, vec!["inner.predictor".to_string()]);
    assert_eq!(mapped_ref_paths, mapped_mut_paths);

    let mut and_then = ChainOfThought::<QA>::new().and_then(
        drop_reasoning_checked as fn(WithReasoning<QAOutput>) -> Result<QAOutput, PredictError>,
    );
    let and_then_mut_paths = named_parameters(&mut and_then)
        .expect("mutable and_then traversal should succeed")
        .into_iter()
        .map(|(path, _)| path)
        .collect::<Vec<_>>();
    let and_then_ref_paths = named_parameters_ref(&and_then)
        .expect("shared and_then traversal should succeed")
        .into_iter()
        .map(|(path, _)| path)
        .collect::<Vec<_>>();
    assert_eq!(and_then_mut_paths, vec!["inner.predictor".to_string()]);
    assert_eq!(and_then_ref_paths, and_then_mut_paths);
}

#[test]
fn named_parameters_react_paths_match_between_mut_and_ref_walkers() {
    let mut react = ReAct::<QA>::new();

    let mut_paths = named_parameters(&mut react)
        .expect("mutable ReAct traversal should succeed")
        .into_iter()
        .map(|(path, _)| path)
        .collect::<Vec<_>>();
    let ref_paths = named_parameters_ref(&react)
        .expect("shared ReAct traversal should succeed")
        .into_iter()
        .map(|(path, _)| path)
        .collect::<Vec<_>>();

    assert_eq!(
        mut_paths,
        vec!["action".to_string(), "extract".to_string()]
    );
    assert_eq!(ref_paths, mut_paths);
}
