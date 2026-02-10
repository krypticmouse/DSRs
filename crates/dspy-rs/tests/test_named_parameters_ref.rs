use dspy_rs::__macro_support::bamltype::facet;
use dspy_rs::{ChainOfThought, Predict, Signature, named_parameters, named_parameters_ref};

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
