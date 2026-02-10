use dspy_rs::__macro_support::bamltype::facet;
use dspy_rs::{Predict, ProgramGraph, Signature, named_parameters_ref};

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
    predictor: Predict<QA>,
}

#[test]
fn from_module_snapshot_then_fit_roundtrip() {
    let mut typed = Wrapper {
        predictor: Predict::<QA>::new(),
    };

    let before = named_parameters_ref(&typed)
        .unwrap()
        .into_iter()
        .find(|(path, _)| path == "predictor")
        .unwrap()
        .1
        .instruction();

    let mut graph = ProgramGraph::from_module(&typed).expect("projection should succeed");

    {
        let node = graph
            .nodes_mut()
            .get_mut("predictor")
            .expect("projected node");
        let mut predictors = node.module.predictors_mut();
        let (_, predictor) = predictors
            .iter_mut()
            .find(|(name, _)| *name == "predictor")
            .expect("dynamic predictor should exist");
        predictor.set_instruction("graph-updated".to_string());
    }

    let after_projection = named_parameters_ref(&typed)
        .unwrap()
        .into_iter()
        .find(|(path, _)| path == "predictor")
        .unwrap()
        .1
        .instruction();
    assert_eq!(after_projection, before);

    graph
        .fit(&mut typed)
        .expect("fit should apply projected state");

    let after_fit = named_parameters_ref(&typed)
        .unwrap()
        .into_iter()
        .find(|(path, _)| path == "predictor")
        .unwrap()
        .1
        .instruction();
    assert_eq!(after_fit, "graph-updated");
}
