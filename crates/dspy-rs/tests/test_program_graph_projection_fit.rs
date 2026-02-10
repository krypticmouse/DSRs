use dspy_rs::__macro_support::bamltype::facet;
use dspy_rs::{
    BamlValue, DynModule, DynPredictor, GraphError, LmError, Node, Predict, PredictError,
    Predicted, ProgramGraph, Signature, SignatureSchema, named_parameters_ref,
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
    predictor: Predict<QA>,
}

#[derive(facet::Facet)]
#[facet(crate = facet)]
struct PairWrapper {
    left: Predict<ProduceAnswer>,
    right: Predict<ConsumeAnswer>,
}

struct MultiLeafDynModule {
    schema: SignatureSchema,
    first: Predict<QA>,
    second: Predict<QA>,
}

impl MultiLeafDynModule {
    fn new() -> Self {
        Self {
            schema: SignatureSchema::of::<QA>().clone(),
            first: Predict::<QA>::new(),
            second: Predict::<QA>::new(),
        }
    }
}

#[async_trait::async_trait]
impl DynModule for MultiLeafDynModule {
    fn schema(&self) -> &SignatureSchema {
        &self.schema
    }

    fn predictors(&self) -> Vec<(&str, &dyn DynPredictor)> {
        vec![
            ("first", &self.first as &dyn DynPredictor),
            ("second", &self.second as &dyn DynPredictor),
        ]
    }

    fn predictors_mut(&mut self) -> Vec<(&str, &mut dyn DynPredictor)> {
        vec![
            ("first", &mut self.first as &mut dyn DynPredictor),
            ("second", &mut self.second as &mut dyn DynPredictor),
        ]
    }

    async fn forward(
        &self,
        _input: BamlValue,
    ) -> std::result::Result<Predicted<BamlValue>, PredictError> {
        Err(PredictError::Lm {
            source: LmError::Provider {
                provider: "test".to_string(),
                message: "unused".to_string(),
                source: None,
            },
        })
    }
}

#[derive(Signature, Clone, Debug, PartialEq, facet::Facet)]
#[facet(crate = facet)]
struct ProduceAnswer {
    #[input]
    question: String,

    #[output]
    answer: String,
}

#[derive(Signature, Clone, Debug, PartialEq, facet::Facet)]
#[facet(crate = facet)]
struct ConsumeAnswer {
    #[input]
    answer: String,

    #[output]
    final_answer: String,
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

#[test]
fn fit_errors_when_graph_is_missing_typed_predictor_path() {
    let mut typed = PairWrapper {
        left: Predict::<ProduceAnswer>::new(),
        right: Predict::<ConsumeAnswer>::new(),
    };

    let mut graph = ProgramGraph::from_module(&typed).expect("projection should succeed");
    graph.nodes_mut().shift_remove("right");

    let err = graph
        .fit(&mut typed)
        .expect_err("fit should fail when the graph omits a typed predictor path");
    assert!(matches!(
        err,
        GraphError::ProjectionMismatch { path, .. } if path == "right"
    ));
}

#[test]
fn fit_errors_when_graph_node_exposes_multiple_predictor_leaves() {
    let mut typed = Wrapper {
        predictor: Predict::<QA>::new(),
    };

    let mut graph = ProgramGraph::new();
    graph
        .add_node(
            "predictor",
            Node {
                schema: SignatureSchema::of::<QA>().clone(),
                module: Box::new(MultiLeafDynModule::new()),
            },
        )
        .expect("graph node insertion should succeed");

    let err = graph
        .fit(&mut typed)
        .expect_err("fit should reject malformed graph nodes");
    assert!(matches!(
        err,
        GraphError::ProjectionMismatch { path, .. } if path == "predictor"
    ));
}
