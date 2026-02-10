use dspy_rs::__macro_support::bamltype::facet;
use dspy_rs::{
    GraphEdgeAnnotation, Predict, ProgramGraph, Signature, register_graph_edge_annotations,
};

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

#[derive(Signature, Clone, Debug, PartialEq, facet::Facet)]
#[facet(crate = facet)]
struct ConsumeCount {
    #[input]
    count: i64,

    #[output]
    final_count: i64,
}

#[derive(facet::Facet)]
#[facet(crate = facet)]
struct AnnotatedModule {
    source: Predict<ProduceAnswer>,
    sink: Predict<ConsumeAnswer>,
}

#[derive(facet::Facet)]
#[facet(crate = facet)]
struct PlainModule {
    source: Predict<ProduceAnswer>,
    sink: Predict<ConsumeAnswer>,
}

#[derive(facet::Facet)]
#[facet(crate = facet)]
struct UnresolvableModule {
    source: Predict<ProduceAnswer>,
    sink: Predict<ConsumeCount>,
}

static EDGE_ANNOTATIONS: &[GraphEdgeAnnotation] = &[GraphEdgeAnnotation {
    from_node: "source",
    from_field: "answer",
    to_node: "sink",
    to_field: "answer",
}];

#[test]
fn from_module_prefers_annotations_and_falls_back_to_inference() {
    register_graph_edge_annotations(
        <AnnotatedModule as facet::Facet<'static>>::SHAPE,
        EDGE_ANNOTATIONS,
    );

    let annotated = AnnotatedModule {
        source: Predict::<ProduceAnswer>::new(),
        sink: Predict::<ConsumeAnswer>::new(),
    };
    let graph = ProgramGraph::from_module(&annotated).expect("projection should succeed");
    assert_eq!(graph.edges().len(), 1);
    assert_eq!(graph.edges()[0].from_node, "source");
    assert_eq!(graph.edges()[0].to_node, "sink");

    let plain = PlainModule {
        source: Predict::<ProduceAnswer>::new(),
        sink: Predict::<ConsumeAnswer>::new(),
    };
    let plain_graph = ProgramGraph::from_module(&plain).expect("projection should succeed");
    assert_eq!(plain_graph.edges().len(), 1);
    assert_eq!(plain_graph.edges()[0].from_node, "source");
    assert_eq!(plain_graph.edges()[0].from_field, "answer");
    assert_eq!(plain_graph.edges()[0].to_node, "sink");
    assert_eq!(plain_graph.edges()[0].to_field, "answer");
}

#[test]
fn from_module_errors_when_multi_node_edges_cannot_be_inferred() {
    let module = UnresolvableModule {
        source: Predict::<ProduceAnswer>::new(),
        sink: Predict::<ConsumeCount>::new(),
    };

    let err = ProgramGraph::from_module(&module)
        .expect_err("projection should fail when no edges can be resolved");
    assert!(matches!(
        err,
        dspy_rs::GraphError::ProjectionMismatch { .. }
    ));
}
