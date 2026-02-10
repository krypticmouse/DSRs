use dspy_rs::__macro_support::bamltype::facet;
use dspy_rs::{GraphEdgeAnnotation, GraphError, Predict, ProgramGraph, Signature};

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

fn source_to_sink_annotation() -> GraphEdgeAnnotation {
    GraphEdgeAnnotation {
        from_node: "source".to_string(),
        from_field: "answer".to_string(),
        to_node: "sink".to_string(),
        to_field: "answer".to_string(),
    }
}

#[test]
fn from_module_with_annotations_applies_edges_without_global_state() {
    let module = PlainModule {
        source: Predict::<ProduceAnswer>::new(),
        sink: Predict::<ConsumeAnswer>::new(),
    };

    let graph = ProgramGraph::from_module_with_annotations(&module, &[source_to_sink_annotation()])
        .expect("projection with explicit annotations should succeed");

    assert_eq!(graph.edges().len(), 1);
    assert_eq!(graph.edges()[0].from_node, "source");
    assert_eq!(graph.edges()[0].from_field, "answer");
    assert_eq!(graph.edges()[0].to_node, "sink");
    assert_eq!(graph.edges()[0].to_field, "answer");
}

#[test]
fn from_module_with_annotations_rejects_invalid_field_paths() {
    let module = PlainModule {
        source: Predict::<ProduceAnswer>::new(),
        sink: Predict::<ConsumeAnswer>::new(),
    };

    let annotations = [GraphEdgeAnnotation {
        from_node: "source".to_string(),
        from_field: "answer".to_string(),
        to_node: "sink".to_string(),
        to_field: "missing".to_string(),
    }];

    let err = ProgramGraph::from_module_with_annotations(&module, &annotations)
        .expect_err("invalid annotation path should fail projection");
    assert!(matches!(err, GraphError::MissingField { .. }));
}

#[test]
fn from_module_without_annotations_falls_back_to_inference() {
    let module = PlainModule {
        source: Predict::<ProduceAnswer>::new(),
        sink: Predict::<ConsumeAnswer>::new(),
    };

    let graph = ProgramGraph::from_module(&module).expect("projection should succeed");
    assert_eq!(graph.edges().len(), 1);
    assert_eq!(graph.edges()[0].from_node, "source");
    assert_eq!(graph.edges()[0].from_field, "answer");
    assert_eq!(graph.edges()[0].to_node, "sink");
    assert_eq!(graph.edges()[0].to_field, "answer");
}

#[test]
fn from_module_with_empty_annotations_falls_back_to_inference() {
    let module = PlainModule {
        source: Predict::<ProduceAnswer>::new(),
        sink: Predict::<ConsumeAnswer>::new(),
    };

    let inferred = ProgramGraph::from_module(&module).expect("projection should succeed");
    let explicit_empty = ProgramGraph::from_module_with_annotations(&module, &[])
        .expect("projection with empty annotations should still infer edges");

    assert_eq!(explicit_empty.edges(), inferred.edges());
}

#[test]
fn projection_is_deterministic_across_repeated_calls_without_registration() {
    let module = PlainModule {
        source: Predict::<ProduceAnswer>::new(),
        sink: Predict::<ConsumeAnswer>::new(),
    };

    let graph_a = ProgramGraph::from_module(&module).expect("first projection should succeed");
    let graph_b = ProgramGraph::from_module(&module).expect("second projection should succeed");

    assert_eq!(graph_a.edges(), graph_b.edges());
}

#[test]
fn from_module_errors_when_multi_node_edges_cannot_be_inferred() {
    let module = UnresolvableModule {
        source: Predict::<ProduceAnswer>::new(),
        sink: Predict::<ConsumeCount>::new(),
    };

    let err = ProgramGraph::from_module(&module)
        .expect_err("projection should fail when no edges can be resolved");
    assert!(matches!(err, GraphError::ProjectionMismatch { .. }));
}
