use dspy_rs::__macro_support::bamltype::facet;
use dspy_rs::{
    BamlValue, DynModule, DynPredictor, GraphError, LmError, Node, PredictError, Predicted,
    ProgramGraph, Signature, SignatureSchema,
};

#[derive(Signature, Clone, Debug, PartialEq, facet::Facet)]
#[facet(crate = facet)]
struct QuestionToAnswer {
    #[input]
    question: String,

    #[output]
    answer: String,
}

#[derive(Signature, Clone, Debug, PartialEq, facet::Facet)]
#[facet(crate = facet)]
struct AnswerToFinal {
    #[input]
    answer: String,

    #[output]
    final_answer: String,
}

#[derive(Signature, Clone, Debug, PartialEq, facet::Facet)]
#[facet(crate = facet)]
struct AnswerPassthrough {
    #[input]
    answer: String,

    #[output]
    answer_out: String,
}

#[derive(Signature, Clone, Debug, PartialEq, facet::Facet)]
#[facet(crate = facet)]
struct CountToFinal {
    #[input]
    count: i64,

    #[output]
    final_answer: String,
}

struct NoopDynModule {
    schema: SignatureSchema,
}

#[async_trait::async_trait]
impl DynModule for NoopDynModule {
    fn schema(&self) -> &SignatureSchema {
        &self.schema
    }

    fn predictors(&self) -> Vec<(&str, &dyn DynPredictor)> {
        Vec::new()
    }

    fn predictors_mut(&mut self) -> Vec<(&str, &mut dyn DynPredictor)> {
        Vec::new()
    }

    async fn forward(
        &self,
        _input: BamlValue,
    ) -> std::result::Result<Predicted<BamlValue>, PredictError> {
        Err(PredictError::Lm {
            source: LmError::Provider {
                provider: "test".to_string(),
                message: "noop".to_string(),
                source: None,
            },
        })
    }
}

fn node_for(schema: &SignatureSchema) -> Node {
    Node {
        schema: schema.clone(),
        module: Box::new(NoopDynModule {
            schema: schema.clone(),
        }),
    }
}

#[test]
fn program_graph_connect_rejects_type_mismatch() {
    let mut graph = ProgramGraph::new();
    graph
        .add_node("a", node_for(SignatureSchema::of::<QuestionToAnswer>()))
        .unwrap();
    graph
        .add_node("b", node_for(SignatureSchema::of::<CountToFinal>()))
        .unwrap();

    let err = graph
        .connect("a", "answer", "b", "count")
        .expect_err("incompatible edge should be rejected");
    assert!(matches!(err, GraphError::TypeMismatch { .. }));
}

#[test]
fn program_graph_replace_node_revalidates_incident_edges() {
    let mut graph = ProgramGraph::new();
    graph
        .add_node("a", node_for(SignatureSchema::of::<QuestionToAnswer>()))
        .unwrap();
    graph
        .add_node("b", node_for(SignatureSchema::of::<AnswerToFinal>()))
        .unwrap();
    graph.connect("a", "answer", "b", "answer").unwrap();

    let err = graph
        .replace_node("b", node_for(SignatureSchema::of::<CountToFinal>()))
        .expect_err("replacement should fail when existing edges become invalid");
    assert!(matches!(
        err,
        GraphError::TypeMismatch { .. } | GraphError::MissingField { .. }
    ));

    let b_node = graph.nodes().get("b").expect("original node should remain");
    assert!(
        b_node.schema.input_field_by_rust("answer").is_some(),
        "failed replacement must keep original node"
    );
    assert_eq!(graph.edges().len(), 1);
}

#[test]
fn program_graph_insert_between_rewires_edge_and_preserves_validity() {
    let mut graph = ProgramGraph::new();
    graph
        .add_node("a", node_for(SignatureSchema::of::<QuestionToAnswer>()))
        .unwrap();
    graph
        .add_node("b", node_for(SignatureSchema::of::<AnswerToFinal>()))
        .unwrap();
    graph.connect("a", "answer", "b", "answer").unwrap();

    graph
        .insert_between(
            "a",
            "b",
            "middle",
            node_for(SignatureSchema::of::<AnswerPassthrough>()),
            "answer",
            "answer",
        )
        .unwrap();

    assert_eq!(graph.edges().len(), 2);
    assert!(
        graph
            .edges()
            .iter()
            .any(|edge| edge.from_node == "a" && edge.to_node == "middle")
    );
    assert!(
        graph
            .edges()
            .iter()
            .any(|edge| edge.from_node == "middle" && edge.to_node == "b")
    );
    assert!(
        graph
            .edges()
            .iter()
            .all(|edge| !(edge.from_node == "a" && edge.to_node == "b"))
    );
}

#[test]
fn program_graph_insert_between_missing_fields_is_atomic() {
    let mut graph = ProgramGraph::new();
    graph
        .add_node("a", node_for(SignatureSchema::of::<QuestionToAnswer>()))
        .unwrap();
    graph
        .add_node("b", node_for(SignatureSchema::of::<AnswerToFinal>()))
        .unwrap();
    graph.connect("a", "answer", "b", "answer").unwrap();

    let passthrough = SignatureSchema::of::<AnswerPassthrough>();
    let missing_input_schema =
        passthrough.with_fields(Vec::new(), passthrough.output_fields().to_vec());
    let err = graph
        .insert_between(
            "a",
            "b",
            "bad_middle",
            node_for(&missing_input_schema),
            "answer",
            "answer",
        )
        .expect_err("insert_between should fail when inserted node has no input");
    assert!(matches!(err, GraphError::ProjectionMismatch { .. }));

    assert!(
        graph.nodes().contains_key("a") && graph.nodes().contains_key("b"),
        "original nodes should remain"
    );
    assert!(
        !graph.nodes().contains_key("bad_middle"),
        "failed insert must not leave inserted node behind"
    );
    assert_eq!(graph.edges().len(), 1);
    assert!(
        graph
            .edges()
            .iter()
            .any(|edge| edge.from_node == "a" && edge.to_node == "b")
    );
}
