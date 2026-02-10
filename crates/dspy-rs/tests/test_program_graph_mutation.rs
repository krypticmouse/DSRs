use dspy_rs::__macro_support::bamltype::facet;
use dspy_rs::{
    BamlType, BamlValue, DynModule, DynPredictor, GraphError, LmError, Node, PredictError,
    Predicted, ProgramGraph, Signature, SignatureSchema, TypeIR,
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

#[derive(Signature, Clone, Debug, PartialEq, facet::Facet)]
#[facet(crate = facet)]
struct OptionalAnswerToFinal {
    #[input]
    answer: Option<String>,

    #[output]
    final_answer: String,
}

#[derive(Signature, Clone, Debug, PartialEq, facet::Facet)]
#[facet(crate = facet)]
struct QuestionToOptionalAnswer {
    #[input]
    question: String,

    #[output]
    answer: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[BamlType]
struct AnswerPayload {
    text: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[BamlType]
struct AlternatePayload {
    text: String,
}

#[derive(Signature, Clone, Debug, PartialEq, facet::Facet)]
#[facet(crate = facet)]
struct QuestionToPayload {
    #[input]
    question: String,

    #[output]
    payload: AnswerPayload,
}

#[derive(Signature, Clone, Debug, PartialEq, facet::Facet)]
#[facet(crate = facet)]
struct PayloadToFinal {
    #[input]
    payload: AnswerPayload,

    #[output]
    final_answer: String,
}

#[derive(Signature, Clone, Debug, PartialEq, facet::Facet)]
#[facet(crate = facet)]
struct AlternatePayloadToFinal {
    #[input]
    payload: AlternatePayload,

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

fn schema_with_input_type(
    schema: &'static SignatureSchema,
    rust_name: &str,
    type_ir: TypeIR,
) -> SignatureSchema {
    let mut input_fields = schema.input_fields().to_vec();
    let field = input_fields
        .iter_mut()
        .find(|field| field.rust_name == rust_name)
        .unwrap_or_else(|| panic!("input field `{rust_name}` not found"));
    field.type_ir = type_ir;
    schema.with_fields(input_fields, schema.output_fields().to_vec())
}

fn schema_with_output_type(
    schema: &'static SignatureSchema,
    rust_name: &str,
    type_ir: TypeIR,
) -> SignatureSchema {
    let mut output_fields = schema.output_fields().to_vec();
    let field = output_fields
        .iter_mut()
        .find(|field| field.rust_name == rust_name)
        .unwrap_or_else(|| panic!("output field `{rust_name}` not found"));
    field.type_ir = type_ir;
    schema.with_fields(schema.input_fields().to_vec(), output_fields)
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
fn program_graph_connect_accepts_non_optional_output_into_optional_input() {
    let mut graph = ProgramGraph::new();
    graph
        .add_node("a", node_for(SignatureSchema::of::<QuestionToAnswer>()))
        .unwrap();
    graph
        .add_node("b", node_for(SignatureSchema::of::<OptionalAnswerToFinal>()))
        .unwrap();

    graph
        .connect("a", "answer", "b", "answer")
        .expect("string should flow into optional string input");
}

#[test]
fn program_graph_connect_rejects_optional_output_into_non_optional_input() {
    let mut graph = ProgramGraph::new();
    graph
        .add_node("a", node_for(SignatureSchema::of::<QuestionToOptionalAnswer>()))
        .unwrap();
    graph
        .add_node("b", node_for(SignatureSchema::of::<AnswerToFinal>()))
        .unwrap();

    let err = graph
        .connect("a", "answer", "b", "answer")
        .expect_err("optional output should not flow into non-optional input");
    assert!(matches!(err, GraphError::TypeMismatch { .. }));
}

#[test]
fn program_graph_connect_requires_matching_custom_payload_labels() {
    let mut graph = ProgramGraph::new();
    graph
        .add_node("a", node_for(SignatureSchema::of::<QuestionToPayload>()))
        .unwrap();
    graph
        .add_node("b", node_for(SignatureSchema::of::<PayloadToFinal>()))
        .unwrap();
    graph
        .connect("a", "payload", "b", "payload")
        .expect("matching payload classes should be assignable");

    graph
        .add_node("c", node_for(SignatureSchema::of::<AlternatePayloadToFinal>()))
        .unwrap();
    let err = graph
        .connect("a", "payload", "c", "payload")
        .expect_err("different payload labels should not be assignable");
    assert!(matches!(err, GraphError::TypeMismatch { .. }));
}

#[test]
fn program_graph_connect_allows_same_list_types() {
    let mut graph = ProgramGraph::new();
    let list_output_schema = schema_with_output_type(
        SignatureSchema::of::<QuestionToAnswer>(),
        "answer",
        TypeIR::list(TypeIR::string()),
    );
    let list_input_schema = schema_with_input_type(
        SignatureSchema::of::<AnswerToFinal>(),
        "answer",
        TypeIR::list(TypeIR::string()),
    );
    graph.add_node("a", node_for(&list_output_schema)).unwrap();
    graph.add_node("b", node_for(&list_input_schema)).unwrap();

    graph
        .connect("a", "answer", "b", "answer")
        .expect("list<string> should be assignable to list<string>");
}

#[test]
fn program_graph_connect_rejects_list_element_mismatch() {
    let mut graph = ProgramGraph::new();
    let list_output_schema = schema_with_output_type(
        SignatureSchema::of::<QuestionToAnswer>(),
        "answer",
        TypeIR::list(TypeIR::string()),
    );
    let list_input_schema = schema_with_input_type(
        SignatureSchema::of::<AnswerToFinal>(),
        "answer",
        TypeIR::list(TypeIR::int()),
    );
    graph.add_node("a", node_for(&list_output_schema)).unwrap();
    graph.add_node("b", node_for(&list_input_schema)).unwrap();

    let err = graph
        .connect("a", "answer", "b", "answer")
        .expect_err("list element mismatch should be rejected");
    assert!(matches!(err, GraphError::TypeMismatch { .. }));
}

#[test]
fn program_graph_connect_requires_matching_map_key_value_types() {
    let mut graph = ProgramGraph::new();
    let map_output_schema = schema_with_output_type(
        SignatureSchema::of::<QuestionToAnswer>(),
        "answer",
        TypeIR::map(TypeIR::string(), TypeIR::string()),
    );
    let map_input_schema = schema_with_input_type(
        SignatureSchema::of::<AnswerToFinal>(),
        "answer",
        TypeIR::map(TypeIR::string(), TypeIR::string()),
    );
    graph.add_node("a", node_for(&map_output_schema)).unwrap();
    graph.add_node("b", node_for(&map_input_schema)).unwrap();
    graph
        .connect("a", "answer", "b", "answer")
        .expect("map<string, string> should be assignable to map<string, string>");

    let mut mismatch_graph = ProgramGraph::new();
    let map_input_mismatch_schema = schema_with_input_type(
        SignatureSchema::of::<AnswerToFinal>(),
        "answer",
        TypeIR::map(TypeIR::string(), TypeIR::int()),
    );
    mismatch_graph
        .add_node("a", node_for(&map_output_schema))
        .unwrap();
    mismatch_graph
        .add_node("b", node_for(&map_input_mismatch_schema))
        .unwrap();
    let err = mismatch_graph
        .connect("a", "answer", "b", "answer")
        .expect_err("map value-type mismatch should be rejected");
    assert!(matches!(err, GraphError::TypeMismatch { .. }));
}

#[test]
fn program_graph_connect_rejects_tuple_length_or_type_mismatch() {
    let mut graph = ProgramGraph::new();
    let tuple_output_schema = schema_with_output_type(
        SignatureSchema::of::<QuestionToAnswer>(),
        "answer",
        TypeIR::tuple(vec![TypeIR::string(), TypeIR::int()]),
    );
    let tuple_input_schema = schema_with_input_type(
        SignatureSchema::of::<AnswerToFinal>(),
        "answer",
        TypeIR::tuple(vec![TypeIR::string(), TypeIR::int()]),
    );
    graph.add_node("a", node_for(&tuple_output_schema)).unwrap();
    graph.add_node("b", node_for(&tuple_input_schema)).unwrap();
    graph
        .connect("a", "answer", "b", "answer")
        .expect("matching tuple arity and element types should connect");

    let mut tuple_type_graph = ProgramGraph::new();
    let tuple_type_mismatch_schema = schema_with_input_type(
        SignatureSchema::of::<AnswerToFinal>(),
        "answer",
        TypeIR::tuple(vec![TypeIR::string(), TypeIR::string()]),
    );
    tuple_type_graph
        .add_node("a", node_for(&tuple_output_schema))
        .unwrap();
    tuple_type_graph
        .add_node("b", node_for(&tuple_type_mismatch_schema))
        .unwrap();
    let type_err = tuple_type_graph
        .connect("a", "answer", "b", "answer")
        .expect_err("tuple element mismatch should be rejected");
    assert!(matches!(type_err, GraphError::TypeMismatch { .. }));

    let mut tuple_len_graph = ProgramGraph::new();
    let tuple_len_mismatch_schema = schema_with_input_type(
        SignatureSchema::of::<AnswerToFinal>(),
        "answer",
        TypeIR::tuple(vec![TypeIR::string(), TypeIR::int(), TypeIR::bool()]),
    );
    tuple_len_graph
        .add_node("a", node_for(&tuple_output_schema))
        .unwrap();
    tuple_len_graph
        .add_node("b", node_for(&tuple_len_mismatch_schema))
        .unwrap();
    let len_err = tuple_len_graph
        .connect("a", "answer", "b", "answer")
        .expect_err("tuple length mismatch should be rejected");
    assert!(matches!(len_err, GraphError::TypeMismatch { .. }));
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
