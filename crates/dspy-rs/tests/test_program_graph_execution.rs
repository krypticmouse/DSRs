use std::sync::LazyLock;

use dspy_rs::__macro_support::bamltype::facet;
use dspy_rs::__macro_support::indexmap::IndexMap;
use dspy_rs::{
    BamlType, BamlValue, CallMetadata, ChainOfThought, ChatAdapter, DynModule, DynPredictor,
    GraphError, LMClient, Node, Predict, PredictError, Predicted, ProgramGraph, Signature,
    SignatureSchema, TestCompletionModel, configure, registry,
};
use rig::completion::{
    AssistantContent, CompletionRequest, Message as RigMessage, message::UserContent,
};
use rig::message::Text;
use tokio::sync::Mutex;

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
struct AnswerToEnriched {
    #[input]
    answer: String,

    #[output]
    enriched: String,
}

#[derive(Signature, Clone, Debug, PartialEq, facet::Facet)]
#[facet(crate = facet)]
struct EnrichedToFinal {
    #[input]
    enriched: String,

    #[output]
    final_answer: String,
}

struct EchoDynModule {
    schema: SignatureSchema,
}

static SETTINGS_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

fn response_with_fields(fields: &[(&str, &str)]) -> String {
    let mut response = String::new();
    for (name, value) in fields {
        response.push_str(&format!("[[ ## {name} ## ]]\n{value}\n\n"));
    }
    response.push_str("[[ ## completed ## ]]\n");
    response
}

fn text_response(text: impl Into<String>) -> AssistantContent {
    AssistantContent::Text(Text { text: text.into() })
}

async fn configure_test_lm(client: TestCompletionModel) {
    unsafe {
        std::env::set_var("OPENAI_API_KEY", "test");
    }

    let lm = dspy_rs::LM::builder()
        .model("openai:gpt-4o-mini".to_string())
        .build()
        .await
        .unwrap()
        .with_client(LMClient::Test(client))
        .await
        .unwrap();

    configure(lm, ChatAdapter {});
}

fn request_system(request: &CompletionRequest) -> String {
    request.preamble.clone().unwrap_or_default()
}

fn request_user_prompt(request: &CompletionRequest) -> String {
    let prompt = request
        .chat_history
        .iter()
        .last()
        .expect("completion request should include a prompt message");

    match prompt {
        RigMessage::User { content } => content
            .iter()
            .find_map(|entry| match entry {
                UserContent::Text(text) => Some(text.text.clone()),
                _ => None,
            })
            .unwrap_or_default(),
        other => panic!("expected prompt to be user message, got: {other:?}"),
    }
}

#[async_trait::async_trait]
impl DynModule for EchoDynModule {
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
        input: BamlValue,
    ) -> std::result::Result<Predicted<BamlValue>, PredictError> {
        let input_field = self
            .schema
            .input_fields()
            .first()
            .expect("test schema must have one input field");
        let output_field = self
            .schema
            .output_fields()
            .first()
            .expect("test schema must have one output field");

        let value = self
            .schema
            .navigate_field(input_field.path(), &input)
            .cloned()
            .unwrap_or(BamlValue::Null);

        let mut out = IndexMap::new();
        insert_baml_at_path(&mut out, output_field.path(), value);

        Ok(Predicted::new(
            BamlValue::Class("EchoOutput".to_string(), out),
            CallMetadata::default(),
        ))
    }
}

fn node_for(schema: &SignatureSchema) -> Node {
    Node {
        schema: schema.clone(),
        module: Box::new(EchoDynModule {
            schema: schema.clone(),
        }),
    }
}

#[tokio::test]
async fn program_graph_execute_routes_fields_topologically() {
    let mut graph = ProgramGraph::new();
    graph
        .add_node("a", node_for(SignatureSchema::of::<QuestionToAnswer>()))
        .unwrap();
    graph
        .add_node("b", node_for(SignatureSchema::of::<AnswerToEnriched>()))
        .unwrap();
    graph
        .add_node("c", node_for(SignatureSchema::of::<EnrichedToFinal>()))
        .unwrap();

    graph.connect("a", "answer", "b", "answer").unwrap();
    graph.connect("b", "enriched", "c", "enriched").unwrap();

    let input = BamlValue::Class(
        "QuestionToAnswerInput".to_string(),
        IndexMap::from([(
            "question".to_string(),
            BamlValue::String("smoke-ok".to_string()),
        )]),
    );

    let output = graph
        .execute(input)
        .await
        .expect("execution should succeed");
    let output_field = graph
        .nodes()
        .get("c")
        .unwrap()
        .schema
        .output_field_by_rust("final_answer")
        .unwrap();
    let final_value = graph
        .nodes()
        .get("c")
        .unwrap()
        .schema
        .navigate_field(output_field.path(), &output)
        .unwrap();

    assert_eq!(final_value, &BamlValue::String("smoke-ok".to_string()));
}

#[tokio::test]
async fn program_graph_execute_cycle_errors() {
    let mut graph = ProgramGraph::new();
    graph
        .add_node("a", node_for(SignatureSchema::of::<QuestionToAnswer>()))
        .unwrap();
    graph
        .add_node("b", node_for(SignatureSchema::of::<AnswerToEnriched>()))
        .unwrap();

    graph.connect("a", "answer", "b", "answer").unwrap();
    graph.connect("b", "enriched", "a", "question").unwrap();

    let input = BamlValue::Class(
        "QuestionToAnswerInput".to_string(),
        IndexMap::from([("question".to_string(), BamlValue::String("x".to_string()))]),
    );

    let err = graph
        .execute(input)
        .await
        .expect_err("cycle should fail before execution");
    assert!(matches!(err, GraphError::Cycle));
}

#[tokio::test]
async fn program_graph_execute_errors_when_graph_has_no_sink() {
    let graph = ProgramGraph::new();
    let input = BamlValue::Class("EmptyInput".to_string(), IndexMap::new());

    let err = graph
        .execute(input)
        .await
        .expect_err("empty graph should not have a sink");
    assert!(matches!(err, GraphError::NoSink));
}

#[tokio::test]
async fn program_graph_execute_errors_when_graph_has_multiple_sinks() {
    let mut graph = ProgramGraph::new();
    graph
        .add_node("a", node_for(SignatureSchema::of::<QuestionToAnswer>()))
        .unwrap();
    graph
        .add_node("b", node_for(SignatureSchema::of::<AnswerToEnriched>()))
        .unwrap();

    let input = BamlValue::Class(
        "QuestionToAnswerInput".to_string(),
        IndexMap::from([(
            "question".to_string(),
            BamlValue::String("ambiguous".to_string()),
        )]),
    );

    let err = graph
        .execute(input)
        .await
        .expect_err("disconnected graph should produce ambiguous sinks");
    assert!(matches!(err, GraphError::AmbiguousSink { .. }));
}

#[tokio::test]
async fn program_graph_execute_accepts_input_pseudonode_edges() {
    let mut graph = ProgramGraph::new();
    graph
        .add_node("a", node_for(SignatureSchema::of::<QuestionToAnswer>()))
        .unwrap();
    graph.connect("input", "question", "a", "question").unwrap();

    let input = BamlValue::Class(
        "QuestionToAnswerInput".to_string(),
        IndexMap::from([(
            "question".to_string(),
            BamlValue::String("via-input".to_string()),
        )]),
    );

    let output = graph
        .execute(input)
        .await
        .expect("execution with input pseudo-node should succeed");
    let output_field = graph
        .nodes()
        .get("a")
        .unwrap()
        .schema
        .output_field_by_rust("answer")
        .unwrap();
    let answer = graph
        .nodes()
        .get("a")
        .unwrap()
        .schema
        .navigate_field(output_field.path(), &output)
        .unwrap();
    assert_eq!(answer, &BamlValue::String("via-input".to_string()));
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn typed_dynamic_prompt_parity_for_predict_and_chain_of_thought() {
    let _lock = SETTINGS_LOCK.lock().await;

    let schema = SignatureSchema::of::<QuestionToAnswer>();
    let typed_input = QuestionToAnswerInput {
        question: "What is 2 + 2?".to_string(),
    };
    let dynamic_input = typed_input.to_baml_value();

    let predict_response = response_with_fields(&[("answer", "4")]);
    let typed_predict_client =
        TestCompletionModel::new(vec![text_response(predict_response.clone())]);
    configure_test_lm(typed_predict_client.clone()).await;
    let typed_predict = Predict::<QuestionToAnswer>::new();
    typed_predict
        .call(typed_input.clone())
        .await
        .expect("typed predict call should succeed");
    let typed_predict_request = typed_predict_client
        .last_request()
        .expect("typed predict request should be captured");

    let dynamic_predict_client =
        TestCompletionModel::new(vec![text_response(predict_response.clone())]);
    configure_test_lm(dynamic_predict_client.clone()).await;
    let dynamic_predict = registry::create("predict", schema, serde_json::json!({}))
        .expect("predict strategy should create");
    dynamic_predict
        .forward(dynamic_input.clone())
        .await
        .expect("dynamic predict call should succeed");
    let dynamic_predict_request = dynamic_predict_client
        .last_request()
        .expect("dynamic predict request should be captured");

    assert_eq!(
        request_system(&typed_predict_request),
        request_system(&dynamic_predict_request)
    );
    assert_eq!(
        request_user_prompt(&typed_predict_request),
        request_user_prompt(&dynamic_predict_request)
    );

    let cot_response = response_with_fields(&[("reasoning", "step-by-step"), ("answer", "4")]);
    let typed_cot_client = TestCompletionModel::new(vec![text_response(cot_response.clone())]);
    configure_test_lm(typed_cot_client.clone()).await;
    let typed_cot = ChainOfThought::<QuestionToAnswer>::new();
    typed_cot
        .call(typed_input)
        .await
        .expect("typed chain_of_thought call should succeed");
    let typed_cot_request = typed_cot_client
        .last_request()
        .expect("typed chain_of_thought request should be captured");

    let dynamic_cot_client = TestCompletionModel::new(vec![text_response(cot_response)]);
    configure_test_lm(dynamic_cot_client.clone()).await;
    let dynamic_cot = registry::create("chain_of_thought", schema, serde_json::json!({}))
        .expect("chain_of_thought strategy should create");
    dynamic_cot
        .forward(dynamic_input)
        .await
        .expect("dynamic chain_of_thought call should succeed");
    let dynamic_cot_request = dynamic_cot_client
        .last_request()
        .expect("dynamic chain_of_thought request should be captured");

    assert_eq!(
        request_system(&typed_cot_request),
        request_system(&dynamic_cot_request)
    );
    assert_eq!(
        request_user_prompt(&typed_cot_request),
        request_user_prompt(&dynamic_cot_request)
    );
}

fn insert_baml_at_path(
    root: &mut IndexMap<String, BamlValue>,
    path: &dspy_rs::FieldPath,
    value: BamlValue,
) {
    let parts: Vec<_> = path.iter().collect();
    if parts.is_empty() {
        return;
    }
    insert_baml_at_parts(root, &parts, value);
}

fn insert_baml_at_parts(
    root: &mut IndexMap<String, BamlValue>,
    parts: &[&'static str],
    value: BamlValue,
) {
    if parts.len() == 1 {
        root.insert(parts[0].to_string(), value);
        return;
    }

    let entry = root
        .entry(parts[0].to_string())
        .or_insert_with(|| BamlValue::Map(IndexMap::new()));
    if !matches!(entry, BamlValue::Map(_) | BamlValue::Class(_, _)) {
        *entry = BamlValue::Map(IndexMap::new());
    }
    let child = match entry {
        BamlValue::Map(map) | BamlValue::Class(_, map) => map,
        _ => unreachable!(),
    };

    insert_baml_at_parts(child, &parts[1..], value);
}
