use std::sync::LazyLock;

use dspy_rs::__macro_support::bamltype::facet;
use dspy_rs::{
    BamlType, ChatAdapter, LM, LMClient, ProgramGraph, Signature, SignatureSchema, StrategyError,
    TestCompletionModel, configure, registry,
};
use rig::completion::AssistantContent;
use rig::message::Text;
use tokio::sync::Mutex;

#[derive(Signature, Clone, Debug, PartialEq, facet::Facet)]
#[facet(crate = facet)]
struct QA {
    #[input]
    question: String,

    #[output]
    answer: String,
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

#[test]
fn registry_list_contains_predict_chain_of_thought_react() {
    let strategies = registry::list();
    assert!(strategies.contains(&"predict"));
    assert!(strategies.contains(&"chain_of_thought"));
    assert!(strategies.contains(&"react"));
}

#[test]
fn registry_create_instantiates_builtins() {
    let schema = SignatureSchema::of::<QA>();

    let predict = registry::create("predict", schema, serde_json::json!({}))
        .expect("predict factory should build");
    assert!(!predict.predictors().is_empty());

    let cot = registry::create("chain_of_thought", schema, serde_json::json!({}))
        .expect("chain_of_thought factory should build");
    assert!(!cot.predictors().is_empty());

    let react = registry::create("react", schema, serde_json::json!({ "max_steps": 2 }))
        .expect("react factory should build");
    let react_predictors = react
        .predictors()
        .into_iter()
        .map(|(name, _)| name)
        .collect::<Vec<_>>();
    assert_eq!(react_predictors, vec!["action", "extract"]);

    let mut graph = ProgramGraph::new();
    graph
        .add_node(
            "react_node",
            registry::create("react", schema, serde_json::json!({ "max_steps": 1 }))
                .expect("react strategy should create"),
        )
        .expect("graph should accept registry module directly");
}

#[test]
fn registry_create_rejects_invalid_react_config() {
    let schema = SignatureSchema::of::<QA>();
    let result = registry::create(
        "react",
        schema,
        serde_json::json!({ "max_steps": "invalid" }),
    );
    match result {
        Ok(_) => panic!("react should reject non-integer max_steps"),
        Err(err) => assert!(matches!(
            err,
            StrategyError::InvalidConfig { strategy, .. } if strategy == "react"
        )),
    }
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn react_factory_runs_action_then_extract_loop() {
    let _lock = SETTINGS_LOCK.lock().await;
    unsafe {
        std::env::set_var("OPENAI_API_KEY", "test");
    }

    let action_response = response_with_fields(&[
        ("thought", "done"),
        ("action", "finish"),
        ("action_input", "ready"),
    ]);
    let extract_response = response_with_fields(&[("answer", "4")]);
    let client = TestCompletionModel::new(vec![
        text_response(action_response),
        text_response(extract_response),
    ]);

    let lm = LM::builder()
        .model("openai:gpt-4o-mini".to_string())
        .build()
        .await
        .unwrap()
        .with_client(LMClient::Test(client))
        .await
        .unwrap();
    configure(lm, ChatAdapter {});

    let schema = SignatureSchema::of::<QA>();
    let react = registry::create("react", schema, serde_json::json!({ "max_steps": 2 }))
        .expect("react factory should build");
    let input = QAInput {
        question: "2+2?".to_string(),
    }
    .to_baml_value();

    let output = react
        .forward(input)
        .await
        .expect("react dynamic module should execute action then extract")
        .into_inner();
    let answer_field = schema.output_field_by_rust("answer").unwrap();
    let answer = schema
        .navigate_field(answer_field.path(), &output)
        .expect("answer field should exist");
    assert_eq!(answer, &dspy_rs::BamlValue::String("4".to_string()));
}
