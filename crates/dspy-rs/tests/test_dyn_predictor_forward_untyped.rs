use std::sync::LazyLock;

use dspy_rs::{
    BamlType, ChatAdapter, LM, LMClient, Predict, Signature, TestCompletionModel, configure,
    named_parameters,
};
use dspy_rs::__macro_support::bamltype::facet;
use rig::completion::AssistantContent;
use rig::message::Text;
use tokio::sync::Mutex;

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

async fn configure_test_lm(responses: Vec<String>) {
    unsafe {
        std::env::set_var("OPENAI_API_KEY", "test");
    }

    let client = TestCompletionModel::new(responses.into_iter().map(text_response));
    let lm = LM::builder()
        .model("openai:gpt-4o-mini".to_string())
        .build()
        .await
        .unwrap()
        .with_client(LMClient::Test(client))
        .await
        .unwrap();

    configure(lm, ChatAdapter {});
}

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

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn dyn_predictor_forward_untyped_returns_baml_and_metadata() {
    let _lock = SETTINGS_LOCK.lock().await;
    let response = response_with_fields(&[("answer", "Paris")]);
    configure_test_lm(vec![response.clone(), response]).await;

    let mut module = Wrapper {
        predictor: Predict::<QA>::new(),
    };
    let input = QAInput {
        question: "What is the capital of France?".to_string(),
    };
    let untyped_input = input.to_baml_value();

    let untyped = {
        let mut params = named_parameters(&mut module).expect("walker should find predictor");
        let (_, predictor) = params
            .iter_mut()
            .find(|(name, _)| name == "predictor")
            .expect("predictor should exist");
        predictor
            .forward_untyped(untyped_input)
            .await
            .expect("untyped call should succeed")
    };
    let typed = module
        .predictor
        .call(input)
        .await
        .expect("typed call should succeed");

    let (untyped_output, untyped_meta) = untyped.into_parts();
    let (typed_output, typed_meta) = typed.into_parts();

    let untyped_output = QAOutput::try_from_baml_value(untyped_output)
        .expect("untyped output should roundtrip to QAOutput");
    assert_eq!(untyped_output.answer, typed_output.answer);
    assert!(!untyped_meta.raw_response.is_empty());
    assert_eq!(untyped_meta.raw_response, typed_meta.raw_response);
}
