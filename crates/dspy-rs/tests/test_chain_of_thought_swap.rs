use dspy_rs::{
    ChainOfThought, ChatAdapter, LM, LMClient, Module, Optimizable, Predict, Reasoning, Signature,
    TestCompletionModel, WithReasoning, configure,
};
use rig::completion::AssistantContent;
use rig::message::Text;
use std::sync::LazyLock;
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

#[derive(Signature, Clone, Debug, PartialEq)]
struct QA {
    #[input]
    question: String,

    #[output]
    answer: String,
}

type QAOutput = __QAOutput;

fn accepts_module<M: Module>(_: &M) {}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn chain_of_thought_swaps_and_returns_with_reasoning() {
    let _lock = SETTINGS_LOCK.lock().await;
    let response = response_with_fields(&[("reasoning", "Think"), ("answer", "Paris")]);
    configure_test_lm(vec![response]).await;

    let _builder = ChainOfThought::<QA>::builder()
        .instruction("Be concise")
        .build();

    let cot = ChainOfThought::<QA>::new();
    accepts_module(&cot);

    let input = QAInput {
        question: "What is the capital of France?".to_string(),
    };
    let outcome = cot.call(input).await;
    let result: WithReasoning<QAOutput> = outcome.into_result().unwrap();

    assert_eq!(result.reasoning, "Think");
    assert_eq!(result.answer, "Paris");

    let _predict = Predict::<dspy_rs::Augmented<QA, Reasoning>>::new();
}

#[test]
fn chain_of_thought_parameters_expose_predictor_for_legacy_optimizers() {
    let mut cot = ChainOfThought::<QA>::new();
    let mut params = cot.parameters();

    let names: Vec<String> = params.keys().cloned().collect();
    assert_eq!(names, vec!["predictor".to_string()]);

    let predictor = params
        .get_mut("predictor")
        .expect("ChainOfThought parameters should expose wrapped predictor");
    predictor
        .update_signature_instruction("updated instruction".to_string())
        .unwrap();

    assert_eq!(
        predictor.get_signature().instruction(),
        "updated instruction"
    );
}
