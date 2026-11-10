use dspy_rs::{LM, LMClient, Signature, TestCompletionModel, with_lm};
use rig::completion::AssistantContent;

/// Answer the question and explain the calculation.
#[dspy_rs::cot]
fn explain(question: String) -> String;

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn cot_attribute_produces_typed_reasoning_and_answer() {
    let client = TestCompletionModel::new([AssistantContent::text(
        "[[ ## reasoning ## ]]\nTwo plus two is four.\n\n[[ ## explain ## ]]\n4\n\n[[ ## completed ## ]]",
    )]);
    let lm = LM::builder()
        .base_url("http://localhost:1/v1".into())
        .model("cot-test".into())
        .cache(false)
        .build()
        .await
        .unwrap()
        .with_client(LMClient::Test(client.clone()))
        .await
        .unwrap();
    let output = with_lm(lm, explain("2+2?".into())).await.unwrap();
    assert_eq!(output.reasoning, "Two plus two is four.");
    assert_eq!(output.explain, "4");
    let request = client.last_request().unwrap();
    let prompt = request.preamble.unwrap();
    assert!(
        prompt.find("[[ ## reasoning ## ]]").unwrap() < prompt.find("[[ ## explain ## ]]").unwrap()
    );
    assert_eq!(explain::Sig::schema().input_fields().len(), 1);
}
