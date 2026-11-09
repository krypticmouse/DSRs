pub async fn proposal_lm() -> dspy_rs::LM {
    use dspy_rs::{LM, LMClient, LMConfig, TestCompletionModel};
    use rig::completion::AssistantContent;
    let model = TestCompletionModel::new([AssistantContent::text(
        "[[ ## instructions ## ]]\n[\"Answer accurately using the provided evidence.\",\"Check the facts before answering.\",\"Return a precise answer.\"]\n[[ ## completed ## ]]",
    )]);
    let config = LMConfig {
        base_url: Some("http://localhost/v1".into()),
        ..LMConfig::default()
    };
    LM::from_config(config)
        .await
        .unwrap()
        .with_client(LMClient::Test(model))
        .await
        .unwrap()
}
