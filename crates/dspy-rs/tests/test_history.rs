use dspy_rs::{
    Chat, LM, LMClient, Message, Module, Predict, PredictError, Predicted, Predictors, Signature,
    TestCompletionModel,
};
use rig::completion::{AssistantContent, ToolDefinition, Usage};
use rig::tool::Tool;
use std::sync::{Arc, LazyLock};
use tokio::sync::Mutex;
use tokio::time::{Duration, timeout};

static SERIAL: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

#[derive(Signature, Clone, Debug)]
/// Answer questions.
struct QA {
    #[input]
    question: String,
    #[output]
    answer: String,
}

fn text(answer: &str) -> AssistantContent {
    AssistantContent::text(format!(
        "[[ ## answer ## ]]\n{answer}\n\n[[ ## completed ## ]]"
    ))
}

async fn lm(model: &str, cache: bool, responses: Vec<AssistantContent>) -> LM {
    let client = TestCompletionModel::new(responses);
    client.set_usage(Usage {
        input_tokens: 5,
        output_tokens: 3,
        total_tokens: 8,
        ..Usage::new()
    });
    LM::builder()
        .base_url("http://user:password@localhost:1/v1?token=secret".into())
        .api_key("private-key".into())
        .model(model.into())
        .cache(cache)
        .build()
        .await
        .unwrap()
        .with_client(LMClient::Test(client))
        .await
        .unwrap()
}

fn chat() -> Chat {
    Chat::new(vec![Message::user("hello")])
}

#[rstest::rstest]
#[case(false)]
#[cfg_attr(feature = "cache", case(true))]
#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn calls_are_recorded_with_and_without_caching(#[case] cache: bool) {
    let _serial = SERIAL.lock().await;
    let lm = lm("history", cache, vec![text("Paris"), text("Paris")]).await;
    let original = lm.call(chat(), vec![]).await.unwrap();
    let second = lm.call(chat(), vec![]).await.unwrap();
    let history = lm.inspect_history(10).await;
    assert_eq!(history.len(), 2);
    assert_ne!(history[0].id, history[1].id);
    assert_eq!(history[0].cache_hit, cache);
    assert_eq!(
        history[0].usage.unwrap().total_tokens,
        if cache { 0 } else { 8 }
    );
    assert_eq!(history[1].usage.unwrap().total_tokens, 8);
    match &history[0].events[0] {
        dspy_rs::SpanEvent::Exchange { usage, .. } => {
            assert_eq!(usage.total_tokens, if cache { 0 } else { 8 });
        }
        _ => panic!("expected completion exchange"),
    }
    assert_eq!(
        history[0].raw_output.as_deref(),
        Some(second.output.content().as_str())
    );
    assert_eq!(
        history[1].raw_output.as_deref(),
        Some(original.output.content().as_str())
    );
    assert_eq!(history[0].messages.len(), 1);
    assert!(history[0].cost.is_none());
    assert!(history[0].timestamp_us > 0);
    assert_eq!(lm.inspect_history(1).await[0].id, history[0].id);
    assert!(lm.inspect_history(0).await.is_empty());
    let json = serde_json::to_string(&history).unwrap();
    assert!(!json.contains("private-key"));
    assert!(!json.contains("password"));
    assert!(!json.contains("token=secret"));
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn global_and_lm_capacities_are_independent() {
    let _serial = SERIAL.lock().await;
    dspy_rs::clear_history();
    dspy_rs::set_history_capacity(2);
    let a = lm("a", false, vec![text("a"), text("a"), text("a")]).await;
    let b = lm("b", false, vec![text("b")]).await;
    a.history.set_capacity(1);
    a.call(chat(), vec![]).await.unwrap();
    b.call(chat(), vec![]).await.unwrap();
    a.call(chat(), vec![]).await.unwrap();
    let global = dspy_rs::inspect_history(10);
    assert_eq!(
        global.iter().map(|e| e.model.as_str()).collect::<Vec<_>>(),
        ["a", "b"]
    );
    assert_eq!(a.inspect_history(10).await.len(), 1);
    a.history.set_capacity(0);
    a.call(chat(), vec![]).await.unwrap();
    assert!(a.inspect_history(1).await.is_empty());
    assert_eq!(dspy_rs::inspect_history(10).len(), 2);
    a.history.clear();
    dspy_rs::clear_history();
    dspy_rs::set_history_capacity(100);
}

struct Pipeline {
    qa: Predict<QA>,
}
dspy_rs::predictors!(Pipeline { qa });
impl Module for Pipeline {
    type Input = QAInput;
    type Output = QAOutput;
    async fn forward(&self, input: QAInput) -> Result<Predicted<QAOutput>, PredictError> {
        self.qa.call(input).await
    }
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn module_and_predictor_history_share_parsed_records_with_lm_and_global() {
    let _serial = SERIAL.lock().await;
    let lm = Arc::new(lm("typed", false, vec![text("Paris")]).await);
    let mut module = Pipeline {
        qa: Predict::builder().named("qa").lm((*lm).clone()).build(),
    }
    .with_history("pipeline", 5);
    assert_eq!(module.predictors().len(), 1);
    assert_eq!(module.predictors_mut().len(), 1);
    let (output, trace) = dspy_rs::capture(|| {
        module.call(QAInput {
            question: "France?".into(),
        })
    })
    .await;
    assert_eq!(output.unwrap().answer, "Paris");
    let entry = &module.inspect_history(1).await[0];
    assert_eq!(entry.module_path, ["pipeline", "qa"]);
    assert_eq!(entry.output.as_ref().unwrap()["answer"], "Paris");
    assert_eq!(entry.output, trace.spans[0].output);
    assert_eq!(entry.id, lm.inspect_history(1).await[0].id);
    assert_eq!(entry.id, module.module.qa.inspect_history(1).await[0].id);
    assert!(
        dspy_rs::inspect_history(100)
            .iter()
            .any(|record| record.id == entry.id)
    );
    module.module.qa.clear_history();
    assert!(module.module.qa.inspect_history(1).await.is_empty());
    assert_eq!(lm.inspect_history(1).await.len(), 1);
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn parsing_and_provider_failures_are_recorded() {
    let _serial = SERIAL.lock().await;
    let lm = Arc::new(lm("errors", false, vec![AssistantContent::text("unformatted")]).await);
    let predict = Predict::<QA>::builder().lm((*lm).clone()).build();
    assert!(
        predict
            .call(QAInput {
                question: "question".into()
            })
            .await
            .is_err()
    );
    let parse = &lm.inspect_history(1).await[0];
    assert_eq!(
        parse.error.as_ref().unwrap().kind,
        dspy_rs::SpanErrorKind::Parse
    );
    assert_eq!(parse.raw_output.as_deref(), Some("unformatted"));
    assert!(parse.output.is_none());
    assert_eq!(parse.usage.unwrap().total_tokens, 8);
    assert!(lm.call(chat(), vec![]).await.is_err()); // canned response queue exhausted
    let provider = &lm.inspect_history(1).await[0];
    assert_eq!(
        provider.error.as_ref().unwrap().kind,
        dspy_rs::SpanErrorKind::Lm
    );
    assert!(provider.usage.is_none());
}

#[derive(Clone)]
struct Wait;
impl Tool for Wait {
    const NAME: &'static str = "wait";
    type Args = serde_json::Value;
    type Output = String;
    type Error = std::convert::Infallible;
    async fn definition(&self, _: String) -> ToolDefinition {
        ToolDefinition {
            name: "wait".into(),
            description: "Wait".into(),
            parameters: serde_json::json!({}),
        }
    }
    async fn call(&self, _: Self::Args) -> Result<String, Self::Error> {
        std::future::pending().await
    }
}

#[derive(Clone)]
struct Ready;
impl Tool for Ready {
    const NAME: &'static str = "ready";
    type Args = serde_json::Value;
    type Output = String;
    type Error = std::convert::Infallible;
    async fn definition(&self, _: String) -> ToolDefinition {
        ToolDefinition {
            name: "ready".into(),
            description: "Ready".into(),
            parameters: serde_json::json!({}),
        }
    }
    async fn call(&self, _: Self::Args) -> Result<String, Self::Error> {
        Ok("done".into())
    }
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn tool_rounds_and_agent_parse_repairs_are_recorded() {
    let _serial = SERIAL.lock().await;
    let lm = Arc::new(
        lm(
            "agent",
            false,
            vec![
                AssistantContent::tool_call("id", "ready", serde_json::json!({})),
                AssistantContent::text("missing markers"),
                text("Paris"),
            ],
        )
        .await,
    );
    let predict = Predict::<QA>::builder()
        .named("agent")
        .lm((*lm).clone())
        .add_tool(Ready)
        .build();
    let output = predict
        .call(QAInput {
            question: "France?".into(),
        })
        .await
        .unwrap();
    assert_eq!(output.answer, "Paris");
    let history = predict.inspect_history(10).await;
    assert_eq!(history.len(), 3);
    assert_eq!(history[0].output.as_ref().unwrap()["answer"], "Paris");
    assert!(history[0].error.is_none());
    assert_eq!(
        history[1].error.as_ref().unwrap().kind,
        dspy_rs::SpanErrorKind::Parse
    );
    assert!(history[2].response.as_ref().unwrap().has_tool_calls());
    assert_eq!(lm.inspect_history(10).await.len(), 3);
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn cancelled_tool_loop_is_recorded() {
    let _serial = SERIAL.lock().await;
    let lm = lm(
        "cancelled",
        false,
        vec![AssistantContent::tool_call(
            "id",
            "wait",
            serde_json::json!({}),
        )],
    )
    .await;
    assert!(
        timeout(
            Duration::from_millis(10),
            lm.call(chat(), vec![Arc::new(Wait)])
        )
        .await
        .is_err()
    );
    let history = lm.inspect_history(1).await;
    assert_eq!(history.len(), 1);
    assert_eq!(
        history[0].error.as_ref().unwrap().kind,
        dspy_rs::SpanErrorKind::Cancelled
    );
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn concurrent_scopes_do_not_mix_module_histories() {
    let _serial = SERIAL.lock().await;
    let lm = lm("concurrent", false, vec![text("a"), text("b")]).await;
    let a = dspy_rs::HistoryBuffer::new(5);
    let b = dspy_rs::HistoryBuffer::new(5);
    let (one, two) = tokio::join!(
        a.scope("a", lm.call(chat(), vec![])),
        b.scope("b", lm.call(chat(), vec![])),
    );
    one.unwrap();
    two.unwrap();
    assert_eq!(a.inspect(5).len(), 1);
    assert_eq!(b.inspect(5).len(), 1);
    assert_eq!(a.inspect(1)[0].module_path, ["a"]);
    assert_eq!(b.inspect(1)[0].module_path, ["b"]);
    assert_eq!(lm.inspect_history(5).await.len(), 2);
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn disabled_history_skips_records_but_an_enabled_caller_still_captures() {
    let _serial = SERIAL.lock().await;
    dspy_rs::clear_history();
    dspy_rs::set_history_capacity(0);
    let lm = lm("disabled", false, vec![text("Paris"), text("Berlin")]).await;
    lm.history.set_capacity(0);
    let response = lm.call(chat(), vec![]).await.unwrap();
    assert!(response.history.is_none());
    assert!(lm.inspect_history(1).await.is_empty());
    assert!(dspy_rs::inspect_history(1).is_empty());

    let caller = dspy_rs::HistoryBuffer::new(1);
    let response = caller
        .scope("caller", lm.call(chat(), vec![]))
        .await
        .unwrap();
    assert!(response.history.is_some());
    assert_eq!(caller.inspect(1)[0].module_path, ["caller"]);
    assert_eq!(
        caller.inspect(1)[0].raw_output.as_deref(),
        Some(response.output.content().as_str())
    );
    assert!(lm.inspect_history(1).await.is_empty());
    assert!(dspy_rs::inspect_history(1).is_empty());
    dspy_rs::set_history_capacity(100);
}
