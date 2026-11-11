use dspy_rs::{
    Chat, LM, LMClient, Module, Predict, PredictError, Predicted, Predictors, Signature,
    TestCompletionModel, configure, get_lm, try_get_lm, with_lm,
};
use futures::FutureExt;
use rig::completion::AssistantContent;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use tokio::time::{Duration, timeout};

#[derive(Signature, Clone, Debug)]
/// Answer questions.
struct QA {
    #[input]
    question: String,
    #[output]
    answer: String,
}

fn input() -> QAInput {
    QAInput {
        question: "Which LM?".into(),
    }
}

#[dspy_rs::predict]
fn scoped_step(question: String) -> String;

#[dspy_rs::Schema]
#[derive(Debug)]
struct ScopedOutput {
    answer: String,
}

#[dspy_rs::module]
async fn scoped_pipeline(question: String) -> Result<ScopedOutput, dspy_rs::ir::RunError> {
    let response = scoped_step(question).await?;
    Ok(ScopedOutput {
        answer: response.scoped_step,
    })
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn generated_ir_modules_bind_scoped_default_lm() {
    let client = TestCompletionModel::new([AssistantContent::text(
        "[[ ## scoped_step ## ]]\nscoped\n\n[[ ## completed ## ]]",
    )]);
    let (base, _) = model("scoped-ir", false).await;
    let lm = (*base)
        .clone()
        .with_client(LMClient::Test(client.clone()))
        .await
        .unwrap();
    let result = with_lm(lm, scoped_pipeline("Question?".into()))
        .await
        .unwrap();
    assert_eq!(result.answer, "scoped");
    assert!(client.last_request().is_some());
}

async fn model(label: &str, cache: bool) -> (Arc<LM>, TestCompletionModel) {
    let text = format!("[[ ## answer ## ]]\n{label}\n\n[[ ## completed ## ]]");
    let client = TestCompletionModel::new((0..8).map(|_| AssistantContent::text(text.clone())));
    let lm = LM::builder()
        .base_url("http://localhost:11434/v1".into())
        .model(label.to_string())
        .temperature(0.3)
        .max_tokens(321)
        .cache(cache)
        .build()
        .await
        .unwrap()
        .with_client(LMClient::Test(client.clone()))
        .await
        .unwrap();
    (Arc::new(lm), client)
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn resolution_precedence_and_restoration() {
    let (global, _) = model("global", false).await;
    let (scoped, _) = model("scoped", false).await;
    let (instance, _) = model("instance", false).await;
    let (explicit, _) = model("explicit", false).await;
    configure((*global).clone());
    let fallback = Predict::<QA>::new();
    let predict = Predict::<QA>::builder().lm((*instance).clone()).build();
    with_lm(scoped.clone(), async {
        assert_eq!(get_lm().config.model, "scoped");
        assert_eq!(fallback.call(input()).await.unwrap().answer, "scoped");
        assert_eq!(predict.call(input()).await.unwrap().answer, "instance");
        assert_eq!(
            predict
                .call_with_lm(input(), explicit)
                .await
                .unwrap()
                .answer,
            "explicit"
        );
        assert_eq!(predict.call(input()).await.unwrap().answer, "instance");
        assert_eq!(fallback.call(input()).await.unwrap().answer, "scoped");
    })
    .await;
    assert_eq!(get_lm().config.model, "global");
    assert_eq!(fallback.call(input()).await.unwrap().answer, "global");
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn scopes_restore_after_error_panic_and_cancellation_and_do_not_leak_to_spawn() {
    let (outer, _) = model("outer", false).await;
    let (inner, _) = model("inner", false).await;
    with_lm(outer.clone(), async {
        let result: Result<(), &str> = with_lm(inner.clone(), async { Err("expected") }).await;
        assert_eq!(result, Err("expected"));
        assert!(Arc::ptr_eq(&get_lm(), &outer));
        let panic = AssertUnwindSafe(with_lm(inner.clone(), async { panic!("expected") }))
            .catch_unwind()
            .await;
        assert!(panic.is_err());
        assert!(Arc::ptr_eq(&get_lm(), &outer));
        assert!(
            timeout(
                Duration::from_millis(1),
                with_lm(inner, std::future::pending::<()>())
            )
            .await
            .is_err()
        );
        assert!(Arc::ptr_eq(&get_lm(), &outer));
        let inherited = tokio::spawn(async { try_get_lm() }).await.unwrap();
        assert!(inherited.as_ref().is_none_or(|lm| !Arc::ptr_eq(lm, &outer)));
    })
    .await;
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn concurrent_scopes_share_a_predictor_without_sharing_lm_bindings() {
    let (a, _) = model("a", false).await;
    let (b, _) = model("b", false).await;
    let predict = Predict::<QA>::new();
    let (a, b) = tokio::join!(
        with_lm(a, predict.call(input())),
        with_lm(b, predict.call(input())),
    );
    assert_eq!(a.unwrap().answer, "a");
    assert_eq!(b.unwrap().answer, "b");
}

struct Pipeline {
    first: Predict<QA>,
    second: Predict<QA>,
}
dspy_rs::predictors!(Pipeline { first, second });

impl Module for Pipeline {
    type Input = QAInput;
    type Output = QAOutput;
    async fn forward(&self, input: QAInput) -> Result<Predicted<QAOutput>, PredictError> {
        let first = self.first.call(input).await?;
        self.second
            .call(QAInput {
                question: first.answer.clone(),
            })
            .await
    }
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn module_defaults_preserve_leaf_overrides_and_explicit_calls_override_both() {
    let (default, default_client) = model("default", false).await;
    let (instance, instance_client) = model("instance", false).await;
    let (explicit, explicit_client) = model("explicit", false).await;
    let mut module = Pipeline {
        first: Predict::new(),
        second: Predict::builder().lm((*instance).clone()).build(),
    }
    .with_lm(default);
    assert_eq!(module.predictors().len(), 2);
    assert_eq!(module.predictors_mut().len(), 2);
    assert_eq!(module.call(input()).await.unwrap().answer, "instance");
    assert!(default_client.last_request().is_some());
    assert!(instance_client.last_request().is_some());
    assert_eq!(
        module.call_with_lm(input(), explicit).await.unwrap().answer,
        "explicit"
    );
    assert!(
        explicit_client
            .last_request()
            .unwrap()
            .chat_history
            .iter()
            .any(|message| format!("{message:?}").contains("explicit"))
    );
    assert_eq!(module.call(input()).await.unwrap().answer, "instance");
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn explicit_conversation_lm_and_call_config_do_not_mutate_predictor() {
    let (instance, instance_client) = model("instance", false).await;
    let (explicit, _) = model("explicit", false).await;
    let predict = Predict::<QA>::builder().lm((*instance).clone()).build();
    let chat = predict
        .build_chat_with_lm(&input(), explicit.clone())
        .await
        .unwrap();
    assert_eq!(
        predict
            .call_and_parse_with_lm(chat, explicit)
            .await
            .unwrap()
            .0
            .answer,
        "explicit"
    );
    assert_eq!(
        predict
            .call_with_config(input(), |config| config.temperature = 0.8)
            .await
            .unwrap()
            .answer,
        "instance"
    );
    let request = instance_client.last_request().unwrap();
    assert_eq!(request.temperature, Some(0.8_f32 as f64));
    assert_eq!(request.max_tokens, Some(321));
    assert_eq!(instance.config.temperature, 0.3);
    predict.call(input()).await.unwrap();
    assert_eq!(
        instance_client.last_request().unwrap().temperature,
        Some(0.3_f32 as f64)
    );
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn copies_keep_injected_clients_and_isolate_cache_and_configuration() {
    let (original, client) = model("original", true).await;
    let chat = Chat::new(vec![dspy_rs::Message::user("hello")]);
    original.call(chat.clone(), vec![]).await.unwrap();
    let copy = original
        .copy_with(|config| config.temperature = 0.8)
        .await
        .unwrap();
    assert_eq!(copy.config.max_tokens, 321);
    assert_eq!(copy.config.model, original.config.model);
    assert_eq!(original.config.temperature, 0.3);
    assert!(!Arc::ptr_eq(
        original.cache_handler.as_ref().unwrap(),
        copy.cache_handler.as_ref().unwrap()
    ));
    assert!(copy.inspect_history(1).await.is_empty());
    copy.call(chat.clone(), vec![]).await.unwrap();
    assert_eq!(
        client.last_request().unwrap().temperature,
        Some(0.8_f32 as f64)
    );
    let uncached = original
        .copy_with(|config| config.cache = false)
        .await
        .unwrap();
    assert!(uncached.cache_handler.is_none());
    assert!(original.cache_handler.is_some());
    let unchanged = original.copy_with(|_| {}).await.unwrap();
    assert!(unchanged.inspect_history(1).await.is_empty());
    assert_eq!(original.inspect_history(1).await.len(), 1);
}
