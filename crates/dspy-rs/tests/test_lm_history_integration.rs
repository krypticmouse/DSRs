use std::sync::Arc;

use dspy_rs::{
    Chat, HistoryBuffer, LM, LMClient, Message, Module, Predict, Predictors, Signature,
    SpanErrorKind, TestCompletionModel,
};
use rig::completion::AssistantContent;

#[derive(Signature, Clone, Debug)]
/// Answer the question.
struct QA {
    #[input]
    question: String,
    #[output]
    answer: String,
}

fn input() -> QAInput {
    QAInput {
        question: "question".into(),
    }
}

fn chat() -> Chat {
    Chat::new(vec![Message::user("question")])
}

async fn model(label: &str, cache: bool) -> Arc<LM> {
    let text = format!("[[ ## answer ## ]]\n{label}\n\n[[ ## completed ## ]]");
    raw_model(label, cache, (0..8).map(|_| text.clone())).await
}

async fn raw_model(
    label: &str,
    cache: bool,
    responses: impl IntoIterator<Item = String>,
) -> Arc<LM> {
    let client = TestCompletionModel::new(responses.into_iter().map(AssistantContent::text));
    Arc::new(
        LM::builder()
            .base_url("http://localhost:1/v1".into())
            .model(label.into())
            .cache(cache)
            .build()
            .await
            .unwrap()
            .with_client(LMClient::Test(client))
            .await
            .unwrap(),
    )
}

#[rstest::rstest]
#[case(false)]
#[cfg_attr(feature = "cache", case(true))]
#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn configuration_copies_have_independent_history(#[case] cache: bool) {
    let original = model("copy", cache).await;
    original.call(chat(), vec![]).await.unwrap();
    let copy = original
        .copy_with(|config| config.temperature = 0.25)
        .await
        .unwrap();
    assert!(copy.inspect_history(10).await.is_empty());
    copy.call(chat(), vec![]).await.unwrap();
    assert_eq!(original.inspect_history(10).await.len(), 1);
    assert_eq!(copy.inspect_history(10).await.len(), 1);
    assert_eq!(copy.inspect_history(1).await[0].config.temperature, 0.25);
    copy.history.clear();
    assert_eq!(original.inspect_history(10).await.len(), 1);
    let clone = (*original).clone();
    clone.call(chat(), vec![]).await.unwrap();
    assert_eq!(original.inspect_history(10).await.len(), 2);
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
#[cfg(not(feature = "cache"))]
async fn copy_cannot_enable_a_missing_cache_feature() {
    let original = model("copy", false).await;
    for change_identity in [false, true] {
        let error = original
            .copy_with(|config| {
                config.cache = true;
                if change_identity {
                    config.model = "other".into();
                }
            })
            .await
            .err()
            .unwrap();
        assert!(
            error
                .to_string()
                .contains("requires the `cache` Cargo feature")
        );
    }
    assert!(original.inspect_history(1).await.is_empty());
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
#[cfg(feature = "cache")]
async fn cache_transitions_in_copies_do_not_replay_parent_responses() {
    let original = raw_model(
        "cache-transitions",
        true,
        ["parent", "uncached", "recached"].map(str::to_owned),
    )
    .await;
    assert_eq!(
        original
            .call(chat(), vec![])
            .await
            .unwrap()
            .output
            .content(),
        "parent"
    );
    let uncached = original
        .copy_with(|config| config.cache = false)
        .await
        .unwrap();
    assert_eq!(
        uncached
            .call(chat(), vec![])
            .await
            .unwrap()
            .output
            .content(),
        "uncached"
    );
    let recached = uncached
        .copy_with(|config| config.cache = true)
        .await
        .unwrap();
    assert_eq!(
        recached
            .call(chat(), vec![])
            .await
            .unwrap()
            .output
            .content(),
        "recached"
    );
    assert!(recached.call(chat(), vec![]).await.unwrap().cache_hit);
    assert_eq!(original.inspect_history(10).await.len(), 1);
    assert_eq!(uncached.inspect_history(10).await.len(), 1);
    assert_eq!(recached.inspect_history(10).await.len(), 2);
}

async fn exercise_wrappers<M: Module<Input = QAInput, Output = QAOutput> + Predictors>(
    module: &mut M,
    history: HistoryBuffer,
    default: Arc<LM>,
    explicit: Arc<LM>,
    cache: bool,
) {
    assert_eq!(module.predictors().len(), 1);
    assert_eq!(module.predictors_mut().len(), 1);
    assert_eq!(module.call(input()).await.unwrap().answer, "default");
    assert_eq!(module.call(input()).await.unwrap().answer, "default");
    assert_eq!(
        module
            .call_with_lm(input(), explicit.clone())
            .await
            .unwrap()
            .answer,
        "explicit"
    );
    let entries = history.inspect(10);
    assert_eq!(entries.len(), 3);
    assert_eq!(entries[0].model, "explicit");
    assert_eq!(entries[1].cache_hit, cache);
    assert_eq!(entries[1].module_path[0], "pipeline");
    assert_eq!(default.inspect_history(10).await.len(), 2);
    assert_eq!(explicit.inspect_history(10).await.len(), 1);
    for entry in entries {
        assert_eq!(entry.module_path.len(), 2);
        assert!(entry.output.is_some());
        assert!(
            dspy_rs::inspect_history(100)
                .iter()
                .any(|global| global.id == entry.id)
        );
    }
}

#[rstest::rstest]
#[case(false)]
#[cfg_attr(feature = "cache", case(true))]
#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn history_outside_lm_wrapper_preserves_overrides_and_leaf_discovery(#[case] cache: bool) {
    let default = model("default", cache).await;
    let explicit = model("explicit", cache).await;
    let mut module = Predict::<QA>::new()
        .with_lm(default.clone())
        .with_history("pipeline", 10);
    let history = module.history.clone();
    exercise_wrappers(&mut module, history, default, explicit, cache).await;
}

#[rstest::rstest]
#[case(false)]
#[cfg_attr(feature = "cache", case(true))]
#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn lm_outside_history_wrapper_preserves_overrides_and_leaf_discovery(#[case] cache: bool) {
    let default = model("default", cache).await;
    let explicit = model("explicit", cache).await;
    let tracked = Predict::<QA>::new().with_history("pipeline", 10);
    let history = tracked.history.clone();
    let mut module = tracked.with_lm(default.clone());
    exercise_wrappers(&mut module, history, default, explicit, cache).await;
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn call_config_parse_errors_stay_in_predictor_and_global_history() {
    let original = raw_model("parse-error", false, ["unformatted".to_owned()]).await;
    let predict = Predict::<QA>::builder().lm((*original).clone()).build();
    assert!(
        predict
            .call_with_config(input(), |config| config.temperature = 0.25)
            .await
            .is_err()
    );
    assert!(original.inspect_history(10).await.is_empty());
    let entries = predict.inspect_history(10).await;
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].config.temperature, 0.25);
    assert_eq!(
        entries[0].error.as_ref().unwrap().kind,
        SpanErrorKind::Parse
    );
    assert!(
        dspy_rs::inspect_history(100)
            .iter()
            .any(|global| global.id == entries[0].id)
    );
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn explicit_conversation_lm_records_only_the_selected_lm() {
    let original = model("original", false).await;
    let explicit = model("explicit", false).await;
    let predict = Predict::<QA>::builder().lm((*original).clone()).build();
    let chat = predict
        .build_chat_with_lm(&input(), explicit.clone())
        .await
        .unwrap();
    assert!(predict.inspect_history(1).await.is_empty());
    let (out, _) = predict
        .call_and_parse_with_lm(chat, explicit.clone())
        .await
        .unwrap();
    assert_eq!(out.answer, "explicit");
    assert!(original.inspect_history(1).await.is_empty());
    let entry = predict.inspect_history(1).await.remove(0);
    assert_eq!(entry.id, explicit.inspect_history(1).await[0].id);
    assert_eq!(entry.output.unwrap()["answer"], "explicit");
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn concurrent_explicit_overrides_keep_history_and_lm_scopes_aligned() {
    let default = model("default", false).await;
    let a = model("a", false).await;
    let b = model("b", false).await;
    let module = Predict::<QA>::new()
        .with_lm(default.clone())
        .with_history("pipeline", 10);
    let (one, two) = tokio::join!(
        module.call_with_lm(input(), a.clone()),
        module.call_with_lm(input(), b.clone()),
    );
    assert_eq!(one.unwrap().answer, "a");
    assert_eq!(two.unwrap().answer, "b");
    assert!(default.inspect_history(1).await.is_empty());
    let entries = module.inspect_history(10).await;
    assert_eq!(entries.len(), 2);
    assert_ne!(entries[0].id, entries[1].id);
    for entry in entries {
        assert_eq!(entry.module_path[0], "pipeline");
        assert_eq!(entry.output.as_ref().unwrap()["answer"], entry.model);
        let selected = if entry.model == "a" { &a } else { &b };
        assert_eq!(selected.inspect_history(1).await[0].id, entry.id);
    }
}

#[rstest::rstest]
#[case(false)]
#[cfg_attr(feature = "cache", case(true))]
#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn replacement_clients_reset_history_without_clearing_existing_clones(#[case] cache: bool) {
    let original = raw_model("replace", cache, ["first", "second"].map(str::to_owned)).await;
    original.call(chat(), vec![]).await.unwrap();
    let retained = (*original).clone();
    let replacement = (*original)
        .clone()
        .with_client(LMClient::Test(TestCompletionModel::new([
            AssistantContent::text("replacement"),
        ])))
        .await
        .unwrap();
    assert!(replacement.inspect_history(10).await.is_empty());
    assert_eq!(
        replacement
            .call(chat(), vec![])
            .await
            .unwrap()
            .output
            .content(),
        "replacement"
    );
    let response = retained.call(chat(), vec![]).await.unwrap();
    assert_eq!(
        response.output.content(),
        if cache { "first" } else { "second" }
    );
    assert_eq!(response.cache_hit, cache);
    assert_eq!(original.inspect_history(10).await.len(), 2);
    assert_eq!(replacement.inspect_history(10).await.len(), 1);
}
