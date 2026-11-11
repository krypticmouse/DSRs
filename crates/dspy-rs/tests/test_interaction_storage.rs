use dspy_rs::{
    Chat, InteractionStore, JsonlInteractionStore, LM, LMClient, Message, Module, Predict,
    Signature, TestCompletionModel, with_interaction_store,
};
use rig::completion::AssistantContent;
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::Arc;

#[derive(Signature, Clone, Debug)]
/// Answer the question.
struct QA {
    #[input]
    question: String,
    #[output]
    answer: String,
}

async fn lm(responses: Vec<AssistantContent>) -> LM {
    LM::builder()
        .base_url("http://127.0.0.1:1/v1".into())
        .model("storage".into())
        .cache(false)
        .build()
        .await
        .unwrap()
        .with_client(LMClient::Test(TestCompletionModel::new(responses)))
        .await
        .unwrap()
}

fn answer(value: &str) -> AssistantContent {
    AssistantContent::text(format!(
        "[[ ## answer ## ]]\n{value}\n[[ ## completed ## ]]"
    ))
}

#[tokio::test]
async fn reopens_complete_typed_records_with_history_disabled() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("interactions.jsonl");
    let store = Arc::new(JsonlInteractionStore::open(&path).unwrap());
    let lm = lm(vec![answer("Paris")]).await;
    lm.history.set_capacity(0);
    dspy_rs::set_history_capacity(0);
    let predict = Predict::<QA>::builder().lm(lm).build();
    predict.set_history_capacity(0);
    let output = with_interaction_store(
        store.clone(),
        predict.call(QAInput {
            question: "France?".into(),
        }),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(output.answer, "Paris");
    let records = store.recent(10).unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].output.as_ref().unwrap()["answer"], "Paris");
    assert!(records[0].error.is_none());
    let id = records[0].id.clone();
    assert_eq!(output.metadata().call_ids, [id.clone()]);
    drop(store);
    let reopened = JsonlInteractionStore::open(&path).unwrap();
    assert_eq!(
        reopened.get(&id).unwrap().unwrap().output.unwrap()["answer"],
        "Paris"
    );
}

#[tokio::test]
async fn failed_parsing_updates_the_same_record() {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(JsonlInteractionStore::open(dir.path().join("calls.jsonl")).unwrap());
    let predict = Predict::<QA>::new().with_lm(lm(vec![AssistantContent::text("invalid")]).await);
    assert!(
        with_interaction_store(
            store.clone(),
            predict.call(QAInput {
                question: "?".into()
            })
        )
        .await
        .unwrap()
        .is_err()
    );
    let records = store.recent(10).unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(
        records[0].error.as_ref().unwrap().kind,
        dspy_rs::SpanErrorKind::Parse
    );
}

#[tokio::test]
async fn torn_tail_is_recovered_but_corruption_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("calls.jsonl");
    let store = Arc::new(JsonlInteractionStore::open(&path).unwrap());
    let lm = lm(vec![answer("one"), answer("two")]).await;
    let call = || lm.call(Chat::new(vec![Message::user("question")]), vec![]);
    with_interaction_store(store.clone(), call())
        .await
        .unwrap()
        .unwrap();
    OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(b"{torn")
        .unwrap();
    with_interaction_store(store.clone(), call())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        JsonlInteractionStore::open(&path)
            .unwrap()
            .recent(10)
            .unwrap()
            .len(),
        2
    );
    OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(b"bad record\n")
        .unwrap();
    assert!(JsonlInteractionStore::open(&path).is_err());
}

#[tokio::test]
async fn independent_handles_do_not_lose_concurrent_updates() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("calls.jsonl");
    let a = Arc::new(JsonlInteractionStore::open(&path).unwrap());
    let b = Arc::new(JsonlInteractionStore::open(&path).unwrap());
    let lm = lm(vec![answer("a"), answer("b")]).await;
    let (one, two) = tokio::join!(
        with_interaction_store(
            a.clone(),
            lm.call(Chat::new(vec![Message::user("a")]), vec![])
        ),
        with_interaction_store(b, lm.call(Chat::new(vec![Message::user("b")]), vec![]))
    );
    one.unwrap().unwrap();
    two.unwrap().unwrap();
    let records = a.recent(10).unwrap();
    assert_eq!(records.len(), 2);
    assert_ne!(records[0].id, records[1].id);
}

#[derive(Debug)]
struct Broken;
impl InteractionStore for Broken {
    fn upsert_trace(&self, _: &dspy_rs::Trace) -> Result<(), dspy_rs::StorageError> {
        Err(std::io::Error::other("disk full").into())
    }
    fn get_trace(&self, _: &str) -> Result<Option<dspy_rs::Trace>, dspy_rs::StorageError> {
        unreachable!()
    }
    fn upsert(&self, _: &dspy_rs::HistoryEntry) -> Result<(), dspy_rs::StorageError> {
        Err(std::io::Error::other("disk full").into())
    }
    fn get(&self, _: &str) -> Result<Option<dspy_rs::HistoryEntry>, dspy_rs::StorageError> {
        unreachable!()
    }
    fn recent(&self, _: usize) -> Result<Vec<dspy_rs::HistoryEntry>, dspy_rs::StorageError> {
        unreachable!()
    }
}

#[tokio::test]
async fn captured_traces_are_reopened_and_joined_to_call_ids() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("calls.jsonl");
    let store = Arc::new(JsonlInteractionStore::open(&path).unwrap());
    let predict = Predict::<QA>::new().with_lm(lm(vec![answer("Paris")]).await);
    let (result, trace) = with_interaction_store(
        store.clone(),
        dspy_rs::capture(|| {
            predict.call(QAInput {
                question: "France?".into(),
            })
        }),
    )
    .await
    .unwrap();
    let id = result.unwrap().metadata().call_ids[0].clone();
    drop(store);
    let reopened = JsonlInteractionStore::open(&path).unwrap();
    let entry = reopened.get(&id).unwrap().unwrap();
    assert_eq!(
        entry.trace_id.as_deref(),
        Some(trace.meta.trace_id.as_str())
    );
    let saved = reopened.get_trace(&trace.meta.trace_id).unwrap().unwrap();
    assert_eq!(saved.spans.len(), 1);
    assert_eq!(saved.spans[0].output.as_ref().unwrap()["answer"], "Paris");
}

#[tokio::test]
async fn persistence_failure_is_returned_to_the_caller() {
    let lm = lm(vec![answer("Paris")]).await;
    let result = with_interaction_store(
        Arc::new(Broken),
        lm.call(Chat::new(vec![Message::user("?")]), vec![]),
    )
    .await;
    let error = result.unwrap_err();
    assert!(error.to_string().contains("disk full"));
    let cause = std::error::Error::source(&error).unwrap();
    assert!(
        cause
            .source()
            .unwrap()
            .downcast_ref::<std::io::Error>()
            .is_some()
    );
}

struct Pending;
impl dspy_rs::CompletionProvider for Pending {
    async fn completion(
        &self,
        _: rig::completion::CompletionRequest,
    ) -> Result<rig::completion::CompletionResponse<()>, rig::completion::CompletionError> {
        std::future::pending().await
    }
}

#[tokio::test]
async fn cancellation_and_provider_failures_are_persistent() {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(JsonlInteractionStore::open(dir.path().join("calls.jsonl")).unwrap());
    let broken = lm(vec![]).await;
    let pending = lm(vec![])
        .await
        .with_client(LMClient::custom(Pending))
        .await
        .unwrap();
    let chat = || Chat::new(vec![Message::user("?")]);
    with_interaction_store(store.clone(), async {
        assert!(broken.call(chat(), vec![]).await.is_err());
        assert!(
            tokio::time::timeout(
                std::time::Duration::from_millis(10),
                pending.call(chat(), vec![])
            )
            .await
            .is_err()
        );
    })
    .await
    .unwrap();
    let records = store.recent(10).unwrap();
    assert_eq!(records.len(), 2);
    assert!(
        records
            .iter()
            .any(|r| r.error.as_ref().unwrap().kind == dspy_rs::SpanErrorKind::Cancelled)
    );
    assert!(
        records
            .iter()
            .any(|r| r.error.as_ref().unwrap().kind == dspy_rs::SpanErrorKind::Lm)
    );
}

#[cfg(feature = "cache")]
#[tokio::test]
async fn cache_hits_have_separate_ids_and_generation_settings_miss() {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(JsonlInteractionStore::open(dir.path().join("calls.jsonl")).unwrap());
    let provider = TestCompletionModel::new([answer("a"), answer("b")]);
    let mut lm = LM::builder()
        .base_url("http://127.0.0.1:1/v1".into())
        .model("storage-cache".into())
        .cache(true)
        .build()
        .await
        .unwrap()
        .with_client(LMClient::Test(provider.clone()))
        .await
        .unwrap();
    with_interaction_store(store.clone(), async {
        for _ in 0..2 {
            lm.call(Chat::new(vec![Message::user("same")]), vec![])
                .await
                .unwrap();
        }
        lm.config.temperature = 0.2;
        lm.call(Chat::new(vec![Message::user("same")]), vec![])
            .await
            .unwrap();
    })
    .await
    .unwrap();
    let records = store.recent(10).unwrap();
    assert_eq!(records.len(), 3);
    assert_eq!(records.iter().filter(|r| r.cache_hit).count(), 1);
    assert_eq!(provider.call_count(), 2);
    assert_eq!(records[0].config.temperature, 0.2);
}
