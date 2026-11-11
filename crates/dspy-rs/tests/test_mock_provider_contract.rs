use dspy_rs::core::lm::CompletionProvider;
use dspy_rs::{LM, LMClient, Predict, PredictError, Signature, TestCompletionModel};
use rig::completion::AssistantContent;

#[derive(Signature, Clone, Debug)]
/// Answer the question.
struct QA {
    #[input]
    question: String,
    #[output]
    answer: String,
}

fn accepts_provider<P: CompletionProvider>(_: &P) {}

struct ExternalProvider(TestCompletionModel);
impl CompletionProvider for ExternalProvider {
    async fn completion(
        &self,
        request: rig::completion::CompletionRequest,
    ) -> Result<rig::completion::CompletionResponse<()>, rig::completion::CompletionError> {
        self.0.completion(request).await
    }
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn mock_and_dispatcher_use_the_same_provider_contract_and_lm_configuration() {
    let client = TestCompletionModel::new([AssistantContent::text(
        "[[ ## answer ## ]]\n4\n\n[[ ## completed ## ]]",
    )]);
    accepts_provider(&client);
    let dispatcher = LMClient::Test(client.clone());
    accepts_provider(&dispatcher);
    let lm = LM::builder()
        .base_url("http://localhost:1/v1".into())
        .model("mock".into())
        .cache(false)
        .build()
        .await
        .unwrap()
        .with_client(dispatcher)
        .await
        .unwrap();
    let predictor = Predict::<QA>::builder().lm(lm).build();
    let output = predictor
        .call_with_config(
            QAInput {
                question: "2+2?".into(),
            },
            |config| {
                config.temperature = 0.0;
                config.max_tokens = 123;
            },
        )
        .await
        .unwrap();
    assert_eq!(output.answer, "4");
    let request = client.last_request().unwrap();
    assert_eq!(request.temperature, Some(0.0));
    assert_eq!(request.max_tokens, Some(123));
    let error = predictor
        .call(QAInput {
            question: "empty queue".into(),
        })
        .await
        .unwrap_err();
    assert!(matches!(error, PredictError::Lm { .. }));
    assert!(!error.is_retryable());
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn third_party_provider_uses_scoped_lm_and_call_overrides() {
    let client = TestCompletionModel::new([AssistantContent::text(
        "[[ ## answer ## ]]\nexternal\n\n[[ ## completed ## ]]",
    )]);
    let lm = LM::builder()
        .base_url("http://localhost:1/v1".into())
        .model("external".into())
        .build()
        .await
        .unwrap()
        .with_client(LMClient::custom(ExternalProvider(client.clone())))
        .await
        .unwrap();
    let output = dspy_rs::with_lm(lm, async {
        Predict::<QA>::new()
            .call_with_config(
                QAInput {
                    question: "q".into(),
                },
                |config| config.max_tokens = 91,
            )
            .await
    })
    .await
    .unwrap();
    assert_eq!(output.answer, "external");
    assert_eq!(client.last_request().unwrap().max_tokens, Some(91));
}

struct FailingProvider {
    status: u16,
    attempts: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}
impl CompletionProvider for FailingProvider {
    async fn completion(
        &self,
        _: rig::completion::CompletionRequest,
    ) -> Result<rig::completion::CompletionResponse<()>, rig::completion::CompletionError> {
        self.attempts
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Err(rig::completion::CompletionError::HttpError(
            rig::http_client::Error::InvalidStatusCodeWithMessage(
                reqwest::StatusCode::from_u16(self.status).unwrap(),
                "original provider detail".into(),
            ),
        ))
    }
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn public_errors_preserve_http_kind_retry_policy_and_nested_source() {
    use dspy_rs::{ErrorClass, LmFailureKind};
    for (status, kind, retryable) in [
        (400, LmFailureKind::BadRequest, false),
        (401, LmFailureKind::Authentication, false),
        (403, LmFailureKind::ContentPolicy, false),
        (429, LmFailureKind::RateLimited, true),
        (503, LmFailureKind::Server, true),
    ] {
        let attempts = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let lm = LM::builder()
            .base_url("http://localhost:1/v1".into())
            .model("external".into())
            .max_retries(1)
            .retry_base_delay_ms(1)
            .build()
            .await
            .unwrap()
            .with_client(LMClient::custom(FailingProvider {
                status,
                attempts: attempts.clone(),
            }))
            .await
            .unwrap();
        let error = Predict::<QA>::builder()
            .lm(lm)
            .build()
            .call(QAInput {
                question: "q".into(),
            })
            .await
            .unwrap_err();
        assert_eq!(error.is_retryable(), retryable);
        if retryable {
            assert_eq!(error.class(), ErrorClass::Temporary);
        }
        let PredictError::Lm { source } = error else {
            panic!("expected LM error")
        };
        let cause = source
            .call_error()
            .expect("interpreter must preserve the typed cause");
        assert_eq!(cause.kind(), kind);
        assert_eq!(cause.http_status(), Some(status));
        assert!(
            std::error::Error::source(cause)
                .unwrap()
                .to_string()
                .contains("original provider detail")
        );
        assert_eq!(
            attempts.load(std::sync::atomic::Ordering::SeqCst),
            if retryable { 2 } else { 1 }
        );
    }
}
