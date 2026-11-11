use dspy_rs::{
    LM, LMClient, LmError, Module, ParseError, Predict, PredictError, Predictors, Signature,
    TestCompletionModel,
};
use rig::completion::{AssistantContent, ToolDefinition, Usage};
use rig::tool::Tool;
use std::num::NonZeroU32;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[derive(Signature, Clone, Debug)]
/// Answer the question and give supporting evidence.
struct QA {
    #[input]
    question: String,
    #[output]
    #[check("this|length >= 4", label = "answer_length")]
    answer: String,
    #[output]
    support: i32,
}

#[derive(Signature, Clone, Debug)]
struct Positive {
    #[input]
    question: String,
    #[output]
    #[assert("this > 0", label = "positive")]
    number: i32,
}

fn answer(text: &str, support: i32) -> String {
    format!("[[ ## answer ## ]]\n{text}\n\n[[ ## support ## ]]\n{support}\n\n[[ ## completed ## ]]")
}

fn input() -> QAInput {
    QAInput {
        question: "good".into(),
    }
}
fn attempts(n: u32) -> NonZeroU32 {
    NonZeroU32::new(n).unwrap()
}

async fn model(responses: Vec<String>, cache: bool) -> (LM, TestCompletionModel) {
    let client = TestCompletionModel::new(responses.into_iter().map(AssistantContent::text));
    client.set_usage(Usage {
        input_tokens: 5,
        output_tokens: 3,
        total_tokens: 8,
        ..Usage::new()
    });
    let lm = LM::builder()
        .base_url("http://localhost:1/v1".into())
        .model("validation".into())
        .cache(cache)
        .max_retries(0)
        .build()
        .await
        .unwrap()
        .with_client(LMClient::Test(client.clone()))
        .await
        .unwrap();
    (lm, client)
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn custom_field_and_signature_rules_send_feedback_and_accumulate_usage() {
    let (lm, client) = model(vec![answer("bad", 0), answer("good", 2)], false).await;
    let module = Predict::<QA>::builder()
        .lm(lm.clone())
        .build()
        .validated()
        .max_attempts(attempts(2))
        .validator(|input, output| {
            if output.answer == input.question {
                Ok(())
            } else {
                Err("Answer must match the requested word.".into())
            }
        })
        .validator(|_, output| {
            if output.support > 0 {
                Ok(())
            } else {
                Err("Provide positive support for the answer.".into())
            }
        });
    let result = module.call(input()).await.unwrap();
    assert_eq!(result.answer, "good");
    assert_eq!(result.metadata().lm_usage.total_tokens, 16);
    let prompt = format!("{:?}", client.last_request().unwrap().chat_history);
    assert!(prompt.contains("Answer must match the requested word."));
    assert!(prompt.contains("Provide positive support"));
    assert!(prompt.contains("bad"));
    assert_eq!(lm.inspect_history(10).await.len(), 2);
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn exhaustion_returns_typed_feedback_last_response_and_total_usage() {
    let (lm, _) = model(
        vec![answer("first", 0), answer("second", 0), answer("unused", 1)],
        false,
    )
    .await;
    let module = Predict::<QA>::builder()
        .lm(lm.clone())
        .build()
        .validated()
        .max_attempts(attempts(2))
        .validator(|_, _| Err("Citations are required.".into()));
    match module.call(input()).await.unwrap_err() {
        PredictError::Parse {
            source: ParseError::ValidationFailed { feedback },
            raw_response,
            lm_usage,
        } => {
            assert_eq!(feedback, "Citations are required.");
            assert!(raw_response.contains("second"));
            assert_eq!(lm_usage.total_tokens, 16);
        }
        other => panic!("unexpected error: {other:?}"),
    }
    assert_eq!(lm.inspect_history(10).await.len(), 2);
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn one_attempt_is_the_default() {
    let (lm, _) = model(vec![answer("first", 0), answer("unused", 1)], false).await;
    let module = Predict::<QA>::builder()
        .lm(lm.clone())
        .build()
        .validated()
        .validator(|_, _| Err("Rejected.".into()));
    assert!(module.call(input()).await.is_err());
    assert_eq!(lm.inspect_history(10).await.len(), 1);
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn soft_checks_stay_informational_unless_enabled() {
    let (lm, _) = model(vec![answer("x", 1)], false).await;
    let module = Predict::<QA>::builder()
        .lm(lm)
        .build()
        .validated()
        .max_attempts(attempts(2));
    let result = module.call(input()).await.unwrap();
    assert_eq!(result.answer, "x");
    assert!(result.metadata().has_failed_checks());
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn enabled_soft_checks_are_sent_back_with_field_and_label() {
    let (lm, client) = model(vec![answer("x", 1), answer("long enough", 1)], false).await;
    let module = Predict::<QA>::builder()
        .lm(lm)
        .build()
        .validated()
        .max_attempts(attempts(2))
        .retry_failed_checks(true);
    let result = module.call(input()).await.unwrap();
    assert!(!result.metadata().has_failed_checks());
    let prompt = format!("{:?}", client.last_request().unwrap().chat_history);
    assert!(prompt.contains("answer_length"));
    assert!(prompt.contains("this|length >= 4"));
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn hard_asserts_are_repaired_and_custom_callbacks_only_see_parsed_values() {
    let (lm, client) = model(
        vec![
            "[[ ## number ## ]]\n-1\n\n[[ ## completed ## ]]".into(),
            "[[ ## number ## ]]\n2\n\n[[ ## completed ## ]]".into(),
        ],
        false,
    )
    .await;
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = calls.clone();
    let module = Predict::<Positive>::builder()
        .lm(lm)
        .build()
        .validated()
        .max_attempts(attempts(2))
        .validator(move |_, output| {
            observed.fetch_add(1, Ordering::SeqCst);
            assert!(output.number > 0);
            Ok(())
        });
    assert_eq!(
        module
            .call(PositiveInput {
                question: "number".into()
            })
            .await
            .unwrap()
            .number,
        2
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let prompt = format!("{:?}", client.last_request().unwrap().chat_history);
    assert!(prompt.contains("positive"));
    assert!(prompt.contains("this > 0"));
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn malformed_output_gets_a_corrective_turn() {
    let (lm, client) = model(vec!["unformatted".into(), answer("good", 1)], false).await;
    let module = Predict::<QA>::builder()
        .lm(lm)
        .build()
        .validated()
        .max_attempts(attempts(2));
    let result = module.call(input()).await.unwrap();
    assert_eq!(result.answer, "good");
    assert_eq!(result.metadata().lm_usage.total_tokens, 16);
    let prompt = format!("{:?}", client.last_request().unwrap().chat_history);
    assert!(prompt.contains("unformatted"));
    assert!(prompt.contains("answer"));
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn provider_failures_are_not_retried_as_validation_failures() {
    let (lm, _) = model(vec![], false).await;
    let module = Predict::<QA>::builder()
        .lm(lm.clone())
        .build()
        .validated()
        .max_attempts(attempts(3));
    assert!(matches!(
        module.call(input()).await,
        Err(PredictError::Lm { .. })
    ));
    assert_eq!(lm.inspect_history(10).await.len(), 1);
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
#[cfg(feature = "cache")]
async fn corrective_messages_do_not_replay_an_invalid_cached_response() {
    let (lm, _) = model(vec![answer("bad", 0), answer("good", 1)], true).await;
    let predict = Predict::<QA>::builder().lm(lm.clone()).build();
    predict.call(input()).await.unwrap();
    let module = predict
        .validated()
        .max_attempts(attempts(2))
        .validator(|_, output| {
            if output.answer == "good" {
                Ok(())
            } else {
                Err("Return good.".into())
            }
        });
    let result = module.call(input()).await.unwrap();
    assert_eq!(result.answer, "good");
    assert_eq!(result.metadata().lm_usage.total_tokens, 8);
    let history = lm.inspect_history(10).await;
    assert_eq!(history.len(), 3);
    assert!(history[1].cache_hit);
    assert!(!history[0].cache_hit);
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn module_overrides_and_optimizer_discovery_are_preserved() {
    let (original, _) = model(vec![], false).await;
    let (explicit, _) = model(vec![answer("good", 1)], false).await;
    let mut module = Predict::<QA>::builder()
        .lm(original.clone())
        .build()
        .validated();
    assert_eq!(module.predictors().len(), 1);
    assert_eq!(module.predictors_mut().len(), 1);
    assert_eq!(
        module.call_with_lm(input(), explicit).await.unwrap().answer,
        "good"
    );
    assert!(original.inspect_history(1).await.is_empty());
}

#[derive(Clone)]
struct Counter(Arc<AtomicUsize>);
impl Tool for Counter {
    const NAME: &'static str = "counter";
    type Args = serde_json::Value;
    type Output = String;
    type Error = std::convert::Infallible;
    async fn definition(&self, _: String) -> ToolDefinition {
        ToolDefinition {
            name: "counter".into(),
            description: "Count".into(),
            parameters: serde_json::json!({}),
        }
    }
    async fn call(&self, _: Self::Args) -> Result<String, Self::Error> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok("done".into())
    }
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn corrective_retries_reject_attached_tools_before_any_execution() {
    let (lm, client) = model(vec![answer("unused", 1)], false).await;
    let calls = Arc::new(AtomicUsize::new(0));
    let module = Predict::<QA>::builder()
        .lm(lm.clone())
        .with_tools(vec![
            Arc::new(Counter(calls.clone())) as Arc<dyn rig::tool::ToolDyn>
        ])
        .build()
        .validated()
        .max_attempts(attempts(2));
    assert!(matches!(module.call(input()).await,
        Err(PredictError::Lm { source: LmError::Provider { provider, .. } })
        if provider == "validation configuration"));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(client.last_request().is_none());
    assert!(lm.inspect_history(1).await.is_empty());
}
