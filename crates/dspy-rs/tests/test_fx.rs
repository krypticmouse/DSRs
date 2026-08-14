//! Functional DSRs (`fx`) coverage: ambient params injection, predictor-instance
//! caching, named trace nodes, FnModule interop with the eval machinery, and
//! Params <-> ModuleState round-tripping.

use anyhow::Result;
use dspy_rs::{
    Example, LM, LMClient, Eval, Module, Predicted, Signature,
    TestCompletionModel, TypedMetric, configure, fx,
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

async fn make_test_lm(responses: Vec<String>) -> (LM, TestCompletionModel) {
    let client = TestCompletionModel::new(responses.into_iter().map(text_response));
    let lm = temp_env::async_with_vars(
        [("OPENAI_API_KEY", Some("test"))],
        LM::builder().model("openai:gpt-4o-mini".to_string()).build(),
    )
    .await
    .unwrap()
    .with_client(LMClient::Test(client.clone()))
    .await
    .unwrap();
    (lm, client)
}

#[derive(Signature, Clone, Debug)]
/// Answer the question.
struct FxQA {
    #[input]
    question: String,

    #[output]
    answer: String,
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn with_params_injects_instruction_ambiently() {
    let _lock = SETTINGS_LOCK.lock().await;
    let (lm, client) = make_test_lm(vec![
        response_with_fields(&[("answer", "default")]),
        response_with_fields(&[("answer", "tuned")]),
    ])
    .await;
    configure(lm);

    // Without a params scope: signature default instruction.
    let out = fx::predict::<FxQA>(
        "inject_test",
        FxQAInput {
            question: "q1".to_string(),
        },
    )
    .await
    .expect("default call should succeed");
    assert_eq!(out.answer, "default");
    let preamble = client.last_request().unwrap().preamble.unwrap_or_default();
    assert!(!preamble.contains("MARKER-INSTRUCTION"));

    // Same function, params injected: the instruction reaches the prompt.
    let mut params = fx::Params::new();
    params.set_instruction("inject_test", "MARKER-INSTRUCTION follow this style");

    let out = fx::with_params(
        params,
        fx::predict::<FxQA>(
            "inject_test",
            FxQAInput {
                question: "q2".to_string(),
            },
        ),
    )
    .await
    .expect("params call should succeed");
    assert_eq!(out.answer, "tuned");
    let preamble = client.last_request().unwrap().preamble.unwrap_or_default();
    assert!(
        preamble.contains("MARKER-INSTRUCTION"),
        "ambient params should override the system instruction"
    );
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn capture_names_spans_after_fx_slots() {
    let _lock = SETTINGS_LOCK.lock().await;
    let (lm, _client) = make_test_lm(vec![
        response_with_fields(&[("answer", "a")]),
        response_with_fields(&[("answer", "b")]),
        response_with_fields(&[("answer", "c")]),
    ])
    .await;
    configure(lm);

    let (result, trace) = dspy_rs::trace::capture(|| async {
        for i in 0..2 {
            fx::predict::<FxQA>(
                "trace_test",
                FxQAInput {
                    question: i.to_string(),
                },
            )
            .await?;
        }
        Ok::<_, dspy_rs::PredictError>(())
    })
    .await;
    result.expect("captured calls should succeed");

    // Spans record the fx slot name as their component, with per-invocation seq.
    assert_eq!(trace.spans.len(), 2);
    assert_eq!(trace.components, vec!["trace_test"]);
    let spans: Vec<_> = trace.for_component("trace_test").collect();
    assert_eq!(spans[0].seq, 0);
    assert_eq!(spans[1].seq, 1);
    // Same slot + same config: both calls share one interned prompt prefix.
    assert_eq!(spans[0].prefix, spans[1].prefix);

    // A different config renders (and records) a different prefix.
    let mut params = fx::Params::new();
    params.set_instruction("trace_test", "different config");
    let (result, trace) = dspy_rs::trace::capture(|| {
        fx::with_params(
            params,
            fx::predict::<FxQA>(
                "trace_test",
                FxQAInput {
                    question: "again".to_string(),
                },
            ),
        )
    })
    .await;
    result.expect("params call should succeed");
    let span = &trace.spans[0];
    let prompt = trace.prompt(span);
    assert!(
        prompt[0].content().contains("different config"),
        "injected instruction should appear in the recorded prompt prefix"
    );
}

struct EchoMatch;

impl<M> TypedMetric<FxQA, M> for EchoMatch
where
    M: Module<Input = FxQAInput, Output = FxQAOutput>,
{
    async fn evaluate(
        &self,
        example: &Example<FxQA>,
        prediction: &Predicted<FxQAOutput>,
        _trace: Option<&dspy_rs::Trace>,
    ) -> Result<Eval> {
        Ok(Eval::score(
            (prediction.answer == example.output.answer) as u8 as f64,
        ))
    }
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn fn_module_plugs_into_evaluate_trainset() {
    let _lock = SETTINGS_LOCK.lock().await;
    let responses: Vec<String> = (0..4)
        .map(|i| response_with_fields(&[("answer", format!("a{i}").as_str())]))
        .collect();
    let (lm, _client) = make_test_lm(responses).await;
    configure(lm);

    async fn harness(input: FxQAInput) -> Result<Predicted<FxQAOutput>, dspy_rs::PredictError> {
        fx::predict::<FxQA>("eval_test", input).await
    }

    let module = fx::module(harness);
    let trainset: Vec<Example<FxQA>> = (0..4)
        .map(|i| {
            Example::new(
                FxQAInput {
                    question: i.to_string(),
                },
                FxQAOutput {
                    answer: format!("a{i}"),
                },
            )
        })
        .collect();

    // Sequential concurrency: the shared test queue pops in call order, so
    // ordered evaluation maps response i to example i.
    let outcomes =
        dspy_rs::evaluate_trainset_with_concurrency(&module, &trainset, &EchoMatch, 1).await;
    let outcomes = outcomes.expect("evaluation should succeed");
    assert_eq!(outcomes.len(), 4);
    assert!(outcomes.iter().all(|o| o.score == 1.0));
}

#[test]
fn params_round_trip_through_module_state() {
    let mut params = fx::Params::new();
    params.set_instruction("stage_a", "tuned instruction");
    assert!(!params.is_empty());

    let state = params.to_module_state();
    let json = state.to_json().expect("serialize");
    let restored = fx::Params::from_module_state(
        dspy_rs::ModuleState::from_json(&json).expect("deserialize"),
    );

    assert_eq!(
        restored.get("stage_a").unwrap().instruction_override.as_deref(),
        Some("tuned instruction")
    );
}
