use dspy_rs::{
    LM, LMClient, Module, Predict, PredictError, Predicted, ReplayMode, Signature,
    TestCompletionModel, Trace, capture, replay,
};
use rig::completion::AssistantContent;

#[derive(Signature, Clone, Debug)]
/// Transform the input.
struct Step {
    #[input]
    question: String,
    #[output]
    answer: String,
}

struct Pipeline {
    first: Predict<Step>,
    second: Predict<Step>,
}
dspy_rs::predictors!(Pipeline { first, second });
impl Module for Pipeline {
    type Input = StepInput;
    type Output = StepOutput;
    async fn forward(&self, input: Self::Input) -> Result<Predicted<Self::Output>, PredictError> {
        let first = self.first.call(input).await?.into_inner();
        self.second
            .call(StepInput {
                question: first.answer,
            })
            .await
    }
}

async fn model(responses: &[&str]) -> LM {
    LM::builder()
        .base_url("http://localhost:1/v1".into())
        .model("trace-history".into())
        .cache(false)
        .build()
        .await
        .unwrap()
        .with_client(LMClient::Test(TestCompletionModel::new(
            responses.iter().map(|answer| {
                AssistantContent::text(format!(
                    "[[ ## answer ## ]]\n{answer}\n\n[[ ## completed ## ]]"
                ))
            }),
        )))
        .await
        .unwrap()
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn runtime_trace_history_and_replay_preserve_composed_dataflow() {
    let lm = model(&["first", "second"]).await;
    let module = Pipeline {
        first: Predict::<Step>::builder()
            .lm(lm.clone())
            .named("first")
            .build(),
        second: Predict::<Step>::builder()
            .lm(lm.clone())
            .named("second")
            .build(),
    }
    .with_history("pipeline", 10);
    let (result, trace) = capture(|| {
        module.call(StepInput {
            question: "seed".into(),
        })
    })
    .await;
    assert_eq!(result.unwrap().answer, "second");
    assert_eq!(trace.spans.len(), 2);
    assert_eq!(trace.spans[0].input.as_ref().unwrap()["question"], "seed");
    assert_eq!(trace.spans[1].input.as_ref().unwrap()["question"], "first");
    let entries = module.inspect_history(10).await;
    assert_eq!(entries.len(), 2);
    for span in &trace.spans {
        let name = trace.component_name(span.component);
        let entry = entries
            .iter()
            .find(|entry| entry.module_path.last().map(String::as_str) == Some(name))
            .unwrap();
        assert!(span.complete);
        assert_eq!(entry.module_path[0], "pipeline");
        assert_eq!(entry.output, span.output);
        assert_eq!(entry.raw_output, span.raw_output);
        assert_eq!(
            serde_json::to_value(&entry.events).unwrap(),
            serde_json::to_value(&span.events).unwrap()
        );
        let prompt = dspy_rs::Chat::new(trace.prompt(span)).to_json();
        assert_eq!(prompt, dspy_rs::Chat::new(entry.messages.clone()).to_json());
    }
    let restored = Trace::from_jsonl(&trace.to_jsonl().unwrap()).unwrap();
    let (result, report) = replay(&restored, ReplayMode::Strict, || {
        module.call(StepInput {
            question: "seed".into(),
        })
    })
    .await;
    assert_eq!(result.unwrap().answer, "second");
    assert_eq!(report.served, 2);
    assert_eq!(report.live, 0);
    assert_eq!(module.inspect_history(10).await.len(), 2);
    assert_eq!(lm.inspect_history(10).await.len(), 2);
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn parse_failure_is_the_same_typed_failure_in_trace_and_history() {
    let client = TestCompletionModel::new([AssistantContent::text("unformatted")]);
    let lm = model(&[])
        .await
        .with_client(LMClient::Test(client))
        .await
        .unwrap();
    let predictor = Predict::<Step>::builder()
        .lm(lm.clone())
        .named("broken")
        .build();
    let (result, trace) = capture(|| {
        predictor.call(StepInput {
            question: "q".into(),
        })
    })
    .await;
    assert!(matches!(result, Err(PredictError::Parse { .. })));
    let entries = lm.inspect_history(1).await;
    assert_eq!(entries.len(), 1);
    let span = &trace.spans[0];
    assert_eq!(
        span.error.as_ref().unwrap().kind,
        dspy_rs::SpanErrorKind::Parse
    );
    assert_eq!(
        entries[0].error.as_ref().unwrap().kind,
        span.error.as_ref().unwrap().kind
    );
    assert_eq!(entries[0].raw_output, span.raw_output);
    assert!(entries[0].output.is_none());
    assert!(span.output.is_none());
}
