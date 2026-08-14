//! Unified trace format (RFC 0001) coverage: capture scoping, span recording,
//! tool-loop event ordering, JSONL round-tripping, and component slicing.

use dspy_rs::{
    Example, LM, LMClient, Message, Predict, Signature, SpanErrorKind, SpanEvent, TestCompletionModel,
    Trace, TraceMeta, begin_span, capture, capture_with_meta, is_capturing,
};
use rig::completion::AssistantContent;
use rig::completion::ToolDefinition;
use rig::message::Text;
use rig::tool::Tool;

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

async fn make_test_lm(responses: Vec<AssistantContent>) -> LM {
    let client = TestCompletionModel::new(responses);
    temp_env::async_with_vars(
        [("OPENAI_API_KEY", Some("test"))],
        LM::builder().model("openai:gpt-4o-mini".to_string()).build(),
    )
    .await
    .unwrap()
    .with_client(LMClient::Test(client))
    .await
    .unwrap()
}

#[derive(Signature, Clone, Debug)]
/// Answer the question.
struct CapQA {
    #[input]
    question: String,

    #[output]
    answer: String,
}

fn answer(text: &str) -> AssistantContent {
    text_response(response_with_fields(&[("answer", text)]))
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn capture_records_spans_with_component_names_and_seq() {
    let drafter = Predict::<CapQA>::builder()
        .named("drafter")
        .demo(Example::new(
            CapQAInput {
                question: "demo-q".to_string(),
            },
            CapQAOutput {
                answer: "demo-a".to_string(),
            },
        ))
        .lm(make_test_lm(vec![answer("a0"), answer("a1")]).await)
        .build();
    let refiner = Predict::<CapQA>::builder()
        .named("refiner")
        .lm(make_test_lm(vec![answer("refined")]).await)
        .build();

    let (result, trace) = capture(|| async {
        drafter
            .call(CapQAInput {
                question: "q0".to_string(),
            })
            .await?;
        drafter
            .call(CapQAInput {
                question: "q1".to_string(),
            })
            .await?;
        refiner
            .call(CapQAInput {
                question: "refine".to_string(),
            })
            .await
    })
    .await;
    result.expect("captured calls should succeed");

    assert_eq!(trace.meta.v, 1);
    assert!(!trace.meta.trace_id.is_empty());
    assert_eq!(trace.spans.len(), 3);
    assert_eq!(trace.components, vec!["drafter", "refiner"]);

    // (component, seq) addressing: drafter called twice, refiner once.
    let drafter_spans: Vec<_> = trace.for_component("drafter").collect();
    assert_eq!(drafter_spans.len(), 2);
    assert_eq!(drafter_spans[0].seq, 0);
    assert_eq!(drafter_spans[1].seq, 1);
    assert_eq!(trace.for_component("refiner").count(), 1);
    assert_eq!(trace.for_component("missing").count(), 0);

    // Sequential links: each span points at its predecessor.
    assert!(trace.spans[0].links.is_empty());
    assert_eq!(trace.spans[1].links, vec![trace.spans[0].id]);
    assert_eq!(trace.spans[2].links, vec![trace.spans[1].id]);

    // Prefix interning: both drafter calls share one prefix entry (system +
    // demo turns); the refiner has its own.
    let d0 = drafter_spans[0];
    let d1 = drafter_spans[1];
    assert_eq!(d0.prefix, d1.prefix);
    let prefix_id = d0.prefix.expect("typed calls record a prefix");
    // system + 1 demo (user/assistant pair)
    assert_eq!(trace.prefixes[prefix_id.0 as usize].messages.len(), 3);

    // prompt() reconstructs prefix ++ suffix.
    let prompt = trace.prompt(d0);
    assert_eq!(prompt.len(), 4);
    assert!(prompt.last().unwrap().content().contains("q0"));

    // Typed input/output captured as JSON maps.
    assert_eq!(d0.input.as_ref().unwrap()["question"], "q0");
    assert_eq!(d0.output.as_ref().unwrap()["answer"], "a0");
    assert_eq!(d0.raw_output.as_deref(), Some(&*response_with_fields(&[("answer", "a0")])));
    assert!(d0.error.is_none());
    assert!(d0.complete);
    assert_eq!(trace.successes().count(), 3);

    // Same component+config, different input → different request hash.
    assert_ne!(d0.request_hash, 0);
    assert_ne!(d0.request_hash, d1.request_hash);

    // Every span carries the final Exchange event even without tools.
    assert!(matches!(d0.events.as_slice(), [SpanEvent::Exchange { .. }]));

    // Model interning: one distinct config per LM instance, deduplicated.
    assert_eq!(trace.models.len(), 1, "identical configs intern to one entry");
    assert!(trace.models[0].config.api_key.is_none());
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn no_scope_means_zero_capture() {
    assert!(!is_capturing());
    let predict = Predict::<CapQA>::builder()
        .named("solo")
        .lm(make_test_lm(vec![answer("out")]).await)
        .build();
    let out = predict
        .call(CapQAInput {
            question: "q".to_string(),
        })
        .await
        .expect("uncaptured call should succeed");
    assert_eq!(out.answer, "out");

    // begin_span outside a scope returns None (zero-overhead path).
    let config = dspy_rs::LMConfig::default();
    let guard = begin_span(dspy_rs::SpanRequest {
        component: "nobody",
        prefix: None,
        suffix: &[],
        input: None,
        model: &config,
    });
    assert!(guard.is_none());
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn nested_capture_scopes_are_exclusive() {
    let predict = Predict::<CapQA>::builder()
        .named("inner_comp")
        .lm(make_test_lm(vec![answer("inner")]).await)
        .build();

    let ((inner_result, inner_trace), outer_trace) = capture(|| async {
        assert!(is_capturing());
        capture(|| async {
            predict
                .call(CapQAInput {
                    question: "q".to_string(),
                })
                .await
        })
        .await
    })
    .await;

    inner_result.expect("inner call should succeed");
    // Innermost wins, exclusively: the outer trace records nothing for the
    // region covered by the inner scope.
    assert_eq!(inner_trace.spans.len(), 1);
    assert_eq!(outer_trace.spans.len(), 0);
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn jsonl_round_trip_preserves_the_trace() {
    let predict = Predict::<CapQA>::builder()
        .named("roundtrip")
        .lm(make_test_lm(vec![answer("serialized")]).await)
        .build();

    let (result, mut trace) = capture(|| async {
        predict
            .call(CapQAInput {
                question: "persist me".to_string(),
            })
            .await
    })
    .await;
    result.expect("call should succeed");
    trace.outcome = Some(dspy_rs::TraceOutcome {
        output: None,
        error: None,
        eval: Some(dspy_rs::Eval::with_feedback(0.8, "good")),
        duration_us: 42,
    });

    let jsonl = trace.to_jsonl().expect("serialize");
    assert_eq!(jsonl.lines().count(), 3, "header + span + footer");
    let parsed = Trace::from_jsonl(&jsonl).expect("deserialize");

    // Serialize→deserialize→serialize equality.
    assert_eq!(jsonl, parsed.to_jsonl().expect("re-serialize"));
    assert_eq!(parsed.spans.len(), 1);
    assert_eq!(parsed.component_name(parsed.spans[0].component), "roundtrip");
    assert_eq!(parsed.spans[0].request_hash, trace.spans[0].request_hash);
    let outcome = parsed.outcome.expect("footer parsed");
    assert_eq!(outcome.eval, Some(dspy_rs::Eval::with_feedback(0.8, "good")));
}

// --- Tool loop event ordering ----------------------------------------------

#[derive(Clone)]
struct EchoTool;

#[derive(Debug)]
struct EchoToolError;

impl std::fmt::Display for EchoToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "echo tool error")
    }
}

impl std::error::Error for EchoToolError {}

impl Tool for EchoTool {
    const NAME: &'static str = "echo";
    type Error = EchoToolError;
    type Args = serde_json::Value;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "echoes its arguments".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "additionalProperties": true
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(format!("echo:{args}"))
    }
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn tool_loop_events_are_ordered_within_one_span() {
    let tool_call = AssistantContent::tool_call(
        "call_1",
        "echo".to_string(),
        serde_json::json!({"payload": "ping"}),
    );
    let predict = Predict::<CapQA>::builder()
        .named("agent")
        .add_tool(EchoTool)
        .lm(make_test_lm(vec![tool_call, answer("pong")]).await)
        .build();

    let (result, trace) = capture(|| async {
        predict
            .call(CapQAInput {
                question: "use the tool".to_string(),
            })
            .await
    })
    .await;
    result.expect("tool-looping call should succeed");

    // One Predict with one tool round-trip = ONE span with three events:
    // Exchange (tool call), ToolRun, Exchange (final text).
    assert_eq!(trace.spans.len(), 1);
    let span = &trace.spans[0];
    assert_eq!(span.events.len(), 3);
    match &span.events[0] {
        SpanEvent::Exchange { message, .. } => assert!(message.has_tool_calls()),
        other => panic!("expected initial Exchange, got {other:?}"),
    }
    match &span.events[1] {
        SpanEvent::ToolRun {
            id,
            name,
            args,
            result,
            error,
            ..
        } => {
            assert_eq!(id, "call_1");
            assert_eq!(name, "echo");
            assert_eq!(args["payload"], "ping");
            assert!(result.contains("echo:"));
            assert!(error.is_none());
        }
        other => panic!("expected ToolRun, got {other:?}"),
    }
    match &span.events[2] {
        SpanEvent::Exchange { message, .. } => {
            assert!(message.content().contains("pong"));
        }
        other => panic!("expected final Exchange, got {other:?}"),
    }

    // Tool structure survives the JSONL round trip.
    let parsed = Trace::from_jsonl(&trace.to_jsonl().unwrap()).unwrap();
    assert_eq!(parsed.spans[0].events.len(), 3);
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn parse_failure_leaves_error_span_with_raw_output() {
    let predict = Predict::<CapQA>::builder()
        .named("flaky")
        .lm(make_test_lm(vec![text_response("prose instead of fields")]).await)
        .build();

    let (result, trace) = capture(|| async {
        predict
            .call(CapQAInput {
                question: "q".to_string(),
            })
            .await
    })
    .await;
    result.expect_err("unparseable response should fail");

    assert_eq!(trace.spans.len(), 1);
    let span = &trace.spans[0];
    let error = span.error.as_ref().expect("span records the failure");
    assert_eq!(error.kind, SpanErrorKind::Parse);
    // The prompt and raw output stay visible for blame assignment/reflection.
    assert_eq!(span.input.as_ref().unwrap()["question"], "q");
    assert_eq!(span.raw_output.as_deref(), Some("prose instead of fields"));
    assert!(span.output.is_none());
    assert_eq!(trace.successes().count(), 0);
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn dropped_guard_marks_span_cancelled() {
    let ((), trace) = capture(|| async {
        let config = dspy_rs::LMConfig::default();
        let guard = begin_span(dspy_rs::SpanRequest {
            component: "doomed",
            prefix: None,
            suffix: &[Message::user("hello")],
            input: None,
            model: &config,
        })
        .expect("scope is active");
        drop(guard); // task dies before finish
    })
    .await;

    assert_eq!(trace.spans.len(), 1);
    let span = &trace.spans[0];
    assert_eq!(
        span.error.as_ref().map(|e| e.kind),
        Some(SpanErrorKind::Cancelled)
    );
    assert!(span.output.is_none());
}

// --- Metric-with-trace feedback path ----------------------------------------

#[derive(facet::Facet)]
#[facet(crate = facet)]
struct OneStep {
    predictor: Predict<CapQA>,
}

impl dspy_rs::Module for OneStep {
    type Input = CapQAInput;
    type Output = CapQAOutput;

    async fn forward(
        &self,
        input: CapQAInput,
    ) -> Result<dspy_rs::Predicted<CapQAOutput>, dspy_rs::PredictError> {
        self.predictor.call(input).await
    }
}

/// A metric that inspects the rollout's intermediate steps via the trace.
struct TraceInspectingMetric;

impl dspy_rs::TypedMetric<CapQA, OneStep> for TraceInspectingMetric {
    async fn evaluate(
        &self,
        _example: &dspy_rs::Example<CapQA>,
        prediction: &dspy_rs::Predicted<CapQAOutput>,
        trace: Option<&Trace>,
    ) -> anyhow::Result<dspy_rs::Eval> {
        let trace = trace.expect("evaluation loop always captures");
        let spans: Vec<_> = trace.for_component("step").collect();
        anyhow::ensure!(spans.len() == 1, "expected one step invocation");
        let feedback = format!(
            "step saw question={} and answered={}",
            spans[0].input.as_ref().unwrap()["question"], prediction.answer
        );
        Ok(dspy_rs::Eval::with_feedback(1.0, feedback))
    }
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn metric_reads_component_subtrace_during_evaluation() {
    let module = OneStep {
        predictor: Predict::<CapQA>::builder()
            .named("step")
            .lm(make_test_lm(vec![answer("42")]).await)
            .build(),
    };
    let trainset = vec![dspy_rs::Example::<CapQA>::new(
        CapQAInput {
            question: "meaning of life".to_string(),
        },
        CapQAOutput {
            answer: "42".to_string(),
        },
    )];

    let evals = dspy_rs::evaluate_trainset(&module, &trainset, &TraceInspectingMetric)
        .await
        .expect("evaluation should succeed");

    assert_eq!(evals.len(), 1);
    assert_eq!(evals[0].score, 1.0);
    let feedback = evals[0].feedback.as_deref().expect("feedback recorded");
    assert!(feedback.contains("meaning of life"));
    assert!(feedback.contains("answered=42"));
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn capture_with_meta_records_rollout_metadata() {
    let mut input = dspy_rs::JsonMap::new();
    input.insert("question".to_string(), serde_json::json!("rollout input"));
    let meta = TraceMeta {
        candidate_hash: Some(99),
        input: Some(input),
        tags: [("optimizer".to_string(), "test".to_string())].into(),
        ..TraceMeta::default()
    };

    let ((), trace) = capture_with_meta(meta, || async {}).await;
    assert_eq!(trace.meta.v, 1);
    assert!(!trace.meta.trace_id.is_empty(), "trace id is minted");
    assert!(trace.meta.started_at_us > 0);
    assert_eq!(trace.meta.candidate_hash, Some(99));
    assert_eq!(trace.meta.input.as_ref().unwrap()["question"], "rollout input");
    assert_eq!(trace.meta.tags["optimizer"], "test");
}
