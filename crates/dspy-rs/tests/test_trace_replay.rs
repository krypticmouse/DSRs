//! Strict replay (RFC 0001 §4d): a recorded trace serves a pipeline's LM calls
//! verbatim — zero provider calls — and any drift from the recording surfaces
//! as a typed divergence error.

use dspy_rs::{
    LM, LMClient, Predict, PredictError, Predicted, ReplayError, ReplayMode, Signature,
    TestCompletionModel, Trace, capture, is_replaying, replay,
};
use rig::completion::AssistantContent;
use rig::message::Text;

fn response_with_fields(fields: &[(&str, &str)]) -> String {
    let mut response = String::new();
    for (name, value) in fields {
        response.push_str(&format!("[[ ## {name} ## ]]\n{value}\n\n"));
    }
    response.push_str("[[ ## completed ## ]]\n");
    response
}

fn answer(text: &str) -> AssistantContent {
    AssistantContent::Text(Text {
        text: response_with_fields(&[("answer", text)]),
    })
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
struct RepQA {
    #[input]
    question: String,

    #[output]
    answer: String,
}

/// A 3-step pipeline: drafter twice (its second input depends on its first
/// output), then a refiner over both drafts.
async fn run_pipeline(
    drafter: &Predict<RepQA>,
    refiner: &Predict<RepQA>,
) -> Result<Predicted<RepQAOutput>, PredictError> {
    let first = drafter
        .call(RepQAInput {
            question: "q0".to_string(),
        })
        .await?;
    let second = drafter
        .call(RepQAInput {
            question: format!("expand: {}", first.answer),
        })
        .await?;
    refiner
        .call(RepQAInput {
            question: format!("refine: {} + {}", first.answer, second.answer),
        })
        .await
}

/// Records the pipeline once on canned responses and returns the trace plus
/// the live run's final answer.
async fn record_pipeline() -> (Trace, String, Predict<RepQA>, Predict<RepQA>) {
    let drafter = Predict::<RepQA>::builder()
        .named("drafter")
        .lm(make_test_lm(vec![answer("a0"), answer("a1")]).await)
        .build();
    let refiner = Predict::<RepQA>::builder()
        .named("refiner")
        .lm(make_test_lm(vec![answer("final")]).await)
        .build();

    let (result, trace) = capture(|| run_pipeline(&drafter, &refiner)).await;
    let out = result.expect("recorded run should succeed");
    assert_eq!(trace.spans.len(), 3);
    (trace, out.answer.clone(), drafter, refiner)
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn strict_replay_serves_pipeline_with_zero_live_calls() {
    let (trace, recorded_answer, drafter, refiner) = record_pipeline().await;

    // Fixture path: replay from the JSONL round trip, exactly what a
    // checked-in `.trace.jsonl` fixture would go through.
    let fixture = trace.to_jsonl().expect("serialize fixture");
    let fixture = Trace::from_jsonl(&fixture).expect("parse fixture");

    // The predictors' canned queues were drained by the recording run: any
    // live LM call inside the replay would fail with an empty-queue error.
    let (result, report) = replay(&fixture, ReplayMode::Strict, || async {
        assert!(is_replaying());
        run_pipeline(&drafter, &refiner).await
    })
    .await;
    assert!(!is_replaying());

    let out = result.expect("strict replay should serve every call");
    assert_eq!(out.answer, recorded_answer, "identical outputs");
    assert_eq!(
        out.metadata().raw_response,
        response_with_fields(&[("answer", "final")]),
        "served metadata carries the recorded raw output"
    );

    assert_eq!(report.served, 3);
    assert_eq!(report.live, 0, "zero live calls");
    assert!(report.diverged_at.is_none());
    assert!(report.divergence.is_none());
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn strict_replay_diverges_on_mutated_instruction() {
    let (trace, _, _, _) = record_pipeline().await;

    // Same pipeline shape, same model config, but the refiner's instruction is
    // mutated — its rendered prompt (and only its) changes.
    let drafter = Predict::<RepQA>::builder()
        .named("drafter")
        .lm(make_test_lm(vec![]).await)
        .build();
    let refiner = Predict::<RepQA>::builder()
        .named("refiner")
        .instruction("Be terse.")
        .lm(make_test_lm(vec![]).await)
        .build();

    let (result, report) =
        replay(&trace, ReplayMode::Strict, || run_pipeline(&drafter, &refiner)).await;

    let err = result.expect_err("mutated instruction must refuse strict replay");
    let refiner_span = &trace.spans[2];
    match err {
        PredictError::Replay {
            source:
                ReplayError::Divergence {
                    component,
                    seq,
                    expected_span,
                    expected_hash,
                    got_hash,
                },
        } => {
            assert_eq!(component, "refiner");
            assert_eq!(seq, 0, "first refiner invocation diverges");
            assert_eq!(expected_span, refiner_span.id);
            assert_eq!(expected_hash, refiner_span.request_hash);
            assert_ne!(got_hash, expected_hash);
        }
        other => panic!("expected Divergence, got {other:?}"),
    }

    // The unchanged prefix was served before the refusal; nothing went live.
    assert_eq!(report.served, 2);
    assert_eq!(report.live, 0);
    assert_eq!(report.diverged_at, Some(refiner_span.id));
    assert!(matches!(
        report.divergence,
        Some(ReplayError::Divergence { ref component, seq: 0, .. }) if component == "refiner"
    ));
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn strict_replay_reports_divergence_seq_for_repeat_invocations() {
    let (trace, _, drafter, refiner) = record_pipeline().await;

    // First drafter call replays; the second sends a different prompt, so the
    // divergence lands at (drafter, seq 1) — loops are addressable.
    let (result, report) = replay(&trace, ReplayMode::Strict, || async {
        let first = drafter
            .call(RepQAInput {
                question: "q0".to_string(),
            })
            .await?;
        drafter
            .call(RepQAInput {
                question: format!("CHANGED: {}", first.answer),
            })
            .await?;
        refiner
            .call(RepQAInput {
                question: "unreached".to_string(),
            })
            .await
    })
    .await;

    let err = result.expect_err("changed second input must diverge");
    match err {
        PredictError::Replay {
            source: ReplayError::Divergence { component, seq, expected_span, .. },
        } => {
            assert_eq!(component, "drafter");
            assert_eq!(seq, 1);
            assert_eq!(expected_span, trace.spans[1].id);
        }
        other => panic!("expected Divergence at drafter seq 1, got {other:?}"),
    }
    assert_eq!(report.served, 1);
    assert_eq!(report.live, 0);
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn strict_replay_refuses_incomplete_spans() {
    let (mut trace, _, drafter, refiner) = record_pipeline().await;
    // Redaction/truncation marks spans incomplete; those refuse strict replay.
    trace.spans[0].complete = false;

    let (result, report) =
        replay(&trace, ReplayMode::Strict, || run_pipeline(&drafter, &refiner)).await;

    let err = result.expect_err("incomplete span must refuse strict replay");
    match err {
        PredictError::Replay {
            source: ReplayError::Incomplete { component, seq, span },
        } => {
            assert_eq!(component, "drafter");
            assert_eq!(seq, 0);
            assert_eq!(span, trace.spans[0].id);
        }
        other => panic!("expected Incomplete, got {other:?}"),
    }
    assert_eq!(report.served, 0);
    assert_eq!(report.live, 0);
    assert_eq!(report.diverged_at, Some(trace.spans[0].id));
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn strict_replay_exhausts_when_live_run_calls_more() {
    let (trace, _, drafter, refiner) = record_pipeline().await;

    let (result, report) = replay(&trace, ReplayMode::Strict, || async {
        run_pipeline(&drafter, &refiner).await?;
        // A fourth call the recording never saw.
        drafter
            .call(RepQAInput {
                question: "q0".to_string(),
            })
            .await
    })
    .await;

    let err = result.expect_err("extra call must exhaust the trace");
    match err {
        PredictError::Replay {
            source: ReplayError::Exhausted { component, seq },
        } => {
            assert_eq!(component, "drafter");
            assert_eq!(seq, 2);
        }
        other => panic!("expected Exhausted, got {other:?}"),
    }
    assert_eq!(report.served, 3, "the recorded pipeline itself replays fine");
    assert!(report.diverged_at.is_none(), "no recorded span to blame");
    assert!(matches!(
        report.divergence,
        Some(ReplayError::Exhausted { .. })
    ));
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn replayed_spans_are_recorded_into_an_inner_capture_scope() {
    let (trace, _, drafter, refiner) = record_pipeline().await;

    // RFC §4e composition: replay outside, capture inside — the replayed
    // rollout is itself a full trace.
    let ((result, new_trace), report) = replay(&trace, ReplayMode::Strict, || {
        capture(|| run_pipeline(&drafter, &refiner))
    })
    .await;
    result.expect("replayed pipeline should succeed");

    assert_eq!(report.served, 3);
    assert_eq!(new_trace.spans.len(), 3, "served spans are re-recorded");
    for (old, new) in trace.spans.iter().zip(&new_trace.spans) {
        assert_eq!(old.request_hash, new.request_hash);
        assert_eq!(old.output, new.output);
        assert_eq!(old.events.len(), new.events.len());
    }
}
