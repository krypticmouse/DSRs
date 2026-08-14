//! Counterfactual replay (RFC 0001 §4e): `UntilDivergence` + a mutated
//! candidate. No "step k" parameter exists — divergence is *detected*, never
//! declared, because a mutated parameter changes the rendered prompt (and
//! therefore the request hash) of exactly the calls it affects. The unchanged
//! prefix is served free from the recording; the mutated call and everything
//! after it go live — and stay live even if a later prompt happens to match
//! the recording again.
//!
//! Uses the functional (`fx`) authoring style: the same pipeline function is
//! recorded once, then re-run under a `fx::Params` instruction overlay inside
//! `replay(.., UntilDivergence, || capture(..))` — the RFC's exact
//! counterfactual composition.

use dspy_rs::{
    LM, LMClient, PredictError, Predicted, ReplayError, ReplayMode, Signature,
    TestCompletionModel, capture, configure, fx, replay,
};
use rig::completion::AssistantContent;
use rig::message::Text;
use std::sync::LazyLock;
use tokio::sync::Mutex;

/// Serializes access to the process-global LM settings across tests.
static SETTINGS_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

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

async fn make_test_lm(responses: Vec<AssistantContent>) -> (LM, TestCompletionModel) {
    let client = TestCompletionModel::new(responses);
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
struct CfQA {
    #[input]
    question: String,

    #[output]
    answer: String,
}

/// A 3-component functional pipeline; each step feeds the next.
async fn pipeline(
    prefix: &'static str,
    question: String,
) -> Result<Predicted<CfQAOutput>, PredictError> {
    let draft = fx::predict::<CfQA>(
        &format!("{prefix}_drafter"),
        CfQAInput { question },
    )
    .await?;
    let mid = fx::predict::<CfQA>(
        &format!("{prefix}_middle"),
        CfQAInput {
            question: format!("improve: {}", draft.answer),
        },
    )
    .await?;
    fx::predict::<CfQA>(
        &format!("{prefix}_refiner"),
        CfQAInput {
            question: format!("polish: {}", mid.answer),
        },
    )
    .await
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn until_divergence_serves_prefix_and_goes_live_from_the_mutated_component() {
    let _lock = SETTINGS_LOCK.lock().await;
    let (lm, client) =
        make_test_lm(vec![answer("draft0"), answer("mid0"), answer("final0")]).await;
    configure(lm);

    // Record the baseline rollout.
    let (result, base_trace) = capture(|| pipeline("cf", "q".to_string())).await;
    let baseline = result.expect("baseline run should succeed");
    assert_eq!(baseline.answer, "final0");
    assert_eq!(base_trace.spans.len(), 3);

    // Counterfactual candidate: overlay ONE mid-pipeline component's
    // instruction through the public params seam.
    let mut candidate = fx::Params::new();
    candidate.set_instruction("cf_middle", "Be terse.");

    // Live responses for the diverged suffix ONLY — the drafter's queue stays
    // empty, so any live drafter call would fail loudly. The middle's live
    // response repeats the recorded answer, which makes the refiner's prompt
    // byte-identical to the recording: it must go live anyway.
    client.push_response(answer("mid0"));
    client.push_response(answer("final1"));

    let ((result, new_trace), report) =
        replay(&base_trace, ReplayMode::UntilDivergence, || {
            capture(|| fx::with_params(candidate, pipeline("cf", "q".to_string())))
        })
        .await;

    let out = result.expect("counterfactual run should succeed");
    assert_eq!(
        out.answer, "final1",
        "the refiner consumed the live queue even though its prompt re-matched the recording"
    );

    // Exactly the unchanged prefix was served; the mutated call and everything
    // after went live.
    assert_eq!(report.served, 1, "drafter only");
    assert_eq!(report.live, 2, "middle + refiner");
    assert_eq!(report.diverged_at, Some(base_trace.spans[1].id));
    match &report.divergence {
        Some(ReplayError::Divergence {
            component,
            seq,
            expected_span,
            expected_hash,
            got_hash,
        }) => {
            assert_eq!(component, "cf_middle");
            assert_eq!(*seq, 0);
            assert_eq!(*expected_span, base_trace.spans[1].id);
            assert_eq!(*expected_hash, base_trace.spans[1].request_hash);
            assert_ne!(*got_hash, *expected_hash);
        }
        other => panic!("expected Divergence at cf_middle seq 0, got {other:?}"),
    }

    // The mutated instruction actually reached the live middle call; the last
    // live request (the refiner's) saw the middle's output.
    let last = client.last_request().expect("live calls hit the test model");
    assert!(
        format!("{:?}", last.chat_history).contains("polish:"),
        "the final live request is the refiner's"
    );

    // The inner capture recorded the full counterfactual rollout: served
    // prefix identical to the recording, mutated span diverged, refiner's
    // prompt re-matched (same hash) yet was answered live.
    assert_eq!(new_trace.spans.len(), 3);
    assert_eq!(
        new_trace.spans[0].request_hash,
        base_trace.spans[0].request_hash
    );
    assert_eq!(
        new_trace.spans[0].output, base_trace.spans[0].output,
        "served span re-records the recorded output"
    );
    assert_ne!(
        new_trace.spans[1].request_hash,
        base_trace.spans[1].request_hash
    );
    assert_eq!(
        new_trace.spans[2].request_hash,
        base_trace.spans[2].request_hash,
        "post-divergence prompt re-matched the recording — and still went live"
    );
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn until_divergence_without_mutation_serves_everything() {
    let _lock = SETTINGS_LOCK.lock().await;
    let (lm, _client) =
        make_test_lm(vec![answer("d"), answer("m"), answer("r")]).await;
    configure(lm);

    let (result, base_trace) = capture(|| pipeline("cfv", "q".to_string())).await;
    result.expect("baseline run should succeed");

    // No candidate overlay: nothing diverges, the whole rollout is free. The
    // queue is drained, so any live call would fail.
    let (result, report) = replay(&base_trace, ReplayMode::UntilDivergence, || {
        pipeline("cfv", "q".to_string())
    })
    .await;

    let out = result.expect("unchanged pipeline should replay fully");
    assert_eq!(out.answer, "r");
    assert_eq!(report.served, 3);
    assert_eq!(report.live, 0);
    assert!(report.diverged_at.is_none());
    assert!(report.divergence.is_none());
}
