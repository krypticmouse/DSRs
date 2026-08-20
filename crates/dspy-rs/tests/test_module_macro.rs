//! RFC 0003 M-2/M-3: `__dsrs_step()` metadata on `#[predict]`/`#[cot]`, and
//! `#[module]` lowering an ordinary async fn body into an IR program — one
//! parse, two projections. Covers: metadata, the lowered artifact (extern
//! hole, leaf names, Main sig), OPACITY, end-to-end execution through the
//! interpreter, and ambient-overlay mutation of a step instruction.

use std::sync::{Arc, LazyLock};

use dspy_rs::ir::{Overlay, StepKind, with_ambient_overlay};
use dspy_rs::trace::capture;
use dspy_rs::{
    LM, LMClient, TestCompletionModel, configure, cot, module, predict,
};
use rig::completion::AssistantContent;
use rig::message::Text;
use tokio::sync::Mutex;

static SETTINGS_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

fn response_with_fields(fields: &[(&str, &str)]) -> AssistantContent {
    let mut response = String::new();
    for (name, value) in fields {
        response.push_str(&format!("[[ ## {name} ## ]]\n{value}\n\n"));
    }
    response.push_str("[[ ## completed ## ]]\n");
    AssistantContent::Text(Text { text: response })
}

async fn make_test_lm(responses: Vec<AssistantContent>) -> (LM, TestCompletionModel) {
    let client = TestCompletionModel::new(responses);
    let lm = temp_env::async_with_vars(
        [("OPENAI_API_KEY", Some("test"))],
        LM::builder()
            .model("openai:gpt-4o-mini".to_string())
            .build(),
    )
    .await
    .unwrap()
    .with_client(LMClient::Test(client.clone()))
    .await
    .unwrap();
    (lm, client)
}

// ---------------------------------------------------------------------------
// Steps
// ---------------------------------------------------------------------------

/// Answer the question.
#[predict]
fn draft(question: String) -> String;

/// Rate the text from 1-10.
#[predict]
fn rate(text: String) -> String;

/// Summarize deeply.
#[cot(model = "@deep")]
fn summarize(text: String) -> String;

// ---------------------------------------------------------------------------
// The module
// ---------------------------------------------------------------------------

#[dspy_rs::Schema]
#[derive(Debug)]
pub struct QaOut {
    pub answer: String,
    pub upper: String,
}

#[module]
async fn qa(question: String) -> Result<QaOut, dspy_rs::ir::RunError> {
    let drafter = draft(question.clone()).await?;
    let upper: String = {
        let d: String = drafter.draft;
        d.to_uppercase()
    };
    let checker = rate(upper.clone()).await?;
    Ok(QaOut {
        answer: checker.rate,
        upper,
    })
}

// ---------------------------------------------------------------------------
// M-2: step metadata
// ---------------------------------------------------------------------------

#[test]
fn predict_step_metadata() {
    let step = draft::__dsrs_step();
    assert_eq!(step.name, "draft");
    assert_eq!(step.kind, StepKind::Predict);
    assert_eq!(step.model, None);
    assert!(step.agent.is_none());
    assert_eq!(step.sig.instruction.as_ref(), "Answer the question.");
    assert_eq!(step.sig.inputs.len(), 1);
    assert_eq!(step.sig.inputs[0].name.as_ref(), "question");
    assert_eq!(step.sig.outputs.len(), 1);
    assert_eq!(step.sig.outputs[0].name.as_ref(), "draft");
}

#[test]
fn cot_step_metadata_with_model_ref() {
    let step = summarize::__dsrs_step();
    assert_eq!(step.kind, StepKind::Cot);
    assert_eq!(step.model, Some("deep"), "leading @ is stripped");
    // Base (un-augmented) signature: lowering adds `reasoning`.
    assert_eq!(step.sig.outputs.len(), 1);
    assert_eq!(step.sig.outputs[0].name.as_ref(), "summarize");
}

// ---------------------------------------------------------------------------
// M-3: the lowered artifact
// ---------------------------------------------------------------------------

#[test]
fn module_lowers_to_a_program_with_an_extern_hole() {
    let program = qa::program();
    let printed = program.to_dsrs();

    // Leaves: two predicts and one extern hole, named after the bindings.
    for leaf in ["drafter", "upper", "checker"] {
        assert!(
            program.param_id(&format!("{leaf}.instruction")).is_some()
                || printed.contains(leaf),
            "leaf `{leaf}` missing from the program"
        );
    }
    assert!(
        printed.contains("extern \""),
        "the hole prints as an extern binding:\n{printed}"
    );
    assert!(printed.contains("hole upper_hole"), "hole sig named after the binding");

    // Main sig inferred from step signatures (never guessed).
    let main = &program.sigs[program.sig];
    assert_eq!(main.inputs.len(), 1);
    assert_eq!(main.inputs[0].name.as_ref(), "question");
    assert_eq!(main.outputs.len(), 2);

    // The optimization surface: the predict steps' instructions are slots.
    assert!(program.param_id("drafter.instruction").is_some());
    assert!(program.param_id("checker.instruction").is_some());

    // Round-trip: the printed artifact reparses to the same hash.
    let reparsed = dspy_rs::ir::Program::from_dsrs(&printed).unwrap();
    assert_eq!(reparsed.meta.program_hash, program.meta.program_hash);
}

#[test]
fn opacity_reports_the_hole() {
    assert_eq!(qa::OPACITY.len(), 1);
    let hole = &qa::OPACITY[0];
    assert_eq!(hole.name, "upper");
    assert_eq!(hole.kind, "host");
    assert!(hole.excerpt.contains("to_uppercase"));
}

// ---------------------------------------------------------------------------
// M-3: execution — both projections, one behavior
// ---------------------------------------------------------------------------

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn module_fn_runs_through_the_interpreter_and_reads_the_ambient_overlay() {
    let _lock = SETTINGS_LOCK.lock().await;
    let (lm, client) = make_test_lm(vec![
        response_with_fields(&[("draft", "hello world")]),
        response_with_fields(&[("rate", "9")]),
    ])
    .await;
    configure(lm);

    // Plain call: typed boundary in, typed boundary out.
    let (out, trace) = capture(|| qa("greet the world".to_string())).await;
    let out = out.expect("module run succeeds");
    assert_eq!(out.upper, "HELLO WORLD", "the host hole ran natively");
    assert_eq!(out.answer, "9");

    // One span per leaf, named after the bindings, in body order.
    assert_eq!(trace.components, vec!["drafter", "upper", "checker"]);
    assert_ne!(trace.spans[1].request_hash, 0, "hole spans carry a preimage");

    // Ambient overlay: mutate the drafter's instruction gene; the program is
    // untouched — the interpreter reads through the overlay at render time.
    let program = qa::program();
    let mut overlay = Overlay::new(program);
    let slot = program
        .slot_of::<dspy_rs::ir::Instruction>("drafter.instruction")
        .expect("drafter.instruction is an Instruction slot");
    overlay.set_instruction(slot, "OVERRIDE-INSTRUCTION-MARKER");

    client.push_response(response_with_fields(&[("draft", "second")]));
    client.push_response(response_with_fields(&[("rate", "3")]));
    let (out, trace) = capture(|| {
        with_ambient_overlay(Arc::new(overlay), qa("again".to_string()))
    })
    .await;
    let out = out.expect("overlaid run succeeds");
    assert_eq!(out.upper, "SECOND");

    // The drafter's rendered system prompt carries the overlay's
    // instruction; the program itself is untouched.
    let drafter_span = &trace.spans[0];
    let prefix = &trace.prefixes[drafter_span.prefix.expect("predicts render a prefix").0 as usize];
    let rendered = format!("{:?}", prefix.messages);
    assert!(
        rendered.contains("OVERRIDE-INSTRUCTION-MARKER"),
        "the rendered prompt carries the overlay's instruction: {rendered}"
    );
    assert!(
        qa::program().to_dsrs().contains("Answer the question."),
        "the artifact keeps the default instruction"
    );

    // Strict replay through the module fn (M-1 ∘ M-3): the recorded run
    // serves every leaf — predicts AND the host hole — with zero live calls.
    client.push_response(response_with_fields(&[("draft", "replayable")]));
    client.push_response(response_with_fields(&[("rate", "7")]));
    let (baseline, base_trace) = capture(|| qa("replay me".to_string())).await;
    let baseline = baseline.expect("recording run succeeds");

    let (replayed, report) = dspy_rs::replay(&base_trace, dspy_rs::ReplayMode::Strict, || {
        qa("replay me".to_string())
    })
    .await;
    let replayed = replayed.expect("strict replay succeeds");
    assert_eq!(replayed.answer, baseline.answer);
    assert_eq!(replayed.upper, baseline.upper);
    assert_eq!(report.served, 3, "drafter + upper (hole) + checker all served");
    assert_eq!(report.live, 0);
}
