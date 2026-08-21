//! Structural (RFC 0004 §6): the sixth strategy — LM-guided edits over the
//! graph-edit calculus. The loop applies a chosen edit via `Program::edited`,
//! migrates the incumbent overlay with `migrate_overlay`, gates the child
//! against the parent on a shared minibatch, and degrades gracefully on every
//! rejection path (edit fails to apply, reflection reply doesn't parse,
//! budget exhausted).

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use dspy_rs::ir::{
    self, DemoRow, Edit, FieldType as T, Interpreter, Node, Overlay, ParamValue, Program,
    ProgramBuilder, RuntimeEnv, SignatureDef,
};
use dspy_rs::trace::JsonMap;
use dspy_rs::{
    Eval, LM, LMClient, LMConfig, ProgramMetric, Structural, TestCompletionModel, Trace,
};
use rig::completion::AssistantContent;
use rig::message::Text;
use serde_json::json;

fn fields(pairs: &[(&str, &str)]) -> String {
    let mut out = String::new();
    for (name, value) in pairs {
        out.push_str(&format!("[[ ## {name} ## ]]\n{value}\n\n"));
    }
    out.push_str("[[ ## completed ## ]]\n");
    out
}

fn text(content: impl Into<String>) -> AssistantContent {
    AssistantContent::Text(Text {
        text: content.into(),
    })
}

async fn canned_lm(responses: Vec<AssistantContent>) -> Arc<LM> {
    Arc::new(canned_lm_owned(responses).await)
}

async fn canned_lm_owned(responses: Vec<AssistantContent>) -> LM {
    let client = TestCompletionModel::new(responses);
    temp_env::async_with_vars(
        [("OPENAI_API_KEY", Some("test"))],
        LM::builder()
            .model("openai:gpt-4o-mini".to_string())
            .build(),
    )
    .await
    .unwrap()
    .with_client(LMClient::Test(client))
    .await
    .unwrap()
}

fn config() -> LMConfig {
    LMConfig {
        model: "openai:gpt-4o-mini".to_string(),
        ..LMConfig::default()
    }
}

fn obj(pairs: &[(&str, serde_json::Value)]) -> JsonMap {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.clone()))
        .collect()
}

fn example(question: &str, answer: &str) -> DemoRow {
    DemoRow {
        input: obj(&[("question", json!(question))]),
        output: obj(&[("answer", json!(answer))]),
    }
}

/// question → answerer (QA predict) → answer. The single-leaf menu, in
/// order: AugmentSig (0), SwapToAgent (1), WrapRetry (2), Remove (3).
fn qa_program() -> Program {
    let mut b = ProgramBuilder::new("structural");
    b.model("m", config());
    let qa = b.sig(
        SignatureDef::build("QA")
            .instruction("Answer the question.")
            .input("question", T::String)
            .output("answer", T::String)
            .finish()
            .unwrap(),
    );
    let node = ir::predict("answerer", qa).bind("question", ir::input("question"));
    b.main(
        qa,
        ir::seq([node]).out("answer", ir::out("answerer", "answer")),
    )
    .unwrap()
}

/// Same pipeline, but the leaf is already `cot`-augmented — `AugmentSig`
/// drops out of the menu, and every remaining option is safe for the canned
/// two-field response.
fn cot_program() -> Program {
    let mut b = ProgramBuilder::new("structural_cot");
    b.model("m", config());
    let main_sig = b.sig(
        SignatureDef::build("Main")
            .input("question", T::String)
            .output("answer", T::String)
            .finish()
            .unwrap(),
    );
    let qa = b.sig(
        SignatureDef::build("QA")
            .instruction("Answer the question.")
            .input("question", T::String)
            .output("answer", T::String)
            .finish()
            .unwrap(),
    );
    let node = ir::cot("answerer", qa).bind("question", ir::input("question"));
    b.main(
        main_sig,
        ir::seq([node]).out("answer", ir::out("answerer", "answer")),
    )
    .unwrap()
}

/// Plain exact-match on the labeled answer, with feedback text.
struct ExactMatch;

impl ProgramMetric for ExactMatch {
    async fn evaluate(
        &self,
        example: &DemoRow,
        output: &JsonMap,
        _trace: Option<&Trace>,
    ) -> Result<Eval> {
        let expected = example.output.get("answer");
        let got = output.get("answer");
        if got == expected {
            Ok(Eval::with_feedback(1.0, "correct"))
        } else {
            Ok(Eval::with_feedback(
                0.0,
                format!("expected {expected:?}, got {got:?}"),
            ))
        }
    }
}

async fn load(program: Program, lm: Arc<LM>) -> Interpreter {
    Interpreter::load(program, RuntimeEnv::new().bind_model("m", lm))
        .await
        .unwrap()
}

// ---------------------------------------------------------------------------
// The loop applies an edit and keeps the child only when it wins
// ---------------------------------------------------------------------------

#[tokio::test]
async fn accepts_the_child_when_it_wins_the_shared_minibatch() {
    // Parent answers wrong (score 0); the CoT-augmented child answers right.
    let parent_lm = canned_lm(vec![text(fields(&[("answer", "wrong")]))]).await;
    let interp = load(qa_program(), parent_lm).await;
    let parent_hash = interp.program().meta.program_hash;

    let child_lm = canned_lm(vec![text(fields(&[
        ("reasoning", "think"),
        ("answer", "right"),
    ]))])
    .await;
    // Reflection chooses option 0: AugmentSig on `answerer`.
    let reflection = canned_lm_owned(vec![text(fields(&[("chosen_option", "0")]))]).await;

    let examples = vec![example("q", "right")];
    let structural = Structural::builder()
        .num_iterations(1)
        .minibatch_size(1)
        .prompt_model(reflection)
        .seed(0)
        .build();

    let report = structural
        .compile_program(&interp, &examples, &ExactMatch, move || {
            RuntimeEnv::new().bind_model("m", child_lm.clone())
        })
        .await
        .unwrap();

    // The child won: the winner is a new program with the reasoning field.
    assert_ne!(report.program.meta.program_hash, parent_hash);
    assert_eq!(report.accepted, 1);
    assert_eq!(report.rejected, 0);
    assert_eq!(report.edits.len(), 1);
    assert!(matches!(report.edits[0], Edit::AugmentSig { .. }));
    assert_eq!(report.baseline_score, 0.0);
    assert_eq!(report.final_score, 1.0);

    let leaf = report.program.leaf_id("answerer").expect("leaf survives");
    let Node::Predict(node) = &report.program.nodes[leaf] else {
        panic!("answerer is still a predict leaf");
    };
    assert!(
        report.program.sigs[node.sig]
            .outputs
            .iter()
            .any(|f| &*f.name == "reasoning"),
        "the accepted edit augmented the leaf signature"
    );

    // Lineage points back at the parent.
    let lineage = report.program.meta.lineage.as_ref().unwrap();
    assert_eq!(
        lineage.parent.as_deref(),
        Some(format!("{parent_hash:016x}").as_str())
    );

    let step = &report.steps[0];
    assert!(step.accepted);
    assert_eq!(step.parent_hash, parent_hash);
    assert_eq!(step.parent_minibatch_score, 0.0);
    assert_eq!(step.child_minibatch_score, Some(1.0));
    assert_eq!(step.full_score, Some(1.0));
}

#[tokio::test]
async fn rejects_the_child_when_it_loses_the_shared_minibatch() {
    // Parent answers right (score 1); the child answers wrong.
    let parent_lm = canned_lm(vec![text(fields(&[("answer", "right")]))]).await;
    let interp = load(qa_program(), parent_lm).await;
    let parent_hash = interp.program().meta.program_hash;

    let child_lm = canned_lm(vec![text(fields(&[
        ("reasoning", "hmm"),
        ("answer", "wrong"),
    ]))])
    .await;
    let reflection = canned_lm_owned(vec![text(fields(&[("chosen_option", "0")]))]).await;

    let examples = vec![example("q", "right")];
    let structural = Structural::builder()
        .num_iterations(1)
        .minibatch_size(1)
        .prompt_model(reflection)
        .seed(0)
        .build();

    let report = structural
        .compile_program(&interp, &examples, &ExactMatch, move || {
            RuntimeEnv::new().bind_model("m", child_lm.clone())
        })
        .await
        .unwrap();

    // The parent stays the incumbent.
    assert_eq!(report.program.meta.program_hash, parent_hash);
    assert_eq!(report.accepted, 0);
    assert_eq!(report.rejected, 1);
    assert!(report.edits.is_empty());
    assert_eq!(report.baseline_score, 1.0);
    assert_eq!(report.final_score, 1.0);
    assert_eq!(report.overlay.base, parent_hash);

    let step = &report.steps[0];
    assert!(!step.accepted);
    assert_eq!(step.child_minibatch_score, Some(0.0));
    assert_eq!(step.full_score, None);
}

// ---------------------------------------------------------------------------
// Overlay migration preserves tuned values for surviving nodes
// ---------------------------------------------------------------------------

#[tokio::test]
async fn migrates_the_tuned_overlay_onto_the_winning_child() {
    let parent_lm = canned_lm(vec![text(fields(&[("answer", "wrong")]))]).await;
    let interp = load(qa_program(), parent_lm).await;
    let program = Arc::clone(interp.program());
    let parent_hash = program.meta.program_hash;

    // A prior value-level optimizer tuned the instruction.
    let slot = program
        .slot_of::<ir::Instruction>("answerer.instruction")
        .unwrap();
    let mut tuned = Overlay::new(&program);
    tuned.set_instruction(slot, "TUNED: answer tersely.");

    let child_lm = canned_lm(vec![text(fields(&[
        ("reasoning", "think"),
        ("answer", "right"),
    ]))])
    .await;
    let reflection = canned_lm_owned(vec![text(fields(&[("chosen_option", "0")]))]).await;

    let examples = vec![example("q", "right")];
    let structural = Structural::builder()
        .num_iterations(1)
        .minibatch_size(1)
        .prompt_model(reflection)
        .seed(0)
        .build();

    let report = structural
        .compile_program_with_overlay(&interp, Some(tuned), &examples, &ExactMatch, move || {
            RuntimeEnv::new().bind_model("m", child_lm.clone())
        })
        .await
        .unwrap();

    // AugmentSig widens outputs, so the tuned instruction survives, re-minted
    // against the winning child.
    assert_ne!(report.program.meta.program_hash, parent_hash);
    assert_eq!(report.overlay.base, report.program.meta.program_hash);
    let id = report.program.param_id("answerer.instruction").unwrap();
    assert_eq!(
        report.overlay.get(id),
        Some(&ParamValue::Instruction {
            text: "TUNED: answer tersely.".to_string()
        })
    );
}

// ---------------------------------------------------------------------------
// Rejection paths never crash the run
// ---------------------------------------------------------------------------

#[tokio::test]
async fn edit_that_fails_validation_is_recorded_and_skipped() {
    // Option 3 is Remove{answerer}: the program's out binding still
    // references the removed leaf, so `edited()` refuses (validate.rs's
    // error) and the generation is skipped without loading a child.
    let parent_lm = canned_lm(vec![text(fields(&[("answer", "right")]))]).await;
    let interp = load(qa_program(), parent_lm).await;
    let parent_hash = interp.program().meta.program_hash;

    let reflection = canned_lm_owned(vec![text(fields(&[("chosen_option", "3")]))]).await;
    let loads = Arc::new(AtomicUsize::new(0));
    let loads_seen = Arc::clone(&loads);

    let examples = vec![example("q", "right")];
    let structural = Structural::builder()
        .num_iterations(1)
        .minibatch_size(1)
        .prompt_model(reflection)
        .seed(0)
        .build();

    let report = structural
        .compile_program(&interp, &examples, &ExactMatch, move || {
            loads.fetch_add(1, Ordering::SeqCst);
            RuntimeEnv::new()
        })
        .await
        .unwrap();

    assert_eq!(report.program.meta.program_hash, parent_hash);
    assert_eq!(report.accepted, 0);
    assert_eq!(report.rejected, 1);
    assert_eq!(loads_seen.load(Ordering::SeqCst), 0, "no child was loaded");

    let step = &report.steps[0];
    assert!(matches!(step.edit, Edit::Remove { .. }));
    assert!(!step.accepted);
    assert_eq!(step.child_minibatch_score, None);
    let rejection = step.rejection.as_deref().expect("rejection recorded");
    assert!(rejection.contains("edit failed"), "got: {rejection}");
}

#[tokio::test]
async fn unparseable_reflection_reply_falls_back_without_crashing() {
    // The leaf is already cot-augmented, so every menu option is safe for
    // the canned two-field response — whatever the seeded fallback picks
    // (SwapToAgent, WrapRetry, or a validation-rejected Remove), the run
    // completes.
    let parent_lm = canned_lm(vec![text(fields(&[
        ("reasoning", "base"),
        ("answer", "right"),
    ]))])
    .await;
    let interp = load(cot_program(), parent_lm).await;

    let child_lm = canned_lm(vec![
        text(fields(&[("reasoning", "child"), ("answer", "right")])),
        text(fields(&[("reasoning", "child"), ("answer", "right")])),
    ])
    .await;
    let reflection = canned_lm_owned(vec![text(fields(&[(
        "chosen_option",
        "definitely the CoT one",
    )]))])
    .await;

    let examples = vec![example("q", "right")];
    let structural = Structural::builder()
        .num_iterations(1)
        .minibatch_size(1)
        .prompt_model(reflection)
        .seed(7)
        .build();

    let report = structural
        .compile_program(&interp, &examples, &ExactMatch, move || {
            RuntimeEnv::new().bind_model("m", child_lm.clone())
        })
        .await
        .expect("fallback choice keeps the run alive");

    assert_eq!(report.steps.len(), 1);
    assert_eq!(report.baseline_score, 1.0);
    // The reflection call was charged against the budget either way.
    assert!(report.spend.lm_calls >= 2);
}

// ---------------------------------------------------------------------------
// Budget bounds
// ---------------------------------------------------------------------------

#[tokio::test]
async fn budget_too_small_for_the_baseline_errors_cleanly() {
    let parent_lm = canned_lm(vec![]).await;
    let interp = load(qa_program(), parent_lm).await;

    let examples = vec![example("q0", "right"), example("q1", "right")];
    let structural = Structural::builder().max_rollouts(1).build();

    let err = structural
        .compile_program(&interp, &examples, &ExactMatch, RuntimeEnv::new)
        .await
        .expect_err("baseline needs 2 rollouts against a cap of 1");
    assert!(err.to_string().contains("budget too small"), "got: {err}");
}

#[tokio::test]
async fn exhausted_budget_stops_before_proposing() {
    // The cap fits exactly the baseline pass: the loop breaks before
    // spending a reflection call or scoring any child.
    let parent_lm = canned_lm(vec![
        text(fields(&[("answer", "right")])),
        text(fields(&[("answer", "right")])),
    ])
    .await;
    let interp = load(qa_program(), parent_lm).await;
    let parent_hash = interp.program().meta.program_hash;

    let examples = vec![example("q0", "right"), example("q1", "right")];
    let structural = Structural::builder()
        .num_iterations(4)
        .minibatch_size(1)
        .max_rollouts(2)
        .seed(0)
        .build();

    let report = structural
        .compile_program(&interp, &examples, &ExactMatch, RuntimeEnv::new)
        .await
        .unwrap();

    assert_eq!(report.program.meta.program_hash, parent_hash);
    assert!(report.steps.is_empty());
    assert_eq!(report.final_score, report.baseline_score);
    assert_eq!(report.spend.metric_calls, 2);
}
