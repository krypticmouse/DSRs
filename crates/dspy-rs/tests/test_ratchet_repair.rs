//! Counterfactual stage-repair end-to-end on canned LM responses: a 2-stage
//! pipeline whose refine stage is broken. The repair teacher blames `refine`
//! and writes the output it should have produced; the engine replays the
//! rollout with the repaired output substituted (draft serves free from the
//! recording, nothing downstream runs live because refine is the last
//! stage), measures full recovery, overrides the round-robin slot choice
//! (which would have mutated `draft`), feeds the verified repair to the
//! Rewrite operator, and ships the repaired-slot child.

use anyhow::Result;
use dspy_rs::optimizer::ratchet::{Ratchet, SlotCredit, TrialOutcome};
use dspy_rs::{
    Eval, LM, LMClient, Module, ModuleState, Predict, PredictError, Predicted, Signature,
    TestCompletionModel, TypedMetric,
};
use rig::completion::AssistantContent;
use rig::message::Text;

#[derive(Signature, Clone, Debug)]
/// Draft an answer.
struct DraftSig {
    #[input]
    question: String,

    #[output]
    draft: String,
}

#[derive(Signature, Clone, Debug)]
/// Refine the draft into a final answer.
struct RefineSig {
    #[input]
    draft: String,

    #[output]
    answer: String,
}

struct TwoStage {
    draft: Predict<DraftSig>,
    refine: Predict<RefineSig>,
}

dspy_rs::predictors!(TwoStage { draft, refine });

impl Module for TwoStage {
    type Input = DraftSigInput;
    type Output = RefineSigOutput;

    async fn forward(&self, input: DraftSigInput) -> Result<Predicted<RefineSigOutput>, PredictError> {
        let drafted = self.draft.call(input).await?;
        self.refine
            .call(RefineSigInput {
                draft: drafted.draft.clone(),
            })
            .await
    }
}

struct WantsOk;

impl TypedMetric<(DraftSigInput, RefineSigOutput), TwoStage> for WantsOk {
    async fn evaluate(
        &self,
        _example: &(DraftSigInput, RefineSigOutput),
        prediction: &Predicted<RefineSigOutput>,
        _trace: Option<&dspy_rs::Trace>,
    ) -> Result<Eval> {
        let score = (prediction.answer.trim() == "ok") as u8 as f64;
        Ok(Eval::with_feedback(score, "the final answer must be the token ok"))
    }
}

fn field_response(fields: &[(&str, &str)]) -> AssistantContent {
    let mut text = String::new();
    for (name, value) in fields {
        text.push_str(&format!("[[ ## {name} ## ]]\n{value}\n\n"));
    }
    text.push_str("[[ ## completed ## ]]\n");
    AssistantContent::Text(Text { text })
}

async fn make_test_lm(client: TestCompletionModel) -> LM {
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

#[tokio::test]
async fn falsified_repair_earns_one_revised_attempt() {
    // Same 2-stage pipeline; the teacher's FIRST repair ("meh") is falsified
    // by re-execution (still not "ok"), so it gets the falsification shown
    // back and its revised repair ("ok") recovers. Both counterfactuals
    // serve the edited refine span — no live student calls.
    let draft_client = TestCompletionModel::new(
        std::iter::repeat_with(|| field_response(&[("draft", "draft")])).take(5),
    );
    let refine_client = TestCompletionModel::new([
        field_response(&[("answer", "bad")]),
        field_response(&[("answer", "bad")]),
        field_response(&[("answer", "bad")]),
        field_response(&[("answer", "ok")]),
        field_response(&[("answer", "ok")]),
    ]);
    let mut module = TwoStage {
        draft: Predict::<DraftSig>::builder()
            .instruction("Draft an answer.")
            .lm(make_test_lm(draft_client).await)
            .build(),
        refine: Predict::<RefineSig>::builder()
            .instruction("Refine the draft.")
            .lm(make_test_lm(refine_client).await)
            .build(),
    };

    let teacher_client = TestCompletionModel::new([
        field_response(&[
            ("blamed_module", "refine"),
            ("repaired_output", "meh"),
            ("hypothesis", "refine is too hesitant"),
        ]),
        field_response(&[
            ("blamed_module", "refine"),
            ("repaired_output", "ok"),
            ("hypothesis", "refine must emit the literal ok token"),
        ]),
        field_response(&[
            ("improved_instruction", "Always answer with the token ok."),
            ("edit_summary", "force the ok token"),
        ]),
    ]);
    let teacher_probe = teacher_client.clone();

    let ratchet = {
        let mut ratchet = Ratchet::builder()
            .num_iterations(1)
            .minibatch_size(1)
            .slot_credit(SlotCredit::Repair)
            .eval_concurrency(1)
            .seed(0)
            .prompt_model(make_test_lm(teacher_client).await)
            .build();
        ratchet.repair_probes = 1;
        ratchet.gate_size = Some(0);
        ratchet.champion_fresh = 0;
        ratchet
    };

    let trainset = vec![
        (
            DraftSigInput { question: "q0".into() },
            RefineSigOutput { answer: "ok".into() },
        ),
        (
            DraftSigInput { question: "q1".into() },
            RefineSigOutput { answer: "ok".into() },
        ),
    ];

    let result = ratchet
        .compile_module(&mut module, &trainset, &WantsOk)
        .await
        .expect("the retry step should succeed on canned responses");

    // One probe, two attempts: the falsified one and the recovering one both
    // land in the slot stats; the probe counts as recovered.
    assert_eq!(result.repair_probes_run, 1);
    assert_eq!(result.repair_recovered, 1);
    let (slot, attempts, delta) = &result.repair_slot_stats[0];
    assert_eq!(slot, "refine");
    assert_eq!(*attempts, 2, "the falsified attempt is recorded too");
    assert!(*delta > 0.9, "0.0 falsified + 1.0 recovered = {delta}");

    // The second diagnosis call saw the falsification verbatim, and the
    // child still shipped from the repaired slot.
    assert_eq!(result.all_candidates[0].module_name, "refine");
    let last = teacher_probe.last_request().expect("teacher was called");
    assert!(
        format!("{:?}", last.chat_history).contains("VERIFIED WEAK-POINT DIAGNOSIS"),
        "rewrite context carries the verified repair"
    );
    assert_eq!(result.best_candidate.slot_text("refine"), "Always answer with the token ok.");
}

#[tokio::test]
async fn repair_blames_verifies_and_mutates_the_broken_stage() {
    // Student stage queues (eval_concurrency 1, sequential): draft always
    // answers "draft"; refine answers "bad" until the mutated instruction's
    // live calls at the end, which answer "ok".
    //
    // draft:  seed sweep ×2, probe base ×1, child race ×2 (the counterfactual
    //         replay serves draft from the recording — no live draft call).
    // refine: seed sweep ×2 (bad), probe base ×1 (bad), child race ×2 (ok);
    //         the counterfactual serves the EDITED refine span — no live call.
    let draft_client = TestCompletionModel::new(
        std::iter::repeat_with(|| field_response(&[("draft", "draft")])).take(5),
    );
    let refine_client = TestCompletionModel::new([
        field_response(&[("answer", "bad")]),
        field_response(&[("answer", "bad")]),
        field_response(&[("answer", "bad")]),
        field_response(&[("answer", "ok")]),
        field_response(&[("answer", "ok")]),
    ]);
    let mut module = TwoStage {
        draft: Predict::<DraftSig>::builder()
            .instruction("Draft an answer.")
            .lm(make_test_lm(draft_client).await)
            .build(),
        refine: Predict::<RefineSig>::builder()
            .instruction("Refine the draft.")
            .lm(make_test_lm(refine_client).await)
            .build(),
    };

    // Teacher queue: one diagnosis (blame refine + the repaired output that
    // provably fixes the rollout), then one rewrite. Gate is disabled, so no
    // fan-out sibling call happens — a third call would exhaust the queue
    // and fail the run.
    let teacher_client = TestCompletionModel::new([
        field_response(&[
            ("blamed_module", "refine"),
            ("repaired_output", "ok"),
            ("hypothesis", "refine never emits the required ok token"),
        ]),
        field_response(&[
            ("improved_instruction", "Always answer with the token ok."),
            ("edit_summary", "force the ok token"),
        ]),
    ]);
    let teacher_probe = teacher_client.clone();

    let ratchet = {
        let mut ratchet = Ratchet::builder()
            .num_iterations(1)
            .minibatch_size(1)
            .slot_credit(SlotCredit::Repair)
            .eval_concurrency(1)
            .seed(0)
            .prompt_model(make_test_lm(teacher_client).await)
            .build();
        ratchet.repair_probes = 1;
        ratchet.gate_size = Some(0);
        ratchet.champion_fresh = 0;
        ratchet
    };

    let trainset = vec![
        (
            DraftSigInput { question: "q0".into() },
            RefineSigOutput { answer: "ok".into() },
        ),
        (
            DraftSigInput { question: "q1".into() },
            RefineSigOutput { answer: "ok".into() },
        ),
    ];

    let result = ratchet
        .compile_module(&mut module, &trainset, &WantsOk)
        .await
        .expect("the repair step should succeed on canned responses");

    // The probe ran, and the counterfactual measurably recovered the rollout.
    assert_eq!(result.repair_probes_run, 1);
    assert_eq!(result.repair_recovered, 1);
    // v3.1: the verified lesson persists in the result (and in later steps'
    // ledgers on longer runs).
    assert_eq!(result.repair_lessons.len(), 1);
    assert_eq!(result.repair_lessons[0].1, "refine");
    assert_eq!(result.repair_slot_stats.len(), 1);
    let (slot, probes, delta) = &result.repair_slot_stats[0];
    assert_eq!(slot, "refine");
    assert_eq!(*probes, 1);
    assert!(*delta > 0.9, "full recovery expected, got {delta}");

    // Verified blame overrode round-robin: generation 0 % 2 leaves would
    // have mutated `draft`; the repair pointed at `refine`.
    assert_eq!(result.all_candidates.len(), 1);
    assert_eq!(result.all_candidates[0].module_name, "refine");
    assert!(matches!(
        result.trials.last().unwrap().outcome,
        TrialOutcome::Promoted { .. }
    ));

    // The verified repair reached the Rewrite prompt.
    let last = teacher_probe
        .last_request()
        .expect("the teacher was called");
    assert!(
        format!("{:?}", last.chat_history).contains("VERIFIED WEAK-POINT DIAGNOSIS"),
        "the rewrite call must carry the verified repair evidence"
    );

    // The repaired-slot child won and was installed: refine carries the new
    // instruction, draft keeps its seed.
    assert_eq!(result.best_candidate.generation, 1);
    assert_eq!(
        result.best_candidate.slot_text("refine"),
        "Always answer with the token ok."
    );
    let state = ModuleState::from_module(&mut module).unwrap();
    assert_eq!(
        state.predictors["refine"].instruction_override.as_deref(),
        Some("Always answer with the token ok.")
    );
    // The joint winner installs every slot; draft keeps its seed text.
    assert_eq!(
        state.predictors["draft"].instruction_override.as_deref(),
        Some("Draft an answer.")
    );
}
