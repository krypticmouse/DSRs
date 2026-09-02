//! P×N parallel proposals end-to-end on canned LM responses: one step with
//! `parents_per_step = 2` and `proposals_per_parent = 2` plans four Rewrite
//! slots against the same pool/ledger snapshot, runs the four reflection
//! calls concurrently, gates each parent's children on its held-out batch,
//! races all survivors together, and commits the whole batch.

use anyhow::Result;
use dspy_rs::optimizer::ratchet::{Ratchet, TrialOutcome};
use dspy_rs::{
    Eval, LM, LMClient, Module, Predict, PredictError, Predicted, Signature,
    TestCompletionModel, TypedMetric,
};
use rig::completion::AssistantContent;
use rig::message::Text;

#[derive(Signature, Clone, Debug)]
/// Answer the prompt.
struct PxnSig {
    #[input]
    prompt: String,

    #[output]
    answer: String,
}

struct PxnModule {
    predictor: Predict<PxnSig>,
}

dspy_rs::predictors!(PxnModule { predictor });

impl Module for PxnModule {
    type Input = PxnSigInput;
    type Output = PxnSigOutput;

    async fn forward(&self, input: PxnSigInput) -> Result<Predicted<PxnSigOutput>, PredictError> {
        self.predictor.call(input).await
    }
}

struct ExactMatch;

impl TypedMetric<(PxnSigInput, PxnSigOutput), PxnModule> for ExactMatch {
    async fn evaluate(
        &self,
        example: &(PxnSigInput, PxnSigOutput),
        prediction: &Predicted<PxnSigOutput>,
        _trace: Option<&dspy_rs::Trace>,
    ) -> Result<Eval> {
        let score = (prediction.answer == example.1.answer) as u8 as f64;
        Ok(Eval::with_feedback(score, "exact-match"))
    }
}

fn answer_response(text: &str) -> AssistantContent {
    AssistantContent::Text(Text {
        text: format!("[[ ## answer ## ]]\n{text}\n\n[[ ## completed ## ]]\n"),
    })
}

fn rewrite_response(instruction: &str) -> AssistantContent {
    AssistantContent::Text(Text {
        text: format!(
            "[[ ## improved_instruction ## ]]\n{instruction}\n\n\
             [[ ## edit_summary ## ]]\ndistinct bet: {instruction}\n\n\
             [[ ## completed ## ]]\n"
        ),
    })
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

fn trainset(n: usize) -> Vec<(PxnSigInput, PxnSigOutput)> {
    (0..n)
        .map(|idx| {
            (
                PxnSigInput {
                    prompt: idx.to_string(),
                },
                PxnSigOutput {
                    answer: idx.to_string(),
                },
            )
        })
        .collect()
}

#[tokio::test]
async fn pxn_step_plans_proposes_and_commits_a_batch() {
    // Student: correct on example 0 in the seed sweep (so the seed holds a
    // Pareto win and the mean argmax), wrong everywhere after — every child
    // ties its parent's 0.0 on the gate batch (lateral pass), survives the
    // single-block race, and promotes with no measured gain.
    let mut student_responses = vec![answer_response("0")];
    student_responses.extend(std::iter::repeat_with(|| answer_response("x")).take(80));
    let student = TestCompletionModel::new(student_responses);
    let mut module = PxnModule {
        predictor: Predict::<PxnSig>::builder()
            .instruction("seed")
            .lm(make_test_lm(student).await)
            .build(),
    };

    // Teacher: exactly four canned rewrites — one per planned slot (2 parents
    // × 2 mutations). A fifth reflection call (e.g. an unintended rewrite
    // fan-out sibling) would exhaust the client and fail the trial, so the
    // promotion count below also guards the N-supersedes-fanout rule.
    let teacher = TestCompletionModel::new([
        rewrite_response("proposal-a"),
        rewrite_response("proposal-b"),
        rewrite_response("proposal-c"),
        rewrite_response("proposal-d"),
    ]);

    let ratchet = {
        let mut ratchet = Ratchet::builder()
            .num_iterations(1)
            .minibatch_size(1)
            .parents_per_step(2)
            .proposals_per_parent(2)
            .eval_concurrency(1)
            .seed(0)
            .prompt_model(make_test_lm(teacher).await)
            .build();
        // 8 examples: 7 stay unsolved, so both parents' reflection (2×2 cols)
        // and gate (2×1 cols) draws all land on unsolved columns and every
        // gate is a winnable 0-vs-0 tie.
        ratchet.champion_fresh = 0;
        ratchet
    };

    let result = ratchet
        .compile_module(&mut module, &trainset(8), &ExactMatch)
        .await
        .expect("the P×N step should succeed on canned responses");

    // All four slots proposed, passed their gates, survived the shared race,
    // and committed together.
    assert_eq!(result.trials.len(), 4, "trials: {:?}", result.trials);
    assert!(
        result
            .trials
            .iter()
            .all(|trial| matches!(trial.outcome, TrialOutcome::Promoted { .. })),
        "every slot should promote: {:?}",
        result.trials
    );
    assert_eq!(result.gate_rejected, 0);
    assert_eq!(result.race_killed, 0);
    assert_eq!(result.all_candidates.len(), 4);

    // The four children carry the four distinct canned proposals.
    let mut instructions: Vec<&str> = result
        .all_candidates
        .iter()
        .map(|candidate| candidate.instruction.as_str())
        .collect();
    instructions.sort_unstable();
    assert_eq!(
        instructions,
        vec!["proposal-a", "proposal-b", "proposal-c", "proposal-d"]
    );

    // Every attempt was a Rewrite (no promotion yet, so Simplify/Merge were
    // ineligible and no slot won its race by a positive delta).
    assert_eq!(
        result.operator_stats,
        vec![("rewrite".to_string(), 4, 0)]
    );

    // The seed keeps the win: it alone scores on example 0, so it is the raw
    // mean argmax and no child displaces it.
    assert_eq!(result.best_candidate.generation, 0);
    assert_eq!(result.best_candidate.instruction, "seed");
}
