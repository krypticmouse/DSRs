//! COMPOUND (v4) end-to-end simulation of the many-tiny-wins regime — the
//! DocsQA economics reproduced deterministically, no LLM in the loop.
//!
//! The world: 50 pinnable facts, each covering ~2% of columns at +0.5 a
//! column (the measured DocsQA density: "a pinned fact helps ~2 questions"
//! of holdout100) — worth ~+.01 val mean apiece, under ±0.25 judge-noise
//! — the exact effect-to-noise ratio class where the win-size law says v2
//! RATCHET structurally refuses the pile
//! (`docs/handoff-verification-economics.md` §0). The student is a
//! scripted LM that answers with whatever fact tokens its instruction
//! carries; the teacher is a scripted LM that pins exactly ONE missing
//! fact per proposal (the tiniest-win teacher). Every proposal is a TRUE
//! improvement, so every kill is a false kill and every refusal refuses
//! real value — the simulation measures the acceptance machinery itself.
//!
//! Both arms get the same task, seed, and rollout budget. The v2 arm runs
//! measured-default RATCHET (z-block racing, 1-SE seed protection). The
//! COMPOUND arm runs `compound()` (e-process racing, sequential paired
//! ship certification on reserved columns).

use std::collections::BTreeSet;

use anyhow::Result;
use dspy_rs::optimizer::ratchet::{
    AcceptMode, LeakPolicy, RaceRule, Ratchet, RatchetResult, SlotCredit, TrialOutcome, compound,
};
use dspy_rs::{
    Eval, LM, LMClient, Module, Predict, PredictError, Predicted, ScriptedCompletionModel,
    Signature, TestCompletionModel, TypedMetric,
};
use rig::completion::AssistantContent;
use rig::message::Text;

const FACTS: usize = 50;
const BASE: f64 = 0.35;
const FACT_LIFT: f64 = 0.5;
const NOISE: f64 = 0.25;
const TRAIN: usize = 240;
const VAL: usize = 60;

fn fact_of(qid: usize) -> usize {
    qid % FACTS
}

fn fact_token(k: usize) -> String {
    format!("F{k:02}")
}

fn facts_in(text: &str) -> BTreeSet<usize> {
    (0..FACTS)
        .filter(|&k| text.contains(&fact_token(k)))
        .collect()
}

/// Deterministic judge noise per (question, answer-content): an FNV hash
/// mapped to uniform ±NOISE. Same candidate content on the same column
/// always scores identically (the rollout cache guarantees that anyway);
/// different content redraws — the paired-noise structure of a real judge.
fn judge_noise(qid: usize, answer: &str) -> f64 {
    let mut hash = 1469598103934665603u64;
    for byte in answer.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(1099511628211);
    }
    hash ^= qid as u64;
    hash = hash.wrapping_mul(1099511628211);
    ((hash % 10_001) as f64 / 10_000.0 - 0.5) * 2.0 * NOISE
}

#[derive(Signature, Clone, Debug)]
/// Answer the question from the docs.
struct QaSig {
    #[input]
    question: String,

    #[output]
    answer: String,
}

struct QaModule {
    predictor: Predict<QaSig>,
}

dspy_rs::predictors!(QaModule { predictor });

impl Module for QaModule {
    type Input = QaSigInput;
    type Output = QaSigOutput;

    async fn forward(&self, input: QaSigInput) -> Result<Predicted<QaSigOutput>, PredictError> {
        self.predictor.call(input).await
    }
}

struct FactMetric;

impl TypedMetric<(QaSigInput, QaSigOutput), QaModule> for FactMetric {
    async fn evaluate(
        &self,
        example: &(QaSigInput, QaSigOutput),
        prediction: &Predicted<QaSigOutput>,
        _trace: Option<&dspy_rs::Trace>,
    ) -> Result<Eval> {
        let qid: usize = example.0.question.trim_start_matches('q').parse()?;
        let needed = fact_of(qid);
        let covered = prediction.answer.contains(&fact_token(needed));
        let lift = if covered { FACT_LIFT } else { 0.0 };
        let score = (BASE + lift + judge_noise(qid, &prediction.answer)).clamp(0.0, 1.0);
        let feedback = if covered {
            "ok".to_string()
        } else {
            format!("MISSING::{}", fact_token(needed))
        };
        Ok(Eval::with_feedback(score, feedback))
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

async fn make_lm(client: ScriptedCompletionModel) -> LM {
    temp_env::async_with_vars(
        [("OPENAI_API_KEY", Some("test"))],
        LM::builder().model("openai:gpt-4o-mini".to_string()).build(),
    )
    .await
    .unwrap()
    .with_client(LMClient::Scripted(client))
    .await
    .unwrap()
}

/// The scripted student: answers with every fact token its rendered prompt
/// (instruction included) carries. Question texts (`q17`) carry none, so
/// the answer is a pure function of the injected instruction.
fn student_client() -> ScriptedCompletionModel {
    ScriptedCompletionModel::new(|request| {
        let haystack = format!("{:?} {:?}", request.preamble, request.chat_history);
        let mut answer = String::from("facts:");
        for k in 0..FACTS {
            let token = fact_token(k);
            if haystack.contains(&token) {
                answer.push(' ');
                answer.push_str(&token);
            }
        }
        Ok(field_response(&[("answer", &answer)]))
    })
}

/// Everything rendered under these field headers is instruction text the
/// proposal builds on (Rewrite/Patch/Simplify use `current_instruction`,
/// Merge gets both sides).
const INSTRUCTION_FIELDS: [&str; 3] = ["current_instruction", "instruction_a", "instruction_b"];

fn field_section(haystack: &str, name: &str) -> String {
    let marker = format!("{name} ## ]]");
    let mut out = String::new();
    let mut rest = haystack;
    while let Some(pos) = rest.find(&marker) {
        let after = &rest[pos + marker.len()..];
        let end = after.find("[[ ##").unwrap_or(after.len());
        out.push_str(&after[..end]);
        out.push('\n');
        rest = &after[end..];
    }
    out
}

/// The tiniest-win teacher: keep every fact the parent instruction(s)
/// already pin, add exactly ONE fact named `MISSING::` in the execution
/// feedback. Every proposal is a true ~+.005 improvement — the pure
/// harvest regime.
fn teacher_client() -> ScriptedCompletionModel {
    ScriptedCompletionModel::new(|request| {
        let haystack = format!("{:?} {:?}", request.preamble, request.chat_history);
        let mut current: BTreeSet<usize> = BTreeSet::new();
        for field in INSTRUCTION_FIELDS {
            current.extend(facts_in(&field_section(&haystack, field)));
        }
        let missing: BTreeSet<usize> = (0..FACTS)
            .filter(|&k| haystack.contains(&format!("MISSING::{}", fact_token(k))))
            .filter(|k| !current.contains(k))
            .collect();
        let (facts, summary) = match missing.iter().next() {
            Some(&new_fact) => {
                let mut facts = current.clone();
                facts.insert(new_fact);
                (facts, format!("pin {}", fact_token(new_fact)))
            }
            None => (current, "no new fact to pin".to_string()),
        };
        let list = facts
            .iter()
            .map(|&k| fact_token(k))
            .collect::<Vec<_>>()
            .join(" ");
        let instruction = format!("Answer the question. Facts: {list}").trim_end().to_string();
        Ok(field_response(&[
            ("improved_instruction", &instruction),
            ("edit_summary", &summary),
        ]))
    })
}

fn dataset(range: std::ops::Range<usize>) -> Vec<(QaSigInput, QaSigOutput)> {
    range
        .map(|qid| {
            (
                QaSigInput {
                    question: format!("q{qid}"),
                },
                // Gold text carries no fact tokens and no leak-class
                // strings; the metric scores against `fact_of(qid)`.
                QaSigOutput { answer: "-".into() },
            )
        })
        .collect()
}

async fn run_arm(mut optimizer: Ratchet) -> Result<(RatchetResult, BTreeSet<usize>)> {
    optimizer.prompt_model = Some(make_lm(teacher_client()).await);
    optimizer.max_rollouts = Some(3200);
    optimizer.seed = Some(0);
    optimizer.eval_concurrency = 16;
    let mut module = QaModule {
        predictor: Predict::<QaSig>::builder()
            .instruction("Answer the question. Facts:")
            .lm(make_lm(student_client()).await)
            .build(),
    };
    let train = dataset(0..TRAIN);
    let val = dataset(TRAIN..TRAIN + VAL);
    let result = optimizer
        .compile_module_with_valset(&mut module, &train, Some(&val), &FactMetric)
        .await?;
    let shipped = facts_in(&result.best_candidate.instruction);
    Ok((result, shipped))
}

/// A trial hypothesis names the fact it pinned; in this world every pinned
/// fact is a true improvement, so a killed "pin F.." trial is a false kill.
fn false_kills(result: &RatchetResult) -> usize {
    result
        .trials
        .iter()
        .filter(|trial| {
            trial.hypothesis.starts_with("pin ")
                && matches!(
                    trial.outcome,
                    TrialOutcome::RaceKilled { .. } | TrialOutcome::GateKilled { .. }
                )
        })
        .count()
}

#[tokio::test]
async fn compound_harvests_the_tiny_win_pile_that_v2_economics_leave_behind() {
    // Arm 1: measured-default RATCHET v2 (z-block race, 1-SE seed
    // protection), budget-matched.
    let v2 = Ratchet::builder().num_iterations(30).build();
    let (v2_result, v2_facts) = run_arm(v2).await.expect("v2 arm runs");

    // Arm 2: COMPOUND.
    let (compound_result, compound_facts) =
        run_arm(compound()).await.expect("compound arm runs");

    println!(
        "v2: shipped {} facts, {} promotions, {} race-killed, {} gate-killed, {} false kills, {} rollouts",
        v2_facts.len(),
        v2_result
            .trials
            .iter()
            .filter(|t| matches!(t.outcome, TrialOutcome::Promoted { .. }))
            .count(),
        v2_result.race_killed,
        v2_result.gate_rejected,
        false_kills(&v2_result),
        v2_result.total_rollouts,
    );
    println!(
        "compound: shipped {} facts, {} promotions, {} race-killed, certified {:?} on {} cols (mean {:+.4}, e {:.1}), {} rollouts, ops {:?}, challenger facts {:?}",
        compound_facts.len(),
        compound_result
            .trials
            .iter()
            .filter(|t| matches!(t.outcome, TrialOutcome::Promoted { .. }))
            .count(),
        compound_result.race_killed,
        compound_result.certified,
        compound_result.certify_cols_used,
        compound_result.certify_mean_delta,
        compound_result.certify_e_value,
        compound_result.total_rollouts,
        compound_result.operator_stats,
        compound_result
            .best_by_mean
            .as_ref()
            .map(|candidate| facts_in(&candidate.instruction).len()),
    );

    // Both arms respected the shared budget.
    assert!(v2_result.total_rollouts <= 3200);
    assert!(compound_result.total_rollouts <= 3200);

    // The v2 refusal machinery fires on true tiny wins: kills of "pin F.."
    // trials are false kills by construction, and each one also lands in
    // the falsified ledger, steering the teacher away from real facts.
    assert!(
        false_kills(&v2_result) >= 3,
        "expected the z-race/gate to false-kill true tiny winners, got {}",
        false_kills(&v2_result)
    );

    // COMPOUND ships a certified pile: strictly more true value than v2 at
    // the same budget, and never an uncertified non-seed winner.
    assert!(
        compound_facts.len() >= v2_facts.len() + 5,
        "compound should out-harvest v2 by a clear margin: compound {} vs v2 {}",
        compound_facts.len(),
        v2_facts.len()
    );
    match compound_result.certified {
        Some(true) => {
            assert!(
                compound_result.certify_e_value >= 10.0,
                "a certified ship must carry the evidence (preset alpha 0.1): e = {}",
                compound_result.certify_e_value
            );
            assert!(compound_result.certify_mean_delta > 0.0);
            assert!(!compound_facts.is_empty());
        }
        other => panic!(
            "compound should certify the harvested pile on this task, got {other:?} \
             after {} columns (mean delta {:+.4})",
            compound_result.certify_cols_used, compound_result.certify_mean_delta
        ),
    }

    // The true (noise-free) quality gap: each fact is worth FACT_LIFT on
    // its share of columns, so shipped fact count is the exact true score.
    let true_gain = |facts: &BTreeSet<usize>| facts.len() as f64 * FACT_LIFT / FACTS as f64;
    assert!(
        true_gain(&compound_facts) - true_gain(&v2_facts) >= 0.025,
        "true shipped gain: compound {:+.4} vs v2 {:+.4}",
        true_gain(&compound_facts),
        true_gain(&v2_facts)
    );
}

// ---------------------------------------------------------------------------
// Verified fact-pinning: a two-stage pipeline whose refine stage must state
// a corpus fact that the leak guard would normally forbid (it is the gold
// answer verbatim). The counterfactual repair proves the fact fixes a real
// failed rollout, which licenses it through the guard; the shipped winner
// carries it as a dated, provenance-backed pin. Under `Forbid` the exact
// same proposal is retried and abandoned — the license is the difference.
// ---------------------------------------------------------------------------

const CAPITAL_FACT: &str = "the capital is quito";

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

    async fn forward(
        &self,
        input: DraftSigInput,
    ) -> Result<Predicted<RefineSigOutput>, PredictError> {
        let drafted = self.draft.call(input).await?;
        self.refine
            .call(RefineSigInput {
                draft: drafted.draft.clone(),
            })
            .await
    }
}

struct WantsCapital;

impl TypedMetric<(DraftSigInput, RefineSigOutput), TwoStage> for WantsCapital {
    async fn evaluate(
        &self,
        _example: &(DraftSigInput, RefineSigOutput),
        prediction: &Predicted<RefineSigOutput>,
        _trace: Option<&dspy_rs::Trace>,
    ) -> Result<Eval> {
        let score = prediction.answer.to_lowercase().contains(CAPITAL_FACT) as u8 as f64;
        Ok(Eval::with_feedback(
            score,
            "the answer must state the capital city fact",
        ))
    }
}

async fn make_canned_lm(client: TestCompletionModel) -> LM {
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

async fn run_pinning_arm(leak_policy: LeakPolicy) -> RatchetResult {
    // Student queues (eval_concurrency 1, sequential): draft always drafts;
    // refine fails until the mutated instruction's live race calls at the
    // end. Under Forbid the race never happens (the proposal is abandoned),
    // so the surplus refine responses are simply never consumed.
    let draft_client = TestCompletionModel::new(
        std::iter::repeat_with(|| field_response(&[("draft", "draft")])).take(5),
    );
    let refine_client = TestCompletionModel::new([
        field_response(&[("answer", "bad")]),
        field_response(&[("answer", "bad")]),
        field_response(&[("answer", "bad")]),
        field_response(&[("answer", CAPITAL_FACT)]),
        field_response(&[("answer", CAPITAL_FACT)]),
    ]);
    let mut module = TwoStage {
        draft: Predict::<DraftSig>::builder()
            .instruction("Draft an answer.")
            .lm(make_canned_lm(draft_client).await)
            .build(),
        refine: Predict::<RefineSig>::builder()
            .instruction("Refine the draft.")
            .lm(make_canned_lm(refine_client).await)
            .build(),
    };

    // Teacher queue: one diagnosis whose verified repair carries the fact,
    // then a rewrite that pins it verbatim, then (Forbid only) the leak
    // retry, which stubbornly pins it again and gets abandoned.
    let pinned_instruction = format!("Always answer exactly: {CAPITAL_FACT}.");
    let teacher_client = TestCompletionModel::new([
        field_response(&[
            ("blamed_module", "refine"),
            ("repaired_output", CAPITAL_FACT),
            ("hypothesis", "refine never states the capital fact"),
        ]),
        field_response(&[
            ("improved_instruction", &pinned_instruction),
            ("edit_summary", "pin the capital fact"),
        ]),
        field_response(&[
            ("improved_instruction", &pinned_instruction),
            ("edit_summary", "pin the capital fact"),
        ]),
    ]);

    let ratchet = {
        let mut ratchet = Ratchet::builder()
            .num_iterations(1)
            .minibatch_size(1)
            .slot_credit(SlotCredit::Repair)
            .leak_policy(leak_policy)
            .eval_concurrency(1)
            .seed(0)
            .prompt_model(make_canned_lm(teacher_client).await)
            .build();
        ratchet.repair_probes = 1;
        ratchet.repair_attempts = 1;
        ratchet.gate_size = Some(0);
        ratchet.champion_fresh = 0;
        ratchet
    };

    let trainset = vec![
        (
            DraftSigInput { question: "q0".into() },
            RefineSigOutput {
                answer: CAPITAL_FACT.into(),
            },
        ),
        (
            DraftSigInput { question: "q1".into() },
            RefineSigOutput {
                answer: CAPITAL_FACT.into(),
            },
        ),
    ];

    ratchet
        .compile_module(&mut module, &trainset, &WantsCapital)
        .await
        .expect("the pinning arm runs on canned responses")
}

#[tokio::test]
async fn verified_repair_licenses_the_fact_through_the_leak_guard() {
    let result = run_pinning_arm(LeakPolicy::AllowWithProvenance).await;

    // The repair recovered, licensing the fact; the pinned rewrite passed
    // the guard without a retry and shipped.
    assert_eq!(result.repair_recovered, 1);
    assert_eq!(result.leak_retries, 0);
    assert_eq!(result.leak_skipped, 0);
    assert!(
        result
            .best_candidate
            .slot_text("refine")
            .to_lowercase()
            .contains(CAPITAL_FACT),
        "the winner must carry the pinned fact"
    );

    // And the pin ships with provenance: the licensing repair's example,
    // generation, verified recovery, and a staleness clock.
    assert_eq!(result.pins.len(), 1, "exactly one pin: {:?}", result.pins);
    let pin = &result.pins[0];
    assert_eq!(pin.text, CAPITAL_FACT);
    assert_eq!(pin.generation, 0);
    assert!(pin.verified_delta > 0.9, "full recovery, got {}", pin.verified_delta);
    assert!(pin.pinned_at_unix > 0);
}

#[tokio::test]
async fn forbid_policy_still_blocks_the_same_pin() {
    let result = run_pinning_arm(LeakPolicy::Forbid).await;

    // Same task, same teacher: the guard retries, the teacher re-pins, the
    // trial is abandoned, and the seed ships pin-free — the license is the
    // only difference between the two arms.
    assert_eq!(result.leak_retries, 1);
    assert_eq!(result.leak_skipped, 1);
    assert!(result.pins.is_empty());
    assert_eq!(result.best_candidate.generation, 0, "the seed ships");
    assert!(
        !result
            .best_candidate
            .slot_text("refine")
            .to_lowercase()
            .contains(CAPITAL_FACT)
    );
}

#[tokio::test]
async fn gate_accept_out_harvests_racing_in_the_accretive_regime() {
    // Port spec v2 §1's cheap-accept mode: same task, same budget, same
    // seed — Gate accept must convert more of the true fact supply than
    // full-val racing, because it neither spends 45 columns per child nor
    // drops true +.01 children on noisy race draws.
    let (race_result, race_facts) = run_arm(compound()).await.expect("race arm runs");

    let mut gate = compound();
    gate.accept_mode = AcceptMode::Gate;
    let (gate_result, gate_facts) = run_arm(gate).await.expect("gate arm runs");

    println!(
        "race-accept: {} facts, {} promotions, certified {:?} | gate-accept: {} facts, {} promotions, certified {:?} (e {:.1}, {} cols)",
        race_facts.len(),
        race_result
            .trials
            .iter()
            .filter(|t| matches!(t.outcome, TrialOutcome::Promoted { .. }))
            .count(),
        race_result.certified,
        gate_facts.len(),
        gate_result
            .trials
            .iter()
            .filter(|t| matches!(t.outcome, TrialOutcome::Promoted { .. }))
            .count(),
        gate_result.certified,
        gate_result.certify_e_value,
        gate_result.certify_cols_used,
    );
    assert_eq!(
        gate_result.certified,
        Some(true),
        "the accretion pile must certify (mean {:+.4}, e {:.1})",
        gate_result.certify_mean_delta,
        gate_result.certify_e_value
    );
    assert!(
        gate_facts.len() > race_facts.len(),
        "gate accept should out-harvest racing: gate {} vs race {}",
        gate_facts.len(),
        race_facts.len()
    );
}

#[tokio::test]
async fn harness_supplied_certify_cols_power_the_ship_gate() {
    // Port spec v2 §2: the harness appends holdout-class rows to the
    // trainset and names them as explicit certification columns. With the
    // in-train carve disabled, the ship gate's ONLY power source is the
    // supplied slice — certification must still run on it and certify.
    let mut optimizer = compound();
    optimizer.prompt_model = Some(make_lm(teacher_client()).await);
    optimizer.max_rollouts = Some(3200);
    optimizer.seed = Some(0);
    optimizer.eval_concurrency = 16;
    optimizer.certify_reserve = 0;
    optimizer.certify_cols = (TRAIN..TRAIN + 80).collect();
    let mut module = QaModule {
        predictor: Predict::<QaSig>::builder()
            .instruction("Answer the question. Facts:")
            .lm(make_lm(student_client()).await)
            .build(),
    };
    // Trainset = the usual 240 rows plus 80 harness-supplied certification
    // rows (fresh question ids, same fact space).
    let mut train = dataset(0..TRAIN);
    train.extend(dataset(400..480));
    let val = dataset(TRAIN..TRAIN + VAL);
    let result = optimizer
        .compile_module_with_valset(&mut module, &train, Some(&val), &FactMetric)
        .await
        .expect("explicit-certify arm runs");

    println!(
        "explicit-certify: certified {:?} on {} cols (mean {:+.4}, e {:.1}), shipped {} facts",
        result.certified,
        result.certify_cols_used,
        result.certify_mean_delta,
        result.certify_e_value,
        facts_in(&result.best_candidate.instruction).len(),
    );
    assert_eq!(
        result.certified,
        Some(true),
        "the harvested pile must certify on the supplied slice alone \
         (mean {:+.4}, e {:.1} after {} cols)",
        result.certify_mean_delta,
        result.certify_e_value,
        result.certify_cols_used
    );
    assert!(result.certify_cols_used > 0);
    assert!(!facts_in(&result.best_candidate.instruction).is_empty());
}

#[tokio::test]
async fn eprocess_race_is_the_mechanism_that_saves_the_tiny_winners() {
    // Ablation inside the simulation: v2 racing vs e-process racing with
    // everything else at v2 defaults. The e-race must false-kill fewer
    // true "pin F.." children than the z-race at the same budget.
    let z_arm = Ratchet::builder().num_iterations(30).build();
    let (z_result, _) = run_arm(z_arm).await.expect("z arm runs");

    let e_arm = Ratchet::builder()
        .num_iterations(30)
        .race_rule(RaceRule::Eprocess)
        .build();
    let (e_result, _) = run_arm(e_arm).await.expect("e arm runs");

    println!(
        "race-kills of true winners: z-block {} vs e-process {}",
        z_result.race_killed, e_result.race_killed
    );
    assert!(
        e_result.race_killed < z_result.race_killed,
        "the anytime-valid kill rule must false-kill fewer true tiny winners: \
         e {} vs z {}",
        e_result.race_killed,
        z_result.race_killed
    );
}
