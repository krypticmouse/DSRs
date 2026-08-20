//! IR-6 (RFC 0002): the eval engine's IR-native path — N candidate overlays
//! evaluated over ONE shared program through the interpreter with true
//! candidate-level parallelism, per-candidate rollout caching keyed on the
//! overlay hash, budget gating, and the minibatch gate.
#![cfg(feature = "ir")]

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use dspy_rs::ir::{
    self, DemoRow, FieldType as T, Interpreter, Overlay, ParamValue, Program, ProgramBuilder,
    RuntimeEnv, SignatureDef,
};
use dspy_rs::trace::JsonMap;
use dspy_rs::{
    BatchEvalOutcome, Budget, Engine, EngineConfig, Eval, EvalOutcome, GateOutcome, LM, LMClient,
    LMConfig, OptimizeTarget, ProgramMetric, TestCompletionModel, Trace,
};
use rig::completion::AssistantContent;
use rig::message::Text;
use serde_json::json;
use tokio::sync::Barrier;

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

async fn canned_lm(responses: Vec<AssistantContent>) -> (Arc<LM>, TestCompletionModel) {
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
    (Arc::new(lm), client)
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

/// One leaf ("answerer"), two declared models — candidates can swap the
/// ModelRef, which is what makes per-candidate outputs deterministic under
/// concurrency (each candidate owns a canned client).
fn two_model_program() -> (Program, ir::ModelId, ir::ModelId) {
    let mut b = ProgramBuilder::new("engine");
    let m1 = b.model("m1", config());
    let m2 = b.model("m2", config());
    let qa = b.sig(
        SignatureDef::build("QA")
            .instruction("Answer the question.")
            .input("question", T::String)
            .output("answer", T::String)
            .finish()
            .unwrap(),
    );
    let node = ir::predict("answerer", qa)
        .model(m1)
        .bind("question", ir::input("question"));
    let program = b
        .main(
            qa,
            ir::seq([node]).out("answer", ir::out("answerer", "answer")),
        )
        .unwrap();
    (program, m1, m2)
}

/// Rendezvous metric: both candidates' rollouts must be in flight at once for
/// the barrier to release. If the engine serialized candidates, the barrier
/// times out and the score goes negative — the test fails loudly instead of
/// hanging. On release, the score encodes which model answered.
struct RendezvousMetric {
    barrier: Barrier,
}

impl ProgramMetric for RendezvousMetric {
    async fn evaluate(
        &self,
        _example: &DemoRow,
        output: &JsonMap,
        _trace: Option<&Trace>,
    ) -> Result<Eval> {
        if tokio::time::timeout(Duration::from_secs(5), self.barrier.wait())
            .await
            .is_err()
        {
            return Ok(Eval::score(-1.0));
        }
        let score = match output.get("answer").and_then(|v| v.as_str()) {
            Some("from m1") => 1.0,
            Some("from m2") => 2.0,
            _ => 0.0,
        };
        Ok(Eval::score(score))
    }
}

/// Plain exact-match on the labeled answer.
struct ExactMatch;

impl ProgramMetric for ExactMatch {
    async fn evaluate(
        &self,
        example: &DemoRow,
        output: &JsonMap,
        _trace: Option<&Trace>,
    ) -> Result<Eval> {
        let score = if output.get("answer") == example.output.get("answer") {
            1.0
        } else {
            0.0
        };
        Ok(Eval::score(score))
    }
}

// ---------------------------------------------------------------------------
// The big one: concurrent candidates over one program instance
// ---------------------------------------------------------------------------

#[tokio::test]
async fn candidates_evaluate_concurrently_with_per_candidate_outputs_and_cache() {
    let (program, _m1, m2) = two_model_program();

    let (lm1, client1) = canned_lm(vec![text(fields(&[("answer", "from m1")]))]).await;
    let (lm2, client2) = canned_lm(vec![text(fields(&[("answer", "from m2")]))]).await;
    let interp = Interpreter::load(
        program,
        RuntimeEnv::new()
            .bind_model("m1", lm1)
            .bind_model("m2", lm2),
    )
    .await
    .unwrap();
    let program = Arc::clone(interp.program());
    let program_hash = program.meta.program_hash;

    let instr = program
        .slot_of::<ir::Instruction>("answerer.instruction")
        .unwrap();

    // Candidate A: new instruction, default model (m1).
    let mut cand_a = Overlay::new(&program);
    cand_a.set_instruction(instr, "CAND A: answer tersely.");
    // Candidate B: different instruction + model swapped to m2.
    let mut cand_b = Overlay::new(&program);
    cand_b.set_instruction(instr, "CAND B: answer verbosely.");
    let model_slot = program.param_id("answerer.model").unwrap();
    cand_b
        .set(&program, model_slot, ParamValue::ModelRef { model: m2 })
        .unwrap();

    let metric = RendezvousMetric {
        barrier: Barrier::new(2),
    };
    let examples = vec![example("q", "any")];
    let target = OptimizeTarget::program(&interp, &examples, &metric);
    let mut engine = Engine::new(EngineConfig::default());
    let a = engine.register_overlay(cand_a);
    let b = engine.register_overlay(cand_b);

    // --- One batch, two candidates, one shared Arc<Program>. ---
    let evals = engine
        .evaluate_many(&target, &[a, b], None)
        .await
        .unwrap()
        .completed()
        .expect("budget is unlimited");

    // Respective outputs: candidate A produced m1's answer, candidate B m2's.
    // (Score -1 would mean the rendezvous timed out — i.e. the engine
    // serialized candidates.)
    assert_eq!(evals.len(), 2);
    assert_eq!(evals[0].candidate, a);
    assert_eq!(evals[0].rollouts[0].eval.score, 1.0);
    assert_eq!(evals[1].candidate, b);
    assert_eq!(evals[1].rollouts[0].eval.score, 2.0);

    let trace_a = evals[0].rollouts[0].trace.as_ref().unwrap();
    assert_eq!(
        trace_a.outcome.as_ref().unwrap().output.as_ref().unwrap()["answer"],
        "from m1"
    );
    assert_eq!(trace_a.meta.candidate_hash, Some(engine.candidate_hash(a)));
    assert_eq!(
        trace_a.meta.tags.get("program"),
        Some(&format!("{program_hash:016x}"))
    );
    let trace_b = evals[1].rollouts[0].trace.as_ref().unwrap();
    assert_eq!(
        trace_b.outcome.as_ref().unwrap().output.as_ref().unwrap()["answer"],
        "from m2"
    );
    assert_eq!(trace_b.meta.candidate_hash, Some(engine.candidate_hash(b)));

    // Each candidate's instruction reached its own model.
    let preamble1 = client1.last_request().unwrap().preamble.unwrap_or_default();
    assert!(preamble1.contains("CAND A"), "m1 saw: {preamble1}");
    let preamble2 = client2.last_request().unwrap().preamble.unwrap_or_default();
    assert!(preamble2.contains("CAND B"), "m2 saw: {preamble2}");

    // The parallelism gauge: two DISTINCT candidates were in flight at the
    // same instant (the module lane is structurally pinned to 1).
    assert!(
        engine.peak_candidate_concurrency() >= 2,
        "expected candidate-level concurrency, gauge read {}",
        engine.peak_candidate_concurrency()
    );

    // Spend: two fresh rollouts, no cache traffic yet.
    assert_eq!(engine.spend().metric_calls, 2);
    assert_eq!(engine.spend().lm_calls, 2);
    assert_eq!(engine.spend().cache_hits, 0);
    // Two distinct cache entries for one (program, example, salt): the
    // overlay hash is in the key.
    assert_eq!(engine.cache().len(), 2);
    assert_ne!(engine.candidate_hash(a), engine.candidate_hash(b));

    // Scores landed in the matrix per candidate row.
    assert_eq!(engine.matrix().score(a, 0), Some(1.0));
    assert_eq!(engine.matrix().score(b, 0), Some(2.0));

    // --- Same batch again: served per-candidate from the cache. ---
    // The canned queues are empty, so any live call would error, and the
    // metric is never re-run for cached rollouts (the barrier stays idle).
    let cached = engine
        .evaluate_many(&target, &[a, b], None)
        .await
        .unwrap()
        .completed()
        .unwrap();
    assert!(cached.iter().all(|c| c.rollouts[0].trace.is_none()));
    assert_eq!(cached[0].rollouts[0].eval.score, 1.0);
    assert_eq!(cached[1].rollouts[0].eval.score, 2.0);
    assert_eq!(engine.spend().cache_hits, 2);
    assert_eq!(engine.spend().metric_calls, 2, "cache hits are free");
}

// ---------------------------------------------------------------------------
// Budget gate
// ---------------------------------------------------------------------------

#[tokio::test]
async fn budget_gate_runs_nothing_when_the_batch_does_not_fit() {
    let (program, _m1, _m2) = two_model_program();
    let (lm, _) = canned_lm(vec![]).await;
    let interp = Interpreter::load(
        program,
        RuntimeEnv::new()
            .bind_model("m1", lm.clone())
            .bind_model("m2", lm),
    )
    .await
    .unwrap();
    let program = Arc::clone(interp.program());
    let instr = program
        .slot_of::<ir::Instruction>("answerer.instruction")
        .unwrap();

    let metric = ExactMatch;
    let examples = vec![example("q", "any")];
    let target = OptimizeTarget::program(&interp, &examples, &metric);
    let mut engine = Engine::new(EngineConfig {
        budget: Budget {
            max_metric_calls: Some(1),
            ..Budget::unlimited()
        },
        ..EngineConfig::default()
    });
    let mut cand_a = Overlay::new(&program);
    cand_a.set_instruction(instr, "A");
    let mut cand_b = Overlay::new(&program);
    cand_b.set_instruction(instr, "B");
    let a = engine.register_overlay(cand_a);
    let b = engine.register_overlay(cand_b);

    // Two pending rollouts against a one-rollout budget: nothing runs.
    let outcome = engine.evaluate_many(&target, &[a, b], None).await.unwrap();
    assert!(matches!(
        outcome,
        BatchEvalOutcome::BudgetExhausted { needed: 2 }
    ));
    assert_eq!(engine.spend().metric_calls, 0);
    assert_eq!(engine.spend().lm_calls, 0);
}

// ---------------------------------------------------------------------------
// Minibatch gate
// ---------------------------------------------------------------------------

#[tokio::test]
async fn minibatch_gate_promotes_and_rejects() {
    let (program, _m1, _m2) = two_model_program();
    // c1 minibatch + c1 full (uncached half) + c2 minibatch = 3 calls.
    let (lm, _) = canned_lm(vec![
        text(fields(&[("answer", "ok")])),
        text(fields(&[("answer", "ok")])),
        text(fields(&[("answer", "ok")])),
    ])
    .await;
    let interp = Interpreter::load(
        program,
        RuntimeEnv::new()
            .bind_model("m1", lm.clone())
            .bind_model("m2", lm),
    )
    .await
    .unwrap();
    let program = Arc::clone(interp.program());
    let instr = program
        .slot_of::<ir::Instruction>("answerer.instruction")
        .unwrap();

    let metric = ExactMatch;
    let examples = vec![example("q0", "ok"), example("q1", "ok")];
    let target = OptimizeTarget::program(&interp, &examples, &metric);
    let mut engine = Engine::new(EngineConfig::default());
    let mut cand_1 = Overlay::new(&program);
    cand_1.set_instruction(instr, "GATE 1");
    let mut cand_2 = Overlay::new(&program);
    cand_2.set_instruction(instr, "GATE 2");
    let c1 = engine.register_overlay(cand_1);
    let c2 = engine.register_overlay(cand_2);

    // Minibatch mean 1.0 > 0.5: promoted to the full set, where the
    // minibatch example replays from the cache.
    match engine.evaluate_gated(&target, c1, &[0], 0.5).await.unwrap() {
        GateOutcome::Promoted { minibatch, full } => {
            assert_eq!(minibatch.mean(), 1.0);
            assert_eq!(full.rollouts.len(), 2);
            assert!(full.rollouts[0].trace.is_none(), "minibatch example cached");
            assert!(full.rollouts[1].trace.is_some(), "second example ran live");
        }
        other => panic!("expected promotion, got {other:?}"),
    }

    // Minibatch mean 1.0 <= 2.0: rejected, no full evaluation.
    match engine.evaluate_gated(&target, c2, &[0], 2.0).await.unwrap() {
        GateOutcome::Rejected { minibatch } => assert_eq!(minibatch.mean(), 1.0),
        other => panic!("expected rejection, got {other:?}"),
    }

    // Single-candidate convenience path reuses the same cache.
    match engine.evaluate(&target, c1, Some(&[0])).await.unwrap() {
        EvalOutcome::Complete(eval) => assert!(eval.rollouts[0].trace.is_none()),
        other => panic!("expected completion, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Failure surfaces
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stale_candidate_fails_the_batch() {
    let (program, _m1, _m2) = two_model_program();
    let (lm, _) = canned_lm(vec![]).await;
    let interp = Interpreter::load(
        program,
        RuntimeEnv::new()
            .bind_model("m1", lm.clone())
            .bind_model("m2", lm),
    )
    .await
    .unwrap();

    let metric = ExactMatch;
    let examples = vec![example("q", "any")];
    let target = OptimizeTarget::program(&interp, &examples, &metric);
    let mut engine = Engine::new(EngineConfig::default());
    // Minted against nothing: the interpreter's base check refuses it.
    let stale = engine.register_overlay(Overlay::default());
    let err = engine
        .evaluate_many(&target, &[stale], None)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("overlay minted against program"));

    let unregistered = engine.evaluate_many(&target, &[7], None).await.unwrap_err();
    assert!(unregistered.to_string().contains("not registered"));
}
