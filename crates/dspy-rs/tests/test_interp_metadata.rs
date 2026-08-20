//! Interpreter per-leaf metadata seam (`Interpreter::run_collecting`):
//! coercion flags, constraint outcomes, raw response text, usage, and model
//! config hash surface per `Predict` leaf, in execution order — the exact
//! parity data the historical static lane kept, so a typed
//! `Predict<S>` routed through the interpreter loses none of `Predicted`'s
//! metadata contract.

use std::sync::Arc;

use dspy_rs::Flag;
use dspy_rs::ir::{
    self, Budget, ConstraintDef, FieldDef, FieldType as T, Interpreter, Program, ProgramBuilder,
    RunError, RuntimeEnv, SignatureDef,
};
use dspy_rs::trace::ModelEntry;
use dspy_rs::{LM, LMClient, LMConfig, TestCompletionModel};
use rig::completion::{AssistantContent, Usage};
use rig::message::Text;
use serde_json::json;

// ---------------------------------------------------------------------------
// Fixtures (mirrors test_ir_interp.rs)
// ---------------------------------------------------------------------------

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

fn obj(pairs: &[(&str, serde_json::Value)]) -> serde_json::Map<String, serde_json::Value> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.clone()))
        .collect()
}

/// question → rater → rating (Int, with the given constraints).
fn rater_program(constraints: Vec<ConstraintDef>) -> Program {
    let mut b = ProgramBuilder::new("rater-pipeline");
    b.model("m", config());
    let main_sig = b.sig(
        SignatureDef::build("Main")
            .input("question", T::String)
            .output("rating", T::Int)
            .finish()
            .unwrap(),
    );
    let mut rating_field = FieldDef::new("rating", T::Int);
    for constraint in constraints {
        rating_field = rating_field.with_constraint(constraint);
    }
    let rate = b.sig(
        SignatureDef::build("Rate")
            .instruction("Rate the thing.")
            .input("question", T::String)
            .output_full(rating_field)
            .finish()
            .unwrap(),
    );
    let rater = ir::predict("rater", rate).bind("question", ir::input("question"));
    b.main(
        main_sig,
        ir::seq([rater]).out("rating", ir::out("rater", "rating")),
    )
    .unwrap()
}

/// question → drafter (QA) → checker (Check) → verdict. Two Predict leaves.
fn seq_program() -> Program {
    let mut b = ProgramBuilder::new("pipeline");
    b.model("m", config());
    let main_sig = b.sig(
        SignatureDef::build("Main")
            .input("question", T::String)
            .output("verdict", T::String)
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
    let check = b.sig(
        SignatureDef::build("Check")
            .instruction("Judge the answer.")
            .input("answer", T::String)
            .output("verdict", T::String)
            .finish()
            .unwrap(),
    );
    let drafter = ir::predict("drafter", qa).bind("question", ir::input("question"));
    let checker = ir::predict("checker", check).bind("answer", ir::out("drafter", "answer"));
    b.main(
        main_sig,
        ir::seq([drafter, checker]).out("verdict", ir::out("checker", "verdict")),
    )
    .unwrap()
}

// ---------------------------------------------------------------------------
// Coercion flags + raw response + model config hash
// ---------------------------------------------------------------------------

#[tokio::test]
async fn coerced_field_surfaces_flags_raw_text_and_model_hash() {
    // "1,000" needs thousands-separator coercion into Int → CoercedFromString.
    let raw_response = fields(&[("rating", "1,000")]);
    let (lm, _client) = canned_lm(vec![text(raw_response.clone())]).await;
    let expected_hash = ModelEntry::from_config(&lm.config).config_hash;
    let interp = Interpreter::load(rater_program(vec![]), RuntimeEnv::new().bind_model("m", lm))
        .await
        .unwrap();

    let run = interp
        .run_collecting(
            obj(&[("question", json!("how many?"))]),
            None,
            Budget::unlimited(),
        )
        .await
        .unwrap();

    assert_eq!(run.output["rating"], 1000);
    assert_eq!(run.leaves.len(), 1);

    let leaf = &run.leaves[0];
    assert_eq!(leaf.name, "rater");
    assert_eq!(leaf.raw_response, raw_response);
    assert_eq!(leaf.model_config_hash, expected_hash);
    assert_ne!(leaf.model_config_hash, 0);

    let meta = &leaf.field_meta["rating"];
    assert_eq!(meta.raw_text, "1,000");
    assert_eq!(meta.flags, vec![Flag::CoercedFromString]);
    assert!(meta.checks.is_empty());
}

#[tokio::test]
async fn clean_field_reports_no_flags() {
    let (lm, _client) = canned_lm(vec![text(fields(&[("rating", "7")]))]).await;
    let interp = Interpreter::load(rater_program(vec![]), RuntimeEnv::new().bind_model("m", lm))
        .await
        .unwrap();

    let run = interp
        .run_collecting(
            obj(&[("question", json!("how many?"))]),
            None,
            Budget::unlimited(),
        )
        .await
        .unwrap();

    assert_eq!(run.output["rating"], 7);
    let meta = &run.leaves[0].field_meta["rating"];
    assert_eq!(meta.raw_text, "7");
    assert!(meta.flags.is_empty());
}

// ---------------------------------------------------------------------------
// Constraint outcomes
// ---------------------------------------------------------------------------

#[tokio::test]
async fn check_outcomes_surface_pass_and_fail() {
    let program = rater_program(vec![
        ConstraintDef::check("positive", "this > 0"),
        ConstraintDef::check("small", "this < 5"),
    ]);
    let (lm, _client) = canned_lm(vec![text(fields(&[("rating", "7")]))]).await;
    let interp = Interpreter::load(program, RuntimeEnv::new().bind_model("m", lm))
        .await
        .unwrap();

    let run = interp
        .run_collecting(
            obj(&[("question", json!("how many?"))]),
            None,
            Budget::unlimited(),
        )
        .await
        .unwrap();

    // Failed #[check]s never abort the run — they surface as outcomes.
    assert_eq!(run.output["rating"], 7);
    let checks = &run.leaves[0].field_meta["rating"].checks;
    assert_eq!(checks.len(), 2);
    assert_eq!(checks[0].label, "positive");
    assert_eq!(checks[0].expression, "this > 0");
    assert!(checks[0].passed);
    assert_eq!(checks[1].label, "small");
    assert_eq!(checks[1].expression, "this < 5");
    assert!(!checks[1].passed);
}

#[tokio::test]
async fn assert_failure_is_a_parse_error() {
    // Same semantics as the static lane: a failed #[assert] is a parse error
    // (no LeafOutcome — the evaluation did not succeed), while a passing
    // assert records no ConstraintResult.
    let program = rater_program(vec![ConstraintDef::assert("this < 5")]);
    let (lm, _client) = canned_lm(vec![text(fields(&[("rating", "7")]))]).await;
    let interp = Interpreter::load(program, RuntimeEnv::new().bind_model("m", lm))
        .await
        .unwrap();

    let err = interp
        .run_collecting(
            obj(&[("question", json!("how many?"))]),
            None,
            Budget::unlimited(),
        )
        .await
        .expect_err("failed assert must fail the run");
    match err {
        RunError::Parse { at, .. } => assert_eq!(&*at, "rater"),
        other => panic!("expected parse error, got {other:?}"),
    }
}

#[tokio::test]
async fn passing_assert_records_no_constraint_result() {
    let program = rater_program(vec![
        ConstraintDef::assert("this > 0"),
        ConstraintDef::check("small", "this < 5"),
    ]);
    let (lm, _client) = canned_lm(vec![text(fields(&[("rating", "3")]))]).await;
    let interp = Interpreter::load(program, RuntimeEnv::new().bind_model("m", lm))
        .await
        .unwrap();

    let run = interp
        .run_collecting(
            obj(&[("question", json!("how many?"))]),
            None,
            Budget::unlimited(),
        )
        .await
        .unwrap();

    // Only #[check] constraints produce ConstraintResults — asserts are
    // pass-or-error, exactly like the historical typed parse path.
    let checks = &run.leaves[0].field_meta["rating"].checks;
    assert_eq!(checks.len(), 1);
    assert_eq!(checks[0].label, "small");
    assert!(checks[0].passed);
}

// ---------------------------------------------------------------------------
// Multi-leaf execution order + usage accumulation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn two_leaf_seq_reports_outcomes_in_execution_order() {
    let drafter_raw = fields(&[("answer", "42")]);
    let checker_raw = fields(&[("verdict", "correct")]);
    let (lm, client) = canned_lm(vec![text(drafter_raw.clone()), text(checker_raw.clone())]).await;
    let mut usage = Usage::new();
    usage.input_tokens = 3;
    usage.output_tokens = 4;
    usage.total_tokens = 7;
    client.set_usage(usage);

    let interp = Interpreter::load(seq_program(), RuntimeEnv::new().bind_model("m", lm))
        .await
        .unwrap();

    let run = interp
        .run_collecting(
            obj(&[("question", json!("what is 6*7?"))]),
            None,
            Budget::unlimited(),
        )
        .await
        .unwrap();

    assert_eq!(run.output["verdict"], "correct");

    // Two leaves, in execution order, each with its own raw response.
    assert_eq!(run.leaves.len(), 2);
    assert_eq!(run.leaves[0].name, "drafter");
    assert_eq!(run.leaves[0].raw_response, drafter_raw);
    assert_eq!(run.leaves[0].field_meta["answer"].raw_text, "42");
    assert_eq!(run.leaves[1].name, "checker");
    assert_eq!(run.leaves[1].raw_response, checker_raw);
    assert_eq!(run.leaves[1].field_meta["verdict"].raw_text, "correct");

    // Both leaves used the same model config.
    assert_eq!(
        run.leaves[0].model_config_hash,
        run.leaves[1].model_config_hash
    );

    // Per-leaf usage is each call's own; the run total accumulates across leaves.
    for leaf in &run.leaves {
        assert_eq!(leaf.usage.prompt_tokens, 3);
        assert_eq!(leaf.usage.completion_tokens, 4);
        assert_eq!(leaf.usage.total_tokens, 7);
    }
    let total: u64 = run.leaves.iter().map(|leaf| leaf.usage.total_tokens).sum();
    assert_eq!(total, 14);
}

// ---------------------------------------------------------------------------
// `run` is unchanged: same output, no collection
// ---------------------------------------------------------------------------

#[tokio::test]
async fn run_output_matches_run_collecting_output() {
    let (lm_a, _c1) = canned_lm(vec![
        text(fields(&[("answer", "42")])),
        text(fields(&[("verdict", "correct")])),
    ])
    .await;
    let (lm_b, _c2) = canned_lm(vec![
        text(fields(&[("answer", "42")])),
        text(fields(&[("verdict", "correct")])),
    ])
    .await;

    let plain = Interpreter::load(seq_program(), RuntimeEnv::new().bind_model("m", lm_a))
        .await
        .unwrap();
    let collecting = Interpreter::load(seq_program(), RuntimeEnv::new().bind_model("m", lm_b))
        .await
        .unwrap();

    let input = obj(&[("question", json!("what is 6*7?"))]);
    let from_run = plain
        .run(input.clone(), None, Budget::unlimited())
        .await
        .unwrap();
    let from_collecting = collecting
        .run_collecting(input, None, Budget::unlimited())
        .await
        .unwrap();

    assert_eq!(from_run, from_collecting.output);
}
