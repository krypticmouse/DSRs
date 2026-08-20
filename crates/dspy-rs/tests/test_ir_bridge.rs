//! IR-2 bridge leftovers (RFC 0002 §2.4 migration contract): `fx::Params` ↔
//! `Overlay` bind/unbind, the `with_overlay` fx-lane scope, and the
//! `ModuleState` ↔ `Overlay` serde projection.

use dspy_rs::ir::{
    self, FieldType as T, Overlay, OverlayError, ParamValue, Program, ProgramBuilder, SignatureDef,
};
use dspy_rs::{LM, LMClient, LMConfig, ModuleState, Signature, TestCompletionModel, configure, fx};
use rig::completion::AssistantContent;
use rig::message::Text;
use serde_json::json;
use std::sync::LazyLock;
use tokio::sync::Mutex;

/// Global-settings lock: `configure` mutates process-wide LM state.
static SETTINGS_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

fn fields(pairs: &[(&str, &str)]) -> String {
    let mut out = String::new();
    for (name, value) in pairs {
        out.push_str(&format!("[[ ## {name} ## ]]\n{value}\n\n"));
    }
    out.push_str("[[ ## completed ## ]]\n");
    out
}

fn obj(pairs: &[(&str, serde_json::Value)]) -> serde_json::Map<String, serde_json::Value> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.clone()))
        .collect()
}

fn config() -> LMConfig {
    LMConfig {
        model: "openai:gpt-4o-mini".to_string(),
        ..LMConfig::default()
    }
}

/// question → drafter (QA) → checker (Check) → verdict. Leaf names double as
/// fx slot names / ModuleState dotted paths.
fn pipeline_program() -> Program {
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
// Params::bind — string names → Overlay
// ---------------------------------------------------------------------------

#[test]
fn params_bind_resolves_names_and_splits_demos() {
    let program = pipeline_program();
    let mut params = fx::Params::new();
    params.set_instruction("drafter", "BOUND: be terse.");
    params.set(
        "checker",
        dspy_rs::PredictState {
            demos: vec![obj(&[
                ("answer", json!("42")),
                ("verdict", json!("correct")),
            ])],
            instruction_override: Some("BOUND: judge hard.".to_string()),
        },
    );

    let overlay = params.bind(&program).unwrap();
    assert_eq!(overlay.base, program.meta.program_hash);

    let drafter_instr = program.param_id("drafter.instruction").unwrap();
    assert_eq!(
        overlay.get(drafter_instr),
        Some(&ParamValue::Instruction {
            text: "BOUND: be terse.".to_string()
        })
    );
    // The drafter's demos slot was not set: default reads through.
    let drafter_demos = program.param_id("drafter.demos").unwrap();
    assert_eq!(overlay.get(drafter_demos), None);

    // The checker's flat demo row split into input/output by signature side.
    let checker_demos = program.param_id("checker.demos").unwrap();
    match overlay.get(checker_demos).unwrap() {
        ParamValue::Demos { rows } => {
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].input, obj(&[("answer", json!("42"))]));
            assert_eq!(rows[0].output, obj(&[("verdict", json!("correct"))]));
        }
        other => panic!("expected demos, got {other:?}"),
    }
}

#[test]
fn bind_unknown_name_is_an_error() {
    let program = pipeline_program();
    let mut params = fx::Params::new();
    params.set_instruction("nonexistent", "x");
    let err = params.bind(&program).unwrap_err();
    assert!(
        matches!(err, OverlayError::UnknownPath { ref path } if path == "nonexistent.instruction")
    );
}

#[test]
fn bind_demo_field_outside_signature_is_an_error() {
    let program = pipeline_program();
    let mut params = fx::Params::new();
    params.set(
        "drafter",
        dspy_rs::PredictState {
            demos: vec![obj(&[("question", json!("q")), ("stray", json!("x"))])],
            instruction_override: None,
        },
    );
    let err = params.bind(&program).unwrap_err();
    assert!(matches!(
        err,
        OverlayError::DemoField { ref path, ref field }
            if path == "drafter.demos" && field == "stray"
    ));
}

#[test]
fn empty_entry_verifies_leaf_but_sets_nothing() {
    let program = pipeline_program();
    let mut params = fx::Params::new();
    params.set("drafter", dspy_rs::PredictState::default());
    let overlay = params.bind(&program).unwrap();
    assert!(overlay.is_empty());
}

// ---------------------------------------------------------------------------
// Params::from_overlay — Overlay → string names
// ---------------------------------------------------------------------------

#[test]
fn params_round_trip_through_overlay() {
    let program = pipeline_program();
    let mut params = fx::Params::new();
    params.set_instruction("drafter", "ROUND: trip.");
    params.set(
        "checker",
        dspy_rs::PredictState {
            demos: vec![obj(&[("answer", json!("a")), ("verdict", json!("v"))])],
            instruction_override: None,
        },
    );

    let overlay = params.bind(&program).unwrap();
    let restored = fx::Params::from_overlay(&program, &overlay).unwrap();

    assert_eq!(
        restored.get("drafter").unwrap().instruction_override,
        Some("ROUND: trip.".to_string())
    );
    let checker = restored.get("checker").unwrap();
    assert_eq!(checker.instruction_override, None);
    assert_eq!(
        checker.demos,
        vec![obj(&[("answer", json!("a")), ("verdict", json!("v"))])]
    );
    // Second bind is stable: same overlay content, same hash.
    assert_eq!(restored.bind(&program).unwrap().hash(), overlay.hash());
}

#[test]
fn from_overlay_is_restricted_to_instruction_and_demos() {
    let mut b = ProgramBuilder::new("modelswap");
    let m1 = b.model("m1", config());
    let m2 = b.model("m2", config());
    let sig = b.sig(
        SignatureDef::build("QA")
            .input("question", T::String)
            .output("answer", T::String)
            .finish()
            .unwrap(),
    );
    let node = ir::predict("answerer", sig)
        .model(m1)
        .bind("question", ir::input("question"));
    let program = b
        .main(
            sig,
            ir::seq([node]).out("answer", ir::out("answerer", "answer")),
        )
        .unwrap();

    let mut overlay = Overlay::new(&program);
    let instr = program.param_id("answerer.instruction").unwrap();
    overlay
        .set(
            &program,
            instr,
            ParamValue::Instruction {
                text: "kept".to_string(),
            },
        )
        .unwrap();
    let model = program.param_id("answerer.model").unwrap();
    overlay
        .set(&program, model, ParamValue::ModelRef { model: m2 })
        .unwrap();

    // ModelRef has no fx representation: skipped, not an error (RFC §2.4
    // "restricted to Instruction/Demos kinds").
    let params = fx::Params::from_overlay(&program, &overlay).unwrap();
    let state = params.get("answerer").unwrap();
    assert_eq!(state.instruction_override, Some("kept".to_string()));
    assert!(state.demos.is_empty());
}

#[test]
fn from_overlay_refuses_a_stale_base() {
    let program = pipeline_program();
    let mut stale = Overlay::default();
    stale.base = 0xdead_beef;
    let err = fx::Params::from_overlay(&program, &stale).unwrap_err();
    assert!(matches!(err, OverlayError::BaseMismatch { .. }));
}

// ---------------------------------------------------------------------------
// with_overlay — one candidate currency across both lanes
// ---------------------------------------------------------------------------

#[derive(Signature, Clone, Debug)]
/// Answer the question.
struct BridgeQA {
    #[input]
    question: String,

    #[output]
    answer: String,
}

/// A program whose leaf name matches the fx slot name and whose signature
/// matches `BridgeQA` — the shared-namespace contract.
fn fx_shaped_program() -> Program {
    let mut b = ProgramBuilder::new("fxshape");
    b.model("m", config());
    let qa = b.sig(
        SignatureDef::build("BridgeQA")
            .instruction("Answer the question.")
            .input("question", T::String)
            .output("answer", T::String)
            .finish()
            .unwrap(),
    );
    let node = ir::predict("bridge_slot", qa).bind("question", ir::input("question"));
    b.main(
        qa,
        ir::seq([node]).out("answer", ir::out("bridge_slot", "answer")),
    )
    .unwrap()
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn with_overlay_drives_an_fx_harness() {
    let _lock = SETTINGS_LOCK.lock().await;
    let client = TestCompletionModel::new(vec![AssistantContent::Text(Text {
        text: fields(&[("answer", "overlaid")]),
    })]);
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
    configure(lm);

    let program = fx_shaped_program();
    let mut overlay = Overlay::new(&program);
    let slot = program
        .slot_of::<ir::Instruction>("bridge_slot.instruction")
        .unwrap();
    overlay.set_instruction(slot, "MARKER-OVERLAY: answer in one word.");

    // The interpreter's candidate drives the fx harness unchanged.
    let out = fx::with_overlay(&program, &overlay, async {
        fx::predict::<BridgeQA>(
            "bridge_slot",
            BridgeQAInput {
                question: "q".to_string(),
            },
        )
        .await
    })
    .await
    .unwrap()
    .unwrap();
    assert_eq!(out.answer, "overlaid");

    let preamble = client.last_request().unwrap().preamble.unwrap_or_default();
    assert!(
        preamble.contains("MARKER-OVERLAY"),
        "overlay instruction should reach the fx prompt: {preamble}"
    );
}

#[tokio::test]
async fn with_overlay_refuses_a_stale_base_before_running() {
    let program = fx_shaped_program();
    let mut stale = Overlay::default();
    stale.base = 1;
    let err = fx::with_overlay(&program, &stale, async { unreachable!() as () })
        .await
        .unwrap_err();
    assert!(matches!(err, OverlayError::BaseMismatch { .. }));
}

// ---------------------------------------------------------------------------
// ModuleState ↔ Overlay projection
// ---------------------------------------------------------------------------

#[test]
fn module_state_round_trips_through_overlay() {
    let program = pipeline_program();
    let mut state = ModuleState::default();
    state.predictors.insert(
        "drafter".to_string(),
        dspy_rs::PredictState {
            demos: vec![obj(&[
                ("question", json!("demo q")),
                ("answer", json!("demo a")),
            ])],
            instruction_override: Some("STATE: draft tersely.".to_string()),
        },
    );
    state.predictors.insert(
        "checker".to_string(),
        dspy_rs::PredictState {
            demos: vec![],
            instruction_override: Some("STATE: check hard.".to_string()),
        },
    );

    let overlay = state.to_overlay(&program).unwrap();
    let drafter_demos = program.param_id("drafter.demos").unwrap();
    match overlay.get(drafter_demos).unwrap() {
        ParamValue::Demos { rows } => {
            assert_eq!(rows[0].input, obj(&[("question", json!("demo q"))]));
            assert_eq!(rows[0].output, obj(&[("answer", json!("demo a"))]));
        }
        other => panic!("expected demos, got {other:?}"),
    }

    let restored = overlay.to_module_state(&program).unwrap();
    assert_eq!(restored.predictors, state.predictors);

    // The serde format is untouched: the round-tripped state serializes to
    // exactly the same JSON the original did.
    assert_eq!(restored.to_json().unwrap(), state.to_json().unwrap());
}

#[test]
fn module_state_unknown_path_is_an_error() {
    let program = pipeline_program();
    let mut state = ModuleState::default();
    state
        .predictors
        .insert("ghost".to_string(), dspy_rs::PredictState::default());
    let err = state.to_overlay(&program).unwrap_err();
    assert!(matches!(err, OverlayError::UnknownPath { ref path } if path == "ghost.instruction"));
}

#[test]
fn overlay_to_module_state_refuses_a_stale_base() {
    let program = pipeline_program();
    let mut stale = Overlay::default();
    stale.base = 7;
    let err = stale.to_module_state(&program).unwrap_err();
    assert!(matches!(err, OverlayError::BaseMismatch { .. }));
}
