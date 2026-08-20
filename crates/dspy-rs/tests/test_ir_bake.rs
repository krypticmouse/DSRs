//! IR-6 (RFC 0002 §5): `Program::bake` — folding an overlay into a new
//! program value, lineage stamping, hash recompute, and behavioral equality
//! of base+overlay vs. baked on canned LMs.

use std::sync::Arc;

use dspy_rs::ir::{
    self, BakeError, Budget, CodeLang, FieldType as T, Interpreter, Lineage, Overlay, ParamValue,
    Program, ProgramBuilder, RuntimeEnv, SignatureDef,
};
use dspy_rs::trace::capture;
use dspy_rs::{LM, LMClient, LMConfig, TestCompletionModel};
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

/// question → drafter (QA) → checker (Check) → verdict.
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

fn note() -> Lineage {
    Lineage {
        optimizer: "test-optimizer-0.1".into(),
        trainset: "toy@1".into(),
        budget: "12 rollouts / $0.00".into(),
        // Deliberately junk: bake must overwrite it with the parent hash.
        parent: Some("junk".into()),
        date: "2026-08-14".into(),
        overlay: None,
    }
}

/// The winning candidate: drafter instruction + demos.
fn winning_overlay(program: &Program) -> Overlay {
    let mut overlay = Overlay::new(program);
    let instr = program
        .slot_of::<ir::Instruction>("drafter.instruction")
        .unwrap();
    overlay.set_instruction(instr, "BAKED: answer in one word.");
    let demos = program.slot_of::<ir::Demos>("drafter.demos").unwrap();
    overlay.set_demos(
        demos,
        vec![ir::DemoRow {
            input: obj(&[("question", json!("demo q"))]),
            output: obj(&[("answer", json!("demo a"))]),
        }],
    );
    overlay
}

// ---------------------------------------------------------------------------
// Behavioral equality: base+overlay == baked
// ---------------------------------------------------------------------------

#[tokio::test]
async fn baked_program_runs_identically_to_base_plus_overlay() {
    let base = pipeline_program();
    let overlay = winning_overlay(&base);
    let baked = base.bake(&overlay, note()).unwrap();

    let responses = || {
        vec![
            text(fields(&[("answer", "42")])),
            text(fields(&[("verdict", "correct")])),
        ]
    };
    let (lm_a, _) = canned_lm(responses()).await;
    let (lm_b, _) = canned_lm(responses()).await;

    let base_interp = Interpreter::load(base, RuntimeEnv::new().bind_model("m", lm_a))
        .await
        .unwrap();
    let baked_interp = Interpreter::load(baked, RuntimeEnv::new().bind_model("m", lm_b))
        .await
        .unwrap();

    let input = obj(&[("question", json!("what is 6*7?"))]);
    let (base_result, base_trace) =
        capture(|| base_interp.run(input.clone(), Some(Arc::new(overlay)), Budget::unlimited()))
            .await;
    let (baked_result, baked_trace) =
        capture(|| baked_interp.run(input.clone(), None, Budget::unlimited())).await;

    // Same outputs.
    assert_eq!(base_result.unwrap(), baked_result.unwrap());

    // Same rendered prompts, span for span — overlay read-through and baked
    // defaults are indistinguishable at the LM boundary.
    assert_eq!(base_trace.spans.len(), baked_trace.spans.len());
    for (a, b) in base_trace.spans.iter().zip(&baked_trace.spans) {
        assert_eq!(
            format!("{:?}", base_trace.prompt(a)),
            format!("{:?}", baked_trace.prompt(b)),
            "prompt mismatch at component {}",
            base_trace.component_name(a.component),
        );
        assert_eq!(a.request_hash, b.request_hash, "request hashes must match");
    }

    // The baked prompt actually carries the overlay values.
    let drafter = baked_trace.for_component("drafter").next().unwrap();
    let prompt = format!("{:?}", baked_trace.prompt(drafter));
    assert!(prompt.contains("BAKED: answer in one word."));
    assert!(prompt.contains("demo q"));
}

// ---------------------------------------------------------------------------
// Defaults folding: every overlay kind lands in the slot default
// ---------------------------------------------------------------------------

#[test]
fn bake_folds_all_slot_kinds_into_defaults() {
    let mut b = ProgramBuilder::new("kinds");
    b.cap("net:search");
    let m1 = b.model("m1", config());
    let m2 = b.model("m2", config());
    let main_sig = b.sig(
        SignatureDef::build("Main")
            .input("question", T::String)
            .output("answer", T::String)
            .finish()
            .unwrap(),
    );
    let search_sig = b.sig(
        SignatureDef::build("Search")
            .input("query", T::String)
            .output("results", T::List(Box::new(T::String)))
            .finish()
            .unwrap(),
    );
    let shout_sig = b.sig(
        SignatureDef::build("Shout")
            .input("question", T::String)
            .output("shout", T::String)
            .finish()
            .unwrap(),
    );
    let search = b.host_tool("search", "old tool desc", search_sig, &["net:search"]);
    let calc_sig = b.sig(
        SignatureDef::build("Calc")
            .input("expression", T::String)
            .output("value", T::Float)
            .finish()
            .unwrap(),
    );
    let calc = b.host_tool("calc", "Evaluate arithmetic", calc_sig, &[]);
    let researcher = ir::agent("researcher", main_sig)
        .model(m1)
        .bind("question", ir::input("question"))
        .tools([search, calc]);
    let shouter = ir::hole("shouter", shout_sig, "(a) => ({shout: a.question})", &[])
        .bind("question", ir::input("question"));
    let program = b
        .main(
            main_sig,
            ir::seq([researcher, shouter]).out("answer", ir::out("researcher", "answer")),
        )
        .unwrap();

    let mut overlay = Overlay::new(&program);
    let model_id = program.param_id("researcher.model").unwrap();
    overlay
        .set(&program, model_id, ParamValue::ModelRef { model: m2 })
        .unwrap();
    let desc_id = program.param_id("tool.search.desc").unwrap();
    overlay
        .set(
            &program,
            desc_id,
            ParamValue::ToolDesc {
                text: "new tool desc".to_string(),
            },
        )
        .unwrap();
    let code_id = program.param_id("shouter.code").unwrap();
    let new_code = ParamValue::code(CodeLang::Js, "(a) => ({shout: a.question.toUpperCase()})");
    overlay.set(&program, code_id, new_code.clone()).unwrap();
    let tool_set_id = program.param_id("researcher.tool_set").unwrap();
    let restricted = ParamValue::ToolSet {
        tools: vec![search],
    };
    overlay
        .set(&program, tool_set_id, restricted.clone())
        .unwrap();

    let baked = program.bake(&overlay, note()).unwrap();

    assert_eq!(
        baked.params[model_id].default,
        ParamValue::ModelRef { model: m2 }
    );
    assert_eq!(
        baked.params[desc_id].default,
        ParamValue::ToolDesc {
            text: "new tool desc".to_string()
        }
    );
    assert_eq!(baked.params[code_id].default, new_code);
    assert_eq!(baked.params[tool_set_id].default, restricted);
    // The restricted selection is a first-class artifact: it prints as a
    // `tool_set` line and the baked program reloads identically.
    let text = baked.to_dsrs();
    assert!(text.contains("tool_set [search]"), "{text}");
    let reloaded = Program::from_dsrs(&text).unwrap();
    assert_eq!(reloaded.meta.program_hash, baked.meta.program_hash);

    // The base program is untouched (bake is pure).
    assert_eq!(
        program.params[model_id].default,
        ParamValue::ModelRef { model: m1 }
    );
    assert_eq!(
        program.params[desc_id].default,
        ParamValue::ToolDesc {
            text: "old tool desc".to_string()
        }
    );
}

// ---------------------------------------------------------------------------
// Lineage, hashing, serde
// ---------------------------------------------------------------------------

#[test]
fn bake_stamps_lineage_and_recomputes_the_hash() {
    let base = pipeline_program();
    let base_hash = base.meta.program_hash;
    let overlay = winning_overlay(&base);
    let overlay_hash = overlay.hash();

    let baked = base.bake(&overlay, note()).unwrap();

    let lineage = baked.meta.lineage.as_ref().unwrap();
    assert_eq!(&*lineage.optimizer, "test-optimizer-0.1");
    assert_eq!(&*lineage.trainset, "toy@1");
    assert_eq!(&*lineage.budget, "12 rollouts / $0.00");
    assert_eq!(&*lineage.date, "2026-08-14");
    // parent/overlay come from bake, never the caller.
    assert_eq!(
        lineage.parent.as_deref(),
        Some(format!("{base_hash:016x}").as_str())
    );
    assert_eq!(
        lineage.overlay.as_deref(),
        Some(format!("{overlay_hash:016x}").as_str())
    );

    // Content changed → hash changed; base untouched.
    assert_ne!(baked.meta.program_hash, base_hash);
    assert_eq!(base.meta.program_hash, base_hash);
    assert!(base.meta.lineage.is_none());

    // Lineage is excluded from the hash preimage: the hash equals the
    // recomputed content hash.
    assert_eq!(baked.meta.program_hash, baked.compute_hash());

    // An overlay minted against the base does not apply to the baked program.
    let stale = winning_overlay(&base);
    let err = baked.bake(&stale, note()).unwrap_err();
    assert!(matches!(err, BakeError::Overlay(_)));
}

#[test]
fn baking_an_empty_overlay_still_restamps_identity() {
    let base = pipeline_program();
    let overlay = Overlay::new(&base);
    let baked = base.bake(&overlay, note()).unwrap();
    // No content change → same content hash; lineage records the (empty)
    // promotion.
    assert_eq!(baked.meta.program_hash, base.meta.program_hash);
    assert!(baked.meta.lineage.is_some());
}

#[test]
fn baked_program_serde_round_trips() {
    let base = pipeline_program();
    let baked = base.bake(&winning_overlay(&base), note()).unwrap();

    let json = serde_json::to_string(&baked).unwrap();
    let loaded: Program = serde_json::from_str(&json).unwrap();

    // The load path re-validates, rebuilds indexes, and recomputes the hash —
    // and lands on the same identity.
    assert_eq!(loaded.meta.program_hash, baked.meta.program_hash);
    assert_eq!(loaded.meta.lineage, baked.meta.lineage);
    let instr = loaded.param_id("drafter.instruction").unwrap();
    assert_eq!(
        loaded.params[instr].default,
        ParamValue::Instruction {
            text: "BAKED: answer in one word.".to_string()
        }
    );
}
