//! The render slot (`<leaf>.render`, `ParamKind::Render`): bare vs. marker
//! rendering as an optimizable parameter — adapter roadmap §5.2/§5.4.
//!
//! Bare mode: instruction = the whole system prompt, raw input = the user
//! turn, whole completion = the single `String` output. Markers stay the
//! default; pre-render-slot programs print, hash, and run unchanged.

use std::sync::Arc;

use dspy_rs::ir::{
    self, Budget, FieldType as T, Interpreter, Overlay, OverlayError, ParamValue, Program,
    ProgramBuilder, RenderMode, RuntimeEnv, SignatureDef,
};
use dspy_rs::{LM, LMClient, LMConfig, TestCompletionModel};
use rig::completion::AssistantContent;
use rig::message::Text;
use serde_json::json;

fn text(content: impl Into<String>) -> AssistantContent {
    AssistantContent::Text(Text {
        text: content.into(),
    })
}

async fn canned_lm(responses: Vec<AssistantContent>) -> Arc<LM> {
    let client = TestCompletionModel::new(responses);
    let lm = temp_env::async_with_vars(
        [("OPENAI_API_KEY", Some("test"))],
        LM::builder()
            .model("openai:gpt-4o-mini".to_string())
            .build(),
    )
    .await
    .unwrap()
    .with_client(LMClient::Test(client))
    .await
    .unwrap();
    Arc::new(lm)
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

fn qa_sig() -> SignatureDef {
    SignatureDef::build("QA")
        .instruction("Answer the question concisely.")
        .input("question", T::String)
        .output("answer", T::String)
        .finish()
        .unwrap()
}

fn qa_program(mode: RenderMode) -> Program {
    let mut b = ProgramBuilder::new("bare_unit");
    b.model("m", config());
    let qa = b.sig(qa_sig());
    let node = ir::predict("answerer", qa)
        .render(mode)
        .bind("question", ir::input("question"));
    b.main(
        qa,
        ir::seq([node]).out("answer", ir::out("answerer", "answer")),
    )
    .unwrap()
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

#[tokio::test]
async fn bare_opening_is_instruction_plus_raw_input() {
    let lm = canned_lm(vec![]).await;
    let interp = Interpreter::load(qa_program(RenderMode::Bare), RuntimeEnv::new().bind_model("m", lm))
        .await
        .unwrap();

    let chat = interp
        .conversation_opening(&obj(&[("question", json!("What is DSRs?"))]), None)
        .unwrap();
    let messages = chat.messages;
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].content(), "Answer the question concisely.");
    assert_eq!(messages[1].content(), "What is DSRs?");
    for message in &messages {
        assert!(
            !message.content().contains("[[ ##"),
            "bare rendering must not leak markers: {}",
            message.content()
        );
    }
}

#[tokio::test]
async fn markers_default_is_unchanged() {
    let lm = canned_lm(vec![]).await;
    let interp = Interpreter::load(
        qa_program(RenderMode::Markers),
        RuntimeEnv::new().bind_model("m", lm),
    )
    .await
    .unwrap();

    let chat = interp
        .conversation_opening(&obj(&[("question", json!("What is DSRs?"))]), None)
        .unwrap();
    assert!(chat.messages[0].content().contains("[[ ## question ## ]]"));
    assert!(chat.messages[1].content().contains("[[ ## question ## ]]"));
}

#[tokio::test]
async fn bare_run_takes_whole_completion_as_the_output() {
    let lm = canned_lm(vec![text("DSRs is a Rust framework for LM pipelines.")]).await;
    let interp = Interpreter::load(qa_program(RenderMode::Bare), RuntimeEnv::new().bind_model("m", lm))
        .await
        .unwrap();

    let output = interp
        .run(
            obj(&[("question", json!("What is DSRs?"))]),
            None,
            Budget::unlimited(),
        )
        .await
        .unwrap();
    assert_eq!(
        output["answer"],
        json!("DSRs is a Rust framework for LM pipelines.")
    );
}

// ---------------------------------------------------------------------------
// Slot semantics: overlay, validation, text form
// ---------------------------------------------------------------------------

#[tokio::test]
async fn overlay_flips_rendering_without_touching_the_program() {
    let program = qa_program(RenderMode::Markers);
    let render_id = program.param_id("answerer.render").unwrap();
    let mut overlay = Overlay::new(&program);
    overlay
        .set(
            &program,
            render_id,
            ParamValue::Render {
                mode: RenderMode::Bare,
            },
        )
        .unwrap();

    let lm = canned_lm(vec![]).await;
    let interp = Interpreter::load(program, RuntimeEnv::new().bind_model("m", lm))
        .await
        .unwrap();
    let chat = interp
        .conversation_opening(
            &obj(&[("question", json!("What is DSRs?"))]),
            Some(Arc::new(overlay)),
        )
        .unwrap();
    assert_eq!(chat.messages[0].content(), "Answer the question concisely.");
    assert_eq!(chat.messages[1].content(), "What is DSRs?");
}

#[test]
fn overlay_refuses_bare_on_a_multi_output_leaf() {
    let mut b = ProgramBuilder::new("wide_unit");
    b.model("m", config());
    let sig = b.sig(
        SignatureDef::build("Wide")
            .input("question", T::String)
            .output("answer", T::String)
            .output("confidence", T::Float)
            .finish()
            .unwrap(),
    );
    let node = ir::predict("wide", sig).bind("question", ir::input("question"));
    let program = b
        .main(
            sig,
            ir::seq([node])
                .out("answer", ir::out("wide", "answer"))
                .out("confidence", ir::out("wide", "confidence")),
        )
        .unwrap();

    let render_id = program.param_id("wide.render").unwrap();
    let mut overlay = Overlay::new(&program);
    let err = overlay
        .set(
            &program,
            render_id,
            ParamValue::Render {
                mode: RenderMode::Bare,
            },
        )
        .unwrap_err();
    assert!(matches!(err, OverlayError::RenderUnsupported { .. }));
}

#[test]
fn builder_refuses_bare_on_a_multi_output_leaf() {
    let mut b = ProgramBuilder::new("wide_unit");
    b.model("m", config());
    let sig = b.sig(
        SignatureDef::build("Wide")
            .input("question", T::String)
            .output("answer", T::String)
            .output("confidence", T::Float)
            .finish()
            .unwrap(),
    );
    let node = ir::predict("wide", sig)
        .render(RenderMode::Bare)
        .bind("question", ir::input("question"));
    let err = b.main(
        sig,
        ir::seq([node])
            .out("answer", ir::out("wide", "answer"))
            .out("confidence", ir::out("wide", "confidence")),
    );
    assert!(err.is_err(), "bare on a two-output sig must not validate");
}

#[test]
fn bare_prints_and_round_trips_markers_stay_silent() {
    let bare = qa_program(RenderMode::Bare);
    let bare_text = bare.to_dsrs();
    assert!(
        bare_text.contains("render \"bare\""),
        "canonical text missing the render opt:\n{bare_text}"
    );
    let reparsed = Program::from_dsrs(&bare_text).unwrap();
    assert_eq!(reparsed.to_dsrs(), bare_text);
    assert_eq!(reparsed.meta.program_hash, bare.meta.program_hash);

    let markers = qa_program(RenderMode::Markers);
    assert!(
        !markers.to_dsrs().contains("render"),
        "default mode must not print (hash stability for existing programs)"
    );
}

#[test]
fn cot_reasoning_field_blocks_bare() {
    let mut b = ProgramBuilder::new("cot_unit");
    b.model("m", config());
    let qa = b.sig(qa_sig());
    // cot augments the sig with `reasoning`, making it two-output.
    let node = ir::cot("thinker", qa)
        .render(RenderMode::Bare)
        .bind("question", ir::input("question"));
    let err = b.main(
        qa,
        ir::seq([node]).out("answer", ir::out("thinker", "answer")),
    );
    assert!(err.is_err(), "bare on a cot leaf must not validate");
}
