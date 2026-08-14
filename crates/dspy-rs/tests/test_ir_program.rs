//! IR-2 (RFC 0002 §2): program construction, load-time validation, parameter
//! addressing, overlays, and the canonical serde round trip.
#![cfg(feature = "ir")]

use dspy_rs::LMConfig;
use dspy_rs::ir::{
    self, BudgetPolicy, BuildError, DemoRow, FieldType as T, NodeBudget, Overlay, OverlayError,
    ParamKind, ParamValue, Program, ProgramBuilder, SignatureDef, ValidateError,
};
use dspy_rs::typesys::{EnumDef, EnumValueDef, TypeTable};

const JS_CITE_FILTER: &str = r#"(a) => ({
  answer: a.draft,
  sources: a.evidence.filter(e => e.startsWith("http")),
})"#;

fn model_config(name: &str) -> LMConfig {
    LMConfig {
        model: name.to_string(),
        ..LMConfig::default()
    }
}

/// The RFC §4.3 worked example: CoT draft → tool-using agent loop → typed hole.
fn qa_program() -> Program {
    let mut b = ProgramBuilder::new("qa");
    b.cap("net:search");
    let fast = b.model("fast", model_config("openai:gpt-4o-mini"));
    let deep = b.model("deep", model_config("anthropic:claude-sonnet-4-5"));

    let main_sig = b.sig(
        SignatureDef::build("Main")
            .input("question", T::String)
            .output("answer", T::String)
            .output("sources", T::List(Box::new(T::String)))
            .finish()
            .unwrap(),
    );
    let draft = b.sig(
        SignatureDef::build("Draft")
            .instruction("Draft a thorough, factual answer.")
            .input("question", T::String)
            .output("answer", T::String)
            .finish()
            .unwrap(),
    );
    let research = b.sig(
        SignatureDef::build("Research")
            .instruction("Verify the draft against sources; collect URLs.")
            .input("question", T::String)
            .input("draft", T::String)
            .output("evidence", T::List(Box::new(T::String)))
            .finish()
            .unwrap(),
    );
    let cite = b.sig(
        SignatureDef::build("CiteCheck")
            .input("draft", T::String)
            .input("evidence", T::List(Box::new(T::String)))
            .output("answer", T::String)
            .output("sources", T::List(Box::new(T::String)))
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

    let search = b.host_tool(
        "search",
        "Web search; returns result snippets with URLs",
        search_sig,
        &["net:search"],
    );

    let drafter = ir::cot("drafter", draft)
        .model(deep)
        .bind("question", ir::input("question"));
    let researcher = ir::agent("researcher", research)
        .model(fast)
        .bind("question", ir::input("question"))
        .bind("draft", ir::out("drafter", "answer"))
        .tools([search])
        .max_turns(6)
        .budget(NodeBudget {
            max_tokens: Some(40_000),
            on_exhausted: BudgetPolicy::Finalize,
            ..NodeBudget::default()
        });
    let checker = ir::hole("checker", cite, JS_CITE_FILTER, &[])
        .bind("draft", ir::out("drafter", "answer"))
        .bind("evidence", ir::out("researcher", "evidence"));

    b.main(
        main_sig,
        ir::seq([drafter, researcher, checker])
            .out("answer", ir::out("checker", "answer"))
            .out("sources", ir::out("checker", "sources")),
    )
    .expect("the worked example builds")
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

#[test]
fn worked_example_builds_and_validates() {
    let program = qa_program();
    assert_eq!(&*program.meta.name, "qa");
    assert_eq!(program.meta.format, 1);
    assert_ne!(program.meta.program_hash, 0);
    // Idempotent re-validation (the loader path).
    program.validate().unwrap();
}

#[test]
fn param_paths_are_addressable() {
    let program = qa_program();
    for path in [
        "drafter.instruction",
        "drafter.demos",
        "drafter.model",
        "researcher.instruction",
        "researcher.demos",
        "researcher.model",
        "researcher.context",
        "checker.code",
        "tool.search.desc",
    ] {
        let id = program
            .param_id(path)
            .unwrap_or_else(|| panic!("param path `{path}` should resolve"));
        assert_eq!(program.param_path(id), path);
    }
    assert!(program.param_id("drafter.code").is_none());

    // Typed handles are kind-checked.
    assert!(
        program
            .slot_of::<ir::Instruction>("drafter.instruction")
            .is_some()
    );
    assert!(
        program
            .slot_of::<ir::Demos>("drafter.instruction")
            .is_none()
    );

    // The optimizer contract: enumerable typed genes.
    let instructions: Vec<&str> = program
        .slots(ParamKind::Instruction)
        .map(|(_, slot)| &*slot.path)
        .collect();
    assert_eq!(
        instructions,
        vec!["drafter.instruction", "researcher.instruction"]
    );
    let code: Vec<&str> = program
        .slots(ParamKind::Code)
        .map(|(_, slot)| &*slot.path)
        .collect();
    assert_eq!(code, vec!["checker.code"]);
}

#[test]
fn cot_is_signature_sugar() {
    let program = qa_program();
    // The drafter's signature was augmented with a leading `reasoning` output;
    // no distinct node kind exists.
    let drafter = program
        .nodes
        .values()
        .find_map(|node| match node {
            ir::Node::Predict(n) if program.syms.get(n.name) == "drafter" => Some(n),
            _ => None,
        })
        .expect("drafter is a plain Predict node");
    let sig = &program.sigs[drafter.sig];
    assert_eq!(&*sig.outputs[0].name, "reasoning");
    assert_eq!(sig.outputs[0].ty, T::String);
    assert_eq!(&*sig.outputs[1].name, "answer");

    // Instruction slot default copied the signature instruction.
    let id = program.param_id("drafter.instruction").unwrap();
    match &program.params[id].default {
        ParamValue::Instruction { text } => {
            assert_eq!(text, "Draft a thorough, factual answer.")
        }
        other => panic!("expected instruction default, got {other:?}"),
    }
}

#[test]
fn program_hash_is_content_addressed() {
    let a = qa_program();
    let b = qa_program();
    assert_eq!(a.meta.program_hash, b.meta.program_hash);
    assert_eq!(a.compute_hash(), a.meta.program_hash);
}

// ---------------------------------------------------------------------------
// Validation rejections
// ---------------------------------------------------------------------------

fn tiny_builder() -> (ProgramBuilder, ir::SigId, ir::SigId) {
    let mut b = ProgramBuilder::new("tiny");
    b.model("only", model_config("openai:gpt-4o-mini"));
    let main_sig = b.sig(
        SignatureDef::build("Main")
            .input("question", T::String)
            .output("answer", T::String)
            .finish()
            .unwrap(),
    );
    let qa = b.sig(
        SignatureDef::build("QA")
            .input("question", T::String)
            .output("answer", T::String)
            .finish()
            .unwrap(),
    );
    (b, main_sig, qa)
}

#[test]
fn dangling_port_is_a_build_error() {
    let (b, main_sig, qa) = tiny_builder();
    let node = ir::predict("answerer", qa).bind("question", ir::out("ghost", "answer"));
    let err = b
        .main(
            main_sig,
            ir::seq([node]).out("answer", ir::out("answerer", "answer")),
        )
        .unwrap_err();
    assert!(matches!(err, BuildError::UnknownNode { name } if name == "ghost"));
}

#[test]
fn forward_reference_is_a_build_error() {
    let (b, main_sig, qa) = tiny_builder();
    // `first` references `second`, which is lowered later — earlier-sibling
    // visibility is structural.
    let first = ir::predict("first", qa).bind("question", ir::out("second", "answer"));
    let second = ir::predict("second", qa).bind("question", ir::input("question"));
    let err = b
        .main(
            main_sig,
            ir::seq([first, second]).out("answer", ir::out("second", "answer")),
        )
        .unwrap_err();
    assert!(matches!(err, BuildError::UnknownNode { name } if name == "second"));
}

#[test]
fn unbound_input_is_rejected() {
    let (b, main_sig, qa) = tiny_builder();
    let node = ir::predict("answerer", qa); // `question` never bound
    let err = b
        .main(
            main_sig,
            ir::seq([node]).out("answer", ir::out("answerer", "answer")),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        BuildError::Invalid(ValidateError::UnboundInput { ref at, ref field })
            if at == "answerer" && field == "question"
    ));
}

#[test]
fn binding_type_mismatch_is_rejected() {
    let mut b = ProgramBuilder::new("tiny");
    b.model("only", model_config("openai:gpt-4o-mini"));
    let main_sig = b.sig(
        SignatureDef::build("Main")
            .input("question", T::String)
            .output("answer", T::String)
            .finish()
            .unwrap(),
    );
    let count_sig = b.sig(
        SignatureDef::build("Count")
            .input("question", T::String)
            .output("count", T::Int)
            .finish()
            .unwrap(),
    );
    let qa = b.sig(
        SignatureDef::build("QA")
            .input("question", T::String)
            .output("answer", T::String)
            .finish()
            .unwrap(),
    );
    let counter = ir::predict("counter", count_sig).bind("question", ir::input("question"));
    // Int output wired into a String input: not a permitted widening.
    let answerer = ir::predict("answerer", qa).bind("question", ir::out("counter", "count"));
    let err = b
        .main(
            main_sig,
            ir::seq([counter, answerer]).out("answer", ir::out("answerer", "answer")),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        BuildError::Invalid(ValidateError::BindingTypeMismatch { ref field, .. })
            if field == "question"
    ));
}

#[test]
fn int_to_float_widening_is_permitted() {
    let mut b = ProgramBuilder::new("widen");
    b.model("only", model_config("openai:gpt-4o-mini"));
    let main_sig = b.sig(
        SignatureDef::build("Main")
            .input("question", T::String)
            .output("verdict", T::String)
            .finish()
            .unwrap(),
    );
    let count_sig = b.sig(
        SignatureDef::build("Count")
            .input("question", T::String)
            .output("count", T::Int)
            .finish()
            .unwrap(),
    );
    let rate_sig = b.sig(
        SignatureDef::build("Rate")
            .input("score", T::Float)
            .output("verdict", T::String)
            .finish()
            .unwrap(),
    );
    let counter = ir::predict("counter", count_sig).bind("question", ir::input("question"));
    let rater = ir::predict("rater", rate_sig).bind("score", ir::out("counter", "count"));
    b.main(
        main_sig,
        ir::seq([counter, rater]).out("verdict", ir::out("rater", "verdict")),
    )
    .expect("Int → Float widening is allowed");
}

#[test]
fn duplicate_leaf_names_are_rejected() {
    let (b, main_sig, qa) = tiny_builder();
    let one = ir::predict("answerer", qa).bind("question", ir::input("question"));
    let two = ir::predict("answerer", qa).bind("question", ir::input("question"));
    let err = b
        .main(
            main_sig,
            ir::seq([one, two]).out("answer", ir::out("answerer", "answer")),
        )
        .unwrap_err();
    // The step-name registry trips first; both surface duplicate naming.
    assert!(matches!(
        err,
        BuildError::DuplicateStepName { ref name } if name == "answerer"
    ));
}

#[test]
fn capability_violation_is_rejected() {
    let (b, main_sig, qa) = tiny_builder();
    // Program ceiling is empty; the hole asks for net:fetch.
    let node = ir::hole(
        "fetcher",
        qa,
        "(a) => ({answer: a.question})",
        &["net:fetch"],
    )
    .bind("question", ir::input("question"));
    let err = b
        .main(
            main_sig,
            ir::seq([node]).out("answer", ir::out("fetcher", "answer")),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        BuildError::Invalid(ValidateError::CapsExceedProgram { ref at, ref missing })
            if at == "fetcher" && missing == &["net:fetch".to_string()]
    ));
}

#[test]
fn unbounded_loops_are_unrepresentable() {
    let result = std::panic::catch_unwind(|| {
        let (_b, _main, qa) = tiny_builder();
        ir::loop_(ir::seq([ir::predict("p", qa)]), 0)
    });
    assert!(result.is_err(), "max_iters = 0 must be rejected by type");
}

#[test]
fn uncovered_route_without_default_is_rejected() {
    let mut b = ProgramBuilder::new("routes");
    b.model("only", model_config("openai:gpt-4o-mini"));
    let mut types = TypeTable::default();
    types.enums.insert(
        "Severity".to_string(),
        EnumDef {
            internal_name: "Severity".to_string(),
            rendered_name: "Severity".to_string(),
            docs: None,
            values: ["Low", "High"]
                .iter()
                .map(|name| EnumValueDef {
                    name: (*name).to_string(),
                    rendered_name: (*name).to_string(),
                    docs: None,
                })
                .collect(),
        },
    );
    b.add_types(&types);

    let main_sig = b.sig(
        SignatureDef::build("Main")
            .input("ticket", T::String)
            .output("reply", T::String)
            .finish()
            .unwrap(),
    );
    let classify = b.sig(
        SignatureDef::build("Classify")
            .input("ticket", T::String)
            .output("severity", T::Enum("Severity".to_string()))
            .finish()
            .unwrap(),
    );
    let reply = b.sig(
        SignatureDef::build("Reply")
            .input("ticket", T::String)
            .output("reply", T::String)
            .finish()
            .unwrap(),
    );

    let classifier = ir::predict("classifier", classify).bind("ticket", ir::input("ticket"));
    let router = ir::route(ir::out("classifier", "severity"))
        .arm(
            "Low",
            ir::predict("low_reply", reply).bind("ticket", ir::input("ticket")),
        )
        .named("router");
    // "High" is uncovered and there is no default.
    let err = b
        .main(
            main_sig,
            ir::seq([classifier, router]).out("reply", ir::out("router", "reply")),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        BuildError::Invalid(ValidateError::RouteUncovered { ref missing, .. })
            if missing == &["High".to_string()]
    ));
}

#[test]
fn missing_model_is_rejected_with_multiple_models() {
    let mut b = ProgramBuilder::new("models");
    b.model("a", model_config("openai:gpt-4o-mini"));
    b.model("b", model_config("openai:gpt-4o"));
    let main_sig = b.sig(
        SignatureDef::build("Main")
            .input("question", T::String)
            .output("answer", T::String)
            .finish()
            .unwrap(),
    );
    let node = ir::predict("answerer", main_sig).bind("question", ir::input("question"));
    let err = b
        .main(
            main_sig,
            ir::seq([node]).out("answer", ir::out("answerer", "answer")),
        )
        .unwrap_err();
    assert!(matches!(err, BuildError::MissingModel { ref at } if at == "answerer"));
}

// ---------------------------------------------------------------------------
// Overlays
// ---------------------------------------------------------------------------

#[test]
fn overlay_reads_through_without_mutation() {
    let program = qa_program();
    let slot = program
        .slot_of::<ir::Instruction>("drafter.instruction")
        .unwrap();

    let mut overlay = Overlay::new(&program);
    overlay.set_instruction(slot, "Be extremely terse.");

    // Resolve reads the overlay; the program default is untouched.
    match overlay.resolve(&program, slot.id) {
        ParamValue::Instruction { text } => assert_eq!(text, "Be extremely terse."),
        other => panic!("unexpected {other:?}"),
    }
    match &program.params[slot.id].default {
        ParamValue::Instruction { text } => {
            assert_eq!(text, "Draft a thorough, factual answer.")
        }
        other => panic!("unexpected {other:?}"),
    }

    // Unset slots fall through to the incumbent.
    let demos = program.param_id("drafter.demos").unwrap();
    assert!(overlay.get(demos).is_none());
    assert!(matches!(
        overlay.resolve(&program, demos),
        ParamValue::Demos { rows } if rows.is_empty()
    ));
}

#[test]
fn overlay_set_is_kind_checked() {
    let program = qa_program();
    let id = program.param_id("drafter.instruction").unwrap();
    let mut overlay = Overlay::new(&program);
    let err = overlay
        .set(
            &program,
            id,
            ParamValue::Demos {
                rows: vec![DemoRow {
                    input: serde_json::Map::new(),
                    output: serde_json::Map::new(),
                }],
            },
        )
        .unwrap_err();
    assert!(matches!(
        err,
        OverlayError::KindMismatch { ref path, expected: ParamKind::Instruction, got: ParamKind::Demos }
            if path == "drafter.instruction"
    ));
}

#[test]
fn overlay_is_base_hash_guarded() {
    let program = qa_program();
    let mut other = ProgramBuilder::new("other");
    other.model("only", model_config("openai:gpt-4o-mini"));
    let sig = other.sig(
        SignatureDef::build("Main")
            .input("question", T::String)
            .output("answer", T::String)
            .finish()
            .unwrap(),
    );
    let node = ir::predict("answerer", sig).bind("question", ir::input("question"));
    let other = other
        .main(
            sig,
            ir::seq([node]).out("answer", ir::out("answerer", "answer")),
        )
        .unwrap();

    let mut overlay = Overlay::new(&program);
    let id = other.param_id("answerer.instruction").unwrap();
    let err = overlay
        .set(
            &other,
            id,
            ParamValue::Instruction {
                text: "stale".into(),
            },
        )
        .unwrap_err();
    assert!(matches!(err, OverlayError::BaseMismatch { .. }));
}

#[test]
fn overlay_hash_and_named_round_trip() {
    let program = qa_program();
    let slot = program
        .slot_of::<ir::Instruction>("drafter.instruction")
        .unwrap();

    let empty = Overlay::new(&program);
    let mut a = Overlay::new(&program);
    a.set_instruction(slot, "Be terse.");
    let mut b = Overlay::new(&program);
    b.set_instruction(slot, "Be verbose.");

    assert_ne!(empty.hash(), a.hash());
    assert_ne!(a.hash(), b.hash());

    let named = a.to_named(&program);
    assert_eq!(named.len(), 1);
    assert!(named.contains_key("drafter.instruction"));
    let restored = Overlay::from_named(&program, named).unwrap();
    assert_eq!(restored.hash(), a.hash());

    let mut unknown = std::collections::BTreeMap::new();
    unknown.insert(
        "ghost.instruction".to_string(),
        ParamValue::Instruction { text: "x".into() },
    );
    assert!(matches!(
        Overlay::from_named(&program, unknown),
        Err(OverlayError::UnknownPath { .. })
    ));
}

// ---------------------------------------------------------------------------
// Serde round trip
// ---------------------------------------------------------------------------

#[test]
fn program_serde_round_trips_canonically() {
    let program = qa_program();
    let json = serde_json::to_string_pretty(&program).unwrap();

    // Secrets are structurally absent from the artifact.
    assert!(!json.contains("api_key"));

    let restored: Program = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.meta.program_hash, program.meta.program_hash);
    assert_eq!(restored.nodes.len(), program.nodes.len());
    assert_eq!(restored.params.len(), program.params.len());
    assert_eq!(
        restored.param_id("drafter.instruction"),
        program.param_id("drafter.instruction")
    );

    // Canonical form: serialize(deserialize(x)) == x.
    let json_again = serde_json::to_string_pretty(&restored).unwrap();
    assert_eq!(json, json_again);
}

#[test]
fn deserialization_is_a_load_and_validates() {
    let program = qa_program();
    let mut value = serde_json::to_value(&program).unwrap();

    // Corrupt the root node id: hostile artifacts fail the load, not a call.
    value["root"] = serde_json::json!(999);
    let err = serde_json::from_value::<Program>(value).unwrap_err();
    assert!(err.to_string().contains("id out of range"));
}
