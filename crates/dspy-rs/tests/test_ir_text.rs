//! IR-5 (RFC 0002 §4/§5): the `.dsrs` text format — parse, canonical print,
//! text-preimage program hash, and parse-error quality.

use dspy_rs::LMConfig;
use dspy_rs::ir::{
    self, BudgetPolicy, DsrsFileError, NodeBudget, Overlay, ParamValue, ParseError, Program,
    ProgramBuilder, SignatureDef,
};
use dspy_rs::typesys::FieldType as T;

const JS_CITE_FILTER: &str = r#"(a) => ({
  answer: a.draft,
  sources: a.evidence.filter(e => e.startsWith("http")),
})"#;

fn fixture(name: &str) -> String {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

fn model_config(name: &str) -> LMConfig {
    LMConfig {
        model: name.to_string(),
        ..LMConfig::default()
    }
}

/// The RFC §4.3 worked example, builder frontend — must equal the parsed
/// `qa.dsrs` golden fixture (same canonical print, same hash).
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
// Golden: text frontend == builder frontend
// ---------------------------------------------------------------------------

#[test]
fn golden_fixture_is_the_builder_programs_canonical_print() {
    let golden = fixture("qa.dsrs");
    let built = qa_program();
    assert_eq!(
        built.to_dsrs(),
        golden,
        "builder canonical print must equal the checked-in golden"
    );
}

#[test]
fn parsed_golden_equals_builder_program() {
    let golden = fixture("qa.dsrs");
    let parsed = Program::from_dsrs(&golden).expect("golden parses");
    let built = qa_program();

    // Same canonical print, same content hash: the dual-frontend rule.
    assert_eq!(parsed.to_dsrs(), built.to_dsrs());
    assert_eq!(parsed.meta.program_hash, built.meta.program_hash);
    assert_eq!(parsed.compute_hash(), built.compute_hash());

    // Same addressable parameter surface.
    for path in [
        "drafter.instruction",
        "drafter.demos",
        "drafter.model",
        "researcher.instruction",
        "researcher.context",
        "checker.code",
        "tool.search.desc",
    ] {
        assert!(
            parsed.param_id(path).is_some(),
            "param path `{path}` resolves on the parsed program"
        );
    }

    // cot re-lowered to the identical augmented signature.
    let id = parsed.param_id("drafter.instruction").unwrap();
    match &parsed.params[id].default {
        ParamValue::Instruction { text } => {
            assert_eq!(text, "Draft a thorough, factual answer.")
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn parse_print_is_identity_on_the_golden() {
    let golden = fixture("qa.dsrs");
    let parsed = Program::from_dsrs(&golden).expect("golden parses");
    assert_eq!(parsed.to_dsrs(), golden, "golden fixture is canonical");
}

#[test]
fn scrambled_variant_canonicalizes_to_the_golden() {
    let golden = fixture("qa.dsrs");
    let scrambled = fixture("qa_scrambled.dsrs");
    assert_ne!(golden, scrambled);
    let parsed = Program::from_dsrs(&scrambled).expect("scrambled variant parses");
    assert_eq!(parsed.to_dsrs(), golden, "print . parse = canonical form");
    assert_eq!(
        parsed.meta.program_hash,
        qa_program().meta.program_hash,
        "reformatting must not move the content hash"
    );
}

/// `parse . print = id` (and hash stability) over every fixture.
#[test]
fn parse_print_round_trips_every_fixture() {
    let dir = format!("{}/tests/fixtures", env!("CARGO_MANIFEST_DIR"));
    let mut checked = 0;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("dsrs") {
            continue;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        let program =
            Program::from_dsrs(&text).unwrap_or_else(|e| panic!("{} parses: {e}", path.display()));
        let canonical = program.to_dsrs();
        let reparsed = Program::from_dsrs(&canonical)
            .unwrap_or_else(|e| panic!("{} canonical form parses: {e}", path.display()));
        assert_eq!(
            reparsed.to_dsrs(),
            canonical,
            "{}: canonical print is a fixed point",
            path.display()
        );
        assert_eq!(
            reparsed.meta.program_hash,
            program.meta.program_hash,
            "{}: hash is stable across the round trip",
            path.display()
        );
        checked += 1;
    }
    assert!(
        checked >= 3,
        "expected at least 3 .dsrs fixtures, saw {checked}"
    );
}

// ---------------------------------------------------------------------------
// Kitchen sink: every node kind survives text -> Program -> text
// ---------------------------------------------------------------------------

#[test]
fn kitchen_sink_covers_every_node_kind() {
    let program = Program::from_dsrs(&fixture("kitchen.dsrs")).expect("kitchen parses");
    let mut kinds: Vec<&str> = program
        .nodes
        .values()
        .map(|n| match n {
            ir::Node::Predict(_) => "predict",
            ir::Node::AgentLoop(_) => "agent",
            ir::Node::Seq(_) => "seq",
            ir::Node::ForkJoin(_) => "fork",
            ir::Node::Route(_) => "route",
            ir::Node::Retry(_) => "retry",
            ir::Node::Refine(_) => "refine",
            ir::Node::Loop(_) => "loop",
            ir::Node::Hole(_) => "hole",
        })
        .collect();
    kinds.sort();
    kinds.dedup();
    assert_eq!(
        kinds,
        vec![
            "agent", "fork", "hole", "loop", "predict", "refine", "retry", "route", "seq"
        ],
        "the kitchen fixture must exercise the whole closed node vocabulary"
    );

    // Lineage round-trips through the text but stays out of the hash.
    let lineage = program.meta.lineage.clone().expect("lineage parsed");
    assert_eq!(&*lineage.optimizer, "gepa-0.3");
    let mut without = program.clone();
    without.meta.lineage = None;
    assert_eq!(
        without.compute_hash(),
        program.compute_hash(),
        "lineage is excluded from the hash preimage"
    );
    let reparsed = Program::from_dsrs(&program.to_dsrs()).unwrap();
    assert_eq!(
        reparsed.meta.lineage.as_ref().map(|l| &*l.optimizer),
        Some("gepa-0.3")
    );
}

#[test]
fn capability_declarations_round_trip_and_are_enforced() {
    let program = Program::from_dsrs(&fixture("kitchen.dsrs")).unwrap();
    assert!(program.caps.contains("fs:read"));
    assert!(program.caps.contains("net:fetch"));

    let reparsed = Program::from_dsrs(&program.to_dsrs()).unwrap();
    assert_eq!(reparsed.caps, program.caps);
    let fetch = reparsed
        .tools
        .values()
        .find(|t| reparsed.syms.get(t.name) == "fetch")
        .expect("fetch tool survives");
    assert!(fetch.caps.contains("net:fetch"));
    let hole_caps = reparsed
        .nodes
        .values()
        .find_map(|n| match n {
            ir::Node::Hole(h) if reparsed.syms.get(h.name) == "redactor" => Some(h.caps.clone()),
            _ => None,
        })
        .expect("redactor hole survives");
    assert!(hole_caps.contains("fs:read"));
}

// ---------------------------------------------------------------------------
// Hash: text preimage, JSON agreement, overlay guards
// ---------------------------------------------------------------------------

#[test]
fn serde_json_path_agrees_with_text_on_the_hash() {
    let built = qa_program();
    let json = serde_json::to_string(&built).unwrap();
    let from_json: Program = serde_json::from_str(&json).unwrap();
    let from_text = Program::from_dsrs(&fixture("qa.dsrs")).unwrap();

    assert_eq!(from_json.meta.program_hash, built.meta.program_hash);
    assert_eq!(from_json.meta.program_hash, from_text.meta.program_hash);
    assert_eq!(from_json.to_dsrs(), from_text.to_dsrs());
}

#[test]
fn overlay_base_guard_works_across_frontends() {
    let built = qa_program();
    let parsed = Program::from_dsrs(&fixture("qa.dsrs")).unwrap();

    // An overlay minted against the builder program applies to the parsed one:
    // both frontends seal the same text-preimage hash.
    let mut overlay = Overlay::new(&built);
    let id = parsed.param_id("drafter.instruction").unwrap();
    overlay
        .set(
            &parsed,
            id,
            ParamValue::Instruction {
                text: "Be terse.".into(),
            },
        )
        .expect("base hashes agree across frontends");

    // And a different program still rejects it.
    let other = Program::from_dsrs(&fixture("kitchen.dsrs")).unwrap();
    let other_id = other.param_id("classifier.instruction").unwrap();
    let err = overlay
        .set(
            &other,
            other_id,
            ParamValue::Instruction { text: "x".into() },
        )
        .unwrap_err();
    assert!(matches!(err, ir::OverlayError::BaseMismatch { .. }));
}

#[test]
fn code_gene_hash_is_recomputed_from_source() {
    let parsed = Program::from_dsrs(&fixture("qa.dsrs")).unwrap();
    let id = parsed.param_id("checker.code").unwrap();
    match &parsed.params[id].default {
        ParamValue::Code { source, hash, .. } => {
            assert_eq!(source, JS_CITE_FILTER);
            assert_eq!(*hash, ir::code_hash(JS_CITE_FILTER));
        }
        other => panic!("unexpected {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// File API
// ---------------------------------------------------------------------------

#[test]
fn save_and_load_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("qa.dsrs");
    let built = qa_program();
    built.save_dsrs(&path).unwrap();
    let loaded = Program::load_dsrs(&path).unwrap();
    assert_eq!(loaded.meta.program_hash, built.meta.program_hash);
    assert_eq!(loaded.to_dsrs(), built.to_dsrs());
}

#[test]
fn load_dsrs_rejects_binary_artifacts() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("qa.dsrs");
    std::fs::write(&path, [0xff, 0xfe, 0x00, 0x01, b'd', b's', b'r', b's']).unwrap();
    let err = Program::load_dsrs(&path).unwrap_err();
    assert!(matches!(err, DsrsFileError::NotText { .. }));
}

#[test]
fn a_json_artifact_is_not_dsrs_text() {
    let err = Program::from_dsrs("{\"meta\":{\"name\":\"qa\"}}").unwrap_err();
    assert_eq!(err.line, 1);
    assert!(err.message.contains("`dsrs`"), "message: {}", err.message);
}

// ---------------------------------------------------------------------------
// Parse-error quality: line + problem, actionable for a generating model
// ---------------------------------------------------------------------------

fn parse_err(src: &str) -> ParseError {
    Program::from_dsrs(src).expect_err("must not parse")
}

#[test]
fn missing_capability_declaration_names_the_line_and_the_cap() {
    let src = r#"dsrs 1
program p
sig Main { in q: string out a: string }
main: Main = seq {
  h = hole Main (q = $.q) caps [net:fetch] js```
(a) => ({ a: a.q })
```
  out { a = h.a }
}
"#;
    let err = parse_err(src);
    assert_eq!(err.line, 5, "points at the caps declaration: {err}");
    assert!(err.message.contains("net:fetch"), "{err}");
    assert!(err.message.contains("caps"), "{err}");
    assert!(err.message.contains("hole `h`"), "{err}");
}

#[test]
fn unknown_top_level_keyword_names_the_line_and_alternatives() {
    let src = "dsrs 1\nprogram p\nwidget Foo { }\n";
    let err = parse_err(src);
    assert_eq!(err.line, 3);
    assert!(err.message.contains("widget"), "{err}");
    assert!(
        err.message.contains("`sig`"),
        "lists the expected keywords: {err}"
    );
}

#[test]
fn unbound_node_reference_names_the_line_and_the_name() {
    let src = r#"dsrs 1
program p
model only = "openai:gpt-4o-mini"
sig Main { in q: string out a: string }
main: Main = seq {
  x = predict Main (q = ghost.answer)
  out { a = x.a }
}
"#;
    let err = parse_err(src);
    assert_eq!(err.line, 6);
    assert!(err.message.contains("ghost"), "{err}");
    assert!(err.message.contains("unknown node"), "{err}");
}

#[test]
fn binding_type_mismatch_names_the_line_and_the_types() {
    let src = r#"dsrs 1
program p
model only = "openai:gpt-4o-mini"
sig Main { in question: string out answer: string }
sig Count { in question: string out count: int }
sig QA { in question: string out answer: string }
main: Main = seq {
  counter = predict Count (question = $.question)
  answerer = predict QA (question = counter.count)
  out { answer = answerer.answer }
}
"#;
    let err = parse_err(src);
    assert_eq!(err.line, 9, "points at the bad binding: {err}");
    assert!(err.message.contains("type mismatch"), "{err}");
    assert!(err.message.contains("question"), "{err}");
    assert!(err.message.contains("string"), "{err}");
    assert!(err.message.contains("int"), "{err}");
}

#[test]
fn unbound_input_names_the_leaf_line() {
    let src = r#"dsrs 1
program p
model only = "openai:gpt-4o-mini"
sig Main { in question: string out answer: string }
main: Main = seq {
  answerer = predict Main
  out { answer = answerer.answer }
}
"#;
    let err = parse_err(src);
    assert_eq!(err.line, 6, "points at the leaf: {err}");
    assert!(err.message.contains("`question`"), "{err}");
    assert!(err.message.contains("not bound"), "{err}");
}

#[test]
fn unknown_sig_model_and_tool_references_are_named() {
    let err = parse_err("dsrs 1\nprogram p\nmain: Main = seq { out { } }\n");
    assert_eq!(err.line, 3);
    assert!(err.message.contains("unknown sig `Main`"), "{err}");

    let err = parse_err(
        "dsrs 1\nprogram p\nsig S { in q: string out a: string }\nmain: S = seq {\n  x = predict S @ghost (q = $.q)\n  out { a = x.a }\n}\n",
    );
    assert_eq!(err.line, 5);
    assert!(err.message.contains("unknown model `@ghost`"), "{err}");

    let err = parse_err(
        "dsrs 1\nprogram p\nmodel m = \"x\"\nsig S { in q: string out a: string }\nmain: S = seq {\n  x = agent S (q = $.q) { tools [nope] }\n  out { a = x.a }\n}\n",
    );
    assert_eq!(err.line, 6);
    assert!(err.message.contains("unknown tool `nope`"), "{err}");
}

#[test]
fn duplicate_names_and_reserved_words_are_rejected_with_positions() {
    let err = parse_err(
        "dsrs 1\nprogram p\nmodel m = \"x\"\nsig S { in q: string out a: string }\nmain: S = seq {\n  x = predict S (q = $.q)\n  x = predict S (q = $.q)\n  out { a = x.a }\n}\n",
    );
    assert_eq!(err.line, 7);
    assert!(err.message.contains("duplicate name `x`"), "{err}");

    let err = parse_err("dsrs 1\nprogram seq\n");
    assert_eq!(err.line, 2);
    assert!(err.message.contains("reserved"), "{err}");
}

#[test]
fn unsupported_format_major_is_rejected() {
    let err = parse_err("dsrs 2\nprogram p\n");
    assert_eq!(err.line, 1);
    assert!(
        err.message.contains("unsupported format major `2`"),
        "{err}"
    );
    assert!(err.message.contains("dsrs 1"), "{err}");
}

#[test]
fn unbounded_loop_is_rejected_at_parse_time() {
    let src = r#"dsrs 1
program p
model m = "x"
sig S { in q: string out a: string }
main: S = seq {
  l = loop (max_iters 0) {
    x = predict S (q = ^q)
    carry { q = x.a }
    join { a = x.a }
  }
  out { a = l.a }
}
"#;
    let err = parse_err(src);
    assert_eq!(err.line, 6);
    assert!(
        err.message.contains("`max_iters` must be at least 1"),
        "{err}"
    );
}

#[test]
fn agent_options_reject_unknown_keys_with_alternatives() {
    let src = "dsrs 1\nprogram p\nmodel m = \"x\"\nsig S { in q: string out a: string }\nmain: S = seq {\n  x = agent S (q = $.q) { max_turnz 3 }\n  out { a = x.a }\n}\n";
    let err = parse_err(src);
    assert_eq!(err.line, 6);
    assert!(err.message.contains("max_turnz"), "{err}");
    assert!(err.message.contains("`max_turns`"), "{err}");
}
