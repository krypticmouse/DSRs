//! IR-1 (RFC 0002 §1) dynamic rendering proof.
//!
//! Two claims under test:
//! 1. **Derive-bridged rendering** — a `SignatureDef` bridged from a derive
//!    ([`SignatureDef::of`]) renders the full prompt protocol: aliases, docs,
//!    `#[format]`/`#[render]` hints, and constraint metadata. (The historical
//!    static `SignatureSchema` render lane was collapsed when `Predict` moved
//!    onto the interpreter; byte-stability is pinned by the golden prompt
//!    tests.)
//! 2. **No `'static` anywhere** — a `SignatureDef` constructed at runtime, never
//!    mentioned in any derive, formats a system prompt + input and round-trips a
//!    canned LM response through parse, with all data owned and dropped after use.

use dspy_rs::ChatAdapter;
use dspy_rs::ir::{ConstraintDef, FieldDef, FieldType, RenderSpec, SignatureDef, TypeTable};
use dspy_rs::typesys::{ClassDef, EnumDef, EnumValueDef, FieldDef as TypeFieldDef};
use dspy_rs::{Schema, Message, ParseError, Signature};

#[derive(Signature, Clone, Debug)]
/// Grade an answer against a question.
struct Graded {
    /// The question being graded.
    #[input]
    question: String,

    #[input]
    #[format("json")]
    evidence: Vec<String>,

    #[output]
    #[alias("score")]
    #[check("this >= 0.0 && this <= 1.0", label = "range")]
    confidence: f64,

    #[output]
    #[assert("this|length > 0")]
    verdict: String,
}

#[derive(Clone, Debug)]
#[Schema]
struct Citation {
    url: String,
    title: String,
}

#[derive(Clone, Debug)]
#[Schema]
enum Stance {
    Support,
    Refute,
}

#[derive(Signature, Clone, Debug)]
/// Extract citations and a stance.
struct Structured {
    #[input]
    document: String,
    #[output]
    citations: Vec<Citation>,
    #[output]
    stance: Stance,
}

#[derive(Signature, Clone, Debug)]
/// Ask with a template.
struct JinjaSig {
    #[input]
    #[render(jinja = "Q: {{ this }} [{{ input.question }}]")]
    question: String,
    #[output]
    answer: String,
}

fn graded_response() -> Message {
    Message::assistant(
        "[[ ## score ## ]]\n0.9\n\n[[ ## verdict ## ]]\nSupported\n\n[[ ## completed ## ]]\n",
    )
}

#[test]
fn derive_bridged_system_prompt_renders_aliases_docs_and_types() {
    let adapter = ChatAdapter;

    let system = adapter.build_system_def(
        SignatureDef::of::<Graded>(),
        SignatureDef::types_of::<Graded>(),
        None,
    );
    assert!(system.contains("`question` (string): The question being graded."));
    assert!(system.contains("[[ ## score ## ]]"));
    assert!(!system.contains("[[ ## confidence ## ]]"));
    assert!(system.contains("Output field `score` should be of type: float"));
    assert!(system.contains("Grade an answer against a question."));

    let system = adapter.build_system_def(
        SignatureDef::of::<Structured>(),
        SignatureDef::types_of::<Structured>(),
        None,
    );
    assert!(system.contains("[[ ## citations ## ]]"));
    assert!(system.contains("url: string,"));
    assert!(system.contains("- Support"));

    let with_override = "Grade strictly.";
    let system = adapter.build_system_def(
        SignatureDef::of::<Graded>(),
        SignatureDef::types_of::<Graded>(),
        Some(with_override),
    );
    assert!(system.contains("Grade strictly."));
    assert!(!system.contains("Grade an answer against a question."));
}

#[test]
fn derive_bridged_input_honors_format_hints() {
    let adapter = ChatAdapter;

    let typed = GradedInput::new(
        "Is the sky blue?".to_string(),
        vec!["observation log".to_string()],
    );
    let value_input = serde_json::to_value(&typed)
        .unwrap()
        .as_object()
        .cloned()
        .unwrap();
    let value_lane = adapter.format_input_def(SignatureDef::of::<Graded>(), &value_input);
    assert!(value_lane.contains("[[ ## question ## ]]\nIs the sky blue?"));
    // `#[format("json")]` renders the list as JSON.
    assert!(value_lane.contains(r#"["observation log"]"#));
    assert!(value_lane.contains("starting with the field `[[ ## score ## ]]`"));
}

#[test]
fn derive_bridged_jinja_input_renders_template() {
    let adapter = ChatAdapter;

    let typed = JinjaSigInput::new("What is 2+2?".to_string());
    let value_input = serde_json::to_value(&typed)
        .unwrap()
        .as_object()
        .cloned()
        .unwrap();
    let value_lane = adapter.format_input_def(SignatureDef::of::<JinjaSig>(), &value_input);
    assert!(value_lane.contains("Q: What is 2+2? [What is 2+2?]"));
}

#[test]
fn derive_bridged_parse_assembles_typed_output_and_checks() {
    let adapter = ChatAdapter;
    let response = graded_response();

    let (value_map, value_meta) = adapter
        .parse_output_def(
            SignatureDef::of::<Graded>(),
            SignatureDef::types_of::<Graded>(),
            &response,
        )
        .unwrap();

    // Canonical field-name keying assembles straight into the typed output.
    let typed: GradedOutput =
        serde_json::from_value(serde_json::Value::Object(value_map)).unwrap();
    assert_eq!(typed.confidence, 0.9);
    assert_eq!(typed.verdict, "Supported");

    // Per-field metadata: raw text and `#[check]` outcomes.
    let confidence = value_meta.get("confidence").unwrap();
    assert_eq!(confidence.raw_text, "0.9");
    assert_eq!(confidence.checks.len(), 1);
    assert_eq!(confidence.checks[0].label, "range");
    assert!(confidence.checks[0].passed);
}

/// A signature that exists nowhere as a type: built at runtime from owned
/// strings, with a hand-built class/enum registry.
fn runtime_only_signature() -> (SignatureDef, TypeTable) {
    let enum_token = String::from("Severity");
    let class_token = String::from("Routing");

    let mut types = TypeTable::default();
    types.enums.insert(
        enum_token.clone(),
        EnumDef {
            internal_name: enum_token.clone(),
            rendered_name: enum_token.clone(),
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
    types.classes.insert(
        class_token.clone(),
        ClassDef {
            internal_name: class_token.clone(),
            rendered_name: class_token.clone(),
            docs: None,
            fields: vec![
                TypeFieldDef {
                    name: "team".to_string(),
                    rendered_name: "team".to_string(),
                    field_type: FieldType::String,
                    docs: None,
                    constraints: Vec::new(),
                },
                TypeFieldDef {
                    name: "priority".to_string(),
                    rendered_name: "priority".to_string(),
                    field_type: FieldType::Int,
                    docs: None,
                    constraints: Vec::new(),
                },
            ],
            constraints: Vec::new(),
        },
    );

    let def = SignatureDef::build("TriageTicket")
        .instruction("Triage the support ticket.")
        .input("ticket", FieldType::String)
        .input_full(
            FieldDef::new(
                "context_notes",
                FieldType::List(Box::new(FieldType::String)),
            )
            .with_render(RenderSpec::Format("json".into())),
        )
        .output("severity", FieldType::Enum(enum_token))
        .output("routing", FieldType::Class(class_token))
        .output_full(
            FieldDef::new("summary", FieldType::String)
                .aliased("tl_dr")
                .with_constraint(ConstraintDef::assert("this|length > 0")),
        )
        .finish()
        .unwrap();

    (def, types)
}

#[test]
fn runtime_only_signature_round_trips() {
    let adapter = ChatAdapter;
    let (def, types) = runtime_only_signature();

    // System prompt straight from the owned value.
    let system = adapter.build_system_def(&def, &types, None);
    assert!(system.contains("Triage the support ticket."));
    assert!(system.contains("[[ ## severity ## ]]"));
    assert!(system.contains("one of:"));
    assert!(system.contains("- High"));
    assert!(system.contains("team: string,"));
    assert!(system.contains("priority: int,"));
    assert!(system.contains("[[ ## tl_dr ## ]]"));

    // Input formatting from a plain JsonMap.
    let mut input = serde_json::Map::new();
    input.insert("ticket".into(), "App crashes on login".into());
    input.insert("context_notes".into(), serde_json::json!(["seen twice"]));
    let user = adapter.format_input_def(&def, &input);
    assert!(user.contains("[[ ## ticket ## ]]\nApp crashes on login"));
    assert!(user.contains("[\"seen twice\"]"));
    assert!(user.contains("starting with the field `[[ ## severity ## ]]`"));

    // Round-trip a canned LM response through parse.
    let response = Message::assistant(
        "[[ ## severity ## ]]\nHigh\n\n\
         [[ ## routing ## ]]\n{\"team\": \"auth\", \"priority\": 1}\n\n\
         [[ ## tl_dr ## ]]\nLogin crash, likely auth service.\n\n\
         [[ ## completed ## ]]\n",
    );
    let (output, metas) = adapter.parse_output_def(&def, &types, &response).unwrap();

    assert_eq!(output["severity"], "High");
    assert_eq!(output["routing"]["team"], "auth");
    assert_eq!(output["routing"]["priority"], 1);
    assert_eq!(output["summary"], "Login crash, likely auth service.");
    assert_eq!(
        metas.get("summary").unwrap().raw_text,
        "Login crash, likely auth service."
    );
}

#[test]
fn runtime_only_signature_enforces_asserts() {
    let adapter = ChatAdapter;
    let (def, types) = runtime_only_signature();

    let response = Message::assistant(
        "[[ ## severity ## ]]\nLow\n\n\
         [[ ## routing ## ]]\n{\"team\": \"auth\", \"priority\": 2}\n\n\
         [[ ## tl_dr ## ]]\n\n\n\
         [[ ## completed ## ]]\n",
    );
    let err = adapter
        .parse_output_def(&def, &types, &response)
        .unwrap_err();
    match err {
        ParseError::Multiple { errors, partial } => {
            assert!(matches!(
                errors.as_slice(),
                [ParseError::AssertFailed { field, .. }] if field == "summary"
            ));
            // The fields that did parse survive as the partial value.
            assert_eq!(partial.unwrap()["severity"], "Low");
        }
        other => panic!("expected Multiple, got {other:?}"),
    }
}
