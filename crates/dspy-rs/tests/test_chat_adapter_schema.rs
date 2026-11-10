//! Def-lane response parsing: canonical field-name keying and dotted
//! (`alias`) markers. Since Predict routes through the IR interpreter, the
//! adapter parses against [`SignatureDef`]s; output maps and metadata key by
//! canonical `FieldDef::name` (Predict translates to `rust_name` keying at
//! its boundary).

use dspy_rs::ir::{ConstraintDef, FieldDef as IrFieldDef, SignatureDef};
use dspy_rs::typesys::{FieldType, TypeTable};
use dspy_rs::{CallMetadata, ChatAdapter, Message, Predicted, Signature};
use serde_json::Value;

#[derive(Signature, Clone, Debug)]
/// Adapter schema parse fixture.
struct ExampleSig {
    #[input]
    question: String,

    #[output]
    answer: String,
}

#[derive(Signature, Clone, Debug)]
/// Alias parse fixture for non-word marker names.
struct AliasSig {
    #[input]
    question: String,

    #[output]
    #[alias("answer.value")]
    answer: String,
}

#[dspy_rs::Schema]
#[derive(Clone, Debug)]
struct City {
    /// The official city name.
    name: String,
}

#[derive(Signature, Clone, Debug)]
struct NestedSig {
    #[input]
    question: String,
    #[output]
    city: City,
}

#[test]
fn complex_native_outputs_keep_nested_schema_descriptions_in_the_prompt() {
    let prompt = ChatAdapter.build_system_json_def(
        SignatureDef::of::<NestedSig>(),
        SignatureDef::types_of::<NestedSig>(),
        None,
        true,
    );
    assert!(prompt.contains("The official city name."));
    assert!(prompt.contains("additionalProperties"));
}

#[test]
fn native_schema_prompt_keeps_semantics_without_repeating_the_api_schema() {
    let def = SignatureDef::build("Prompt")
        .instruction("Compare cities.")
        .input_full(
            IrFieldDef::new("question", FieldType::String)
                .aliased("query")
                .with_docs("The question."),
        )
        .output_full(
            IrFieldDef::new("answer", FieldType::String)
                .aliased("answer.value")
                .with_docs("City name."),
        )
        .finish()
        .unwrap();
    let types = TypeTable::default();
    let native =
        ChatAdapter.build_system_json_def(&def, &types, Some("Use current capitals."), true);
    for text in [
        "query",
        "answer.value",
        "The question.",
        "City name.",
        "Use current capitals.",
    ] {
        assert!(native.contains(text), "missing {text}");
    }
    assert!(!native.contains("Compare cities."));
    assert!(!native.contains("additionalProperties"));
    let json_mode = ChatAdapter.build_system_json_def(&def, &types, None, false);
    assert!(json_mode.contains("Compare cities."));
    assert!(json_mode.contains("additionalProperties"));
    assert!(json_mode.contains("answer.value"));
}

#[test]
fn parse_output_def_uses_canonical_field_names() {
    let adapter = ChatAdapter;
    let response = Message::assistant("[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]\n");

    let (output_map, field_meta) = adapter
        .parse_output_def(
            SignatureDef::of::<ExampleSig>(),
            SignatureDef::types_of::<ExampleSig>(),
            &response,
        )
        .expect("def parse should succeed");

    assert_eq!(output_map.get("answer"), Some(&Value::from("Paris")));
    let output: ExampleSigOutput =
        serde_json::from_value(Value::Object(output_map)).expect("typed assembly");
    assert_eq!(output.answer, "Paris");
    let answer_meta = field_meta.get("answer").expect("answer field metadata");
    assert_eq!(answer_meta.raw_text.trim(), "Paris");

    let metadata = CallMetadata::new(
        response.content(),
        dspy_rs::LmUsage::default(),
        Vec::new(),
        Vec::new(),
        None,
        field_meta,
    );
    let predicted = Predicted::new(output, metadata);

    assert_eq!(predicted.metadata().field_raw("answer"), Some("Paris"));
    assert!(!predicted.metadata().has_failed_checks());
    assert_eq!(predicted.into_inner().answer, "Paris");
}

#[test]
fn parse_output_def_accepts_dotted_field_markers() {
    let adapter = ChatAdapter;
    let response = Message::assistant("[[ ## answer.value ## ]]\nParis\n\n[[ ## completed ## ]]\n");

    let (output_map, field_meta) = adapter
        .parse_output_def(
            SignatureDef::of::<AliasSig>(),
            SignatureDef::types_of::<AliasSig>(),
            &response,
        )
        .expect("def parse should succeed for dotted aliases");

    let output: AliasSigOutput =
        serde_json::from_value(Value::Object(output_map)).expect("typed assembly");
    assert_eq!(output.answer, "Paris");
    assert_eq!(
        field_meta
            .get("answer")
            .expect("answer field metadata")
            .raw_text
            .trim(),
        "Paris"
    );
}

#[test]
fn native_json_parser_preserves_string_values_and_field_metadata() {
    let def = SignatureDef::build("Native")
        .input("question", FieldType::String)
        .output_full(
            IrFieldDef::new("answer", FieldType::Optional(Box::new(FieldType::String)))
                .aliased("answer.value"),
        )
        .output_full(
            IrFieldDef::new("accepted", FieldType::Bool)
                .with_constraint(ConstraintDef::check("truthy", "this == true")),
        )
        .output_full(
            IrFieldDef::new("positive", FieldType::Int)
                .with_constraint(ConstraintDef::assert("this > 0")),
        )
        .finish()
        .unwrap();
    let adapter = ChatAdapter;
    let response =
        Message::assistant(r#"{"answer.value":"quoted \"value\"","accepted":true,"positive":2}"#);
    let (output, meta) = adapter
        .parse_output_json_def(&def, &TypeTable::default(), &response)
        .unwrap();
    assert_eq!(output["answer"], "quoted \"value\"");
    assert_eq!(output["accepted"], true);
    assert!(meta["accepted"].checks[0].passed);
    assert_eq!(meta["answer"].flags.len(), 0);

    let bad = Message::assistant(r#"{"answer.value":null,"accepted":false,"positive":0}"#);
    let error = adapter
        .parse_output_json_def(&def, &TypeTable::default(), &bad)
        .unwrap_err();
    match error {
        dspy_rs::ParseError::Multiple { errors, .. } => assert!(errors.iter().any(|error| matches!(error, dspy_rs::ParseError::AssertFailed { field, .. } if field == "positive"))),
        other => panic!("expected collected parse errors, got {other}"),
    }
}
