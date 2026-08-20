//! Def-lane response parsing: canonical field-name keying and dotted
//! (`alias`) markers. Since Predict routes through the IR interpreter, the
//! adapter parses against [`SignatureDef`]s; output maps and metadata key by
//! canonical `FieldDef::name` (Predict translates to `rust_name` keying at
//! its boundary).

use dspy_rs::ir::SignatureDef;
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
