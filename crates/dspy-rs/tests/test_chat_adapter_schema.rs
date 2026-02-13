use dspy_rs::{CallMetadata, ChatAdapter, Message, Predicted, Signature};

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
fn parse_response_typed_uses_schema_field_names() {
    let adapter = ChatAdapter;
    let response = Message::assistant("[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]\n");

    let (output, field_meta) = adapter
        .parse_response_typed::<ExampleSig>(&response)
        .expect("typed parse should succeed");

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
fn parse_response_typed_accepts_dotted_field_markers() {
    let adapter = ChatAdapter;
    let response = Message::assistant("[[ ## answer.value ## ]]\nParis\n\n[[ ## completed ## ]]\n");

    let (output, field_meta) = adapter
        .parse_response_typed::<AliasSig>(&response)
        .expect("typed parse should succeed for dotted aliases");

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
