use dspy_rs::{
    CallMetadata, CallOutcome, CallOutcomeErrorKind, ConstraintResult, FieldMeta, LmUsage,
    ParseError,
};
use indexmap::IndexMap;

#[test]
fn error_outcome_preserves_metadata() {
    let metadata = CallMetadata::new(
        "raw response".to_string(),
        LmUsage::default(),
        Vec::new(),
        Vec::new(),
        Some(42),
        IndexMap::new(),
    );

    let outcome: CallOutcome<String> = CallOutcome::err(
        CallOutcomeErrorKind::Parse(ParseError::MissingField {
            field: "answer".to_string(),
            raw_response: "raw response".to_string(),
        }),
        metadata.clone(),
    );

    let err = outcome.into_result().expect_err("expected parse failure");
    assert_eq!(err.metadata.raw_response, metadata.raw_response);
    assert_eq!(err.metadata.node_id, Some(42));

    match err.kind {
        CallOutcomeErrorKind::Parse(ParseError::MissingField { field, .. }) => {
            assert_eq!(field, "answer")
        }
        other => panic!("unexpected error kind: {other:?}"),
    }
}

#[test]
fn success_outcome_exposes_field_metadata() {
    let mut field_meta = IndexMap::new();
    field_meta.insert(
        "answer".to_string(),
        FieldMeta {
            raw_text: "Paris".to_string(),
            flags: Vec::new(),
            checks: vec![ConstraintResult {
                label: "non_empty".to_string(),
                expression: "this.len() > 0".to_string(),
                passed: true,
            }],
        },
    );

    let metadata = CallMetadata::new(
        "raw response".to_string(),
        LmUsage::default(),
        Vec::new(),
        Vec::new(),
        None,
        field_meta,
    );

    let outcome = CallOutcome::ok("Paris".to_string(), metadata);
    assert_eq!(outcome.metadata().field_raw("answer"), Some("Paris"));
    assert!(!outcome.metadata().has_failed_checks());

    let output = outcome.into_result().expect("expected success");
    assert_eq!(output, "Paris");
}
