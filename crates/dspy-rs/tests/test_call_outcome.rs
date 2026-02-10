use dspy_rs::{
    CallMetadata, ConstraintResult, FieldMeta, LmUsage, ParseError, PredictError, Predicted,
};
use indexmap::IndexMap;

#[test]
fn parse_error_preserves_raw_response_and_usage() {
    let usage = LmUsage {
        prompt_tokens: 5,
        completion_tokens: 7,
        total_tokens: 12,
    };
    let err = PredictError::Parse {
        source: ParseError::MissingField {
            field: "answer".to_string(),
            raw_response: "raw response".to_string(),
        },
        raw_response: "raw response".to_string(),
        lm_usage: usage.clone(),
    };

    match err {
        PredictError::Parse {
            source: ParseError::MissingField { field, .. },
            raw_response,
            lm_usage,
        } => {
            assert_eq!(field, "answer");
            assert_eq!(raw_response, "raw response");
            assert_eq!(lm_usage.prompt_tokens, usage.prompt_tokens);
            assert_eq!(lm_usage.completion_tokens, usage.completion_tokens);
            assert_eq!(lm_usage.total_tokens, usage.total_tokens);
        }
        other => panic!("unexpected error type: {other:?}"),
    }
}

#[test]
fn predicted_exposes_field_metadata() {
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

    let predicted = Predicted::new("Paris".to_string(), metadata);
    assert_eq!(predicted.metadata().field_raw("answer"), Some("Paris"));
    assert!(!predicted.metadata().has_failed_checks());

    let output = predicted.into_inner();
    assert_eq!(output, "Paris");
}
