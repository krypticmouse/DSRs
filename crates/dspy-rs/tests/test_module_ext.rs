use dspy_rs::{
    CallMetadata, CallOutcome, CallOutcomeErrorKind, Module, ModuleExt, ParseError,
};

struct MaybeFails;

impl Module for MaybeFails {
    type Input = i32;
    type Output = i32;

    async fn forward(&self, input: Self::Input) -> CallOutcome<Self::Output> {
        let metadata = CallMetadata::new(
            format!("raw:{input}"),
            dspy_rs::LmUsage::default(),
            Vec::new(),
            Vec::new(),
            Some(input as usize),
            indexmap::IndexMap::new(),
        );

        if input < 0 {
            CallOutcome::err(
                CallOutcomeErrorKind::Parse(ParseError::MissingField {
                    field: "value".to_string(),
                    raw_response: format!("raw:{input}"),
                }),
                metadata,
            )
        } else {
            CallOutcome::ok(input * 2, metadata)
        }
    }
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn map_transforms_success_and_preserves_metadata() {
    let mapped = MaybeFails.map(|value| format!("v={value}"));

    let success = mapped.forward(3).await;
    assert_eq!(success.metadata().raw_response, "raw:3");
    assert_eq!(success.into_result().expect("success expected"), "v=6");

    let failure = mapped.forward(-7).await;
    let err = failure.into_result().expect_err("failure expected");
    assert_eq!(err.metadata.raw_response, "raw:-7");
    match err.kind {
        CallOutcomeErrorKind::Parse(ParseError::MissingField { field, .. }) => {
            assert_eq!(field, "value")
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn and_then_applies_fallible_transform_and_keeps_metadata() {
    let module = MaybeFails.and_then(|value| {
        if value >= 4 {
            Ok(value.to_string())
        } else {
            Err(CallOutcomeErrorKind::Parse(ParseError::MissingField {
                field: "transformed".to_string(),
                raw_response: "transform".to_string(),
            }))
        }
    });

    let success = module.forward(3).await;
    assert_eq!(success.metadata().raw_response, "raw:3");
    assert_eq!(success.into_result().expect("success expected"), "6");

    let transformed_error = module.forward(1).await;
    let err = transformed_error
        .into_result()
        .expect_err("transform error expected");
    assert_eq!(err.metadata.raw_response, "raw:1");
    match err.kind {
        CallOutcomeErrorKind::Parse(ParseError::MissingField { field, .. }) => {
            assert_eq!(field, "transformed")
        }
        other => panic!("unexpected error: {other:?}"),
    }
}
