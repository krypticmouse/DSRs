use dspy_rs::{
    BamlType, CallMetadata, CallOutcome, CallOutcomeErrorKind, Module, ModuleExt, ParseError,
};

struct MaybeFails;

#[derive(Clone, Debug, PartialEq, Eq)]
#[BamlType]
struct IntPayload {
    value: i32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[BamlType]
struct TextPayload {
    value: String,
}

impl Module for MaybeFails {
    type Input = IntPayload;
    type Output = IntPayload;

    async fn forward(&self, input: Self::Input) -> CallOutcome<Self::Output> {
        let input_value = input.value;
        let metadata = CallMetadata::new(
            format!("raw:{input_value}"),
            dspy_rs::LmUsage::default(),
            Vec::new(),
            Vec::new(),
            Some(input_value.max(0) as usize),
            indexmap::IndexMap::new(),
        );

        if input_value < 0 {
            CallOutcome::err(
                CallOutcomeErrorKind::Parse(ParseError::MissingField {
                    field: "value".to_string(),
                    raw_response: format!("raw:{input_value}"),
                }),
                metadata,
            )
        } else {
            CallOutcome::ok(
                IntPayload {
                    value: input_value * 2,
                },
                metadata,
            )
        }
    }
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn map_transforms_success_and_preserves_metadata() {
    let mapped = MaybeFails.map(|value| TextPayload {
        value: format!("v={}", value.value),
    });

    let success = mapped.forward(IntPayload { value: 3 }).await;
    assert_eq!(success.metadata().raw_response, "raw:3");
    assert_eq!(
        success.into_result().expect("success expected"),
        TextPayload {
            value: "v=6".to_string()
        }
    );

    let failure = mapped.forward(IntPayload { value: -7 }).await;
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
        if value.value >= 4 {
            Ok(TextPayload {
                value: value.value.to_string(),
            })
        } else {
            Err(CallOutcomeErrorKind::Parse(ParseError::MissingField {
                field: "transformed".to_string(),
                raw_response: "transform".to_string(),
            }))
        }
    });

    let success = module.forward(IntPayload { value: 3 }).await;
    assert_eq!(success.metadata().raw_response, "raw:3");
    assert_eq!(
        success.into_result().expect("success expected"),
        TextPayload {
            value: "6".to_string()
        }
    );

    let transformed_error = module.forward(IntPayload { value: 1 }).await;
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
