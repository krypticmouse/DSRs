use dspy_rs::{BamlType, CallMetadata, Module, ModuleExt, ParseError, PredictError, Predicted};

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

    async fn forward(&self, input: Self::Input) -> Result<Predicted<Self::Output>, PredictError> {
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
            Err(PredictError::Parse {
                source: ParseError::MissingField {
                    field: "value".to_string(),
                    raw_response: format!("raw:{input_value}"),
                },
                raw_response: format!("raw:{input_value}"),
                lm_usage: dspy_rs::LmUsage::default(),
            })
        } else {
            Ok(Predicted::new(
                IntPayload {
                    value: input_value * 2,
                },
                metadata,
            ))
        }
    }
}

#[expect(
    clippy::result_large_err,
    reason = "Tests ModuleExt::and_then using the crate's public PredictError type."
)]
fn transform_int_payload(value: IntPayload) -> Result<TextPayload, PredictError> {
    if value.value >= 4 {
        Ok(TextPayload {
            value: value.value.to_string(),
        })
    } else {
        Err(PredictError::Parse {
            source: ParseError::MissingField {
                field: "transformed".to_string(),
                raw_response: "transform".to_string(),
            },
            raw_response: "transform".to_string(),
            lm_usage: dspy_rs::LmUsage::default(),
        })
    }
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn map_transforms_success_and_preserves_metadata() {
    let mapped = MaybeFails.map(|value| TextPayload {
        value: format!("v={}", value.value),
    });

    let success = mapped.call(IntPayload { value: 3 }).await.unwrap();
    assert_eq!(success.metadata().raw_response, "raw:3");
    assert_eq!(
        success.into_inner(),
        TextPayload {
            value: "v=6".to_string()
        }
    );

    let err = mapped
        .call(IntPayload { value: -7 })
        .await
        .expect_err("failure expected");
    match err {
        PredictError::Parse {
            source: ParseError::MissingField { field, .. },
            raw_response,
            ..
        } => {
            assert_eq!(field, "value");
            assert_eq!(raw_response, "raw:-7");
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn and_then_applies_fallible_transform_and_keeps_metadata() {
    let module = MaybeFails
        .and_then(transform_int_payload as fn(IntPayload) -> Result<TextPayload, PredictError>);

    let success = module.call(IntPayload { value: 3 }).await.unwrap();
    assert_eq!(success.metadata().raw_response, "raw:3");
    assert_eq!(
        success.into_inner(),
        TextPayload {
            value: "6".to_string()
        }
    );

    let err = module
        .call(IntPayload { value: 1 })
        .await
        .expect_err("transform error expected");
    match err {
        PredictError::Parse {
            source: ParseError::MissingField { field, .. },
            raw_response,
            ..
        } => {
            assert_eq!(field, "transformed");
            assert_eq!(raw_response, "transform");
        }
        other => panic!("unexpected error: {other:?}"),
    }
}
