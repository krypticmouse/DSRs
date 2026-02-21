use std::time::Duration;

use dspy_rs::{BamlType, CallMetadata, Chat, Module, PredictError, Predicted, forward_all};
use tokio::time::sleep;

struct DelayEcho;

#[derive(Clone, Debug, PartialEq, Eq)]
#[BamlType]
struct DelayInput {
    value: i64,
    delay_ms: i64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[BamlType]
struct DelayOutput {
    value: i64,
}

impl Module for DelayEcho {
    type Input = DelayInput;
    type Output = DelayOutput;

    async fn forward(&self, input: Self::Input) -> Result<Predicted<Self::Output>, PredictError> {
        sleep(Duration::from_millis(input.delay_ms.max(0) as u64)).await;
        Ok(Predicted::new(
            DelayOutput { value: input.value },
            CallMetadata::default(),
            Chat::new(vec![]),
        ))
    }
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn forward_all_preserves_input_order() {
    let module = DelayEcho;
    let inputs = vec![
        DelayInput {
            value: 0,
            delay_ms: 60,
        },
        DelayInput {
            value: 1,
            delay_ms: 10,
        },
        DelayInput {
            value: 2,
            delay_ms: 40,
        },
        DelayInput {
            value: 3,
            delay_ms: 5,
        },
    ];

    let outcomes = forward_all(&module, inputs, 2).await;
    let outputs = outcomes
        .into_iter()
        .map(|outcome| outcome.expect("forward should succeed").into_inner().value)
        .collect::<Vec<_>>();

    assert_eq!(outputs, vec![0, 1, 2, 3]);
}
