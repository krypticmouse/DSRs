use std::time::Duration;

use dspy_rs::{CallMetadata, CallOutcome, Module, forward_all};
use tokio::time::sleep;

struct DelayEcho;

impl Module for DelayEcho {
    type Input = (usize, u64);
    type Output = usize;

    async fn forward(&self, input: Self::Input) -> CallOutcome<Self::Output> {
        let (value, delay_ms) = input;
        sleep(Duration::from_millis(delay_ms)).await;
        CallOutcome::ok(value, CallMetadata::default())
    }
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn forward_all_preserves_input_order() {
    let module = DelayEcho;
    let inputs = vec![(0, 60), (1, 10), (2, 40), (3, 5)];

    let outcomes = forward_all(&module, inputs, 2).await;
    let outputs = outcomes
        .into_iter()
        .map(|outcome| outcome.into_result().expect("forward should succeed"))
        .collect::<Vec<_>>();

    assert_eq!(outputs, vec![0, 1, 2, 3]);
}
