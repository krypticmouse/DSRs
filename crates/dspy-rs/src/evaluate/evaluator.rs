use crate::core::Module;
use crate::data::{example::Example, prediction::Prediction};
use futures::future::join_all;

#[allow(async_fn_in_trait)]
pub trait Evaluator: Module {
    const MAX_CONCURRENCY: usize = 32;
    const DISPLAY_PROGRESS: bool = true;

    async fn metric(&self, example: &Example, prediction: &Prediction) -> f32;

    async fn evaluate(&self, examples: Vec<Example>) -> f32 {
        let predictions = self
            .batch(
                examples.clone(),
                Self::MAX_CONCURRENCY,
                Self::DISPLAY_PROGRESS,
            )
            .await
            .unwrap();

        let futures: Vec<_> = examples
            .iter()
            .zip(predictions.iter())
            .map(|(example, prediction)| {
                let prediction = prediction.clone();
                async move { self.metric(example, &prediction).await }
            })
            .collect();

        let metrics = join_all(futures).await;
        metrics.iter().sum::<f32>() / examples.len() as f32
    }
}
