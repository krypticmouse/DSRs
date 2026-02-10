use crate::core::{Module, forward_all_with_progress};
use crate::data::{example::Example, prediction::Prediction};
use futures::stream::{self, StreamExt};
use tracing::{debug, warn};

#[allow(async_fn_in_trait)]
pub trait Evaluator: Module<Input = Example, Output = Prediction> {
    const MAX_CONCURRENCY: usize = 32;
    const DISPLAY_PROGRESS: bool = true;

    async fn metric(&self, example: &Example, prediction: &Prediction) -> f32;

    #[tracing::instrument(
        name = "dsrs.evaluate",
        level = "debug",
        skip(self, examples),
        fields(
            examples = examples.len(),
            max_concurrency = Self::MAX_CONCURRENCY,
            display_progress = Self::DISPLAY_PROGRESS
        )
    )]
    async fn evaluate(&self, examples: Vec<Example>) -> f32 {
        let outcomes = forward_all_with_progress(
            self,
            examples.clone(),
            Self::MAX_CONCURRENCY,
            Self::DISPLAY_PROGRESS,
        )
        .await;
        let mut predictions = Vec::with_capacity(outcomes.len());
        for (idx, outcome) in outcomes.into_iter().enumerate() {
            match outcome {
                Ok(prediction) => predictions.push(prediction.into_inner()),
                Err(err) => {
                    warn!(idx, error = %err, "evaluation failed while generating predictions");
                    panic!("evaluation failed: {err}");
                }
            }
        }

        let total = examples.len();

        // Pair examples with predictions and evaluate with controlled concurrency
        let metrics: Vec<f32> = stream::iter(examples.iter().zip(predictions.iter()).enumerate())
            .map(|(idx, (example, prediction))| {
                let prediction = prediction.clone();
                async move {
                    let score = self.metric(example, &prediction).await;
                    debug!(idx, score, "evaluation metric computed");
                    score
                }
            })
            .buffer_unordered(Self::MAX_CONCURRENCY)
            .collect()
            .await;

        let average_score = metrics.iter().sum::<f32>() / total as f32;
        debug!(average_score, "evaluation complete");
        average_score
    }
}
