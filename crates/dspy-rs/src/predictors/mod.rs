pub mod predict;

pub use predict::*;

use crate::{Example, LM, LmUsage, Prediction};
use anyhow::Result;
use futures::stream::{self, StreamExt};
use std::sync::Arc;
use tracing::debug;

#[allow(async_fn_in_trait)]
pub trait Predictor: Send + Sync {
    async fn forward(&self, inputs: Example) -> anyhow::Result<Prediction>;
    async fn forward_with_config(&self, inputs: Example, lm: Arc<LM>)
    -> anyhow::Result<Prediction>;

    #[tracing::instrument(
        name = "dsrs.predictor.batch",
        level = "debug",
        skip(self, inputs),
        fields(total_inputs = inputs.len(), max_concurrency = 32)
    )]
    async fn batch(&self, inputs: Vec<Example>) -> Result<Vec<Prediction>> {
        let indexed_results: Vec<(usize, Result<Prediction>)> =
            stream::iter(inputs.into_iter().enumerate())
                .map(|(idx, input)| async move {
                    let result = self.forward(input).await;
                    (idx, result)
                })
                .buffer_unordered(32) // Match MAX_CONCURRENCY from Evaluator
                .collect()
                .await;

        // Sort results back to original order
        let mut indexed_results = indexed_results;
        indexed_results.sort_by_key(|(idx, _)| *idx);

        // Collect predictions and handle errors
        let mut predictions = Vec::with_capacity(indexed_results.len());
        for (idx, result) in indexed_results {
            match result {
                Ok(prediction) => predictions.push(prediction),
                Err(err) => {
                    debug!(idx, error = %err, "predictor batch item failed");
                    return Err(err);
                }
            }
        }
        debug!(predictions = predictions.len(), "predictor batch completed");
        Ok(predictions)
    }

    #[tracing::instrument(
        name = "dsrs.predictor.batch_with_config",
        level = "debug",
        skip(self, inputs, lm),
        fields(total_inputs = inputs.len(), max_concurrency = 32)
    )]
    async fn batch_with_config(
        &self,
        inputs: Vec<Example>,
        lm: Arc<LM>,
    ) -> Result<Vec<Prediction>> {
        let lm_ref = lm.clone();
        let indexed_results: Vec<(usize, Result<Prediction>)> =
            stream::iter(inputs.into_iter().enumerate())
                .map(|(idx, input)| {
                    let lm_clone = lm_ref.clone();
                    async move {
                        let result = self.forward_with_config(input, lm_clone).await;
                        (idx, result)
                    }
                })
                .buffer_unordered(32) // Match MAX_CONCURRENCY from Evaluator
                .collect()
                .await;

        // Sort results back to original order
        let mut indexed_results = indexed_results;
        indexed_results.sort_by_key(|(idx, _)| *idx);

        // Collect predictions and handle errors
        let mut predictions = Vec::with_capacity(indexed_results.len());
        for (idx, result) in indexed_results {
            match result {
                Ok(prediction) => predictions.push(prediction),
                Err(err) => {
                    debug!(idx, error = %err, "predictor batch_with_config item failed");
                    return Err(err);
                }
            }
        }
        debug!(
            predictions = predictions.len(),
            "predictor batch_with_config completed"
        );
        Ok(predictions)
    }
}

pub struct DummyPredict;

impl Predictor for DummyPredict {
    async fn forward(&self, inputs: Example) -> anyhow::Result<Prediction> {
        Ok(Prediction::new(inputs.data, LmUsage::default()))
    }

    #[allow(unused_variables)]
    async fn forward_with_config(
        &self,
        inputs: Example,
        lm: Arc<LM>,
    ) -> anyhow::Result<Prediction> {
        Ok(Prediction::new(inputs.data, LmUsage::default()))
    }
}
