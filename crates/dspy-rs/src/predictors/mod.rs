pub mod predict;

pub use predict::*;

use crate::{Example, LM, LmUsage, Prediction};
use anyhow::Result;
use futures::future::join_all;
use std::sync::Arc;

#[allow(async_fn_in_trait)]
pub trait Predictor: Send + Sync {
    async fn forward(&self, inputs: Example) -> anyhow::Result<Prediction>;
    async fn forward_with_config(&self, inputs: Example, lm: Arc<LM>)
    -> anyhow::Result<Prediction>;

    async fn batch(&self, inputs: Vec<Example>) -> Result<Vec<Prediction>> {
        let futures: Vec<_> = inputs
            .iter()
            .map(|input| self.forward(input.clone()))
            .collect();
        let predictions = join_all(futures)
            .await
            .into_iter()
            .collect::<Result<Vec<Prediction>>>()?;
        Ok(predictions)
    }

    async fn batch_with_config(
        &self,
        inputs: Vec<Example>,
        lm: Arc<LM>,
    ) -> Result<Vec<Prediction>> {
        let futures: Vec<_> = inputs
            .iter()
            .map(|input| self.forward_with_config(input.clone(), lm.clone()))
            .collect();
        let predictions = join_all(futures)
            .await
            .into_iter()
            .collect::<Result<Vec<Prediction>>>()?;
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
