pub mod predict;

pub use predict::*;

use crate::{Example, LM, LmUsage, Prediction};
use std::sync::Arc;

#[allow(async_fn_in_trait)]
pub trait Predictor: Send + Sync {
    async fn forward(&self, inputs: Example) -> anyhow::Result<Prediction>;
    async fn forward_with_config(&self, inputs: Example, lm: Arc<LM>)
    -> anyhow::Result<Prediction>;
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
