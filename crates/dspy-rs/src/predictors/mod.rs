pub mod predict;

pub use predict::*;

use crate::{Example, LM, LmUsage, Prediction};

#[allow(async_fn_in_trait)]
pub trait Predictor {
    async fn forward(&self, inputs: Example) -> anyhow::Result<Prediction>;
    async fn forward_with_config(&self, inputs: Example, lm: &mut LM)
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
        lm: &mut LM,
    ) -> anyhow::Result<Prediction> {
        Ok(Prediction::new(inputs.data, LmUsage::default()))
    }
}
