pub mod predict;

pub use predict::*;

use crate::{Module, Example, Prediction, LmUsage};

pub struct DummyPredict;

impl Module for DummyPredict {
    async fn forward(
        &self,
        inputs: Example
    ) -> anyhow::Result<Prediction> {
        Ok(Prediction::new(inputs.data, LmUsage::default()))
    }
}