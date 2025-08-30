use crate::data::{example::Example, prediction::Prediction};
use anyhow::Result;

#[allow(async_fn_in_trait)]
pub trait Module {
    async fn forward(&self, inputs: Example) -> Result<Prediction>;
}
