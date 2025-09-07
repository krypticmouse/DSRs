use crate::{Example, Prediction};
use anyhow::Result;
use std::collections::HashMap;

#[allow(async_fn_in_trait)]
pub trait Module {
    async fn forward(&self, inputs: Example) -> Result<Prediction>;
}

#[allow(unused_variables)]
pub trait Optimizable {
    fn parameters(&mut self) -> HashMap<String, &mut dyn Optimizable>;

    fn update_signature_instruction(&mut self, instruction: String) -> anyhow::Result<()> {
        todo!()
    }
}
