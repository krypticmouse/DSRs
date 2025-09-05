use crate::data::{example::Example, prediction::Prediction};
use anyhow::Result;
use std::collections::HashMap;

#[allow(async_fn_in_trait)]
pub trait Module {
    async fn forward(&self, inputs: Example) -> Result<Prediction>;
}

pub trait Optimizable {
    fn parameters(&mut self) -> HashMap<String, &mut dyn Optimizable>;

    fn optimize(&mut self) {
        let params = self.parameters();
        for (_name, param) in params {
            param.optimize();
        }
    }
}
