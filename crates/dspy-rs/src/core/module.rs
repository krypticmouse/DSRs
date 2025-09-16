use anyhow::Result;
use futures::future::join_all;
use indexmap::IndexMap;

use crate::{Example, Prediction, core::MetaSignature};

#[allow(async_fn_in_trait)]
pub trait Module: Send + Sync {
    async fn forward(&self, inputs: Example) -> Result<Prediction>;

    async fn batch(&self, inputs: Vec<Example>) -> Result<Vec<Prediction>> {
        let futures: Vec<_> = inputs
            .iter()
            .map(|input| self.forward(input.clone()))
            .collect::<Vec<_>>();
        
        let predictions = join_all(futures).await;
        let predictions = predictions.into_iter().collect::<Result<Vec<_>>>()?;

        Ok(predictions)
    }
}

#[allow(unused_variables)]
pub trait Optimizable {
    fn get_signature(&self) -> &dyn MetaSignature {
        todo!()
    }

    fn parameters(&mut self) -> IndexMap<String, &mut dyn Optimizable>;

    fn update_signature_instruction(&mut self, instruction: String) -> anyhow::Result<()> {
        todo!()
    }
}
