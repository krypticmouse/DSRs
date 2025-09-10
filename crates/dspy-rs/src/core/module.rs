use anyhow::Result;
use indexmap::IndexMap;

use crate::{Example, Prediction, core::MetaSignature};

#[allow(async_fn_in_trait)]
pub trait Module: Send + Sync {
    async fn forward(&self, inputs: Example) -> Result<Prediction>;
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
