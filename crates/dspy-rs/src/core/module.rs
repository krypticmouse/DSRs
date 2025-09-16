use anyhow::Result;
use futures::future::join_all;
use indexmap::IndexMap;
use kdam::tqdm;

use crate::{Example, Prediction, core::MetaSignature};

#[allow(async_fn_in_trait)]
pub trait Module: Send + Sync {
    async fn forward(&self, inputs: Example) -> Result<Prediction>;

    async fn batch(
        &self,
        inputs: Vec<Example>,
        max_concurrency: usize,
        display_progress: bool,
    ) -> Result<Vec<Prediction>> {
        let batches = inputs.chunks(max_concurrency).collect::<Vec<_>>();
        let mut predictions = Vec::new();

        for batch in tqdm!(
            batches.iter(),
            desc = "Processing Batch",
            disable = !display_progress
        ) {
            let futures: Vec<_> = batch
                .iter()
                .map(|example| self.forward(example.clone()))
                .collect();

            predictions.extend(
                join_all(futures)
                    .await
                    .into_iter()
                    .map(|prediction| prediction.unwrap())
                    .collect::<Vec<_>>(),
            );
        }

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
