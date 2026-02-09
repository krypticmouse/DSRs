use futures::stream::{self, StreamExt};
use indexmap::IndexMap;
use kdam::{BarExt, tqdm};
use tracing::debug;

use crate::{CallOutcome, core::MetaSignature};

#[allow(async_fn_in_trait)]
pub trait Module: Send + Sync {
    type Input: Send + Sync + 'static;
    type Output: Send + Sync + 'static;

    async fn forward(&self, input: Self::Input) -> CallOutcome<Self::Output>;
}

#[tracing::instrument(
    name = "dsrs.forward_all",
    level = "debug",
    skip(module, inputs),
    fields(total_inputs = inputs.len(), max_concurrency)
)]
pub async fn forward_all<M>(
    module: &M,
    inputs: Vec<M::Input>,
    max_concurrency: usize,
) -> Vec<CallOutcome<M::Output>>
where
    M: Module + ?Sized,
{
    forward_all_with_progress(module, inputs, max_concurrency, true).await
}

#[tracing::instrument(
    name = "dsrs.forward_all_with_progress",
    level = "debug",
    skip(module, inputs),
    fields(total_inputs = inputs.len(), max_concurrency, display_progress)
)]
pub async fn forward_all_with_progress<M>(
    module: &M,
    inputs: Vec<M::Input>,
    max_concurrency: usize,
    display_progress: bool,
) -> Vec<CallOutcome<M::Output>>
where
    M: Module + ?Sized,
{
    let total = inputs.len();
    let mut pb = if display_progress {
        Some(tqdm!(total = total, desc = "Processing"))
    } else {
        None
    };

    let mut indexed_results: Vec<(usize, CallOutcome<M::Output>)> =
        stream::iter(inputs.into_iter().enumerate())
            .map(|(idx, input)| async move { (idx, module.forward(input).await) })
            .buffer_unordered(max_concurrency)
            .inspect(|_| {
                if let Some(ref mut progress) = pb {
                    let _ = progress.update(1);
                }
            })
            .collect()
            .await;

    indexed_results.sort_by_key(|(idx, _)| *idx);

    let outcomes = indexed_results
        .into_iter()
        .map(|(_, outcome)| outcome)
        .collect::<Vec<_>>();
    debug!(outcomes = outcomes.len(), "forward_all completed");
    outcomes
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
