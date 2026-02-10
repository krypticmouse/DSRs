pub mod copro;
pub mod gepa;
pub mod mipro;
pub mod pareto;

pub use copro::*;
pub use gepa::*;
pub use mipro::*;
pub use pareto::*;

use anyhow::Result;
use anyhow::anyhow;

use crate::core::{DynPredictor, named_parameters};
use crate::{Example, Facet, Module};
use crate::evaluate::{MetricOutcome, TypedMetric, evaluate_trainset};

#[allow(async_fn_in_trait)]
pub trait Optimizer {
    type Report;

    async fn compile<M, MT>(
        &self,
        module: &mut M,
        trainset: Vec<Example>,
        metric: &MT,
    ) -> Result<Self::Report>
    where
        M: Module + for<'a> Facet<'a>,
        MT: TypedMetric<M>;
}

pub(crate) async fn evaluate_module_with_metric<M, MT>(
    module: &M,
    trainset: &[Example],
    metric: &MT,
) -> Result<Vec<MetricOutcome>>
where
    M: Module,
    MT: TypedMetric<M>,
{
    evaluate_trainset(module, trainset, metric).await
}

pub(crate) fn predictor_names<M>(module: &mut M) -> Result<Vec<String>>
where
    M: for<'a> Facet<'a>,
{
    Ok(named_parameters(module)?
        .into_iter()
        .map(|(name, _)| name)
        .collect())
}

pub(crate) fn with_named_predictor<M, R, F>(
    module: &mut M,
    predictor_name: &str,
    f: F,
) -> Result<R>
where
    M: for<'a> Facet<'a>,
    F: FnOnce(&mut dyn DynPredictor) -> Result<R>,
{
    let mut predictors = named_parameters(module)?;
    let (_, predictor) = predictors
        .iter_mut()
        .find(|(name, _)| name == predictor_name)
        .ok_or_else(|| anyhow!("predictor `{predictor_name}` not found"))?;
    f(*predictor)
}
