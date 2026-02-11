//! Automatic prompt optimization.
//!
//! An optimizer takes a module, a training set, and a metric, then searches for better
//! instructions (and in some cases, demos) for each [`Predict`](crate::Predict) leaf.
//! The module is mutated in-place — after optimization, calling it produces better results
//! without any code changes.
//!
//! The [`Optimizer::compile`] method takes `&mut module` (exclusive access — no concurrent
//! `call()` during optimization) and returns a report. The specific report type depends
//! on the optimizer: [`COPRO`] returns `()`, [`GEPA`] returns [`GEPAResult`] with full
//! evolution history, [`MIPROv2`] returns `()`.
//!
//! # How it works internally
//!
//! 1. The optimizer calls `named_parameters` to discover all `Predict` leaves via
//!    Facet reflection
//! 2. For each leaf, it reads the current instruction and generates candidates
//! 3. Each candidate is evaluated by setting the instruction, running the module on the
//!    trainset, and scoring with the metric
//! 4. The best instruction (per optimizer's strategy) is kept
//!
//! Users never see this machinery — they call `optimizer.compile(&mut module, trainset, &metric)`
//! and their module gets better.
//!
//! # Choosing an optimizer
//!
//! | Optimizer | Strategy | Needs feedback? | Cost |
//! |-----------|----------|-----------------|------|
//! | [`COPRO`] | Breadth-first instruction search | No | Low (breadth × depth × trainset) |
//! | [`GEPA`] | Genetic-Pareto evolution with feedback | **Yes** | Medium-high (iterations × eval) |
//! | [`MIPROv2`] | Trace-guided candidate generation | No | Medium (candidates × trials × trainset) |

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
use crate::{Facet, Module, Signature};
use crate::evaluate::{MetricOutcome, TypedMetric, evaluate_trainset};
use crate::predictors::Example;

/// Tunes a module's [`Predict`](crate::Predict) leaves for better performance.
///
/// Takes exclusive `&mut` access to the module during optimization — you cannot call
/// the module concurrently. After `compile` returns, the module's instructions and/or
/// demos have been mutated in-place. Just call the module as before; no code changes needed.
///
/// ```ignore
/// let optimizer = COPRO::builder().breadth(10).depth(3).build();
/// optimizer.compile(&mut module, trainset, &metric).await?;
/// // module is now optimized — call it as usual
/// let result = module.call(input).await?;
/// ```
///
/// # Errors
///
/// Returns an error if:
/// - No optimizable `Predict` leaves are found in the module
/// - The metric evaluation fails on any training example
/// - An LM call fails during candidate evaluation
#[allow(async_fn_in_trait)]
pub trait Optimizer {
    type Report;

    async fn compile<S, M, MT>(
        &self,
        module: &mut M,
        trainset: Vec<Example<S>>,
        metric: &MT,
    ) -> Result<Self::Report>
    where
        S: Signature,
        S::Input: Clone,
        M: Module<Input = S::Input> + for<'a> Facet<'a>,
        MT: TypedMetric<S, M>;
}

/// Evaluates a module on a trainset using a typed metric.
///
/// Thin wrapper around [`evaluate_trainset`](crate::evaluate::evaluate_trainset) for
/// internal optimizer use. Returns one [`MetricOutcome`] per training example.
pub(crate) async fn evaluate_module_with_metric<S, M, MT>(
    module: &M,
    trainset: &[Example<S>],
    metric: &MT,
) -> Result<Vec<MetricOutcome>>
where
    S: Signature,
    S::Input: Clone,
    M: Module<Input = S::Input>,
    MT: TypedMetric<S, M>,
{
    evaluate_trainset(module, trainset, metric).await
}

/// Returns the dotted-path names of all [`Predict`](crate::Predict) leaves in a module.
///
/// Convenience wrapper around [`named_parameters`](crate::core::dyn_predictor::named_parameters)
/// that discards the mutable handles and returns just the names.
pub(crate) fn predictor_names<M>(module: &mut M) -> Result<Vec<String>>
where
    M: for<'a> Facet<'a>,
{
    Ok(named_parameters(module)?
        .into_iter()
        .map(|(name, _)| name)
        .collect())
}

/// Looks up a single named predictor and applies a closure to it.
///
/// # Errors
///
/// Returns an error if the predictor name doesn't match any discovered leaf.
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
