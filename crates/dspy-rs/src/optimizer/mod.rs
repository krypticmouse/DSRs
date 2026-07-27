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
//! 1. The optimizer calls `visit_named_predictors_mut` to discover all `Predict`
//!    leaves via Facet reflection
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
use std::collections::HashMap;
use std::ops::ControlFlow;

use crate::core::{DynPredictor, visit_named_predictors_mut};
use crate::evaluate::{MetricOutcome, TypedMetric, evaluate_examples};
use crate::predictors::Example;
use crate::{Facet, Module, Signature};

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

/// Evaluates a module on a set of examples using a typed metric.
///
/// Thin wrapper around the concurrent evaluation core for internal optimizer use.
/// Accepts borrowed examples so optimizers can pass sampled minibatches without
/// cloning rows. Returns one [`MetricOutcome`] per example, in input order.
pub(crate) async fn evaluate_module_with_metric<'a, S, M, MT, I>(
    module: &M,
    examples: I,
    metric: &MT,
    max_concurrency: usize,
) -> Result<Vec<MetricOutcome>>
where
    S: Signature,
    S::Input: Clone,
    M: Module<Input = S::Input>,
    MT: TypedMetric<S, M>,
    I: IntoIterator<Item = &'a Example<S>>,
{
    evaluate_examples(module, examples, metric, max_concurrency).await
}

/// Returns the dotted-path names of all [`Predict`](crate::Predict) leaves in a module.
///
/// Convenience wrapper around
/// [`visit_named_predictors_mut`](crate::core::dyn_predictor::visit_named_predictors_mut)
/// that collects discovered paths.
pub(crate) fn predictor_names<M>(module: &mut M) -> Result<Vec<String>>
where
    M: for<'a> Facet<'a>,
{
    let mut names = Vec::new();
    visit_named_predictors_mut(module, |name, _predictor| {
        names.push(name.to_string());
        ControlFlow::Continue(())
    })?;
    Ok(names)
}

/// Maps each [`Predict`](crate::Predict) leaf's instance address to its dotted path.
///
/// Trace nodes record the same address as
/// [`NodeType::Predict::instance_key`](crate::trace::NodeType), so this map joins
/// per-node trace data (inputs/outputs per LM call) back to named predictors for
/// demo bootstrapping and credit assignment. Addresses are only stable while the
/// module value is not moved — build the map and consume traces under the same
/// `&mut` borrow.
pub fn predictor_instance_keys<M>(module: &mut M) -> Result<HashMap<usize, String>>
where
    M: for<'a> Facet<'a>,
{
    let mut keys = HashMap::new();
    visit_named_predictors_mut(module, |name, predictor| {
        let address = std::ptr::from_mut(predictor).cast::<()>() as usize;
        keys.insert(address, name.to_string());
        ControlFlow::Continue(())
    })?;
    Ok(keys)
}

/// Looks up a single named predictor and applies a closure to it.
///
/// # Errors
///
/// Returns an error if the predictor name doesn't match any discovered leaf.
pub(crate) fn with_named_predictor<M, R, F>(module: &mut M, predictor_name: &str, f: F) -> Result<R>
where
    M: for<'a> Facet<'a>,
    F: FnOnce(&mut dyn DynPredictor) -> Result<R>,
{
    let mut apply = Some(f);
    let mut result = None;

    visit_named_predictors_mut(module, |name, predictor| {
        if name != predictor_name {
            return ControlFlow::Continue(());
        }

        let f = apply.take().expect("selector closure should only run once");
        result = Some(f(predictor));
        ControlFlow::Break(())
    })?;

    result.unwrap_or_else(|| Err(anyhow!("predictor `{predictor_name}` not found")))
}
