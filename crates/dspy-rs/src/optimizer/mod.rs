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
//! 3. Each candidate becomes an overlay [`Candidate`] evaluated on the shared
//!    [`EvalEngine`] — cached, budget-metered, bounded-concurrency traced
//!    rollouts through the `DynPredictor::apply_update` mutation seam
//! 4. The best candidate (per optimizer's strategy) is installed through
//!    [`apply_candidate`]
//!
//! Users never see this machinery — they call `optimizer.compile(&mut module, trainset, &metric)`
//! and their module gets better.
//!
//! # Choosing an optimizer
//!
//! | Optimizer | Strategy | Needs feedback? | Cost |
//! |-----------|----------|-----------------|------|
//! | [`BootstrapFewShot`] | One-shot demo harvesting from a teacher pass | No | Low (2 × trainset) |
//! | [`COPRO`] | Breadth-first instruction search | No | Low (breadth × depth × trainset) |
//! | [`SIMBA`] | Minibatch introspective ascent (demos + rules) | No | Low (steps × minibatch) |
//! | [`GEPA`] | Genetic-Pareto evolution with feedback | **Yes** | Medium-high (iterations × eval) |
//! | [`MIPROv2`] | Trace-guided candidate generation | No | Medium (candidates × trials × trainset) |

pub mod bootstrap;
pub mod copro;
pub mod engine;
pub mod gepa;
pub(crate) mod harvest;
pub mod mipro;
pub mod pareto;
#[cfg(feature = "ir")]
pub mod program_engine;
pub mod simba;

pub use bootstrap::*;
pub use copro::*;
pub use engine::*;
pub use gepa::*;
pub use mipro::*;
pub use pareto::*;
#[cfg(feature = "ir")]
pub use program_engine::*;
pub use simba::*;

use anyhow::Result;
use anyhow::anyhow;
use std::ops::ControlFlow;

use crate::core::{DynPredictor, visit_named_predictors_mut};
use crate::evaluate::TypedMetric;
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

/// Returns the dotted-path names of all [`Predict`](crate::Predict) leaves in a
/// module, and assigns each leaf its path as trace-span component name.
///
/// The naming pass is what joins traces back to predictors: spans record the
/// same string [`with_named_predictor`] addresses, so demo harvesting and
/// per-component reflection need no pointer-identity bookkeeping.
pub(crate) fn predictor_names<M>(module: &mut M) -> Result<Vec<String>>
where
    M: for<'a> Facet<'a>,
{
    let mut names = Vec::new();
    visit_named_predictors_mut(module, |name, predictor| {
        predictor.set_trace_name(name);
        names.push(name.to_string());
        ControlFlow::Continue(())
    })?;
    Ok(names)
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
