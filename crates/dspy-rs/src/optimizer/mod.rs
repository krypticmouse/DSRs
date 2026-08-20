//! Automatic prompt optimization.
//!
//! An optimizer takes an [`OptimizeTarget`] (a typed module + trainset +
//! metric, or an interpreter-loaded IR program + examples + metric), searches
//! for better instructions (and in some cases demos) for each optimizable
//! leaf, and returns a [`Report`]. Candidates are **data, never mutation**:
//!
//! 1. Leaves are discovered explicitly — modules declare them via
//!    [`Predictors`](crate::Predictors) (see the `predictors!` macro); the
//!    target snapshots their names, schemas, and current values as
//!    [`LeafInfo`]s and stamps each leaf's trace name once per run.
//! 2. Each candidate is a name-keyed [`Candidate`] evaluated on the shared
//!    [`Engine`] — cached, budget-metered, bounded-concurrency traced
//!    rollouts with the candidate injected *ambiently* per rollout
//!    ([`fx::with_params`](crate::fx::with_params)); different candidates
//!    evaluate concurrently because nothing is ever applied to shared state.
//! 3. The winner is installed exactly once at the end
//!    ([`OptimizeTarget::install`]) — the module lane's one mutation; the
//!    program lane's winner is an [`ir::Overlay`](crate::ir::Overlay) for
//!    [`Program::bake`](crate::ir::Program::bake).
//!
//! The convenience entry point for the common case is each optimizer's
//! `compile_module` inherent method:
//!
//! ```ignore
//! let copro = COPRO::builder().breadth(10).depth(3).build();
//! copro.compile_module(&mut module, &trainset, &metric).await?;
//! // module is now optimized — call it as usual
//! ```
//!
//! For composition (`Box<dyn Optimizer>` pipelines sharing one engine budget)
//! use the object-safe [`Optimizer`] trait directly.
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
pub mod simba;
pub mod target;

pub use bootstrap::*;
pub use copro::*;
pub use engine::*;
pub use gepa::*;
pub use mipro::*;
pub use simba::*;
pub use target::{LeafInfo, OptimizeTarget, ProgramMetric};

use anyhow::Result;
use rand::{SeedableRng, rngs::StdRng};

/// A tuning strategy over the shared [`Engine`].
///
/// Object-safe by design: optimizers compose (`Box<dyn Optimizer>` pipelines
/// can share one [`Engine`] — one budget, one rollout cache, one score
/// matrix — across stages). The target carries the thing under optimization
/// and its example set *by reference*; the engine carries the spend.
///
/// # Errors
///
/// Returns an error if:
/// - The target has no optimizable leaves
/// - The metric evaluation fails on any example
/// - An LM call fails during candidate evaluation
#[async_trait::async_trait(?Send)]
pub trait Optimizer: Send + Sync {
    /// The engine configuration this optimizer wants when the caller doesn't
    /// supply an engine explicitly (the `compile_module` convenience path).
    fn engine_config(&self) -> EngineConfig {
        EngineConfig::default()
    }

    /// Runs the strategy: proposes candidates, evaluates them through
    /// `engine`, installs the winner on `target`, and reports what happened.
    async fn compile(
        &self,
        target: &mut OptimizeTarget<'_>,
        engine: &mut Engine,
    ) -> Result<Report>;
}

/// What an optimization run produced. Strategy-specific payloads for the
/// optimizers that report more than "done".
#[derive(Clone, Debug)]
pub enum Report {
    /// Nothing beyond the installed winner (COPRO, MIPROv2).
    None,
    Gepa(GEPAResult),
    Simba(SimbaReport),
    Bootstrap(BootstrapReport),
    /// Extension point for third-party strategies.
    Custom(serde_json::Value),
}

impl Report {
    pub fn into_gepa(self) -> Option<GEPAResult> {
        match self {
            Self::Gepa(report) => Some(report),
            _ => None,
        }
    }

    pub fn into_simba(self) -> Option<SimbaReport> {
        match self {
            Self::Simba(report) => Some(report),
            _ => None,
        }
    }

    pub fn into_bootstrap(self) -> Option<BootstrapReport> {
        match self {
            Self::Bootstrap(report) => Some(report),
            _ => None,
        }
    }
}

/// The engine/RNG knobs shared by every optimizer builder: evaluation
/// concurrency, budget caps, cache salt, and the sampling seed. Each
/// optimizer assembles one from its builder fields; [`engine_config`]
/// and [`rng`] replace the per-strategy construction boilerplate.
///
/// [`engine_config`]: OptimizerCommon::engine_config
/// [`rng`]: OptimizerCommon::rng
#[derive(Clone, Copy, Debug, Default)]
pub struct OptimizerCommon {
    pub eval_concurrency: usize,
    pub max_metric_calls: Option<usize>,
    pub max_lm_calls: Option<usize>,
    pub cache_salt: u64,
    pub seed: Option<u64>,
}

impl OptimizerCommon {
    pub fn engine_config(&self) -> EngineConfig {
        EngineConfig {
            concurrency: self.eval_concurrency.max(1),
            budget: Budget {
                max_metric_calls: self.max_metric_calls,
                max_lm_calls: self.max_lm_calls,
                max_tokens: None,
            },
            cache_salt: self.cache_salt,
        }
    }

    pub fn rng(&self) -> StdRng {
        match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        }
    }
}
