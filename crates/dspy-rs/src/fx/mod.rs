//! Functional DSRs (experimental): harnesses as plain async functions.
//!
//! The struct world (`Predict<S>` fields + `Module::forward`) remains fully
//! supported — this module is an *additional* authoring style layered on top of
//! the same machinery, in the spirit of JAX: pure functions over inputs, with
//! optimizable parameters living **outside** the function in a [`Params`] pytree
//! that is injected ambiently per call tree.
//!
//! ```ignore
//! use dspy_rs::fx;
//!
//! async fn pipeline(question: String) -> Result<Predicted<RefineOutput>, PredictError> {
//!     let draft = fx::predict::<Draft>("drafter", DraftInput { question }).await?;
//!     fx::predict::<Refine>("refiner", RefineInput { draft: draft.answer.clone() }).await
//! }
//!
//! // Eager, default params:
//! let out = pipeline("hi".into()).await?;
//!
//! // Same function, candidate params injected — nothing is `&mut`, so
//! // different candidates can evaluate concurrently:
//! let mut params = fx::Params::new();
//! params.set_instruction("drafter", "Draft a thorough answer.");
//! let out = fx::with_params(params, pipeline("hi".into())).await?;
//!
//! // Plug into the existing eval machinery:
//! let module = fx::module(|input: DraftInput| pipeline(input.question));
//! evaluate_trainset(&module, &trainset, &metric).await?;
//! ```
//!
//! Predictors are addressed by **name** instead of struct-field path; the same
//! names appear on trace nodes (`NodeType::Predict::param_name`), and [`Params`]
//! converts losslessly to/from [`ModuleState`], so persistence works across both
//! styles. Like [`trace()`](crate::trace::trace), the ambient params scope is a
//! tokio task-local: spawned subtasks do not inherit it.

use std::any::{Any, TypeId};
use std::collections::{BTreeMap, HashMap};
use std::hash::{DefaultHasher, Hasher};
use std::marker::PhantomData;
use std::sync::{Arc, LazyLock, RwLock};

use tokio::task_local;

use crate::core::{DynPredictor, ModuleState, PredictState};
use crate::{Facet, LmError, Module, Predict, PredictError, Predicted, Schema, Signature};

task_local! {
    static CURRENT_PARAMS: Arc<Params>;
}

/// The optimizable state of a functional harness: named [`PredictState`]s —
/// instructions and demos keyed by the names passed to [`predict`].
///
/// The functional analogue of JAX's params pytree. A harness function is pure
/// with respect to its `Params`: evaluating a different candidate means
/// injecting a different `Params` value via [`with_params`], never mutating a
/// module in place.
#[derive(Clone, Debug, Default)]
pub struct Params {
    /// (config hash, state) per predictor name. The hash keys the predictor
    /// instance cache so unchanged configs reuse fully-warmed `Predict`s.
    entries: BTreeMap<String, (u64, PredictState)>,
}

impl Params {
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the full state (instruction + demos) for a named predictor.
    pub fn set(&mut self, name: impl Into<String>, state: PredictState) {
        let hash = hash_state(&state);
        self.entries.insert(name.into(), (hash, state));
    }

    /// Convenience: overrides just the instruction for a named predictor,
    /// preserving any demos already set.
    pub fn set_instruction(&mut self, name: impl Into<String>, instruction: impl Into<String>) {
        let name = name.into();
        let mut state = self
            .entries
            .remove(&name)
            .map(|(_, state)| state)
            .unwrap_or_default();
        state.instruction_override = Some(instruction.into());
        self.set(name, state);
    }

    /// Returns the state configured for `name`, if any.
    pub fn get(&self, name: &str) -> Option<&PredictState> {
        self.entries.get(name).map(|(_, state)| state)
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Converts to the persistence format shared with struct-based modules —
    /// `Params` round-trips through [`ModuleState::save`]/[`ModuleState::load`].
    pub fn to_module_state(&self) -> ModuleState {
        ModuleState {
            predictors: self
                .entries
                .iter()
                .map(|(name, (_, state))| (name.clone(), state.clone()))
                .collect(),
        }
    }

    /// Builds `Params` from a saved [`ModuleState`].
    pub fn from_module_state(state: ModuleState) -> Self {
        let mut params = Self::new();
        for (name, predictor_state) in state.predictors {
            params.set(name, predictor_state);
        }
        params
    }

    fn entry(&self, name: &str) -> Option<(u64, &PredictState)> {
        self.entries.get(name).map(|(hash, state)| (*hash, state))
    }
}

fn hash_state(state: &PredictState) -> u64 {
    let mut hasher = DefaultHasher::new();
    if let Some(instruction) = &state.instruction_override {
        hasher.write(instruction.as_bytes());
    }
    for demo in &state.demos {
        let serialized = serde_json::to_string(demo).unwrap_or_default();
        hasher.write(serialized.as_bytes());
    }
    hasher.write_usize(state.demos.len());
    hasher.finish()
}

/// Runs a future with `params` as the ambient parameter set for every
/// [`predict`] call inside it.
///
/// Scoped via a tokio task-local, mirroring [`trace()`](crate::trace::trace):
/// only `predict` calls on the same task see the params; spawned subtasks do not
/// inherit them. Nesting replaces the outer scope for the inner future.
pub async fn with_params<Fut: Future>(params: Params, fut: Fut) -> Fut::Output {
    CURRENT_PARAMS.scope(Arc::new(params), fut).await
}

type PredictorCacheKey = (TypeId, String, u64);
type PredictorCache = HashMap<PredictorCacheKey, Arc<dyn Any + Send + Sync>>;

/// Predictor-instance cache: (signature type, name, config hash) → `Arc<Predict<S>>`.
///
/// This is what keeps the functional path on the same performance envelope as
/// structs — a cache hit reuses a fully-warmed `Predict` (prompt prefix, toolset)
/// instead of rebuilding per call.
static PREDICTOR_CACHE: LazyLock<RwLock<PredictorCache>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

/// Backstop against unbounded growth across many optimizer candidates.
const PREDICTOR_CACHE_CAP: usize = 1024;

#[allow(clippy::result_large_err)]
fn resolve_predictor<S>(name: &str) -> Result<Arc<Predict<S>>, PredictError>
where
    S: Signature,
    S::Input: Schema,
    S::Output: Schema,
{
    let (config_hash, state) = CURRENT_PARAMS
        .try_with(|params| {
            params
                .entry(name)
                .map(|(hash, state)| (hash, Some(state.clone())))
        })
        .ok()
        .flatten()
        .unwrap_or((0, None));

    let key = (TypeId::of::<S>(), name.to_string(), config_hash);
    {
        let cache = PREDICTOR_CACHE.read().expect("fx predictor cache poisoned");
        if let Some(cached) = cache.get(&key) {
            return Ok(cached
                .clone()
                .downcast::<Predict<S>>()
                .expect("fx predictor cache entry has matching TypeId"));
        }
    }

    let mut predictor = Predict::<S>::builder().named(name).build();
    if let Some(state) = state {
        predictor.load_state(state).map_err(|err| PredictError::Lm {
            source: LmError::Provider {
                provider: "fx".to_string(),
                message: format!("params for `{name}` don't fit signature: {err}"),
                source: None,
            },
        })?;
    }
    let predictor = Arc::new(predictor);

    let mut cache = PREDICTOR_CACHE.write().expect("fx predictor cache poisoned");
    if cache.len() >= PREDICTOR_CACHE_CAP {
        cache.clear();
    }
    let entry = cache
        .entry(key)
        .or_insert_with(|| predictor.clone() as Arc<dyn Any + Send + Sync>);
    Ok(entry
        .clone()
        .downcast::<Predict<S>>()
        .expect("fx predictor cache entry has matching TypeId"))
}

/// The atomic LM call of functional DSRs: one signature, one named parameter
/// slot, one prediction.
///
/// Configuration (instruction override + demos) comes from the ambient
/// [`Params`] injected by [`with_params`]; with no scope active, the signature's
/// defaults apply. The LM is resolved exactly like struct-based `Predict` calls
/// (global [`configure`](crate::configure)d LM). Under a
/// [`trace()`](crate::trace::trace) scope the node records `name` as its
/// `param_name`, so traces from functional harnesses are addressable by the
/// same names the optimizer would mutate.
pub async fn predict<S>(name: &str, input: S::Input) -> Result<Predicted<S::Output>, PredictError>
where
    S: Signature,
    S::Input: Schema,
    S::Output: Schema,
{
    let predictor = resolve_predictor::<S>(name)?;
    predictor.forward(input).await
}

/// Adapts a plain async function into a [`Module`], so functional harnesses
/// plug into `evaluate_trainset`, metrics, and every other module consumer.
pub struct FnModule<I, O, F> {
    f: F,
    _marker: PhantomData<fn(I) -> O>,
}

impl<I, O, F, Fut> Module for FnModule<I, O, F>
where
    I: Schema + for<'a> Facet<'a> + Send + Sync,
    O: Schema + for<'a> Facet<'a> + Send + Sync,
    F: Fn(I) -> Fut + Send + Sync,
    Fut: Future<Output = Result<Predicted<O>, PredictError>> + Send,
{
    type Input = I;
    type Output = O;

    async fn forward(&self, input: I) -> Result<Predicted<O>, PredictError> {
        (self.f)(input).await
    }
}

/// Wraps an async function as a [`Module`]. See [`FnModule`].
pub fn module<I, O, F, Fut>(f: F) -> FnModule<I, O, F>
where
    I: Schema + for<'a> Facet<'a> + Send + Sync,
    O: Schema + for<'a> Facet<'a> + Send + Sync,
    F: Fn(I) -> Fut + Send + Sync,
    Fut: Future<Output = Result<Predicted<O>, PredictError>> + Send,
{
    FnModule {
        f,
        _marker: PhantomData,
    }
}
