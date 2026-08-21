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
//! names appear as trace-span components (`Trace::for_component(name)`), and
//! [`Params`] converts losslessly to/from [`ModuleState`], so persistence works
//! across both styles. Like [`capture()`](crate::trace::capture), the ambient
//! params scope is a tokio task-local: spawned subtasks do not inherit it.

use std::any::{Any, TypeId};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::hash::{DefaultHasher, Hasher};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, RwLock};

use tokio::task_local;

use crate::core::{ModuleState, PredictState, PredictorInfo};
use crate::{Facet, Module, Predict, PredictError, Predicted, Schema, Signature};

/// Runs a future with an [`ir::Overlay`](crate::ir::Overlay) as the ambient
/// candidate — the overlay is unbound against the program into [`Params`] and
/// scoped exactly like [`with_params`]. See
/// [`ir::bridge`](crate::ir::bridge).
pub use crate::ir::bridge::with_overlay;

task_local! {
    static CURRENT_PARAMS: Arc<Params>;
}

/// One named parameter slot inside [`Params`]: a [`PredictState`] plus the
/// explicit-clear markers that make "reset to the signature default"
/// expressible (plain `PredictState` semantics are "None/empty = leave the
/// incumbent alone").
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct ParamsEntry {
    pub state: PredictState,
    /// Explicitly clear the instruction override back to the signature
    /// default (wins over any instance override when injected ambiently).
    pub clear_instruction: bool,
    /// `state.demos` is an explicit *set* — an empty vec means "no demos",
    /// overriding any instance demos — rather than "non-empty means set".
    pub explicit_demos: bool,
}

/// The optimizable state of a functional harness: named [`PredictState`]s —
/// instructions and demos keyed by the names passed to [`predict`].
///
/// The functional analogue of JAX's params pytree. A harness function is pure
/// with respect to its `Params`: evaluating a different candidate means
/// injecting a different `Params` value via [`with_params`], never mutating a
/// module in place.
///
/// Struct-held [`Predict`](crate::Predict) leaves consult the ambient
/// `Params` too: each leaf binds the entry matching its component name (the
/// name stamped by [`Predictors`](crate::Predictors) discovery or
/// [`PredictBuilder::named`](crate::predictors::PredictBuilder::named)) at
/// call time, with ambient values winning over instance state per slot. This
/// is the optimizer's candidate-injection currency.
#[derive(Clone, Debug, Default)]
pub struct Params {
    /// (config hash, entry) per predictor name. The hash keys the predictor
    /// instance cache so unchanged configs reuse fully-warmed `Predict`s.
    entries: BTreeMap<String, (u64, ParamsEntry)>,
}

impl Params {
    pub fn new() -> Self {
        Self::default()
    }

    fn upsert(&mut self, name: String, mutate: impl FnOnce(&mut ParamsEntry)) {
        let mut entry = self
            .entries
            .remove(&name)
            .map(|(_, entry)| entry)
            .unwrap_or_default();
        mutate(&mut entry);
        let hash = hash_entry(&entry);
        self.entries.insert(name, (hash, entry));
    }

    /// Sets the full state (instruction + demos) for a named predictor.
    ///
    /// `PredictState` semantics: `instruction_override: None` and empty
    /// `demos` mean "leave the incumbent alone". For explicit resets use
    /// [`clear_instruction`](Params::clear_instruction) /
    /// [`set_demos`](Params::set_demos).
    pub fn set(&mut self, name: impl Into<String>, state: PredictState) {
        self.upsert(name.into(), |entry| {
            *entry = ParamsEntry {
                state,
                clear_instruction: false,
                explicit_demos: false,
            };
        });
    }

    /// Convenience: overrides just the instruction for a named predictor,
    /// preserving any demos already set.
    pub fn set_instruction(&mut self, name: impl Into<String>, instruction: impl Into<String>) {
        let instruction = instruction.into();
        self.upsert(name.into(), |entry| {
            entry.state.instruction_override = Some(instruction);
            entry.clear_instruction = false;
        });
    }

    /// Explicitly clears the instruction override back to the signature
    /// default, preserving any demos already set. Unlike leaving the
    /// instruction unset (which lets an instance override read through), this
    /// wins over instance state when injected ambiently.
    pub fn clear_instruction(&mut self, name: impl Into<String>) {
        self.upsert(name.into(), |entry| {
            entry.state.instruction_override = None;
            entry.clear_instruction = true;
        });
    }

    /// Explicitly sets the demo set for a named predictor, preserving any
    /// instruction already set. An empty vec means "no demos" and wins over
    /// instance demos when injected ambiently.
    pub fn set_demos(&mut self, name: impl Into<String>, demos: Vec<crate::trace::JsonMap>) {
        self.upsert(name.into(), |entry| {
            entry.state.demos = demos;
            entry.explicit_demos = true;
        });
    }

    /// Returns the state configured for `name`, if any.
    pub fn get(&self, name: &str) -> Option<&PredictState> {
        self.entries.get(name).map(|(_, entry)| &entry.state)
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
                .map(|(name, (_, entry))| (name.clone(), entry.state.clone()))
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

    fn entry(&self, name: &str) -> Option<(u64, &ParamsEntry)> {
        self.entries.get(name).map(|(hash, entry)| (*hash, entry))
    }

    /// All named states, for the IR bridge (`Params::bind`).
    pub(crate) fn iter_states(&self) -> impl Iterator<Item = (&str, &PredictState)> {
        self.entries
            .iter()
            .map(|(name, (_, entry))| (name.as_str(), &entry.state))
    }
}

fn hash_entry(entry: &ParamsEntry) -> u64 {
    let mut hasher = DefaultHasher::new();
    if let Some(instruction) = &entry.state.instruction_override {
        hasher.write(instruction.as_bytes());
    }
    for demo in &entry.state.demos {
        let serialized = serde_json::to_string(demo).unwrap_or_default();
        hasher.write(serialized.as_bytes());
    }
    hasher.write_usize(entry.state.demos.len());
    hasher.write_u8(entry.clear_instruction as u8);
    hasher.write_u8(entry.explicit_demos as u8);
    hasher.finish()
}

/// The ambient [`ParamsEntry`] for `name`, if a [`with_params`] scope is
/// active on this task. Read by struct-held [`Predict`](crate::Predict)
/// leaves at call time (each leaf binds only its own entry).
pub(crate) fn ambient_entry(name: &str) -> Option<ParamsEntry> {
    CURRENT_PARAMS
        .try_with(|params| params.entry(name).map(|(_, entry)| entry.clone()))
        .ok()
        .flatten()
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

/// [`with_params`] without re-wrapping: scopes an already-shared `Params`.
/// Used by the optimizer engine, which evaluates many rollouts under one
/// candidate concurrently.
pub(crate) async fn with_params_shared<Fut: Future>(params: Arc<Params>, fut: Fut) -> Fut::Output {
    CURRENT_PARAMS.scope(params, fut).await
}

type PredictorCacheKey = (TypeId, String, u64);

/// One cached predictor plus its CLOCK reference bit. The bit is set on every
/// hit (atomically, so the shared read lock suffices) and buys the entry a
/// second chance when the eviction hand sweeps past it.
struct CacheSlot {
    predictor: Arc<dyn Any + Send + Sync>,
    referenced: AtomicBool,
}

/// Bounded predictor cache with second-chance (CLOCK) eviction.
///
/// The previous design cleared the whole map at capacity, which flushed every
/// warm predictor mid-optimizer-run. CLOCK evicts one *cold* entry per insert
/// instead: recently-hit entries keep circulating, so an optimizer sweeping
/// many candidates retains its working set.
#[derive(Default)]
struct PredictorCache {
    map: HashMap<PredictorCacheKey, CacheSlot>,
    /// The clock ring: keys in sweep order. The hand is the front; entries
    /// granted a second chance rotate to the back.
    ring: VecDeque<PredictorCacheKey>,
}

impl PredictorCache {
    fn get(&self, key: &PredictorCacheKey) -> Option<Arc<dyn Any + Send + Sync>> {
        self.map.get(key).map(|slot| {
            slot.referenced.store(true, Ordering::Relaxed);
            slot.predictor.clone()
        })
    }

    /// Inserts `predictor` under `key`, returning the cached instance (the
    /// incumbent, if a concurrent writer got there first). Evicts at most one
    /// cold entry when at capacity.
    fn insert(
        &mut self,
        key: PredictorCacheKey,
        predictor: Arc<dyn Any + Send + Sync>,
    ) -> Arc<dyn Any + Send + Sync> {
        if let Some(slot) = self.map.get(&key) {
            slot.referenced.store(true, Ordering::Relaxed);
            return slot.predictor.clone();
        }
        if self.map.len() >= PREDICTOR_CACHE_CAP {
            self.evict_one();
        }
        self.ring.push_back(key.clone());
        self.map.insert(
            key,
            CacheSlot {
                predictor: predictor.clone(),
                referenced: AtomicBool::new(false),
            },
        );
        predictor
    }

    /// Advances the clock hand until it finds an entry whose reference bit is
    /// clear, and evicts it. Bits are cleared as the hand passes, so this
    /// terminates within one lap plus one step even if every entry was hot.
    fn evict_one(&mut self) {
        while let Some(key) = self.ring.pop_front() {
            let Some(slot) = self.map.get(&key) else {
                // Stale ring key with no map entry: drop it and keep sweeping.
                continue;
            };
            if slot.referenced.swap(false, Ordering::Relaxed) {
                self.ring.push_back(key);
            } else {
                self.map.remove(&key);
                return;
            }
        }
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.map.len()
    }

    #[cfg(test)]
    fn contains(&self, key: &PredictorCacheKey) -> bool {
        self.map.contains_key(key)
    }
}

/// Predictor-instance cache: (signature type, name, config hash) → `Arc<Predict<S>>`.
///
/// This is what keeps the functional path on the same performance envelope as
/// structs — a cache hit reuses a fully-warmed `Predict` (prompt prefix, toolset)
/// instead of rebuilding per call.
static PREDICTOR_CACHE: LazyLock<RwLock<PredictorCache>> =
    LazyLock::new(|| RwLock::new(PredictorCache::default()));

/// Capacity bound; reached, the CLOCK sweep evicts one cold entry per insert.
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
                .map(|(hash, entry)| (hash, Some(entry.state.clone())))
        })
        .ok()
        .flatten()
        .unwrap_or((0, None));

    let key = (TypeId::of::<S>(), name.to_string(), config_hash);
    {
        let cache = PREDICTOR_CACHE.read().expect("fx predictor cache poisoned");
        if let Some(cached) = cache.get(&key) {
            return Ok(cached
                .downcast::<Predict<S>>()
                .expect("fx predictor cache entry has matching TypeId"));
        }
    }

    let mut predictor = Predict::<S>::builder().named(name).build();
    if let Some(state) = state {
        PredictorInfo::load_state(&mut predictor, state).map_err(|err| PredictError::Params {
            name: name.to_string(),
            source: err.into(),
        })?;
    }
    let predictor = Arc::new(predictor);

    let mut cache = PREDICTOR_CACHE.write().expect("fx predictor cache poisoned");
    let entry = cache.insert(key, predictor as Arc<dyn Any + Send + Sync>);
    Ok(entry
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
/// [`capture()`](crate::trace::capture) scope the span records `name` as its
/// component, so traces from functional harnesses are addressable by the
/// same names the optimizer would mutate.
pub async fn predict<S>(name: &str, input: S::Input) -> Result<Predicted<S::Output>, PredictError>
where
    S: Signature,
    S::Input: Schema,
    S::Output: Schema,
{
    let predictor = resolve_predictor::<S>(name)?;
    predictor.call(input).await
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

#[cfg(test)]
mod tests {
    use super::*;

    fn key(n: u64) -> PredictorCacheKey {
        (TypeId::of::<()>(), format!("predictor-{n}"), n)
    }

    fn slot() -> Arc<dyn Any + Send + Sync> {
        Arc::new(()) as Arc<dyn Any + Send + Sync>
    }

    #[test]
    fn cache_stays_bounded_at_cap() {
        let mut cache = PredictorCache::default();
        for n in 0..(PREDICTOR_CACHE_CAP as u64 + 200) {
            cache.insert(key(n), slot());
        }
        assert_eq!(cache.len(), PREDICTOR_CACHE_CAP);
    }

    #[test]
    fn eviction_spares_the_warm_working_set() {
        let mut cache = PredictorCache::default();
        for n in 0..PREDICTOR_CACHE_CAP as u64 {
            cache.insert(key(n), slot());
        }
        // A warm working set: the first 16 entries keep getting hit.
        let working_set: Vec<_> = (0..16).map(key).collect();
        for k in &working_set {
            assert!(cache.get(k).is_some());
        }
        // Sweep in twice the capacity of fresh candidates; each insert evicts
        // one cold entry. The warm set must survive the first sweep wave, and
        // as long as it keeps getting hit between waves, every wave.
        for n in 0..PREDICTOR_CACHE_CAP as u64 {
            cache.insert(key(1_000_000 + n), slot());
            if n % 64 == 0 {
                for k in &working_set {
                    assert!(cache.get(k).is_some(), "warm entry evicted mid-sweep");
                }
            }
        }
        for k in &working_set {
            assert!(cache.contains(k), "warm entry evicted by candidate sweep");
        }
        assert_eq!(cache.len(), PREDICTOR_CACHE_CAP);
    }

    #[test]
    fn reinserting_existing_key_returns_incumbent() {
        let mut cache = PredictorCache::default();
        let first = slot();
        let incumbent = cache.insert(key(1), first.clone());
        assert!(Arc::ptr_eq(&incumbent, &first));
        let second = slot();
        let returned = cache.insert(key(1), second);
        assert!(Arc::ptr_eq(&returned, &first), "incumbent must win");
        assert_eq!(cache.len(), 1);
    }
}
