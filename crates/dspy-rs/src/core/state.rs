//! Persistence for optimized module state.
//!
//! After an optimizer tunes a module, the improved instructions and demos live only
//! in memory. [`ModuleState`] snapshots every [`Predict`](crate::Predict) leaf into a
//! serializable value so an optimized program can be saved to disk and reloaded
//! without re-running optimization:
//!
//! ```ignore
//! // After optimization:
//! ModuleState::from_module(&module)?.save("optimized.json")?;
//!
//! // In production:
//! let mut module = MyPipeline::new();
//! ModuleState::load("optimized.json")?.apply(&mut module)?;
//! ```
//!
//! Leaves are discovered through the explicit [`Predictors`] contract — state
//! is keyed by the names the module declares for its leaves.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::core::{PredictState, Predictors};

/// Serializable snapshot of every [`Predict`](crate::Predict) leaf in a module,
/// keyed by the leaf name the module declares via [`Predictors`].
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ModuleState {
    /// Per-predictor state, keyed by leaf name (`BTreeMap` keeps JSON output stable).
    pub predictors: BTreeMap<String, PredictState>,
}

impl ModuleState {
    /// Snapshots the current state (instruction overrides + demos) of every
    /// `Predict` leaf in `module`.
    pub fn from_module<M>(module: &M) -> Result<Self>
    where
        M: Predictors + ?Sized,
    {
        let mut predictors = BTreeMap::new();
        for (name, info) in module.predictors() {
            predictors.insert(name, info.dump_state());
        }
        Ok(Self { predictors })
    }

    /// Applies this state to a module in place, stamping each restored leaf's
    /// trace name with its declared name.
    ///
    /// Every name in the state must resolve to a `Predict` leaf in `module` —
    /// unknown names are an error, since they mean the saved state and the module
    /// structure have diverged. Predictors not named in the state are left untouched.
    pub fn apply<M>(&self, module: &mut M) -> Result<()>
    where
        M: Predictors + ?Sized,
    {
        let mut remaining: BTreeMap<&str, &PredictState> = self
            .predictors
            .iter()
            .map(|(name, state)| (name.as_str(), state))
            .collect();

        for (name, info) in module.predictors_mut() {
            if let Some(state) = remaining.remove(name.as_str()) {
                info.set_trace_name(&name);
                info.load_state(state.clone())
                    .map_err(|err| anyhow!("failed to load state for `{name}`: {err}"))?;
            }
        }

        if !remaining.is_empty() {
            let missing = remaining.keys().copied().collect::<Vec<_>>().join("`, `");
            return Err(anyhow!(
                "state refers to predictors not present in the module: `{missing}`"
            ));
        }
        Ok(())
    }

    /// Serializes to pretty-printed JSON.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).context("failed to serialize module state")
    }

    /// Deserializes from JSON produced by [`to_json`](ModuleState::to_json).
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).context("failed to deserialize module state")
    }

    /// Writes the state to a JSON file.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        std::fs::write(path, self.to_json()?)
            .with_context(|| format!("failed to write module state to `{}`", path.display()))
    }

    /// Reads state from a JSON file written by [`save`](ModuleState::save).
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let json = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read module state from `{}`", path.display()))?;
        Self::from_json(&json)
    }
}
