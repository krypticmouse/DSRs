use std::{collections::BTreeMap, marker::PhantomData, ops::ControlFlow};

use dsrs_core::{Facet, Module, Signature, visit_named_predictors_mut};
use leaven_core::{ArtifactIdentity, CacheIdentity};
use leaven_kernel::{ContentId, FingerprintBuilder};

use crate::{
    DsrsLeavenError, DsrsModuleFactory, DsrsPredictorEdit, DsrsPredictorSnapshot,
    DsrsProgramChange, DsrsProgramLayout, DsrsProgramState, PredictorPath,
};

/// Immutable Leaven artifact for a DSRs module's optimizer-visible predictor state.
pub struct DsrsProgramArtifact<S, M, F>
where
    S: Signature,
    M: Module,
    F: DsrsModuleFactory<M>,
{
    factory: F,
    layout: DsrsProgramLayout,
    state: DsrsProgramState,
    _phantom: PhantomData<(S, M)>,
}

impl<S, M, F> Clone for DsrsProgramArtifact<S, M, F>
where
    S: Signature,
    M: Module,
    F: DsrsModuleFactory<M>,
{
    fn clone(&self) -> Self {
        Self {
            factory: self.factory.clone(),
            layout: self.layout.clone(),
            state: self.state.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<S, M, F> DsrsProgramArtifact<S, M, F>
where
    S: Signature,
    M: Module + for<'a> Facet<'a>,
    F: DsrsModuleFactory<M>,
{
    /// Captures the mutable predictor state from a caller-owned module.
    pub fn capture(factory: F, module: &mut M) -> Result<Self, DsrsLeavenError> {
        let (layout, state) = capture_program_state(module)?;
        let artifact = Self {
            factory,
            layout,
            state,
            _phantom: PhantomData,
        };
        artifact.validate_state()?;
        Ok(artifact)
    }

    /// Builds a fresh module and installs this artifact's predictor snapshots into it.
    pub fn materialize_module(&self) -> Result<M, DsrsLeavenError> {
        let mut module = self.factory.fresh_module();
        self.install_into(&mut module)?;
        Ok(module)
    }

    /// Installs this artifact's predictor snapshots into an existing module.
    pub fn install_into(&self, module: &mut M) -> Result<(), DsrsLeavenError> {
        install_program_state(module, &self.layout, &self.state)
    }

    #[must_use]
    pub fn layout(&self) -> &DsrsProgramLayout {
        &self.layout
    }

    #[must_use]
    pub fn state(&self) -> &DsrsProgramState {
        &self.state
    }

    fn content_id(&self) -> ContentId {
        let encoded = serde_json::to_vec(&(&self.layout, &self.state))
            .expect("DSRs program artifact state should serialize");
        let mut builder = FingerprintBuilder::new();
        builder
            .update(b"dsrs-leaven-program-artifact-v1")
            .update(encoded);
        let fingerprint = builder.finish();
        ContentId::from_bytes(fingerprint.0)
    }

    fn validate_state(&self) -> Result<(), DsrsLeavenError> {
        let state_paths = self.state.predictors().keys().collect::<Vec<_>>();
        let layout_paths = self.layout.paths().iter().collect::<Vec<_>>();
        if layout_paths.is_empty() {
            return Err(DsrsLeavenError::NoPredictors);
        }
        if state_paths != layout_paths {
            return Err(DsrsLeavenError::LayoutStateMismatch);
        }
        let mut module = self.factory.fresh_module();
        self.install_into(&mut module)
    }
}

impl<S, M, F> leaven_core::Artifact for DsrsProgramArtifact<S, M, F>
where
    S: Signature,
    M: Module + for<'a> Facet<'a> + Send + Sync + 'static,
    F: DsrsModuleFactory<M>,
{
    type Change = DsrsProgramChange;
    type ApplyError = DsrsLeavenError;

    fn identity(&self) -> ArtifactIdentity {
        ArtifactIdentity::Content(self.content_id())
    }

    fn cache_identity(&self) -> Option<CacheIdentity> {
        Some(CacheIdentity::Content(self.content_id()))
    }

    fn validate(&self) -> Result<(), Self::ApplyError> {
        self.validate_state()
    }

    fn apply_change(&self, change: &Self::Change) -> Result<Self, Self::ApplyError> {
        if !self.state.predictors().contains_key(change.path()) {
            return Err(DsrsLeavenError::UnknownPredictorPath(change.path().clone()));
        }

        let mut next = self.clone();
        let snapshot = next
            .state
            .predictor_mut(change.path())
            .ok_or_else(|| DsrsLeavenError::UnknownPredictorPath(change.path().clone()))?;
        match change.edit() {
            DsrsPredictorEdit::ReplaceInstruction(instruction) => {
                *snapshot = snapshot.with_instruction(instruction.clone());
            }
        }
        next.validate()?;
        Ok(next)
    }
}

fn capture_program_state<M>(
    module: &mut M,
) -> Result<(DsrsProgramLayout, DsrsProgramState), DsrsLeavenError>
where
    M: Module + for<'a> Facet<'a>,
{
    let mut paths = Vec::new();
    let mut snapshots = BTreeMap::new();
    visit_named_predictors_mut(module, |raw_path, predictor| {
        match PredictorPath::new(raw_path.to_string()) {
            Ok(path) => {
                paths.push(path.clone());
                snapshots.insert(
                    path,
                    DsrsPredictorSnapshot::new(predictor.instruction(), predictor.dump_state()),
                );
                ControlFlow::Continue(())
            }
            Err(_) => ControlFlow::Break(()),
        }
    })?;

    if paths.is_empty() {
        return Err(DsrsLeavenError::NoPredictors);
    }

    Ok((
        DsrsProgramLayout::new(paths),
        DsrsProgramState::new(snapshots),
    ))
}

fn install_program_state<M>(
    module: &mut M,
    layout: &DsrsProgramLayout,
    state: &DsrsProgramState,
) -> Result<(), DsrsLeavenError>
where
    M: Module + for<'a> Facet<'a>,
{
    let mut seen = Vec::new();
    let mut error = None;
    visit_named_predictors_mut(module, |raw_path, predictor| {
        let path = match PredictorPath::new(raw_path.to_string()) {
            Ok(path) => path,
            Err(err) => {
                error = Some(err);
                return ControlFlow::Break(());
            }
        };
        seen.push(path.clone());
        let Some(snapshot) = state.predictors().get(&path) else {
            error = Some(DsrsLeavenError::UnknownPredictorPath(path));
            return ControlFlow::Break(());
        };
        if let Err(err) = predictor.load_state(snapshot.state.clone()) {
            error = Some(DsrsLeavenError::LoadState { path, source: err });
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    })?;

    if let Some(error) = error {
        return Err(error);
    }
    if seen != layout.paths() {
        return Err(DsrsLeavenError::LayoutStateMismatch);
    }
    Ok(())
}
