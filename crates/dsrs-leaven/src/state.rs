use std::collections::BTreeMap;

use dsrs_core::PredictState;

use crate::PredictorPath;

/// Immutable snapshot for one mutable DSRs predictor leaf.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct DsrsPredictorSnapshot {
    pub instruction: String,
    pub state: PredictState,
}

impl DsrsPredictorSnapshot {
    #[must_use]
    pub fn new(instruction: String, state: PredictState) -> Self {
        Self { instruction, state }
    }

    #[must_use]
    pub fn with_instruction(&self, instruction: String) -> Self {
        let mut state = self.state.clone();
        state.instruction_override = Some(instruction.clone());
        Self { instruction, state }
    }
}

/// Ordered immutable map of predictor paths to snapshots.
#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
pub struct DsrsProgramState {
    predictors: BTreeMap<PredictorPath, DsrsPredictorSnapshot>,
}

impl DsrsProgramState {
    #[must_use]
    pub fn new(predictors: BTreeMap<PredictorPath, DsrsPredictorSnapshot>) -> Self {
        Self { predictors }
    }

    #[must_use]
    pub fn predictors(&self) -> &BTreeMap<PredictorPath, DsrsPredictorSnapshot> {
        &self.predictors
    }

    pub(crate) fn predictor_mut(
        &mut self,
        path: &PredictorPath,
    ) -> Option<&mut DsrsPredictorSnapshot> {
        self.predictors.get_mut(path)
    }
}

/// Stable layout of predictor parts exposed by the DSRs program surface.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct DsrsProgramLayout {
    paths: Vec<PredictorPath>,
}

impl DsrsProgramLayout {
    #[must_use]
    pub fn new(paths: Vec<PredictorPath>) -> Self {
        Self { paths }
    }

    #[must_use]
    pub fn paths(&self) -> &[PredictorPath] {
        &self.paths
    }
}
