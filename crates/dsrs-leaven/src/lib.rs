//! DSRs to Leaven integration.
//!
//! This crate is the DSRs-owned bridge from mutable DSRs modules to Leaven's
//! immutable optimization contracts.

pub mod artifact;
pub mod change;
pub mod evaluator;
pub mod evidence;
pub mod factory;
pub mod state;
pub mod surface;

pub use artifact::DsrsProgramArtifact;
pub use change::{DsrsPredictorEdit, DsrsProgramChange, PredictorPath};
pub use evaluator::{DsrsEvaluator, DsrsLeavenProblem};
pub use evidence::DsrsCaseEvidence;
pub use factory::DsrsModuleFactory;
pub use state::{DsrsPredictorSnapshot, DsrsProgramLayout, DsrsProgramState};
pub use surface::DsrsProgramSurface;

#[derive(Debug, thiserror::Error)]
pub enum DsrsLeavenError {
    #[error("invalid predictor path `{0}`")]
    InvalidPredictorPath(String),
    #[error("DSRs module has no discoverable predictors")]
    NoPredictors,
    #[error("unknown predictor path `{0}`")]
    UnknownPredictorPath(PredictorPath),
    #[error("program layout and predictor state do not match")]
    LayoutStateMismatch,
    #[error("failed to discover DSRs predictors: {0}")]
    PredictorDiscovery(#[from] dsrs_core::NamedParametersError),
    #[error("failed to load predictor state at `{path}`: {source}")]
    LoadState {
        path: PredictorPath,
        #[source]
        source: anyhow::Error,
    },
}
