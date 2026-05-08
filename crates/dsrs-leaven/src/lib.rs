//! DSRs to leaven integration scaffolding.
//!
//! Bodies are deliberately `unimplemented!()` until the leaven-side optimizer
//! path is real. This crate exists to keep the capability trait signatures
//! compiling against the current leaven crates.

pub mod artifact;
pub mod change;
pub mod evaluator;
pub mod evidence;
pub mod surface;

pub use artifact::DsrsProgramArtifact;
pub use change::DsrsProgramChange;
pub use evaluator::{DsrsEvaluator, DsrsLeavenProblem};
pub use evidence::DsrsEvidence;
pub use surface::DsrsProgramSurface;

#[derive(Debug, thiserror::Error)]
pub enum DsrsLeavenError {
    #[error("dsrs-leaven scaffold is not implemented yet: {0}")]
    Unimplemented(&'static str),
}
