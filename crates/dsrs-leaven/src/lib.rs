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

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use dsrs_core::{CallMetadata, Module, PredictError, Predicted, Signature};
    use leaven_core::{Artifact, OptimizationProblem};

    #[derive(Signature, Clone, Debug)]
    struct TestSig {
        #[input]
        prompt: String,

        #[output]
        answer: String,
    }

    struct TestModule;

    impl Module for TestModule {
        type Input = TestSigInput;
        type Output = TestSigOutput;

        async fn forward(
            &self,
            input: Self::Input,
        ) -> Result<Predicted<Self::Output>, PredictError> {
            Ok(Predicted::new(
                TestSigOutput {
                    answer: input.prompt,
                },
                CallMetadata::default(),
            ))
        }
    }

    #[test]
    fn scaffold_constructors_are_cloneable_markers() {
        let artifact = DsrsProgramArtifact::<TestSig, TestModule>::scaffold();
        let _clone = artifact.clone();
        let _surface = DsrsProgramSurface::<TestSig, TestModule>::scaffold();
        let _evaluator = DsrsEvaluator::<TestSig, TestModule>::scaffold();
    }

    #[test]
    fn problem_associated_types_match_dsrs_scaffold() {
        fn assert_problem<P: OptimizationProblem>() {}
        assert_problem::<DsrsLeavenProblem<TestSig, TestModule>>();
    }

    #[test]
    #[should_panic(expected = "dsrs-leaven: artifact identity")]
    fn artifact_identity_is_explicit_scaffold_panic() {
        let artifact = DsrsProgramArtifact::<TestSig, TestModule>::scaffold();
        let _ = artifact.identity();
    }

    #[test]
    fn change_and_evidence_round_trip_json_payloads() {
        let change = DsrsProgramChange {
            address: "predictor.instruction".to_string(),
            replacement: serde_json::json!("new instruction"),
        };
        let encoded = serde_json::to_string(&change).unwrap();
        let decoded: DsrsProgramChange = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded.address, "predictor.instruction");
        assert_eq!(decoded.replacement, "new instruction");

        let evidence = DsrsEvidence {
            payload: serde_json::json!({"score": 1.0}),
        };
        assert_eq!(evidence.payload["score"], 1.0);
    }
}
