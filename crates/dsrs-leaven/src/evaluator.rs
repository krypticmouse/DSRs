use std::marker::PhantomData;

use dsrs_core::{Module, Signature};

use crate::{DsrsEvidence, DsrsProgramArtifact};

#[derive(Clone, Debug)]
pub struct DsrsEvaluator<S, M>
where
    S: Signature,
    M: Module,
{
    _phantom: PhantomData<(S, M)>,
}

impl<S, M> DsrsEvaluator<S, M>
where
    S: Signature,
    M: Module,
{
    pub const fn scaffold() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DsrsLeavenProblem<S, M>
where
    S: Signature,
    M: Module,
{
    _phantom: PhantomData<(S, M)>,
}

impl<S, M> leaven_core::OptimizationProblem for DsrsLeavenProblem<S, M>
where
    S: Signature,
    M: Module + Send + Sync + 'static,
{
    type Artifact = DsrsProgramArtifact<S, M>;
    type Case = serde_json::Value;
    type Evidence = DsrsEvidence;
    type ProposalAnnotations = serde_json::Value;
}
