use std::marker::PhantomData;

use dsrs_core::{Module, Signature};

use crate::{DsrsLeavenError, DsrsProgramChange};

#[derive(Debug)]
pub struct DsrsProgramArtifact<S, M>
where
    S: Signature,
    M: Module,
{
    _phantom: PhantomData<(S, M)>,
}

impl<S, M> Clone for DsrsProgramArtifact<S, M>
where
    S: Signature,
    M: Module,
{
    fn clone(&self) -> Self {
        Self::scaffold()
    }
}

impl<S, M> DsrsProgramArtifact<S, M>
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

impl<S, M> leaven_core::Artifact for DsrsProgramArtifact<S, M>
where
    S: Signature,
    M: Module + Send + Sync + 'static,
{
    type Change = DsrsProgramChange;
    type ApplyError = DsrsLeavenError;

    fn identity(&self) -> leaven_core::ArtifactIdentity {
        unimplemented!("dsrs-leaven: artifact identity")
    }

    fn apply_change(&self, _change: &Self::Change) -> Result<Self, Self::ApplyError> {
        unimplemented!("dsrs-leaven: artifact apply_change")
    }
}
