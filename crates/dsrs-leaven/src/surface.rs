use std::marker::PhantomData;

use dsrs_core::{Module, Signature};

use crate::{DsrsProgramArtifact, DsrsProgramChange};

#[derive(Clone, Debug)]
pub struct DsrsProgramSurface<S, M>
where
    S: Signature,
    M: Module,
{
    _phantom: PhantomData<(S, M)>,
}

impl<S, M> DsrsProgramSurface<S, M>
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

impl<S, M> leaven_surface::EditSurface<DsrsProgramArtifact<S, M>> for DsrsProgramSurface<S, M>
where
    S: Signature,
    M: Module + Send + Sync + 'static,
{
    type PartId = String;
    type Address = String;
    type View<'a>
        = &'a str
    where
        DsrsProgramArtifact<S, M>: 'a;
    type Edit = serde_json::Value;

    fn fingerprint(&self) -> leaven_surface::SurfaceFingerprint {
        unimplemented!("dsrs-leaven: surface fingerprint")
    }

    fn parts<'a>(
        &self,
        _artifact: &'a DsrsProgramArtifact<S, M>,
    ) -> Result<Vec<leaven_surface::Part<Self::PartId, Self::Address, Self::View<'a>>>, leaven_surface::SurfaceError>
    {
        unimplemented!("dsrs-leaven: surface parts")
    }

    fn change_part(
        &self,
        _artifact: &DsrsProgramArtifact<S, M>,
        _id: Self::PartId,
        _edit: Self::Edit,
    ) -> Result<DsrsProgramChange, leaven_surface::SurfaceError> {
        unimplemented!("dsrs-leaven: surface change_part")
    }
}
