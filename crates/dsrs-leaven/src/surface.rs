use dsrs_core::{Facet, Module, Signature};
use leaven_kernel::FingerprintBuilder;
use leaven_surface::{Part, SurfaceError, SurfaceFingerprint};

use crate::{DsrsModuleFactory, DsrsProgramArtifact, DsrsProgramChange, PredictorPath};

/// Leaven edit surface exposing DSRs predictor instructions as editable parts.
#[derive(Clone, Debug, Default)]
pub struct DsrsProgramSurface;

impl<S, M, F> leaven_surface::EditSurface<DsrsProgramArtifact<S, M, F>> for DsrsProgramSurface
where
    S: Signature,
    M: Module + for<'a> Facet<'a> + Send + Sync + 'static,
    F: DsrsModuleFactory<M>,
{
    type PartId = PredictorPath;
    type Address = PredictorPath;
    type View<'a>
        = &'a str
    where
        DsrsProgramArtifact<S, M, F>: 'a;
    type Edit = String;

    fn fingerprint(&self) -> SurfaceFingerprint {
        let mut builder = FingerprintBuilder::new();
        builder.update(b"dsrs-leaven-program-surface-v1");
        SurfaceFingerprint(builder.finish())
    }

    fn parts<'a>(
        &self,
        artifact: &'a DsrsProgramArtifact<S, M, F>,
    ) -> Result<Vec<Part<Self::PartId, Self::Address, Self::View<'a>>>, SurfaceError> {
        artifact
            .layout()
            .paths()
            .iter()
            .map(|path| {
                let snapshot = artifact
                    .state()
                    .predictors()
                    .get(path)
                    .ok_or(SurfaceError::UnknownPart)?;
                Ok(Part {
                    id: path.clone(),
                    address: path.clone(),
                    view: snapshot.instruction.as_str(),
                })
            })
            .collect()
    }

    fn change_part(
        &self,
        artifact: &DsrsProgramArtifact<S, M, F>,
        id: Self::PartId,
        edit: Self::Edit,
    ) -> Result<DsrsProgramChange, SurfaceError> {
        if !artifact.state().predictors().contains_key(&id) {
            return Err(SurfaceError::UnknownPart);
        }
        Ok(DsrsProgramChange::replace_instruction(id, edit))
    }
}
