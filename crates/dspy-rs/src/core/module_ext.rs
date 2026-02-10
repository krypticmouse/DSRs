use std::sync::Arc;

use crate::{BamlType, Facet, PredictError, Predicted};

use super::Module;

pub trait ModuleExt: Module + Sized {
    fn map<F, T>(self, map: F) -> Map<Self, T>
    where
        F: Fn(Self::Output) -> T + Send + Sync + 'static,
        T: BamlType + for<'a> Facet<'a> + Send + Sync,
    {
        Map {
            inner: self,
            map: Arc::new(map),
        }
    }

    fn and_then<F, T>(self, and_then: F) -> AndThen<Self, T>
    where
        F: Fn(Self::Output) -> Result<T, PredictError> + Send + Sync + 'static,
        T: BamlType + for<'a> Facet<'a> + Send + Sync,
    {
        AndThen {
            inner: self,
            and_then: Arc::new(and_then),
        }
    }
}

impl<M: Module> ModuleExt for M {}

#[derive(facet::Facet)]
#[facet(crate = facet)]
pub struct Map<M, T: 'static>
where
    M: Module,
{
    pub(crate) inner: M,
    #[facet(opaque, skip)]
    map: Arc<dyn Fn(M::Output) -> T + Send + Sync>,
}

#[allow(async_fn_in_trait)]
impl<M, T> Module for Map<M, T>
where
    M: Module,
    T: BamlType + for<'a> Facet<'a> + Send + Sync,
{
    type Input = M::Input;
    type Output = T;

    async fn forward(&self, input: Self::Input) -> Result<Predicted<Self::Output>, PredictError> {
        let predicted = self.inner.call(input).await?;
        let (output, metadata) = predicted.into_parts();
        Ok(Predicted::new((self.map)(output), metadata))
    }
}

#[derive(facet::Facet)]
#[facet(crate = facet)]
pub struct AndThen<M, T: 'static>
where
    M: Module,
{
    pub(crate) inner: M,
    #[facet(opaque, skip)]
    and_then: Arc<dyn Fn(M::Output) -> Result<T, PredictError> + Send + Sync>,
}

#[allow(async_fn_in_trait)]
impl<M, T> Module for AndThen<M, T>
where
    M: Module,
    T: BamlType + for<'a> Facet<'a> + Send + Sync,
{
    type Input = M::Input;
    type Output = T;

    async fn forward(&self, input: Self::Input) -> Result<Predicted<Self::Output>, PredictError> {
        let predicted = self.inner.call(input).await?;
        let (output, metadata) = predicted.into_parts();
        let transformed = (self.and_then)(output)?;
        Ok(Predicted::new(transformed, metadata))
    }
}
