use crate::{CallOutcome, CallOutcomeErrorKind};

use super::Module;

pub trait ModuleExt: Module + Sized {
    fn map<F, T>(self, map: F) -> Map<Self, F>
    where
        F: Fn(Self::Output) -> T + Send + Sync + 'static,
        T: Send + Sync + 'static,
    {
        Map { inner: self, map }
    }

    fn and_then<F, T>(self, and_then: F) -> AndThen<Self, F>
    where
        F: Fn(Self::Output) -> Result<T, CallOutcomeErrorKind> + Send + Sync + 'static,
        T: Send + Sync + 'static,
    {
        AndThen {
            inner: self,
            and_then,
        }
    }
}

impl<M: Module> ModuleExt for M {}

#[derive(facet::Facet)]
#[facet(crate = facet)]
pub struct Map<M, F> {
    pub(crate) inner: M,
    #[facet(skip)]
    map: F,
}

#[allow(async_fn_in_trait)]
impl<M, F, T> Module for Map<M, F>
where
    M: Module,
    F: Fn(M::Output) -> T + Send + Sync + 'static,
    T: Send + Sync + 'static,
{
    type Input = M::Input;
    type Output = T;

    async fn forward(&self, input: Self::Input) -> CallOutcome<Self::Output> {
        let (result, metadata) = self.inner.forward(input).await.into_parts();
        match result {
            Ok(output) => CallOutcome::ok((self.map)(output), metadata),
            Err(err) => CallOutcome::err(err, metadata),
        }
    }
}

#[derive(facet::Facet)]
#[facet(crate = facet)]
pub struct AndThen<M, F> {
    pub(crate) inner: M,
    #[facet(skip)]
    and_then: F,
}

#[allow(async_fn_in_trait)]
impl<M, F, T> Module for AndThen<M, F>
where
    M: Module,
    F: Fn(M::Output) -> Result<T, CallOutcomeErrorKind> + Send + Sync + 'static,
    T: Send + Sync + 'static,
{
    type Input = M::Input;
    type Output = T;

    async fn forward(&self, input: Self::Input) -> CallOutcome<Self::Output> {
        let (result, metadata) = self.inner.forward(input).await.into_parts();
        match result {
            Ok(output) => match (self.and_then)(output) {
                Ok(transformed) => CallOutcome::ok(transformed, metadata),
                Err(err) => CallOutcome::err(err, metadata),
            },
            Err(err) => CallOutcome::err(err, metadata),
        }
    }
}
