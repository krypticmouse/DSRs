use std::sync::Arc;

use crate::{BamlType, Facet, PredictError, Predicted};

use super::Module;

/// Output transformation combinators for any [`Module`].
///
/// Post-process a module's output without writing a full `impl Module`. This is
/// the intermediate step between "use a library module" and "author your own" —
/// if you just need to reshape the output, a closure is enough.
///
/// The inner module's [`crate::Predict`] leaves remain visible to the Facet walker,
/// so optimizer discovery works through these wrappers.
///
/// ```ignore
/// // Transform output without impl Module
/// let confident = cot.map(|r| ConfidentAnswer {
///     answer: r.answer.clone(),
///     confidence: 0.9,
/// });
/// let result = confident.call(input).await?;
/// ```
pub trait ModuleExt: Module + Sized {
    /// Transforms the output with an infallible closure. Returns a [`Map`] wrapper.
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

    /// Transforms the output with a fallible closure. Returns an [`AndThen`] wrapper.
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

/// Output transformation wrapper created by [`ModuleExt::map`].
///
/// Delegates to the inner module, then applies the closure to the output.
/// The inner module's [`crate::Predict`] leaves remain visible to Facet reflection
/// (the `inner` field is a real struct field), so optimizers can still discover and
/// tune parameters through this wrapper.
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
        let (output, metadata, chat) = predicted.into_parts();
        Ok(Predicted::new((self.map)(output), metadata, chat))
    }
}

/// Fallible output transformation wrapper created by [`ModuleExt::and_then`].
///
/// Like [`Map`], but the closure returns `Result<T, PredictError>`.
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
        let (output, metadata, chat) = predicted.into_parts();
        let transformed = (self.and_then)(output)?;
        Ok(Predicted::new(transformed, metadata, chat))
    }
}
