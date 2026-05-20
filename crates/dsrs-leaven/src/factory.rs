/// Builds fresh DSRs module instances for immutable Leaven candidate evaluation.
pub trait DsrsModuleFactory<M>: Clone + Send + Sync + 'static {
    fn fresh_module(&self) -> M;
}

impl<M, F> DsrsModuleFactory<M> for F
where
    F: Fn() -> M + Clone + Send + Sync + 'static,
{
    fn fresh_module(&self) -> M {
        self()
    }
}
