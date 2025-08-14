use super::Signature;
use super::{GLOBAL_SETTINGS, Settings};
use anyhow::Result;
use std::future::Future;

pub trait Module {
    type SIG: Signature;

    fn forward(
        &self,
        inputs: &<<Self as Module>::SIG as Signature>::Inputs,
    ) -> Result<<<Self as Module>::SIG as Signature>::Outputs> {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(self.aforward(inputs))
        })
    }

    fn forward_with_settings(
        &self,
        inputs: &<<Self as Module>::SIG as Signature>::Inputs,
        settings: &mut Settings,
    ) -> Result<<<Self as Module>::SIG as Signature>::Outputs> {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current()
                .block_on(self.aforward_with_settings(inputs, settings))
        })
    }

    fn aforward(
        &self,
        inputs: &<<Self as Module>::SIG as Signature>::Inputs,
    ) -> impl Future<Output = Result<<<Self as Module>::SIG as Signature>::Outputs>> {
        async move {
            let mut guard = GLOBAL_SETTINGS.write().unwrap();
            let settings = guard.as_mut().ok_or_else(|| {
                anyhow::anyhow!(
                    "Global settings not configured. Please call `configure` before using modules."
                )
            })?;
            let result = self.aforward_with_settings(inputs, settings).await;
            result
        }
    }

    fn aforward_with_settings(
        &self,
        inputs: &<<Self as Module>::SIG as Signature>::Inputs,
        settings: &mut Settings,
    ) -> impl Future<Output = Result<<<Self as Module>::SIG as Signature>::Outputs>>;
}
