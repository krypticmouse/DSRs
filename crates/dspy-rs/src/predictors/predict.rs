use crate::core::{Adapter, Module, Signature};

pub struct Predict<S: Signature> {
    pub signature: S,
}

impl<S: Signature> Module for Predict<S> {
    type SIG = S;

    async fn aforward_with_settings(
        &self,
        inputs: &<<Self as Module>::SIG as Signature>::Inputs,
        settings: &mut crate::core::Settings,
    ) -> anyhow::Result<<<Self as Module>::SIG as Signature>::Outputs> {
        let adapter = &settings.adapter;
        let lm = &mut settings.lm;
        let result = adapter.call(lm, &self.signature, inputs).await;
        result
    }
}
