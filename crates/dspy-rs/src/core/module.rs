use super::Signature;
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

    fn aforward(
        &self,
        inputs: &<<Self as Module>::SIG as Signature>::Inputs,
    ) -> impl Future<Output = Result<<<Self as Module>::SIG as Signature>::Outputs>>;
}
