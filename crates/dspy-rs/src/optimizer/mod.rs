use anyhow::Result;
use crate::core::MetaSignature;

pub trait Optimizer {
    fn compile(&self, signature: MetaSignature) -> Result<()>;
}