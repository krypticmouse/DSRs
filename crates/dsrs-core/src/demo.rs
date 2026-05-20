use crate::Signature;

/// A typed input/output pair for few-shot prompting.
#[derive(Clone, Debug, facet::Facet)]
#[facet(crate = facet)]
pub struct Example<S: Signature> {
    pub input: S::Input,
    pub output: S::Output,
}

impl<S: Signature> Example<S> {
    pub fn new(input: S::Input, output: S::Output) -> Self {
        Self { input, output }
    }
}
