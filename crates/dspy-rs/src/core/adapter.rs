use super::{Chat, LM, Message, Signature};
use anyhow::Result;

pub trait Adapter: Default + Clone {
    fn call<S: Signature>(
        &self,
        lm: &mut LM,
        signature: &S,
        inputs: &S::Inputs,
    ) -> impl Future<Output = Result<S::Outputs>> {
        async move {
            let messages = self.format(signature, inputs);
            let response = lm.call(messages, signature.metadata().clone()).await?;
            self.parse(signature, response)
        }
    }

    fn format<S: Signature>(&self, signature: &S, inputs: &S::Inputs) -> Chat;

    fn parse<S: Signature>(&self, signature: &S, response: Message) -> Result<S::Outputs>;
}
