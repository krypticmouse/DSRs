use super::{Chat, LM, Message, Signature};
use crate::data::{example::Example, prediction::Prediction};
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
            let response = lm.call(&messages, signature.metadata().clone()).await?;
            self.parse(signature, response)
        }
    }

    fn format<S: Signature>(&self, signature: &S, inputs: &S::Inputs) -> Chat {
        let system_message = self.format_system_message(signature);
        let user_message = self.format_user_message(signature, inputs);

        let mut chat = Chat::new(vec![]);
        chat.push("system", system_message);
        chat.push("user", user_message);

        chat
    }

    fn parse<S: Signature>(&self, signature: &S, response: Message) -> Result<S::Outputs> {
        let prediction = self.parse_response(signature, response);
        prediction.into_outputs()
    }

    fn format_field_description(&self, signature: &impl Signature) -> String;
    fn format_field_structure(&self, signature: &impl Signature) -> String;
    fn format_task_description(&self, signature: &impl Signature) -> String;

    fn format_system_message(&self, signature: &impl Signature) -> String {
        let field_description = self.format_field_description(signature);
        let field_structure = self.format_field_structure(signature);
        let task_description = self.format_task_description(signature);

        format!("{field_description}\n{field_structure}\n{task_description}")
    }

    fn format_user_message(&self, signature: &impl Signature, inputs: Example) -> String;
}
