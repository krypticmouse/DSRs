use crate::clients::chat::Chat;
use crate::data::{example::Example, prediction::Prediction};
use crate::internal::MetaSignature;
use async_openai::types::CreateChatCompletionResponse;

pub trait Adapter {
    fn format(&self, signature: &MetaSignature, inputs: Example) -> Chat {
        let system_message = self.format_system_message(signature);
        let user_message = self.format_user_message(signature, inputs);

        let mut chat = Chat::new(vec![]);
        chat.push("system", system_message);
        chat.push("user", user_message);

        chat
    }

    fn format_field_description(&self, signature: &MetaSignature) -> String;
    fn format_field_structure(&self, signature: &MetaSignature) -> String;
    fn format_task_description(&self, signature: &MetaSignature) -> String;

    fn format_system_message(&self, signature: &MetaSignature) -> String {
        let field_description = self.format_field_description(signature);
        let field_structure = self.format_field_structure(signature);
        let task_description = self.format_task_description(signature);

        format!("{field_description}\n{field_structure}\n{task_description}")
    }

    fn format_user_message(&self, signature: &MetaSignature, inputs: Example) -> String;

    fn parse_response(
        &self,
        signature: &MetaSignature,
        response: CreateChatCompletionResponse,
    ) -> Prediction;
}
