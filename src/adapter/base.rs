use openrouter_rs::{
    api::chat::Message,
    types::{CompletionsResponse, Role},
};

use crate::clients::chat::Chat;
use crate::data::{example::Example, prediction::Prediction};
use crate::signature::signature::Signature;

pub trait Adapter {
    fn format(&self, signature: Signature, inputs: Example) -> Chat {
        let system_message = self.format_system_message(signature.clone());
        let user_message = self.format_user_message(signature.clone(), inputs.clone());

        Chat {
            messages: vec![
                Message {
                    role: Role::System,
                    content: system_message,
                },
                Message {
                    role: Role::User,
                    content: user_message,
                },
            ],
        }
    }

    fn format_field_description(&self, signature: Signature) -> String;
    fn format_field_structure(&self, signature: Signature) -> String;
    fn format_task_description(&self, signature: Signature) -> String;

    fn format_system_message(&self, signature: Signature) -> String {
        let field_description = self.format_field_description(signature.clone());
        let field_structure = self.format_field_structure(signature.clone());
        let task_description = self.format_task_description(signature.clone());

        format!("{field_description}\n{field_structure}\n{task_description}")
    }

    fn format_user_message(&self, signature: Signature, inputs: Example) -> String;

    fn parse_response(&self, signature: Signature, response: CompletionsResponse) -> Prediction;
}
