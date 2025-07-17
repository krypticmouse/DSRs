use crate::data::example::Example;
use crate::signature::signature::Signature;

pub trait Adapter {
    pub fn format(&self, signature: Signature, demos: Vec<Example>) -> String {
        let system_message = self.format_system_message(signatures);
        let user_message = self.format_user_message(signature, demos);
        
        vec![
            ChatCompletionMessage {
                role: MessageRole::system,
                content: Content::Text(system_message),
            },
            ChatCompletionMessage {
                role: MessageRole::user,
                content: Content::Text(user_message),
            },
        ]
    }

    pub fn format_field_description(&self, signature: Signature) -> String;
    pub fn format_field_structure(&self, signature: Signature) -> String;
    pub fn format_task_description(&self, signature: Signature) -> String;

    pub fn format_system_message(&self, signature: Signature) -> String {
        let field_description = self.format_field_description(signature);
        let field_structure = self.format_field_structure(signature);
        let task_description = self.format_task_description(signature);

        format!("{field_description}\n{field_structure}\n{task_description}")
    }
    
    pub fn format_user_message(&self, signature: Signature, inputs: HashMap<String, String>) -> String;
    pub fn format_demos(&self, demos: Vec<Example>) -> String;

    pub fn parse_response(&self, signature: Signature, response: String) -> HashMap<String, String>;
}