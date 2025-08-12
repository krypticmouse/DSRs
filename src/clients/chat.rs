use async_openai::types::{
    ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
};
use serde_json::json;

#[derive(Debug, Clone)]
pub struct Chat {
    pub messages: Vec<ChatCompletionRequestMessage>,
}

impl Chat {
    pub fn new(messages: Vec<ChatCompletionRequestMessage>) -> Self {
        Self { messages }
    }

    fn build_message(&self, role: &str, content: String) -> ChatCompletionRequestMessage {
        match role {
            "system" => ChatCompletionRequestSystemMessageArgs::default()
                .content(content)
                .build()
                .unwrap()
                .into(),
            "user" => ChatCompletionRequestUserMessageArgs::default()
                .content(content)
                .build()
                .unwrap()
                .into(),
            "assistant" => ChatCompletionRequestAssistantMessageArgs::default()
                .content(content)
                .build()
                .unwrap()
                .into(),
            _ => panic!("Invalid role: {role}"),
        }
    }

    pub fn push(&mut self, role: &str, content: String) {
        self.messages.push(self.build_message(role, content));
    }

    pub fn pop(&mut self) -> Option<ChatCompletionRequestMessage> {
        self.messages.pop()
    }

    pub fn len(&self) -> usize {
        self.messages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    fn get_message_json(&self, message: &ChatCompletionRequestMessage) -> serde_json::Value {
        match message {
            ChatCompletionRequestMessage::System(system_message) => {
                json!({ "role": "system", "content": system_message.content })
            }
            ChatCompletionRequestMessage::User(user_message) => {
                json!({ "role": "user", "content": user_message.content })
            }
            ChatCompletionRequestMessage::Assistant(assistant_message) => {
                json!({ "role": "assistant", "content": assistant_message.content })
            }
            _ => panic!("Invalid message type"),
        }
    }

    pub fn to_json(&self) -> serde_json::Value {
        let json_messages = self
            .messages
            .iter()
            .map(|message| self.get_message_json(message))
            .collect::<Vec<serde_json::Value>>();

        json!(json_messages)
    }

    pub fn from_json(&self, json_dump: &str) -> Result<Self, serde_json::Error> {
        let parsed: serde_json::Value = serde_json::from_str(json_dump)?;
        let json_messages = parsed.as_array().unwrap();

        let mut messages: Vec<ChatCompletionRequestMessage> = Vec::new();
        for message in json_messages {
            let role = message["role"].as_str().unwrap();
            let content = message["content"].as_str().unwrap();
            messages.push(self.build_message(role, content.to_string()));
        }

        Ok(Self { messages })
    }
}
