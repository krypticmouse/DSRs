use anyhow::Result;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use async_openai::types::{
    ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
    ChatCompletionResponseMessage, Role,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Message {
    System { content: String },
    User { content: String },
    Assistant { content: String },
}

impl Message {
    pub fn new(role: &str, content: &str) -> Self {
        match role {
            "system" => Message::system(content),
            "user" => Message::user(content),
            "assistant" => Message::assistant(content),
            _ => panic!("Invalid role: {role}"),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Message::User {
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Message::Assistant {
            content: content.into(),
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Message::System {
            content: content.into(),
        }
    }

    pub fn content(&self) -> String {
        match self {
            Message::System { content } => content.clone(),
            Message::User { content } => content.clone(),
            Message::Assistant { content } => content.clone(),
        }
    }

    pub fn get_message_turn(&self) -> ChatCompletionRequestMessage {
        match self {
            Message::System { content } => ChatCompletionRequestSystemMessageArgs::default()
                .content(content.as_str())
                .build()
                .unwrap()
                .into(),
            Message::User { content } => ChatCompletionRequestUserMessageArgs::default()
                .content(content.as_str())
                .build()
                .unwrap()
                .into(),
            Message::Assistant { content } => ChatCompletionRequestAssistantMessageArgs::default()
                .content(content.as_str())
                .build()
                .unwrap()
                .into(),
        }
    }

    pub fn to_json(&self) -> Value {
        match self {
            Message::System { content } => json!({ "role": "system", "content": content }),
            Message::User { content } => json!({ "role": "user", "content": content }),
            Message::Assistant { content } => json!({ "role": "assistant", "content": content }),
        }
    }
}

impl From<ChatCompletionResponseMessage> for Message {
    fn from(message: ChatCompletionResponseMessage) -> Self {
        match message.role {
            Role::System => Message::System {
                content: message.content.unwrap(),
            },
            Role::User => Message::User {
                content: message.content.unwrap(),
            },
            Role::Assistant => Message::Assistant {
                content: message.content.unwrap(),
            },
            _ => panic!("Invalid role: {:?}", message.role),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Chat {
    pub messages: Vec<Message>,
}

impl Chat {
    pub fn new(messages: Vec<Message>) -> Self {
        Self { messages }
    }

    pub fn len(&self) -> usize {
        self.messages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    pub fn push(&mut self, role: &str, content: &str) {
        self.messages.push(Message::new(role, content));
    }

    pub fn pop(&mut self) -> Option<Message> {
        self.messages.pop()
    }

    pub fn from_json(&self, json_dump: Value) -> Result<Self> {
        let messages = json_dump.as_array().unwrap();
        let messages = messages
            .iter()
            .map(|message| {
                Message::new(
                    message["role"].as_str().unwrap(),
                    message["content"].as_str().unwrap(),
                )
            })
            .collect();
        Ok(Self { messages })
    }

    pub fn to_json(&self) -> Value {
        let messages = self
            .messages
            .iter()
            .map(|message| message.to_json())
            .collect::<Vec<Value>>();
        json!(messages)
    }

    pub fn get_async_openai_messages(&self) -> Vec<ChatCompletionRequestMessage> {
        self.messages
            .iter()
            .map(|message| message.get_message_turn())
            .collect()
    }
}
