use serde::{Deserialize, Serialize};
use std::fmt;

use openrouter_rs::{api::chat::Message, types::Role};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chat {
    pub messages: Vec<Message>,
}

impl Chat {
    pub fn new(messages: Vec<Message>) -> Self {
        Self { messages }
    }

    pub fn push(&mut self, role: Role, content: String) {
        self.messages.push(Message { role, content });
    }

    pub fn pop(&mut self) -> Option<Message> {
        self.messages.pop()
    }

    pub fn len(&self) -> usize {
        self.messages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Serialize the chat messages to JSON string
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(&self.messages)
    }

    /// Deserialize chat messages from JSON string
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        let messages: Vec<Message> = serde_json::from_str(json)?;
        Ok(Self { messages })
    }
}

impl fmt::Display for Chat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for message in &self.messages {
            write!(f, "{}", serde_json::to_string(message).unwrap())?;
        }
        Ok(())
    }
}
