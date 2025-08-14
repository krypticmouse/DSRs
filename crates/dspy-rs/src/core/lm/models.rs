use crate::core::SignatureMetadata;
use anyhow::Result;

use bon::Builder;

use serde::{Deserialize, Serialize};
use serde_json::json;

use std::collections::HashMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ContentTypes {
    Text(String),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCallMessage {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Message {
    System {
        content: ContentTypes,
    },
    User {
        content: ContentTypes,
    },
    Assistant {
        content: Option<ContentTypes>,
        tool_calls: Option<Vec<ToolCallMessage>>,
    },
    Tool {
        content: ContentTypes,
        tool_call_id: String,
    },
}

impl Message {
    pub fn user(content: impl Into<String>) -> Self {
        Message::User {
            content: ContentTypes::Text(content.into()),
        }
    }

    pub fn assistant(
        content: Option<impl Into<String>>,
        tool_calls: Option<Vec<ToolCallMessage>>,
    ) -> Self {
        Message::Assistant {
            content: content.map(|c| ContentTypes::Text(c.into())),
            tool_calls,
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Message::System {
            content: ContentTypes::Text(content.into()),
        }
    }

    pub fn tool(content: impl Into<String>, tool_call_id: impl Into<String>) -> Self {
        Message::Tool {
            content: ContentTypes::Text(content.into()),
            tool_call_id: tool_call_id.into(),
        }
    }
}

#[derive(Clone, Debug, Builder)]
pub struct LMConfig {
    #[builder(default = "gpt-4o-mini".to_string())]
    pub model: String,

    #[builder(default = 0.7)]
    pub temperature: f32,
    #[builder(default = 1.0)]
    pub top_p: f32,
    #[builder(default = 512)]
    pub max_tokens: u32,
    #[builder(default = 512)]
    pub max_completion_tokens: u32,
    #[builder(default = 1)]
    pub n: u8,
    #[builder(default = 0.0)]
    pub presence_penalty: f32,
    #[builder(default = 0.0)]
    pub frequency_penalty: f32,
    #[builder(default = 42)]
    pub seed: i64,
    pub logit_bias: Option<HashMap<String, serde_json::Value>>,
    // removed stream until we decide on how to provide a method to receive stream data back
    // could allow users to pass a broadcast tx in module settings and stream back on that by default
    // pub stream: Option<bool>,
    // pub stream_options: Option<ChatCompletionStreamOptions>,

    // removed reasoning effort for now to keep async_openai out of core
    // pub reasoning_effort: Option<ReasoningEffort>,
}

impl Default for LMConfig {
    fn default() -> Self {
        LMConfig::builder().build()
    }
}

#[derive(Clone)]
pub struct LMInvocation {
    pub chat: Chat,
    pub config: LMConfig,
    pub signature: SignatureMetadata,
    pub output: Message,
}

#[derive(Debug, Clone)]
pub struct Chat {
    pub messages: Vec<Message>,
}

impl Chat {
    pub fn new(messages: Vec<Message>) -> Self {
        Self { messages }
    }

    fn build_message(&self, role: &str, content: String) -> Message {
        match role {
            "system" => Message::system(content),
            "user" => Message::user(content),
            "assistant" => Message::assistant(Some(content), None),
            _ => panic!("Invalid role: {role}"),
        }
    }

    pub fn push(&mut self, role: &str, content: String) {
        self.messages.push(self.build_message(role, content));
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

    pub fn to_json(&self) -> serde_json::Value {
        let json_messages = self
            .messages
            .iter()
            .map(|message| serde_json::to_value(message).unwrap())
            .collect::<Vec<serde_json::Value>>();

        json!(json_messages)
    }

    pub fn from_json(&self, json_dump: &str) -> Result<Self> {
        let parsed: serde_json::Value = serde_json::from_str(json_dump)?;
        let json_messages = parsed.as_array().unwrap();

        let mut messages: Vec<Message> = Vec::new();
        for message in json_messages {
            let role = message["role"].as_str().unwrap();
            let content = message["content"].as_str().unwrap();
            messages.push(self.build_message(role, content.to_string()));
        }

        Ok(Self { messages })
    }
}
