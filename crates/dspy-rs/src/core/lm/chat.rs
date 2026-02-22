use anyhow::Result;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use rig::OneOrMany;
use rig::message::{
    AssistantContent, Message as RigMessage, Reasoning, ToolCall, ToolResult, ToolResultContent,
    UserContent,
};

// ---------------------------------------------------------------------------
// ContentBlock — one piece of content within a message
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text { text: String },
    ToolCall { tool_call: ToolCall },
    ToolResult { tool_result: ToolResult },
    Reasoning { reasoning: Reasoning },
}

impl ContentBlock {
    pub fn text(t: impl Into<String>) -> Self {
        ContentBlock::Text { text: t.into() }
    }

    pub fn tool_call(tc: ToolCall) -> Self {
        ContentBlock::ToolCall { tool_call: tc }
    }

    pub fn tool_result(tr: ToolResult) -> Self {
        ContentBlock::ToolResult { tool_result: tr }
    }

    pub fn reasoning(r: Reasoning) -> Self {
        ContentBlock::Reasoning { reasoning: r }
    }
}

// ---------------------------------------------------------------------------
// Role
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

// ---------------------------------------------------------------------------
// Message — a single turn in a conversation
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
    /// Provider-assigned message ID (e.g. Anthropic thinking turn IDs).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub id: Option<String>,
}

impl Message {
    /// Creates a text-only message for a typed role.
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: vec![ContentBlock::text(content)],
            id: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![ContentBlock::text(content)],
            id: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: vec![ContentBlock::text(content)],
            id: None,
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: vec![ContentBlock::text(content)],
            id: None,
        }
    }

    /// Creates an assistant message containing a single tool call.
    pub fn tool_call(tool_call: ToolCall) -> Self {
        Self {
            role: Role::Assistant,
            content: vec![ContentBlock::tool_call(tool_call)],
            id: None,
        }
    }

    /// Creates a user message containing a single tool result.
    pub fn tool_result(tool_result: ToolResult) -> Self {
        Self {
            role: Role::User,
            content: vec![ContentBlock::tool_result(tool_result)],
            id: None,
        }
    }

    /// Creates an assistant message containing a single reasoning block.
    pub fn reasoning(reasoning: Reasoning) -> Self {
        Self {
            role: Role::Assistant,
            content: vec![ContentBlock::reasoning(reasoning)],
            id: None,
        }
    }

    /// Creates a message with arbitrary content blocks.
    pub fn with_content(role: Role, content: Vec<ContentBlock>) -> Self {
        Self {
            role,
            content,
            id: None,
        }
    }

    // -- Accessors -----------------------------------------------------------

    /// Returns a string representation of the message's content.
    ///
    /// For text-only messages, returns the text. For multi-content messages,
    /// returns all blocks formatted and joined with newlines.
    pub fn content(&self) -> String {
        let parts: Vec<String> = self
            .content
            .iter()
            .map(|block| match block {
                ContentBlock::Text { text } => text.clone(),
                ContentBlock::ToolCall { tool_call } => {
                    format!(
                        "{}({})",
                        tool_call.function.name, tool_call.function.arguments
                    )
                }
                ContentBlock::ToolResult { tool_result } => tool_result
                    .content
                    .iter()
                    .filter_map(|item| match item {
                        ToolResultContent::Text(text) => Some(text.text.as_str()),
                        ToolResultContent::Image(_) => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
                ContentBlock::Reasoning { reasoning } => reasoning.display_text(),
            })
            .collect();
        parts.join("\n")
    }

    /// Returns only the text content, ignoring tool calls, tool results,
    /// and reasoning blocks. Used by the parser to extract structured output.
    pub fn text_content(&self) -> String {
        self.content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    // -- Content query helpers -----------------------------------------------

    /// Returns `true` if this message contains at least one tool call.
    pub fn has_tool_calls(&self) -> bool {
        self.content
            .iter()
            .any(|b| matches!(b, ContentBlock::ToolCall { .. }))
    }

    /// Returns `true` if this message contains at least one tool result.
    pub fn has_tool_results(&self) -> bool {
        self.content
            .iter()
            .any(|b| matches!(b, ContentBlock::ToolResult { .. }))
    }

    /// Returns `true` if this message contains at least one reasoning block.
    pub fn has_reasoning(&self) -> bool {
        self.content
            .iter()
            .any(|b| matches!(b, ContentBlock::Reasoning { .. }))
    }

    /// Extracts all tool calls from this message.
    pub fn tool_calls(&self) -> Vec<&ToolCall> {
        self.content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::ToolCall { tool_call } => Some(tool_call),
                _ => None,
            })
            .collect()
    }

    // -- Rig conversion ------------------------------------------------------

    /// Converts this message to a rig message for provider API calls.
    ///
    /// Returns `None` for system messages (rig handles them as preamble).
    pub(crate) fn to_rig_message(&self) -> Option<RigMessage> {
        match self.role {
            Role::System => None,
            Role::User => {
                let user_content: Vec<UserContent> = self
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::Text { text } => Some(UserContent::text(text.clone())),
                        ContentBlock::ToolResult { tool_result } => {
                            Some(UserContent::ToolResult(tool_result.clone()))
                        }
                        // ToolCall/Reasoning don't belong in user messages; skip gracefully
                        _ => None,
                    })
                    .collect();
                if user_content.is_empty() {
                    return Some(RigMessage::user(String::new()));
                }
                Some(RigMessage::User {
                    content: OneOrMany::many(user_content)
                        .unwrap_or_else(|_| OneOrMany::one(UserContent::text(String::new()))),
                })
            }
            Role::Assistant => {
                let asst_content: Vec<AssistantContent> = self
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::Text { text } => Some(AssistantContent::text(text.clone())),
                        ContentBlock::ToolCall { tool_call } => {
                            Some(AssistantContent::ToolCall(tool_call.clone()))
                        }
                        ContentBlock::Reasoning { reasoning } => {
                            Some(AssistantContent::Reasoning(reasoning.clone()))
                        }
                        // ToolResult doesn't belong in assistant messages; skip gracefully
                        _ => None,
                    })
                    .collect();
                if asst_content.is_empty() {
                    return Some(RigMessage::assistant(String::new()));
                }
                Some(RigMessage::Assistant {
                    id: self.id.clone(),
                    content: OneOrMany::many(asst_content)
                        .unwrap_or_else(|_| OneOrMany::one(AssistantContent::text(String::new()))),
                })
            }
        }
    }

    // -- JSON serialization --------------------------------------------------

    pub fn to_json(&self) -> Value {
        let content_json: Vec<Value> = self
            .content
            .iter()
            .map(|block| match block {
                ContentBlock::Text { text } => json!({ "type": "text", "text": text }),
                ContentBlock::ToolCall { tool_call } => {
                    json!({ "type": "tool_call", "tool_call": tool_call })
                }
                ContentBlock::ToolResult { tool_result } => {
                    json!({ "type": "tool_result", "tool_result": tool_result })
                }
                ContentBlock::Reasoning { reasoning } => {
                    json!({ "type": "reasoning", "reasoning": reasoning })
                }
            })
            .collect();

        let mut msg = json!({
            "role": self.role.as_str(),
            "content": content_json,
        });

        if let Some(id) = &self.id {
            msg.as_object_mut()
                .unwrap()
                .insert("id".to_string(), json!(id));
        }

        msg
    }

    fn from_json_value(message: &Value) -> Result<Self> {
        let role_str = message
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("chat message missing string role"))?;

        let role = match role_str {
            "system" => Role::System,
            "user" => Role::User,
            "assistant" => Role::Assistant,
            other => return Err(anyhow::anyhow!("unsupported chat message role: {other}")),
        };

        let id = message.get("id").and_then(Value::as_str).map(String::from);

        let content = message
            .get("content")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow::anyhow!("chat message content must be an array"))?
            .iter()
            .map(parse_content_block)
            .collect::<Result<Vec<_>>>()?;

        Ok(Self { role, content, id })
    }
}

fn parse_content_block(value: &Value) -> Result<ContentBlock> {
    let block_type = value
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("content block missing type"))?;

    match block_type {
        "text" => {
            let text = value
                .get("text")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow::anyhow!("text block missing text field"))?;
            Ok(ContentBlock::text(text))
        }
        "tool_call" => {
            let tc: ToolCall = serde_json::from_value(value["tool_call"].clone())?;
            Ok(ContentBlock::tool_call(tc))
        }
        "tool_result" => {
            let tr: ToolResult = serde_json::from_value(value["tool_result"].clone())?;
            Ok(ContentBlock::tool_result(tr))
        }
        "reasoning" => {
            let r: Reasoning = serde_json::from_value(value["reasoning"].clone())?;
            Ok(ContentBlock::reasoning(r))
        }
        other => Err(anyhow::anyhow!("unsupported content block type: {other}")),
    }
}

// ---------------------------------------------------------------------------
// From<RigMessage> — grouped conversion, one rig message → one DSRs message
// ---------------------------------------------------------------------------

impl From<RigMessage> for Message {
    fn from(message: RigMessage) -> Self {
        match message {
            RigMessage::User { content } => {
                let blocks: Vec<ContentBlock> = content
                    .into_iter()
                    .filter_map(|item| match item {
                        UserContent::Text(text) => Some(ContentBlock::text(text.text)),
                        UserContent::ToolResult(result) => Some(ContentBlock::tool_result(result)),
                        UserContent::Image(_)
                        | UserContent::Audio(_)
                        | UserContent::Video(_)
                        | UserContent::Document(_) => None,
                    })
                    .collect();
                Message {
                    role: Role::User,
                    content: if blocks.is_empty() {
                        vec![ContentBlock::text(String::new())]
                    } else {
                        blocks
                    },
                    id: None,
                }
            }
            RigMessage::Assistant { id, content } => {
                let blocks: Vec<ContentBlock> = content
                    .into_iter()
                    .filter_map(|item| match item {
                        AssistantContent::Text(text) => Some(ContentBlock::text(text.text)),
                        AssistantContent::ToolCall(tc) => Some(ContentBlock::tool_call(tc)),
                        AssistantContent::Reasoning(r) => Some(ContentBlock::reasoning(r)),
                        AssistantContent::Image(_) => None,
                    })
                    .collect();
                Message {
                    role: Role::Assistant,
                    content: if blocks.is_empty() {
                        vec![ContentBlock::text(String::new())]
                    } else {
                        blocks
                    },
                    id,
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Chat — ordered sequence of messages
// ---------------------------------------------------------------------------

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

    pub fn push(&mut self, role: Role, content: impl Into<String>) {
        self.messages.push(Message::new(role, content));
    }

    pub fn push_message(&mut self, message: Message) {
        self.messages.push(message);
    }

    pub fn push_all(&mut self, chat: &Chat) {
        self.messages.extend(chat.messages.clone());
    }

    pub fn pop(&mut self) -> Option<Message> {
        self.messages.pop()
    }

    pub fn from_json(&self, json_dump: Value) -> Result<Self> {
        let messages = json_dump
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("chat dump must be an array"))?;
        let messages = messages
            .iter()
            .map(Message::from_json_value)
            .collect::<Result<Vec<_>>>()?;
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

    // -- Rig interop ---------------------------------------------------------

    /// Extracts the system prompt text from the first system message.
    pub(crate) fn system_prompt(&self) -> String {
        self.messages
            .iter()
            .find_map(|message| {
                if message.role == Role::System {
                    Some(message.text_content())
                } else {
                    None
                }
            })
            .unwrap_or_default()
    }

    /// Converts all non-system messages to rig messages for provider API calls.
    pub(crate) fn to_rig_chat_history(&self) -> Vec<RigMessage> {
        self.messages
            .iter()
            .filter_map(Message::to_rig_message)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rig::OneOrMany;
    use rig::message::{ToolFunction, ToolResultContent};
    use serde_json::json;

    #[test]
    fn rig_conversion_preserves_assistant_reasoning_and_tool_calls() {
        let original = Message::with_content(
            Role::Assistant,
            vec![
                ContentBlock::reasoning(Reasoning::new("step 1")),
                ContentBlock::reasoning(Reasoning::new("step 2")),
                ContentBlock::tool_call(ToolCall::new(
                    "tc-1".to_string(),
                    ToolFunction {
                        name: "search".to_string(),
                        arguments: json!({"q": "rust ownership"}),
                    },
                )),
            ],
        );

        let rig_msg = original
            .to_rig_message()
            .expect("assistant message should convert to rig");
        let roundtripped = Message::from(rig_msg);

        assert_eq!(roundtripped.role, Role::Assistant);
        assert_eq!(roundtripped.content.len(), 3);
        assert!(matches!(
            &roundtripped.content[0],
            ContentBlock::Reasoning { .. }
        ));
        assert!(matches!(
            &roundtripped.content[1],
            ContentBlock::Reasoning { .. }
        ));
        assert!(
            matches!(&roundtripped.content[2], ContentBlock::ToolCall { tool_call } if tool_call.function.name == "search")
        );
    }

    #[test]
    fn rig_conversion_preserves_user_text_and_tool_result() {
        let original = Message::with_content(
            Role::User,
            vec![
                ContentBlock::text("Here is context"),
                ContentBlock::tool_result(ToolResult {
                    id: "tr-1".to_string(),
                    call_id: Some("tc-1".to_string()),
                    content: OneOrMany::one(ToolResultContent::text("search result")),
                }),
            ],
        );

        let rig_msg = original
            .to_rig_message()
            .expect("user should convert to rig");
        let roundtripped = Message::from(rig_msg);

        assert_eq!(roundtripped.role, Role::User);
        assert_eq!(roundtripped.content.len(), 2);
        assert!(
            matches!(&roundtripped.content[0], ContentBlock::Text { text } if text == "Here is context")
        );
        assert!(roundtripped.has_tool_results());
    }

    #[test]
    fn rig_chat_history_excludes_system_message() {
        let chat = Chat::new(vec![
            Message::system("You are a helpful assistant."),
            Message::user("What is the capital of France?"),
            Message::assistant("Paris."),
        ]);

        let rig_history = chat.to_rig_chat_history();
        assert_eq!(rig_history.len(), 2);
    }

    #[test]
    fn system_messages_are_not_converted_to_rig_messages() {
        let msg = Message::system("You are helpful");
        assert!(msg.to_rig_message().is_none());
    }

    #[test]
    fn assistant_message_id_survives_rig_roundtrip() {
        let mut msg = Message::assistant("some text");
        msg.id = Some("msg_abc123".to_string());

        let rig_msg = msg
            .to_rig_message()
            .expect("assistant should convert to rig");
        let roundtripped = Message::from(rig_msg);

        assert_eq!(roundtripped.id, Some("msg_abc123".to_string()));
    }
}
