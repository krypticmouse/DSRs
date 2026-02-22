//! Public-API tests for the grouped Message model.
//!
//! These tests validate message/content behavior through stable public methods
//! (`to_json`/`from_json`, content accessors) without calling crate-internal
//! rig conversion helpers.

use dspy_rs::{Chat, ContentBlock, Message, Role};
use rig::OneOrMany;
use rig::message::{Reasoning, ToolCall, ToolFunction, ToolResult, ToolResultContent};
use serde_json::json;

#[test]
fn grouped_message_json_roundtrip() {
    let original = Chat::new(vec![
        Message::system("Be helpful"),
        Message::with_content(
            Role::Assistant,
            vec![
                ContentBlock::reasoning(Reasoning::new("let me think")),
                ContentBlock::text("the answer is 42"),
                ContentBlock::tool_call(ToolCall::new(
                    "tc-1".to_string(),
                    ToolFunction {
                        name: "verify".to_string(),
                        arguments: json!({"answer": 42}),
                    },
                )),
            ],
        ),
        Message::with_content(
            Role::User,
            vec![
                ContentBlock::tool_result(ToolResult {
                    id: "tc-1".to_string(),
                    call_id: None,
                    content: OneOrMany::one(ToolResultContent::text("confirmed")),
                }),
                ContentBlock::text("Thanks! Can you also check 43?"),
            ],
        ),
    ]);

    let json = original.to_json();
    let reparsed = Chat::new(vec![]).from_json(json).unwrap();

    assert_eq!(reparsed.len(), 3);

    let asst = &reparsed.messages[1];
    assert_eq!(asst.role, Role::Assistant);
    assert_eq!(asst.content.len(), 3);
    assert!(asst.has_reasoning());
    assert!(asst.has_tool_calls());

    let user = &reparsed.messages[2];
    assert_eq!(user.role, Role::User);
    assert_eq!(user.content.len(), 2);
    assert!(user.has_tool_results());
}

#[test]
fn multi_turn_json_roundtrip_preserves_earlier_reasoning() {
    let chat = Chat::new(vec![
        Message::system("You are a helpful assistant."),
        Message::user("What is the capital of France?"),
        Message::with_content(
            Role::Assistant,
            vec![
                ContentBlock::reasoning(Reasoning::new("The user is asking about geography.")),
                ContentBlock::text("The capital of France is Paris."),
            ],
        ),
        Message::user("And Germany?"),
        Message::assistant("The capital of Germany is Berlin."),
    ]);

    let reparsed = Chat::new(vec![]).from_json(chat.to_json()).unwrap();
    assert_eq!(reparsed.len(), 5);

    let turn1_reply = &reparsed.messages[2];
    assert_eq!(turn1_reply.role, Role::Assistant);
    assert!(turn1_reply.has_reasoning());
    assert_eq!(turn1_reply.content.len(), 2);
}

#[test]
fn legacy_plain_string_json_is_rejected() {
    let legacy_json = json!([
        {"role": "system", "content": "Be helpful"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]);

    let err = Chat::new(vec![]).from_json(legacy_json).unwrap_err();
    assert!(err.to_string().contains("content must be an array"));
}

#[test]
fn text_content_excludes_non_text_blocks() {
    let msg = Message::with_content(
        Role::Assistant,
        vec![
            ContentBlock::reasoning(Reasoning::new("internal monologue")),
            ContentBlock::text("visible answer"),
            ContentBlock::tool_call(ToolCall::new(
                "tc".to_string(),
                ToolFunction {
                    name: "search".to_string(),
                    arguments: json!({}),
                },
            )),
        ],
    );

    assert_eq!(msg.text_content(), "visible answer");
    let full = msg.content();
    assert!(full.contains("internal monologue"));
    assert!(full.contains("visible answer"));
    assert!(full.contains("search"));
}

#[test]
fn tool_calls_accessor_returns_all_tool_calls() {
    let msg = Message::with_content(
        Role::Assistant,
        vec![
            ContentBlock::reasoning(Reasoning::new("planning")),
            ContentBlock::tool_call(ToolCall::new(
                "tc-1".to_string(),
                ToolFunction {
                    name: "search".to_string(),
                    arguments: json!({"q": "a"}),
                },
            )),
            ContentBlock::tool_call(ToolCall::new(
                "tc-2".to_string(),
                ToolFunction {
                    name: "calculate".to_string(),
                    arguments: json!({"expr": "1+1"}),
                },
            )),
        ],
    );

    let calls = msg.tool_calls();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].function.name, "search");
    assert_eq!(calls[1].function.name, "calculate");
}

#[test]
fn empty_content_message_does_not_panic() {
    let msg = Message::with_content(Role::Assistant, vec![]);
    assert_eq!(msg.content(), "");
    assert_eq!(msg.text_content(), "");
    assert!(!msg.has_tool_calls());
    assert!(!msg.has_reasoning());
}

#[test]
fn message_id_survives_json_roundtrip() {
    let mut msg = Message::assistant("some text");
    msg.id = Some("msg_abc123".to_string());

    let chat = Chat::new(vec![msg]);
    let reparsed = Chat::new(vec![]).from_json(chat.to_json()).unwrap();

    assert_eq!(reparsed.messages.len(), 1);
    assert_eq!(reparsed.messages[0].id, Some("msg_abc123".to_string()));
}
