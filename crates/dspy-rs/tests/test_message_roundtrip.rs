//! Round-trip tests for the new Message model.
//!
//! Verifies that the grouped Role + ContentBlock representation preserves
//! all content through: DSRs Message → rig Message → DSRs Message, and
//! through JSON serialization/deserialization.

use dspy_rs::core::lm::chat::{Chat, ContentBlock, Message, Role};
use rig::OneOrMany;
use rig::message::{
    Message as RigMessage, Reasoning, ToolCall, ToolFunction, ToolResult, ToolResultContent,
};
use serde_json::json;

// ---------------------------------------------------------------------------
// Reasoning continuity round-trip
// ---------------------------------------------------------------------------

/// Anthropic's thinking turns produce [Reasoning, Reasoning, ToolCall] in a
/// single assistant turn. The entire chain of thought must survive:
///   DSRs Message → rig → DSRs Message
#[test]
fn reasoning_chain_survives_rig_roundtrip() {
    let original = Message::with_content(
        Role::Assistant,
        vec![
            ContentBlock::reasoning(Reasoning::new("step 1: analyze the query")),
            ContentBlock::reasoning(Reasoning::new("step 2: plan the search")),
            ContentBlock::tool_call(ToolCall::new(
                "tc-1".to_string(),
                ToolFunction {
                    name: "search".to_string(),
                    arguments: json!({"q": "rust ownership"}),
                },
            )),
        ],
    );

    // Forward: DSRs → rig
    let rig_msg = original
        .to_rig_message()
        .expect("assistant message should convert to rig");

    // Backward: rig → DSRs
    let roundtripped = Message::from(rig_msg);

    assert_eq!(roundtripped.role, Role::Assistant);
    assert_eq!(
        roundtripped.content.len(),
        3,
        "all three content blocks must survive: got {:?}",
        roundtripped.content
    );

    assert!(
        matches!(&roundtripped.content[0], ContentBlock::Reasoning { reasoning } if reasoning.display_text().contains("step 1")),
        "first reasoning block lost"
    );
    assert!(
        matches!(&roundtripped.content[1], ContentBlock::Reasoning { reasoning } if reasoning.display_text().contains("step 2")),
        "second reasoning block lost"
    );
    assert!(
        matches!(&roundtripped.content[2], ContentBlock::ToolCall { tool_call } if tool_call.function.name == "search"),
        "tool call lost"
    );
}

/// A reasoning-only assistant turn (no text, no tool call) must round-trip.
#[test]
fn reasoning_only_turn_roundtrips() {
    let original = Message::with_content(
        Role::Assistant,
        vec![ContentBlock::reasoning(Reasoning::new(
            "just thinking out loud",
        ))],
    );

    let rig_msg = original.to_rig_message().unwrap();
    let roundtripped = Message::from(rig_msg);

    assert_eq!(roundtripped.role, Role::Assistant);
    assert_eq!(roundtripped.content.len(), 1);
    assert!(roundtripped.has_reasoning());
    assert!(!roundtripped.has_tool_calls());
}

// ---------------------------------------------------------------------------
// Multi-content user messages
// ---------------------------------------------------------------------------

/// A user message with both text and a tool result must preserve both.
#[test]
fn user_text_plus_tool_result_roundtrips() {
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

    let rig_msg = original.to_rig_message().unwrap();
    let roundtripped = Message::from(rig_msg);

    assert_eq!(roundtripped.role, Role::User);
    assert_eq!(
        roundtripped.content.len(),
        2,
        "both text and tool result must survive"
    );
    assert!(matches!(
        &roundtripped.content[0],
        ContentBlock::Text { text } if text == "Here is context"
    ));
    assert!(roundtripped.has_tool_results());
}

// ---------------------------------------------------------------------------
// Multi-turn conversation with reasoning in history
// ---------------------------------------------------------------------------

/// Build a multi-turn conversation where an earlier assistant turn has
/// reasoning blocks. Convert the full chat to rig format and back.
/// The reasoning from earlier turns must be preserved.
#[test]
fn multi_turn_conversation_preserves_earlier_reasoning() {
    let chat = Chat::new(vec![
        Message::system("You are a helpful assistant."),
        Message::user("What is the capital of France?"),
        // Turn 1 reply: reasoning + text answer
        Message::with_content(
            Role::Assistant,
            vec![
                ContentBlock::reasoning(Reasoning::new("The user is asking about geography.")),
                ContentBlock::text("The capital of France is Paris."),
            ],
        ),
        // User follow-up
        Message::user("And Germany?"),
        // Turn 2 reply: just text
        Message::assistant("The capital of Germany is Berlin."),
    ]);

    // Convert to rig and back
    let rig_history = chat.to_rig_chat_history();
    // rig_history should have 4 messages (system excluded)
    assert_eq!(rig_history.len(), 4);

    // Reconstruct from rig history
    let mut reconstructed = Chat::new(vec![Message::system(chat.system_prompt())]);
    for rig_msg in rig_history {
        reconstructed.push_message(Message::from(rig_msg));
    }

    assert_eq!(reconstructed.len(), 5);

    // Verify turn 1's reasoning survived
    let turn1_reply = &reconstructed.messages[2];
    assert_eq!(turn1_reply.role, Role::Assistant);
    assert!(
        turn1_reply.has_reasoning(),
        "turn 1 reasoning must survive rig round-trip"
    );
    assert_eq!(
        turn1_reply.content.len(),
        2,
        "turn 1 must have both reasoning and text"
    );
}

// ---------------------------------------------------------------------------
// JSON serialization round-trip
// ---------------------------------------------------------------------------

/// Full multi-content message survives JSON serialization.
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

    // Verify the assistant message preserved all 3 content blocks
    let asst = &reparsed.messages[1];
    assert_eq!(asst.role, Role::Assistant);
    assert_eq!(asst.content.len(), 3);
    assert!(asst.has_reasoning());
    assert!(asst.has_tool_calls());

    // Verify the user message preserved both blocks
    let user = &reparsed.messages[2];
    assert_eq!(user.role, Role::User);
    assert_eq!(user.content.len(), 2);
    assert!(user.has_tool_results());
}

/// Legacy JSON format (content as plain string) still parses correctly.
#[test]
fn legacy_plain_string_json_parses_into_new_model() {
    let legacy_json = json!([
        {"role": "system", "content": "Be helpful"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]);

    let chat = Chat::new(vec![]).from_json(legacy_json).unwrap();
    assert_eq!(chat.len(), 3);
    assert_eq!(chat.messages[0].role, Role::System);
    assert_eq!(chat.messages[0].content(), "Be helpful");
    assert_eq!(chat.messages[2].text_content(), "Hi there!");
}

// ---------------------------------------------------------------------------
// Accessor correctness
// ---------------------------------------------------------------------------

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
    // content() includes everything
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

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

/// Empty content vec (pathological) should not panic.
#[test]
fn empty_content_message_does_not_panic() {
    let msg = Message::with_content(Role::Assistant, vec![]);
    assert_eq!(msg.content(), "");
    assert_eq!(msg.text_content(), "");
    assert!(!msg.has_tool_calls());
    assert!(!msg.has_reasoning());

    // Rig conversion should produce an assistant message with empty text
    let rig_msg = msg.to_rig_message().unwrap();
    match rig_msg {
        RigMessage::Assistant { content, .. } => {
            assert_eq!(content.iter().count(), 1); // empty text fallback
        }
        _ => panic!("expected assistant message"),
    }
}

/// System messages return None from to_rig_message (handled as preamble).
#[test]
fn system_message_excluded_from_rig_conversion() {
    let msg = Message::system("You are helpful");
    assert!(msg.to_rig_message().is_none());
}

/// Message ID (e.g. Anthropic thinking turn IDs) survives round-trip.
#[test]
fn message_id_survives_rig_roundtrip() {
    let mut msg = Message::assistant("some text");
    msg.id = Some("msg_abc123".to_string());

    let rig_msg = msg.to_rig_message().unwrap();
    let roundtripped = Message::from(rig_msg);

    // Note: rig's User messages don't carry IDs, but Assistant messages do
    assert_eq!(roundtripped.id, Some("msg_abc123".to_string()));
}
