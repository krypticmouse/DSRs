use dspy_rs::core::lm::chat::{Chat, ContentBlock, Message, Role};
use rig::OneOrMany;
use rig::message::{
    AssistantContent, Message as RigMessage, Reasoning, ToolCall, ToolFunction, ToolResult,
    ToolResultContent, UserContent,
};
use rstest::*;
use serde_json::json;

#[rstest]
fn test_chat_init() {
    let chat = Chat::new(vec![
        Message::system("You are a helpful assistant."),
        Message::user("Hello, world!"),
        Message::assistant("Hello, world to you!"),
    ]);

    assert_eq!(chat.len(), 3);
    assert!(!chat.is_empty());
    assert_eq!(chat.messages[0].role, Role::System);
    assert_eq!(chat.messages[0].content(), "You are a helpful assistant.");
    assert_eq!(chat.messages[1].role, Role::User);
    assert_eq!(chat.messages[1].content(), "Hello, world!");
    assert_eq!(chat.messages[2].role, Role::Assistant);
    assert_eq!(chat.messages[2].content(), "Hello, world to you!");
}

#[rstest]
fn test_chat_push() {
    let mut chat = Chat::new(vec![]);
    chat.push(Role::User, "Hello, world!");

    assert_eq!(chat.len(), 1);
    assert_eq!(chat.messages[0].role, Role::User);
    assert_eq!(chat.messages[0].content(), "Hello, world!");
}

#[rstest]
fn test_chat_pop() {
    let mut chat = Chat::new(vec![]);
    chat.push(Role::User, "Hello, world!");
    chat.pop();

    assert_eq!(chat.len(), 0);
}

#[rstest]
fn test_chat_to_json_and_back() {
    let chat = Chat::new(vec![
        Message::system("You are a helpful assistant."),
        Message::user("Hello, world!"),
        Message::assistant("Hello, world to you!"),
    ]);
    let json_dump = chat.to_json();
    let reparsed = Chat::new(vec![]).from_json(json_dump).unwrap();

    assert_eq!(reparsed.len(), 3);
    assert_eq!(reparsed.messages[0].role, Role::System);
    assert_eq!(
        reparsed.messages[0].content(),
        "You are a helpful assistant."
    );
    assert_eq!(reparsed.messages[1].role, Role::User);
    assert_eq!(reparsed.messages[1].content(), "Hello, world!");
    assert_eq!(reparsed.messages[2].role, Role::Assistant);
    assert_eq!(reparsed.messages[2].content(), "Hello, world to you!");
}

#[rstest]
fn test_chat_from_json_requires_grouped_content() {
    let json = json!([
        {"role":"system","content":"You are a helpful assistant."},
        {"role":"user","content":"Hello, world!"},
        {"role":"assistant","content":"Hello, world to you!"}
    ]);
    let err = Chat::new(vec![]).from_json(json).unwrap_err();
    assert!(err.to_string().contains("content must be an array"));
}

#[rstest]
fn test_chat_push_all() {
    let mut chat1 = Chat::new(vec![
        Message::system("You are a helpful assistant."),
        Message::user("Hello!"),
    ]);

    let chat2 = Chat::new(vec![
        Message::assistant("Hi there!"),
        Message::user("How are you?"),
        Message::assistant("I'm doing well, thank you!"),
    ]);

    chat1.push_all(&chat2);

    assert_eq!(chat1.len(), 5);
    assert_eq!(chat1.messages[0].role, Role::System);
    assert_eq!(chat1.messages[0].content(), "You are a helpful assistant.");
    assert_eq!(chat1.messages[1].role, Role::User);
    assert_eq!(chat1.messages[1].content(), "Hello!");
    assert_eq!(chat1.messages[2].role, Role::Assistant);
    assert_eq!(chat1.messages[2].content(), "Hi there!");
    assert_eq!(chat1.messages[3].role, Role::User);
    assert_eq!(chat1.messages[3].content(), "How are you?");
    assert_eq!(chat1.messages[4].role, Role::Assistant);
    assert_eq!(chat1.messages[4].content(), "I'm doing well, thank you!");
}

#[rstest]
fn test_chat_push_all_empty() {
    let mut chat1 = Chat::new(vec![Message::system("System message")]);

    let empty_chat = Chat::new(vec![]);
    chat1.push_all(&empty_chat);

    assert_eq!(chat1.len(), 1);
    assert_eq!(chat1.messages[0].role, Role::System);
    assert_eq!(chat1.messages[0].content(), "System message");
}

#[rstest]
fn test_new_variants_round_trip_json() {
    let call = ToolCall::new(
        "call-1".to_string(),
        ToolFunction {
            name: "lookup".to_string(),
            arguments: json!({ "query": "rust" }),
        },
    );
    let result = ToolResult {
        id: "call-1".to_string(),
        call_id: Some("provider-call-1".to_string()),
        content: OneOrMany::one(ToolResultContent::text("result payload")),
    };
    let reasoning = Reasoning::new("thinking...");

    let chat = Chat::new(vec![
        Message::system("You are a tool-using assistant."),
        Message::tool_call(call.clone()),
        Message::tool_result(result.clone()),
        Message::reasoning(reasoning.clone()),
    ]);

    let json_dump = chat.to_json();
    let reparsed = Chat::new(vec![]).from_json(json_dump).unwrap();
    assert_eq!(reparsed.len(), 4);

    assert_eq!(reparsed.messages[0].role, Role::System);

    assert_eq!(reparsed.messages[1].role, Role::Assistant);
    assert!(reparsed.messages[1].has_tool_calls());
    let reparsed_calls = reparsed.messages[1].tool_calls();
    assert_eq!(reparsed_calls[0].function.name, call.function.name);

    assert_eq!(reparsed.messages[2].role, Role::User);
    assert!(reparsed.messages[2].has_tool_results());

    assert_eq!(reparsed.messages[3].role, Role::Assistant);
    assert!(reparsed.messages[3].has_reasoning());
}

#[rstest]
fn test_from_rig_message_preserves_all_content() {
    // User with text + tool result — both preserved
    let user_msg = RigMessage::User {
        content: OneOrMany::many(vec![
            UserContent::text("some context"),
            UserContent::ToolResult(ToolResult {
                id: "id-1".to_string(),
                call_id: None,
                content: OneOrMany::one(ToolResultContent::text("ok")),
            }),
        ])
        .unwrap(),
    };
    let converted = Message::from(user_msg);
    assert_eq!(converted.role, Role::User);
    assert_eq!(converted.content.len(), 2);
    assert!(matches!(converted.content[0], ContentBlock::Text { .. }));
    assert!(matches!(
        converted.content[1],
        ContentBlock::ToolResult { .. }
    ));

    // Assistant with reasoning + tool call — both preserved (was lossy before)
    let assistant_msg = RigMessage::Assistant {
        id: Some("asst-123".to_string()),
        content: OneOrMany::many(vec![
            AssistantContent::Reasoning(Reasoning::new("step by step")),
            AssistantContent::ToolCall(ToolCall::new(
                "tool-2".to_string(),
                ToolFunction {
                    name: "search".to_string(),
                    arguments: json!({ "q": "x" }),
                },
            )),
        ])
        .unwrap(),
    };
    let converted = Message::from(assistant_msg);
    assert_eq!(converted.role, Role::Assistant);
    assert_eq!(converted.id, Some("asst-123".to_string()));
    assert_eq!(converted.content.len(), 2);
    assert!(converted.has_reasoning());
    assert!(converted.has_tool_calls());
}

#[rstest]
fn test_text_content_filters_non_text_blocks() {
    let msg = Message::with_content(
        Role::Assistant,
        vec![
            ContentBlock::reasoning(Reasoning::new("thinking")),
            ContentBlock::text("the answer is 42"),
        ],
    );

    // text_content() returns only Text blocks
    assert_eq!(msg.text_content(), "the answer is 42");
    // content() returns everything
    assert!(msg.content().contains("thinking"));
    assert!(msg.content().contains("the answer is 42"));
}
