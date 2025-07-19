use dspy_rs::clients::chat::Chat;
use openrouter_rs::{api::chat::Message, types::Role};
use rstest::*;

#[rstest]
fn test_chat_init() {
    let chat = Chat::new(vec![
        Message {
            role: Role::System,
            content: "You are a helpful assistant.".to_string(),
        },
        Message {
            role: Role::User,
            content: "Hello, world!".to_string(),
        },
        Message {
            role: Role::Assistant,
            content: "Hello, world to you!".to_string(),
        },
    ]);

    assert_eq!(chat.len(), 3);
    assert_eq!(chat.messages[0].role.to_string(), Role::System.to_string());
    assert_eq!(
        chat.messages[0].content,
        "You are a helpful assistant.".to_string()
    );
    assert_eq!(chat.messages[1].role.to_string(), Role::User.to_string());
    assert_eq!(chat.messages[1].content, "Hello, world!".to_string());
    assert_eq!(
        chat.messages[2].role.to_string(),
        Role::Assistant.to_string()
    );
    assert_eq!(chat.messages[2].content, "Hello, world to you!".to_string());
}

#[rstest]
fn test_chat_push() {
    let mut chat = Chat::new(vec![]);
    chat.push(Role::User, "Hello, world!".to_string());
    assert_eq!(chat.len(), 1);
    assert_eq!(chat.messages[0].role.to_string(), Role::User.to_string());
    assert_eq!(chat.messages[0].content, "Hello, world!".to_string());
}

#[rstest]
fn test_chat_pop() {
    let mut chat = Chat::new(vec![]);
    chat.push(Role::User, "Hello, world!".to_string());
    chat.pop();
    assert_eq!(chat.len(), 0);
}

#[rstest]
fn test_chat_to_json() {
    let chat = Chat::new(vec![
        Message {
            role: Role::System,
            content: "You are a helpful assistant.".to_string(),
        },
        Message {
            role: Role::User,
            content: "Hello, world!".to_string(),
        },
        Message {
            role: Role::Assistant,
            content: "Hello, world to you!".to_string(),
        },
    ]);
    let json = chat.to_json().unwrap();
    assert_eq!(
        json,
        "[{\"role\":\"system\",\"content\":\"You are a helpful assistant.\"},{\"role\":\"user\",\"content\":\"Hello, world!\"},{\"role\":\"assistant\",\"content\":\"Hello, world to you!\"}]"
    );
}

#[rstest]
fn test_chat_from_json() {
    let json = "[{\"role\":\"system\",\"content\":\"You are a helpful assistant.\"},{\"role\":\"user\",\"content\":\"Hello, world!\"},{\"role\":\"assistant\",\"content\":\"Hello, world to you!\"}]";
    let chat = Chat::from_json(json).unwrap();
    assert_eq!(chat.len(), 3);
    assert_eq!(chat.messages[0].role.to_string(), Role::System.to_string());
    assert_eq!(
        chat.messages[0].content,
        "You are a helpful assistant.".to_string()
    );
    assert_eq!(chat.messages[1].role.to_string(), Role::User.to_string());
    assert_eq!(chat.messages[1].content, "Hello, world!".to_string());
    assert_eq!(
        chat.messages[2].role.to_string(),
        Role::Assistant.to_string()
    );
    assert_eq!(chat.messages[2].content, "Hello, world to you!".to_string());
}
