use dspy_rs::core::{Chat, Message};
use rstest::*;
use serde_json::json;

#[rstest]
fn test_chat_init() {
    let chat = Chat::new(vec![
        Message::system("You are a helpful assistant."),
        Message::user("Hello, world!"),
        Message::assistant("Hello, world to you!"),
    ]);

    let json_value = chat.to_json();
    let json = json_value.as_array().unwrap();

    assert_eq!(chat.len(), 3);
    assert_eq!(json[0]["role"], "system");
    assert!(!chat.is_empty());
    assert_eq!(
        json[0]["content"],
        "You are a helpful assistant.".to_string()
    );
    assert_eq!(json[1]["role"], "user");
    assert_eq!(json[1]["content"], "Hello, world!".to_string());
    assert_eq!(json[2]["role"], "assistant");
    assert_eq!(json[2]["content"], "Hello, world to you!".to_string());
}

#[rstest]
fn test_chat_push() {
    let mut chat = Chat::new(vec![]);
    chat.push("user", "Hello, world!");

    let json_value = chat.to_json();
    let json = json_value.as_array().unwrap();
    assert_eq!(json.len(), 1);
    assert_eq!(json[0]["role"], "user");
    assert_eq!(json[0]["content"], "Hello, world!".to_string());
}

#[rstest]
fn test_chat_pop() {
    let mut chat = Chat::new(vec![]);
    chat.push("user", "Hello, world!");
    chat.pop();

    let json_value = chat.to_json();
    let json = json_value.as_array().unwrap();
    assert_eq!(json.len(), 0);
}

#[rstest]
fn test_chat_to_json() {
    let chat = Chat::new(vec![
        Message::system("You are a helpful assistant."),
        Message::user("Hello, world!"),
        Message::assistant("Hello, world to you!"),
    ]);
    let json = chat.to_json();
    assert_eq!(
        json.to_string(),
        "[{\"role\":\"system\",\"content\":\"You are a helpful assistant.\"},{\"role\":\"user\",\"content\":\"Hello, world!\"},{\"role\":\"assistant\",\"content\":\"Hello, world to you!\"}]"
    );
}

#[rstest]
fn test_chat_from_json() {
    let json = json!([
        {"role":"system","content":"You are a helpful assistant."},
        {"role":"user","content":"Hello, world!"},
        {"role":"assistant","content":"Hello, world to you!"}
    ]);
    let empty_chat = Chat::new(vec![]);
    let chat = empty_chat.from_json(json).unwrap();

    let json_value = chat.to_json();
    let json = json_value.as_array().unwrap();

    assert_eq!(chat.len(), 3);
    assert_eq!(json[0]["role"], "system");
    assert_eq!(
        json[0]["content"],
        "You are a helpful assistant.".to_string()
    );
    assert_eq!(json[1]["role"], "user");
    assert_eq!(json[1]["content"], "Hello, world!".to_string());
    assert_eq!(json[2]["content"], "Hello, world to you!".to_string());
}
