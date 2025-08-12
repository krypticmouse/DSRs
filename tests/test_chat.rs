use async_openai::types::{
    ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
};
use dspy_rs::clients::chat::Chat;
use rstest::*;

#[rstest]
fn test_chat_init() {
    let chat = Chat::new(vec![
        ChatCompletionRequestMessage::System(
            ChatCompletionRequestSystemMessageArgs::default()
                .content("You are a helpful assistant.".to_string())
                .build()
                .unwrap(),
        ),
        ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessageArgs::default()
                .content("Hello, world!".to_string())
                .build()
                .unwrap(),
        ),
        ChatCompletionRequestMessage::Assistant(
            ChatCompletionRequestAssistantMessageArgs::default()
                .content("Hello, world to you!".to_string())
                .build()
                .unwrap(),
        ),
    ]);

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
    assert_eq!(json[2]["role"], "assistant");
    assert_eq!(json[2]["content"], "Hello, world to you!".to_string());
}

#[rstest]
fn test_chat_push() {
    let mut chat = Chat::new(vec![]);
    chat.push("user", "Hello, world!".to_string());

    let json_value = chat.to_json();
    let json = json_value.as_array().unwrap();
    assert_eq!(json.len(), 1);
    assert_eq!(json[0]["role"], "user");
    assert_eq!(json[0]["content"], "Hello, world!".to_string());
}

#[rstest]
fn test_chat_pop() {
    let mut chat = Chat::new(vec![]);
    chat.push("user", "Hello, world!".to_string());
    chat.pop();

    let json_value = chat.to_json();
    let json = json_value.as_array().unwrap();
    assert_eq!(json.len(), 0);
}

#[rstest]
fn test_chat_to_json() {
    let chat = Chat::new(vec![
        ChatCompletionRequestMessage::System(
            ChatCompletionRequestSystemMessageArgs::default()
                .content("You are a helpful assistant.".to_string())
                .build()
                .unwrap(),
        ),
        ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessageArgs::default()
                .content("Hello, world!".to_string())
                .build()
                .unwrap(),
        ),
        ChatCompletionRequestMessage::Assistant(
            ChatCompletionRequestAssistantMessageArgs::default()
                .content("Hello, world to you!".to_string())
                .build()
                .unwrap(),
        ),
    ]);
    let json = chat.to_json();
    assert_eq!(
        json.to_string(),
        "[{\"content\":\"You are a helpful assistant.\",\"role\":\"system\"},{\"content\":\"Hello, world!\",\"role\":\"user\"},{\"content\":\"Hello, world to you!\",\"role\":\"assistant\"}]"
    );
}

#[rstest]
fn test_chat_from_json() {
    let json = "[{\"role\":\"system\",\"content\":\"You are a helpful assistant.\"},{\"role\":\"user\",\"content\":\"Hello, world!\"},{\"role\":\"assistant\",\"content\":\"Hello, world to you!\"}]";
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
