use dspy_rs::{
    ChatAdapter, LM, LMClient, Message, Predict, Role, Signature, TestCompletionModel, configure,
};
use rig::completion::{AssistantContent, CompletionRequest};
use rig::message::{Message as RigMessage, Text, UserContent};
use std::sync::LazyLock;
use tokio::sync::Mutex;

static SETTINGS_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

fn response_with_fields(fields: &[(&str, &str)]) -> String {
    let mut response = String::new();
    for (name, value) in fields {
        response.push_str(&format!("[[ ## {name} ## ]]\n{value}\n\n"));
    }
    response.push_str("[[ ## completed ## ]]\n");
    response
}

fn text_response(text: impl Into<String>) -> AssistantContent {
    AssistantContent::Text(Text { text: text.into() })
}

async fn configure_test_lm(responses: Vec<String>) -> TestCompletionModel {
    let client = TestCompletionModel::new(responses.into_iter().map(text_response));
    let lm = temp_env::async_with_vars(
        [("OPENAI_API_KEY", Some("test"))],
        LM::builder()
            .model("openai:gpt-4o-mini".to_string())
            .build(),
    )
    .await
    .unwrap()
    .with_client(LMClient::Test(client.clone()))
    .await
    .unwrap();

    configure(lm, ChatAdapter {});

    client
}

fn request_contains_text(request: &CompletionRequest, needle: &str) -> bool {
    if request
        .preamble
        .as_ref()
        .is_some_and(|preamble| preamble.contains(needle))
    {
        return true;
    }

    for message in request.chat_history.iter() {
        match message {
            RigMessage::User { content } => {
                for item in content.iter() {
                    if let UserContent::Text(text) = item
                        && text.text.contains(needle)
                    {
                        return true;
                    }
                }
            }
            RigMessage::Assistant { content, .. } => {
                for item in content.iter() {
                    match item {
                        AssistantContent::Text(text) if text.text.contains(needle) => return true,
                        AssistantContent::Reasoning(reasoning)
                            if reasoning.display_text().contains(needle) =>
                        {
                            return true;
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    false
}

#[derive(Signature, Clone, Debug, PartialEq)]
/// Conversational QA test signature.
struct ConversationQA {
    #[input]
    question: String,

    #[output]
    answer: String,
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn forward_returns_chat_and_prediction() {
    let _lock = SETTINGS_LOCK.lock().await;
    let response = response_with_fields(&[("answer", "Paris")]);
    let _client = configure_test_lm(vec![response]).await;

    let predict = Predict::<ConversationQA>::new();
    let input = ConversationQAInput {
        question: "What is the capital of France?".to_string(),
    };

    let chat = predict
        .build_chat(&input)
        .expect("build_chat should succeed");
    let (predicted, chat) = predict
        .call_and_parse(chat)
        .await
        .expect("first turn should succeed");

    assert_eq!(predicted.into_inner().answer, "Paris");
    assert_eq!(chat.len(), 3);
    assert_eq!(chat.messages[0].role, Role::System);
    assert_eq!(chat.messages[1].role, Role::User);
    assert_eq!(chat.messages[2].role, Role::Assistant);
}

#[cfg_attr(miri, ignore = "MIRI has issues with tokio's I/O driver")]
#[tokio::test]
async fn forward_continue_supports_two_turn_roundtrip() {
    let _lock = SETTINGS_LOCK.lock().await;
    let first_response = response_with_fields(&[("answer", "First turn answer")]);
    let second_response = response_with_fields(&[("answer", "Second turn answer")]);
    let client = configure_test_lm(vec![first_response, second_response]).await;

    let predict = Predict::<ConversationQA>::new();
    let first_input = ConversationQAInput {
        question: "Turn 1 question".to_string(),
    };

    // First turn: build fresh chat
    let chat = predict
        .build_chat(&first_input)
        .expect("build_chat should succeed");
    let (first_predicted, mut chat) = predict
        .call_and_parse(chat)
        .await
        .expect("first turn should succeed");
    assert_eq!(first_predicted.into_inner().answer, "First turn answer");

    // Second turn: append follow-up, continue conversation
    let caller_follow_up = "Caller follow-up message";
    chat.push_message(Message::user(caller_follow_up));

    let (second_predicted, second_chat) = predict
        .forward_continue(chat)
        .await
        .expect("second turn should succeed");

    assert_eq!(second_predicted.into_inner().answer, "Second turn answer");
    assert!(second_chat.len() >= 5);

    // Verify the follow-up text was sent to the LM
    let last_request = client
        .last_request()
        .expect("test model should capture last request");
    assert!(request_contains_text(&last_request, caller_follow_up));
}
