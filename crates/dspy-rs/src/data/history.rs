use crate::clients::chat::Chat;
use async_openai::types::CreateChatCompletionResponse;

#[derive(Clone, Debug)]
pub struct History {
    pub input: Chat,
    pub output: CreateChatCompletionResponse,
    pub signature: String,

    pub model: String,
}
