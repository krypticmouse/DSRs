use crate::premitives::lm::LMProvider;
use openai_api_rs::v1::chat_completion::ChatCompletionMessage;

#[derive(Clone, Debug)]
pub struct History {
    pub input: Vec<ChatCompletionMessage>,
    pub output: String,
    pub signature: String,

    pub model: String,
    pub provider: LMProvider,
}
