use crate::clients::chat::Chat;
use openrouter_rs::types::CompletionsResponse;

#[derive(Clone, Debug)]
pub struct History {
    pub input: Chat,
    pub output: CompletionsResponse,
    pub signature: String,

    pub model: String,
}
