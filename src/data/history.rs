use crate::clients::chat::Chat;
use openrouter_rs::types::CompletionsResponse;

#[derive(Clone, Debug)]
pub struct History<'a> {
    pub input: Chat,
    pub output: CompletionsResponse,
    pub signature: &'a str,

    pub model: &'a str,
}
