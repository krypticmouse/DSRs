use super::{Chat, LMConfig, Message};
use anyhow::Result;

pub trait CompletionProvider {
    fn complete(&self, messages: Chat, config: LMConfig) -> impl Future<Output = Result<Message>>;
}
