use crate::core::{Chat, CompletionProvider};
use anyhow::Result;

#[derive(Clone, Debug)]
pub struct DummyProvider;

impl CompletionProvider for DummyProvider {
    fn complete(
        &self,
        _messages: Chat,
        _config: crate::core::LMConfig,
    ) -> impl Future<Output = Result<crate::core::Message>> {
        async move {
            Ok(crate::core::Message::assistant(
                Some("dummy response"),
                None,
            ))
        }
    }
}
