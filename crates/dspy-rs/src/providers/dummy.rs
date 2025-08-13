use crate::core::{Chat, CompletionProvider};
use anyhow::Result;

#[derive(Clone, Debug)]
pub struct DummyProvider;

#[allow(deprecated)]
impl CompletionProvider for DummyProvider {
    fn complete(
        &self,
        messages: Chat,
        config: crate::core::LMConfig,
    ) -> impl Future<Output = Result<crate::core::Message>> {
        async move { Ok(crate::core::Message::assistant("dummy response", None)) }
    }
}
