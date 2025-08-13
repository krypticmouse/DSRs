pub mod dummy;
pub mod openai;

pub use dummy::*;
pub use openai::*;

use crate::core::CompletionProvider;

pub enum ConcreteProvider {
    Dummy(DummyProvider),
    OpenAI(OpenAIProvider),
}

impl CompletionProvider for ConcreteProvider {
    fn complete(
        &self,
        messages: crate::core::Chat,
        config: crate::core::LMConfig,
    ) -> impl Future<Output = anyhow::Result<crate::core::Message>> {
        match self {
            ConcreteProvider::Dummy(provider) => provider.complete(messages, config),
            ConcreteProvider::OpenAI(provider) => provider.complete(messages, config),
        }
    }
}
