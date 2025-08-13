pub mod models;
pub mod providers;

use crate::core::SignatureMetadata;
use crate::providers::ConcreteProvider;
pub use models::*;
pub use providers::*;

use anyhow::Result;
use bon::Builder;

#[derive(Clone, Debug, Builder)]
pub struct LM {
    pub provider: ConcreteProvider,

    #[builder(default = "openai/gpt-4o-mini".to_string())]
    pub model: String,

    #[builder(default = LMConfig::default())]
    pub config: LMConfig,

    #[builder(default = Vec::new())]
    pub history: Vec<LMInvocation>,
}

impl LM {
    pub async fn call(
        &mut self,
        chat: &Chat,
        signature_metadata: SignatureMetadata,
    ) -> Result<Message> {
        let response = self
            .provider
            .complete(chat.clone(), self.config.clone())
            .await?;

        self.history.push(LMInvocation {
            chat: chat.clone(),
            output: response.clone(),
            signature: signature_metadata,
            config: self.config.clone(),
        });

        Ok(response)
    }

    pub fn inspect_history(&self, n: usize) -> Vec<LMInvocation> {
        self.history.iter().rev().take(n).collect()
    }

    pub fn builder() -> LMBuilder {
        LMBuilder::default()
    }
}
