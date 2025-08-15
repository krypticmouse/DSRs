pub mod chat;
pub mod config;

pub use chat::*;
pub use config::*;

use anyhow::Result;
use async_openai::types::CreateChatCompletionRequestArgs;
use async_openai::{Client, config::OpenAIConfig};
use bon::Builder;
use secrecy::{ExposeSecretMut, SecretString};

use crate::core::SignatureMetadata;

#[derive(Clone)]
pub struct LMResponse {
    pub chat: Chat,
    pub config: LMConfig,
    pub output: Message,
    pub signature_metadata: SignatureMetadata, // TODO: remove this, we just need enough data for DAG curation
}

#[derive(Clone, Builder)]
pub struct LM {
    pub api_key: SecretString,
    #[builder(default = "https://api.openai.com/v1".to_string())]
    pub base_url: String,
    #[builder(default = LMConfig::default())]
    pub config: LMConfig,
    #[builder(default = Vec::new())]
    pub history: Vec<LMResponse>,
    client: Client<OpenAIConfig>,
}

impl LM {
    fn setup_client(&mut self) {
        let config = OpenAIConfig::new()
            .with_api_key(self.api_key.expose_secret_mut().to_string())
            .with_api_base(self.base_url.clone());

        self.client = Client::with_config(config);
    }

    pub async fn call(
        &mut self,
        messages: Chat,
        signature_metadata: SignatureMetadata,
    ) -> Result<Message> {
        let request_messages = messages.get_async_openai_messages();

        let mut builder = CreateChatCompletionRequestArgs::default();

        let request = builder
            .model(self.config.model.clone())
            .messages(request_messages)
            .temperature(self.config.temperature)
            .top_p(self.config.top_p)
            .n(self.config.n)
            .max_completion_tokens(self.config.max_completion_tokens)
            .max_tokens(self.config.max_tokens)
            .presence_penalty(self.config.presence_penalty)
            .frequency_penalty(self.config.frequency_penalty)
            .seed(self.config.seed)
            .logit_bias(self.config.logit_bias.clone().unwrap_or_default())
            .build()?;

        let response = self.client.chat().create(request).await?;
        let first_choice = Message::from(response.choices.first().unwrap().message.clone());

        self.history.push(LMResponse {
            chat: messages.clone(),
            output: first_choice.clone(),
            config: self.config.clone(),
            signature_metadata: signature_metadata.clone(),
        });

        Ok(first_choice)
    }

    pub fn inspect_history(&self, n: usize) -> Vec<&LMResponse> {
        self.history.iter().rev().take(n).collect()
    }
}
