pub mod chat;
pub mod config;

pub use chat::*;
pub use config::*;

use bon::Builder;
use anyhow::Result;
use secrecy::{ExposeSecretMut, SecretString};
use async_openai::{Client, config::OpenAIConfig};
use async_openai::types::{CreateChatCompletionRequestArgs};

#[derive(Clone)]
pub struct LMResponse {
    pub chat: Chat,
    pub config: LMConfig,
    pub output: Message,
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

    async fn call(
        &mut self,
        messages: Chat,
        config: LMConfig,
    ) -> Result<Message> {
        let request_messages = messages.get_async_openai_messages();

        let mut builder = CreateChatCompletionRequestArgs::default();

        let request = builder
            .model(config.model)
            .messages(request_messages)
            .temperature(config.temperature)
            .top_p(config.top_p)
            .n(config.n)
            .max_completion_tokens(config.max_completion_tokens)
            .max_tokens(config.max_tokens)
            .presence_penalty(config.presence_penalty)
            .frequency_penalty(config.frequency_penalty)
            .seed(config.seed)
            .logit_bias(config.logit_bias.clone().unwrap_or_default())
            .build()?;

        let response = self.client.chat().create(request).await?;
        let first_choice = Message::from(response.choices.first().unwrap().message.clone());

        self.history.push(LMResponse {
            chat: messages.clone(),
            output: first_choice.clone(),
            config: self.config.clone(),
        });

        Ok(first_choice)
    }

    pub fn inspect_history(&self, n: usize) -> Vec<&LMResponse> {
        self.history.iter().rev().take(n).collect()
    }
}
