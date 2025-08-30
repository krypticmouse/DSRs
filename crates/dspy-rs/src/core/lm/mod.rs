pub mod chat;
pub mod config;
pub mod usage;

pub use chat::*;
pub use config::*;
pub use usage::*;

use anyhow::Result;
use async_openai::types::CreateChatCompletionRequestArgs;
use async_openai::{Client, config::OpenAIConfig};

use bon::Builder;
use secrecy::{ExposeSecretMut, SecretString};

#[derive(Clone)]
pub struct LMResponse {
    pub chat: Chat,
    pub config: LMConfig,
    pub output: Message,
    pub signature: String,
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
    client: Option<Client<OpenAIConfig>>,
}

impl LM {
    fn setup_client(&mut self) {
        let config = OpenAIConfig::new()
            .with_api_key(self.api_key.expose_secret_mut().to_string())
            .with_api_base(self.base_url.clone());

        self.client = Some(Client::with_config(config));
    }

    pub async fn call(&mut self, messages: Chat, signature: &str) -> Result<(Message, LmUsage)> {
        if self.client.is_none() {
            self.setup_client();
        }

        let request_messages = messages.get_async_openai_messages();

        let mut builder = CreateChatCompletionRequestArgs::default();

        let request = builder
            .model(self.config.model.clone())
            .messages(request_messages)
            .temperature(self.config.temperature)
            .top_p(self.config.top_p)
            .n(self.config.n)
            .max_tokens(self.config.max_tokens)
            .presence_penalty(self.config.presence_penalty)
            .frequency_penalty(self.config.frequency_penalty)
            .seed(self.config.seed)
            .logit_bias(self.config.logit_bias.clone().unwrap_or_default())
            .build()?;

        let response = self.client.as_ref().unwrap().chat().create(request).await?;
        let first_choice = Message::from(response.choices.first().unwrap().message.clone());
        let usage = LmUsage::from(response.usage.unwrap());

        self.history.push(LMResponse {
            chat: messages.clone(),
            output: first_choice.clone(),
            config: self.config.clone(),
            signature: signature.to_string(),
        });

        Ok((first_choice, usage))
    }

    pub fn inspect_history(&self, n: usize) -> Vec<&LMResponse> {
        self.history.iter().rev().take(n).collect()
    }
}

#[derive(Clone, Builder, Default)]
pub struct DummyLM {
    pub api_key: SecretString,
    #[builder(default = "https://api.openai.com/v1".to_string())]
    pub base_url: String,
    #[builder(default = LMConfig::default())]
    pub config: LMConfig,
    #[builder(default = Vec::new())]
    pub history: Vec<LMResponse>,
}

impl DummyLM {
    pub async fn call(
        &mut self,
        messages: Chat,
        signature: &str,
        prediction: String,
    ) -> Result<(Message, LmUsage)> {
        self.history.push(LMResponse {
            chat: messages.clone(),
            output: Message::Assistant {
                content: prediction.clone(),
            },
            config: self.config.clone(),
            signature: signature.to_string(),
        });

        Ok((
            Message::Assistant {
                content: prediction.clone(),
            },
            LmUsage::default(),
        ))
    }

    pub fn inspect_history(&self, n: usize) -> Vec<&LMResponse> {
        self.history.iter().rev().take(n).collect()
    }
}
