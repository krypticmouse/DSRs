use openrouter_rs::{
    OpenRouterClient, api::chat::ChatCompletionRequest, types::CompletionsResponse,
};
use smart_default::SmartDefault;

use crate::{clients::chat::Chat, data::history::History};
use std::collections::HashMap;
use std::error::Error;

#[derive(Clone, Debug, SmartDefault)]
pub struct LMConfig {
    #[default = 0.7]
    pub temperature: f64,
    #[default = 1.0]
    pub top_p: f64,
    #[default = 512]
    pub max_tokens: u32,
    #[default = 0.0]
    pub presence_penalty: f64,
    #[default = 0.0]
    pub frequency_penalty: f64,
    pub stream: Option<bool>,
    pub seed: Option<u32>,
    pub top_k: Option<u32>,
    pub repetition_penalty: Option<f64>,
    pub logit_bias: Option<HashMap<String, f64>>,
    pub min_p: Option<f64>,
    pub top_a: Option<f64>,
}

#[derive(Clone, Debug, SmartDefault)]
pub struct LM<'a> {
    #[default = "https://openrouter.ai/api/v1"]
    pub base_url: &'a str,

    pub api_key: &'a str,
    #[default = "openai/gpt-4o-mini"]
    pub model: &'a str,
    #[default(Vec::new())]
    pub history: Vec<History<'a>>,
    #[default(LMConfig::default())]
    pub config: LMConfig,
}

impl<'a> LM<'a> {
    pub async fn call(
        &mut self,
        chat: &Chat,
        signature: &'a str,
    ) -> Result<CompletionsResponse, Box<dyn Error>> {
        let client = OpenRouterClient::builder()
            .api_key(self.api_key)
            .base_url(self.base_url)
            .build()?;

        let request = ChatCompletionRequest::builder()
            .model(self.model)
            .messages(chat.messages.clone())
            .temperature(self.config.temperature)
            .top_p(self.config.top_p)
            .max_tokens(self.config.max_tokens)
            .presence_penalty(self.config.presence_penalty)
            .frequency_penalty(self.config.frequency_penalty)
            .seed(self.config.seed.unwrap_or_default())
            .top_k(self.config.top_k.unwrap_or_default())
            .repetition_penalty(self.config.repetition_penalty.unwrap_or_default())
            .logit_bias(self.config.logit_bias.clone().unwrap_or_default())
            .min_p(self.config.min_p.unwrap_or_default())
            .top_a(self.config.top_a.unwrap_or_default())
            .build()?;

        let response = client.send_chat_completion(&request).await?;

        self.history.push(History {
            input: chat.clone(),
            output: response.clone(),
            signature,
            model: self.model,
        });

        Ok(response)
    }

    pub fn inspect_history(&self, n: usize) -> Vec<&History> {
        self.history.iter().rev().take(n).collect()
    }
}
