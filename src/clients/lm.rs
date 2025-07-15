use openrouter_rs::{
    OpenRouterClient,
    api::chat::ChatCompletionRequest,
    types::{CompletionsResponse, ResponseFormat},
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
    pub top_logprobs: Option<u32>,
    pub min_p: Option<f64>,
    pub top_a: Option<f64>,
    pub response_format: Option<ResponseFormat>,
}

#[derive(Clone, Debug, SmartDefault)]
pub struct LM {
    #[default = "https://openrouter.ai/api/v1"]
    pub base_url: String,

    pub api_key: String,
    #[default = "openai/gpt-4o-mini"]
    pub model: String,
    #[default(Vec::new())]
    pub history: Vec<History>,
    #[default(LMConfig::default())]
    pub config: LMConfig,
}

impl LM {
    pub async fn call(
        &mut self,
        chat: &Chat,
        signature: String,
    ) -> Result<CompletionsResponse, Box<dyn Error>> {
        let client = OpenRouterClient::builder()
            .api_key("your_api_key")
            .base_url("https://openrouter.ai/api/v1") // optional
            .http_referer("your_referer") // optional
            .x_title("your_app") // optional
            .build()?;

        let request = ChatCompletionRequest::builder()
            .model(self.model.clone())
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
            .top_logprobs(self.config.top_logprobs.unwrap_or_default())
            .min_p(self.config.min_p.unwrap_or_default())
            .top_a(self.config.top_a.unwrap_or_default())
            .response_format(self.config.response_format.clone().unwrap_or_default())
            .build()?;

        let response = client.send_chat_completion(&request).await?;

        self.history.push(History {
            input: chat.clone(),
            output: response.clone(),
            signature,
            model: self.model.clone(),
        });

        Ok(response)
    }

    pub fn inspect_history(&self, n: usize) -> Vec<&History> {
        self.history.iter().rev().take(n).collect()
    }
}
