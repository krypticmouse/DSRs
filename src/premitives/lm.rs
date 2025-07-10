use openai_api_rs::v1::api::OpenAIClient;
use openai_api_rs::v1::chat_completion::{ChatCompletionMessage, ChatCompletionRequest};

use crate::data::history::History;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum LMProvider {
    OpenAI,
    Anthropic,
    VertexAI,
    DeepSeek,
    XAI,
    Phind,
    Groq,
    Mistral,
    Cohere,
    Anyscale,
    ElevenLabs,
    Perplexity,
    FireworksAI,
    DeepInfra,
    TogetherAI,
    OpenRouter,
    AI21,
    VoyageAI,
    JinaAI,
    AlephAlpha,
    Baseten,
    Petals,
    NLPCloud,
    Replicate,
    SambaNova,
    Clarifai,
}

pub fn get_base_url(provider: LMProvider, base_url: Option<String>) -> String {
    let base_url = base_url.unwrap_or_default();

    match provider {
        LMProvider::OpenAI => "https://api.openai.com/v1".to_string(),
        LMProvider::Anthropic => "https://api.anthropic.com/v1".to_string(),
        LMProvider::DeepSeek => "https://api.deepseek.com/v1".to_string(),
        LMProvider::XAI => "https://api.x.ai/v1".to_string(),
        LMProvider::Phind => "https://api.phind.com/v1".to_string(),
        LMProvider::Groq => "https://api.groq.com/openai/v1".to_string(),
        LMProvider::Mistral => "https://api.mistral.ai/v1".to_string(),
        LMProvider::Cohere => "https://api.cohere.com/v2".to_string(),
        LMProvider::Anyscale => "https://app.endpoints.anyscale.com".to_string(),
        LMProvider::ElevenLabs => "https://api.elevenlabs.io/v1".to_string(),
        LMProvider::Perplexity => "https://api.perplexity.ai".to_string(),
        LMProvider::FireworksAI => "https://api.fireworks.ai/v1".to_string(),
        LMProvider::DeepInfra => "https://api.deepinfra.com/v1".to_string(),
        LMProvider::TogetherAI => "https://api.together.xyz/v1".to_string(),
        LMProvider::OpenRouter => "https://openrouter.ai/api/v1".to_string(),
        LMProvider::AI21 => "https://api.ai21.com/studio/v1".to_string(),
        LMProvider::VoyageAI => "https://api.voyageai.com/v1".to_string(),
        LMProvider::JinaAI => "https://api.jina.ai/v1".to_string(),
        LMProvider::AlephAlpha => "https://api.aleph-alpha.com/v1".to_string(),
        LMProvider::Baseten => "https://app.baseten.co/v1".to_string(),
        LMProvider::Petals => "https://api.petals.dev/v1".to_string(),
        LMProvider::NLPCloud => "https://api.nlpcloud.io/v1".to_string(),
        LMProvider::Replicate => "https://api.replicate.com/v1".to_string(),
        LMProvider::SambaNova => "https://cloud.sambanova.ai/api/v1".to_string(),
        LMProvider::Clarifai => "https://api.clarifai.com/v2".to_string(),
        _ => base_url,
    }
}

#[derive(Clone, Debug)]
pub struct LMConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub max_tokens: u32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub stop: Option<String>,
    pub n: u32,
}

impl Default for LMConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 1.0,
            max_tokens: 1000,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            stop: None,
            n: 1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LM {
    pub provider: LMProvider,
    pub base_url: Option<String>,
    pub api_key: String,
    pub model: String,
    pub lm_config: LMConfig,
    pub history: Vec<History>,
}

impl LM {
    pub fn new(
        provider: Option<LMProvider>,
        api_key: String,
        model: String,
        base_url: Option<String>,
        lm_config: LMConfig,
    ) -> Self {
        let history = Vec::<History>::new();

        assert!(
            provider.is_some() || base_url.is_some(),
            "Either provider or base_url must be provided"
        );

        let provider = provider.unwrap();
        let base_url_str = get_base_url(provider, base_url);

        Self {
            provider,
            base_url: Some(base_url_str),
            api_key,
            model,
            lm_config,
            history,
        }
    }

    pub async fn forward(
        &mut self,
        messages: Vec<ChatCompletionMessage>,
        signature: String,
    ) -> String {
        let endpoint = self
            .base_url
            .clone()
            .unwrap_or_else(|| get_base_url(self.provider, None));

        assert!(!endpoint.is_empty(), "Base URL is not set");

        let mut client = OpenAIClient::builder()
            .with_endpoint(&endpoint)
            .with_api_key(self.api_key.clone())
            .build()
            .unwrap();

        let request = ChatCompletionRequest {
            model: self.model.clone(),
            messages: messages.clone(),
            temperature: Some(self.lm_config.temperature as f64),
            top_p: Some(self.lm_config.top_p as f64),
            max_tokens: Some(self.lm_config.max_tokens as i64),
            presence_penalty: Some(self.lm_config.presence_penalty as f64),
            frequency_penalty: Some(self.lm_config.frequency_penalty as f64),
            stop: self.lm_config.stop.as_ref().map(|s| vec![s.clone()]),
            n: Some(self.lm_config.n as i64),
            logit_bias: None,
            user: None,
            seed: None,
            tools: None,
            parallel_tool_calls: None,
            tool_choice: None,
            response_format: None,
            stream: None,
        };

        let response = client.chat_completion(request).await.unwrap();
        let output = response.choices[0]
            .message
            .content
            .clone()
            .unwrap_or_default();

        self.history.push(History {
            input: messages,
            output: output.clone(),
            signature,
            model: self.model.clone(),
            provider: self.provider,
        });

        output
    }

    pub fn inspect_history(&self, n: usize) -> Vec<&History> {
        self.history.iter().rev().take(n).collect()
    }
}
