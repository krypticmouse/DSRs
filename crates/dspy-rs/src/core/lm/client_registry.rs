use anyhow::Result;
use enum_dispatch::enum_dispatch;
use rig::{
    completion::{CompletionError, CompletionRequest, CompletionResponse},
    providers::{anthropic, cohere, gemini, groq, openai, perplexity},
};
use reqwest;

#[enum_dispatch]
#[allow(async_fn_in_trait)]
pub trait CompletionProvider {
    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<()>, CompletionError>;
}

#[enum_dispatch(CompletionProvider)]
#[derive(Clone)]
pub enum LMClient {
    OpenAI(openai::completion::CompletionModel),
    Anthropic(anthropic::completion::CompletionModel),
    Cohere(cohere::completion::CompletionModel),
    Gemini(gemini::completion::CompletionModel),
    Groq(groq::CompletionModel<reqwest::Client>),
    Perplexity(perplexity::CompletionModel<reqwest::Client>),
}

// Implement the trait for each concrete provider type using the CompletionModel trait from rig
impl CompletionProvider for openai::completion::CompletionModel {
    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<()>, CompletionError> {
        let response = rig::completion::CompletionModel::completion(self, request).await?;
        // Convert the typed response to unit type
        Ok(CompletionResponse {
            choice: response.choice,
            usage: response.usage,
            raw_response: (),
        })
    }
}

impl CompletionProvider for anthropic::completion::CompletionModel {
    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<()>, CompletionError> {
        let response = rig::completion::CompletionModel::completion(self, request).await?;
        Ok(CompletionResponse {
            choice: response.choice,
            usage: response.usage,
            raw_response: (),
        })
    }
}

impl CompletionProvider for cohere::completion::CompletionModel {
    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<()>, CompletionError> {
        let response = rig::completion::CompletionModel::completion(self, request).await?;
        Ok(CompletionResponse {
            choice: response.choice,
            usage: response.usage,
            raw_response: (),
        })
    }
}

impl CompletionProvider for gemini::completion::CompletionModel {
    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<()>, CompletionError> {
        let response = rig::completion::CompletionModel::completion(self, request).await?;
        Ok(CompletionResponse {
            choice: response.choice,
            usage: response.usage,
            raw_response: (),
        })
    }
}

impl CompletionProvider for groq::CompletionModel<reqwest::Client> {
    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<()>, CompletionError> {
        let response = rig::completion::CompletionModel::completion(self, request).await?;
        Ok(CompletionResponse {
            choice: response.choice,
            usage: response.usage,
            raw_response: (),
        })
    }
}

impl CompletionProvider for perplexity::CompletionModel<reqwest::Client> {
    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<()>, CompletionError> {
        let response = rig::completion::CompletionModel::completion(self, request).await?;
        Ok(CompletionResponse {
            choice: response.choice,
            usage: response.usage,
            raw_response: (),
        })
    }
}

impl LMClient {
    pub fn from_model_string(model_str: &str) -> Result<Self> {
        let parts: Vec<&str> = model_str.split(':').collect();
        if parts.len() != 2 {
            anyhow::bail!("Model string must be in format 'provider:model_name'");
        }

        let provider = parts[0];
        let model_id = parts[1];

        match provider {
            "openai" => {
                let api_key = std::env::var("OPENAI_API_KEY")
                    .map_err(|_| anyhow::anyhow!("OPENAI_API_KEY environment variable not set"))?;
                let client = openai::ClientBuilder::new(&api_key).build();
                Ok(LMClient::OpenAI(openai::completion::CompletionModel::new(client, model_id)))
            }
            "anthropic" => {
                let api_key = std::env::var("ANTHROPIC_API_KEY")
                    .map_err(|_| anyhow::anyhow!("ANTHROPIC_API_KEY environment variable not set"))?;
                let client = anthropic::ClientBuilder::new(&api_key).build()?;
                Ok(LMClient::Anthropic(anthropic::completion::CompletionModel::new(client, model_id)))
            }
            "cohere" => {
                let api_key = std::env::var("COHERE_API_KEY")
                    .map_err(|_| anyhow::anyhow!("COHERE_API_KEY environment variable not set"))?;
                let client = cohere::client::ClientBuilder::new(&api_key).build();
                Ok(LMClient::Cohere(cohere::completion::CompletionModel::new(client, model_id)))
            }
            "gemini" => {
                let api_key = std::env::var("GEMINI_API_KEY")
                    .map_err(|_| anyhow::anyhow!("GEMINI_API_KEY environment variable not set"))?;
                let client = gemini::client::ClientBuilder::<reqwest::Client>::new(&api_key).build()?;
                Ok(LMClient::Gemini(gemini::completion::CompletionModel::new(client, model_id)))
            }
            "groq" => {
                let api_key = std::env::var("GROQ_API_KEY")
                    .map_err(|_| anyhow::anyhow!("GROQ_API_KEY environment variable not set"))?;
                let client = groq::ClientBuilder::new(&api_key).build();
                Ok(LMClient::Groq(groq::CompletionModel::new(client, model_id)))
            }
            "perplexity" => {
                let api_key = std::env::var("PERPLEXITY_API_KEY")
                    .map_err(|_| anyhow::anyhow!("PERPLEXITY_API_KEY environment variable not set"))?;
                let client = perplexity::ClientBuilder::new(&api_key).build();
                Ok(LMClient::Perplexity(perplexity::CompletionModel::new(client, model_id)))
            }
            _ => anyhow::bail!("Unsupported provider: {}", provider),
        }
    }
}
