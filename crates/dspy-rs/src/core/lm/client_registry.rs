use anyhow::Result;
use enum_dispatch::enum_dispatch;
use reqwest;
use rig::{
    OneOrMany,
    client::Nothing,
    completion::{AssistantContent, CompletionError, CompletionRequest, CompletionResponse, Usage},
    providers::*,
};
use std::{
    borrow::Cow,
    collections::VecDeque,
    sync::{Arc, Mutex},
};
use tracing::{debug, trace, warn};

#[enum_dispatch]
#[allow(async_fn_in_trait)]
pub trait CompletionProvider {
    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<()>, CompletionError>;
}

#[derive(Clone, Default)]
pub struct TestCompletionModel {
    responses: Arc<Mutex<VecDeque<AssistantContent>>>,
    last_request: Arc<Mutex<Option<CompletionRequest>>>,
}

impl TestCompletionModel {
    pub fn new(responses: impl IntoIterator<Item = AssistantContent>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(responses.into_iter().collect())),
            last_request: Arc::new(Mutex::new(None)),
        }
    }

    pub fn push_response(&self, response: AssistantContent) {
        self.responses.lock().unwrap().push_back(response);
    }

    pub fn last_request(&self) -> Option<CompletionRequest> {
        self.last_request.lock().unwrap().clone()
    }
}

impl CompletionProvider for TestCompletionModel {
    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<()>, CompletionError> {
        *self.last_request.lock().unwrap() = Some(request);
        let response = self.responses.lock().unwrap().pop_front().ok_or_else(|| {
            CompletionError::ResponseError("test response queue is empty".to_string())
        })?;
        Ok(CompletionResponse {
            choice: OneOrMany::one(response),
            usage: Usage::new(),
            raw_response: (),
        })
    }
}

#[enum_dispatch(CompletionProvider)]
#[derive(Clone)]
pub enum LMClient {
    OpenAI(openai::completion::CompletionModel),
    Gemini(gemini::completion::CompletionModel),
    Anthropic(anthropic::completion::CompletionModel),
    Groq(groq::CompletionModel<reqwest::Client>),
    OpenRouter(openrouter::completion::CompletionModel),
    Ollama(ollama::CompletionModel<reqwest::Client>),
    Azure(azure::CompletionModel<reqwest::Client>),
    XAI(xai::completion::CompletionModel),
    Cohere(cohere::completion::CompletionModel),
    Mistral(mistral::completion::CompletionModel),
    Together(together::completion::CompletionModel),
    Deepseek(deepseek::CompletionModel<reqwest::Client>),
    Test(TestCompletionModel),
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

impl CompletionProvider for openrouter::completion::CompletionModel {
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

impl CompletionProvider for ollama::CompletionModel<reqwest::Client> {
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

impl CompletionProvider for azure::CompletionModel<reqwest::Client> {
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
impl CompletionProvider for xai::completion::CompletionModel {
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

impl CompletionProvider for mistral::completion::CompletionModel {
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

impl CompletionProvider for together::completion::CompletionModel {
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

impl CompletionProvider for deepseek::CompletionModel<reqwest::Client> {
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
    #[tracing::instrument(
        name = "dsrs.lm.client.get_api_key",
        level = "trace",
        skip(provided, env_var),
        fields(env_var, provided = provided.is_some())
    )]
    fn get_api_key<'a>(provided: Option<&'a str>, env_var: &str) -> Result<Cow<'a, str>> {
        match provided {
            Some(k) => Ok(Cow::Borrowed(k)),
            None => Ok(Cow::Owned(std::env::var(env_var).map_err(|_| {
                anyhow::anyhow!("{} environment variable not set", env_var)
            })?)),
        }
    }

    /// Build case 1: OpenAI-compatible API from base_url + api_key
    #[tracing::instrument(
        name = "dsrs.lm.client.from_openai_compatible",
        level = "debug",
        skip(base_url, api_key, model),
        fields(model, base_url_present = true, api_key_present = true)
    )]
    pub fn from_openai_compatible(base_url: &str, api_key: &str, model: &str) -> Result<Self> {
        trace!(base_url, "creating openai-compatible client");
        let client = openai::CompletionsClient::builder()
            .api_key(api_key)
            .base_url(base_url)
            .build()?;
        debug!("openai-compatible client ready");
        Ok(LMClient::OpenAI(openai::completion::CompletionModel::new(
            client, model,
        )))
    }

    /// Build case 2: Local OpenAI-compatible model from base_url (vLLM, etc.)
    /// Uses a dummy API key since local servers don't require authentication
    #[tracing::instrument(
        name = "dsrs.lm.client.from_local",
        level = "debug",
        skip(base_url, model),
        fields(model, base_url_present = true)
    )]
    pub fn from_local(base_url: &str, model: &str) -> Result<Self> {
        trace!(base_url, "creating local openai-compatible client");
        let client = openai::CompletionsClient::builder()
            .api_key("dummy-key-for-local-server")
            .base_url(base_url)
            .build()?;
        debug!("local openai-compatible client ready");
        Ok(LMClient::OpenAI(openai::completion::CompletionModel::new(
            client, model,
        )))
    }

    /// Build case 3: From provider via model name (provider:model format)
    #[tracing::instrument(
        name = "dsrs.lm.client.from_model_string",
        level = "debug",
        skip(model_str, api_key),
        fields(
            provider = tracing::field::Empty,
            model_id = tracing::field::Empty,
            api_key_present = api_key.is_some()
        )
    )]
    pub fn from_model_string(model_str: &str, api_key: Option<&str>) -> Result<Self> {
        let (provider, model_id) = model_str.split_once(':').ok_or(anyhow::anyhow!(
            "Model string must be in format 'provider:model_name'"
        ))?;
        tracing::Span::current().record("provider", &tracing::field::display(provider));
        tracing::Span::current().record("model_id", &tracing::field::display(model_id));

        match provider {
            "openai" => {
                debug!("selecting openai provider");
                let key = Self::get_api_key(api_key, "OPENAI_API_KEY")?;
                let client = openai::CompletionsClient::builder()
                    .api_key(key.as_ref())
                    .build()?;
                Ok(LMClient::OpenAI(openai::completion::CompletionModel::new(
                    client, model_id,
                )))
            }
            "anthropic" => {
                debug!("selecting anthropic provider");
                let key = Self::get_api_key(api_key, "ANTHROPIC_API_KEY")?;
                let client = anthropic::Client::builder().api_key(key.as_ref()).build()?;
                Ok(LMClient::Anthropic(
                    anthropic::completion::CompletionModel::new(client, model_id),
                ))
            }
            "gemini" => {
                debug!("selecting gemini provider");
                let key = Self::get_api_key(api_key, "GEMINI_API_KEY")?;
                let client = gemini::Client::<reqwest::Client>::builder()
                    .api_key(key.as_ref())
                    .build()?;
                Ok(LMClient::Gemini(gemini::completion::CompletionModel::new(
                    client, model_id,
                )))
            }
            "ollama" => {
                debug!("selecting ollama provider");
                let client = ollama::Client::builder().api_key(Nothing).build()?;
                Ok(LMClient::Ollama(ollama::CompletionModel::new(
                    client, model_id,
                )))
            }
            "openrouter" => {
                debug!("selecting openrouter provider");
                let key = Self::get_api_key(api_key, "OPENROUTER_API_KEY")?;
                let client = openrouter::Client::builder()
                    .api_key(key.as_ref())
                    .build()?;
                Ok(LMClient::OpenRouter(
                    openrouter::completion::CompletionModel::new(client, model_id),
                ))
            }
            "groq" => {
                debug!("selecting groq provider");
                let key = Self::get_api_key(api_key, "GROQ_API_KEY")?;
                let client = groq::Client::builder().api_key(key.as_ref()).build()?;
                Ok(LMClient::Groq(groq::CompletionModel::new(client, model_id)))
            }
            _ => {
                warn!(provider, "unsupported provider");
                anyhow::bail!(
                    "Unsupported provider: {}. Supported providers are: openai, anthropic, gemini, groq, openrouter, ollama",
                    provider
                );
            }
        }
    }

    /// Convert a concrete completion model to LMClient
    ///
    /// This function accepts concrete types that can be converted to LMClient.
    /// The enum_dispatch macro automatically generates From implementations for
    /// each variant type, so you can use this with any concrete completion model.
    pub fn from_custom<T: Into<LMClient>>(client: T) -> Self {
        client.into()
    }
}
