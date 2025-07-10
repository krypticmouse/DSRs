use openai_api_rs::v1::chat_completion::{ChatCompletionMessage, Content, MessageRole};

use crate::data::history::History;
use crate::premitives::lm::{LMConfig, LMProvider, get_base_url};

#[derive(Clone, Debug)]
pub struct DummyLM {
    pub provider: LMProvider,
    pub base_url: Option<String>,
    pub api_key: String,
    pub model: String,
    pub lm_config: LMConfig,
    pub history: Vec<History>,
}

impl DummyLM {
    pub fn new(
        provider: Option<LMProvider>,
        api_key: String,
        model: String,
        lm_config: LMConfig,
        base_url: Option<String>,
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

    pub async fn forward(&mut self, input: String, output: String, signature: String) -> String {
        self.history.push(History {
            input: vec![ChatCompletionMessage {
                role: MessageRole::user,
                content: Content::Text(input.clone()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            }],
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
