use async_openai::types::{
    ChatChoice, ChatCompletionResponseMessage, CreateChatCompletionResponse, FinishReason, Role,
};
use smart_default::SmartDefault;
use std::error::Error;

use crate::clients::{chat::Chat, lm::LMConfig};
use crate::data::history::History;

#[derive(Clone, Debug, SmartDefault)]
pub struct DummyLM {
    #[default = "dummy/model"]
    pub model: String,
    #[default(Vec::new())]
    pub history: Vec<History>,
    #[default(LMConfig::default())]
    pub config: LMConfig,
}

#[allow(deprecated)]
impl DummyLM {
    pub async fn call(
        &mut self,
        chat: &Chat,
        output: &str,
        signature: &str,
    ) -> Result<CreateChatCompletionResponse, Box<dyn Error>> {
        let response = CreateChatCompletionResponse {
            id: "dummy_id".to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatCompletionResponseMessage {
                    role: Role::Assistant,
                    content: Some(output.to_string()),
                    refusal: None,
                    tool_calls: None,
                    function_call: None,
                    audio: None,
                },
                finish_reason: Some(FinishReason::Stop),
                logprobs: None,
            }],
            created: 0,
            model: self.model.to_string(),
            system_fingerprint: None,
            usage: None,
            object: "dummy_chat.completion".to_string(),
            service_tier: None,
        };

        self.history.push(History {
            input: chat.clone(),
            output: response.clone(),
            signature: signature.to_string(),
            model: self.model.clone(),
        });
        Ok(response)
    }

    pub fn inspect_history(&self, n: usize) -> Vec<&History> {
        self.history.iter().rev().take(n).collect()
    }
}
