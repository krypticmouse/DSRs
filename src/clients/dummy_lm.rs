use openrouter_rs::types::{
    Choice, CompletionsResponse, FinishReason, Message, NonStreamingChoice, ObjectType,
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

impl DummyLM {
    pub async fn call(
        &mut self,
        chat: &Chat,
        output: &str,
        signature: &str,
    ) -> Result<CompletionsResponse, Box<dyn Error>> {
        let response = CompletionsResponse {
            id: "dummy_id".to_string(),
            choices: vec![Choice::NonStreaming(NonStreamingChoice {
                finish_reason: Some(FinishReason::Stop),
                native_finish_reason: None,
                error: None,
                message: Message {
                    role: Some("assistant".to_string()),
                    content: Some(output.to_string()),
                    tool_calls: None,
                    reasoning: None,
                    reasoning_details: None,
                },
            })],
            created: 0,
            model: self.model.to_string(),
            object_type: ObjectType::ChatCompletion,
            provider: None,
            system_fingerprint: None,
            usage: None,
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
