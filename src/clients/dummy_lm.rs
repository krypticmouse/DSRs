use openrouter_rs::types::{
    Choice, CompletionsResponse, FinishReason, Message, NonStreamingChoice, ObjectType,
};
use smart_default::SmartDefault;
use std::error::Error;

use crate::clients::{chat::Chat, lm::LMConfig};
use crate::data::history::History;

#[derive(Clone, Debug, SmartDefault)]
pub struct DummyLM<'a> {
    #[default = "dummy/model"]
    pub model: &'a str,
    #[default(Vec::new())]
    pub history: Vec<History<'a>>,
    #[default(LMConfig::default())]
    pub config: LMConfig,
}

impl<'a> DummyLM<'a> {
    pub async fn call(
        &mut self,
        chat: &Chat,
        output: &'a str,
        signature: &'a str,
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
            signature,
            model: self.model,
        });
        Ok(response)
    }

    pub fn inspect_history(&self, n: usize) -> Vec<&History<'a>> {
        self.history.iter().rev().take(n).collect()
    }
}
