use async_openai::types::CompletionUsage;
use serde::{Deserialize, Serialize};
use std::ops::Add;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LmUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    pub reasoning_tokens: Option<u32>,
}

impl From<CompletionUsage> for LmUsage {
    fn from(usage: CompletionUsage) -> Self {
        LmUsage {
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
            reasoning_tokens: usage
                .completion_tokens_details
                .and_then(|details| details.reasoning_tokens),
        }
    }
}

impl Add for LmUsage {
    type Output = LmUsage;

    fn add(self, other: LmUsage) -> Self {
        LmUsage {
            prompt_tokens: self.prompt_tokens + other.prompt_tokens,
            completion_tokens: self.completion_tokens + other.completion_tokens,
            total_tokens: self.total_tokens + other.total_tokens,
            reasoning_tokens: self.reasoning_tokens.or(other.reasoning_tokens),
        }
    }
}
