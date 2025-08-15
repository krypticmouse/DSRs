use bon::Builder;
use serde_json::Value;
use std::collections::HashMap;

#[derive(Clone, Debug, Builder)]
pub struct LMConfig {
    #[builder(default = "gpt-4o-mini".to_string())]
    pub model: String,
    #[builder(default = 0.7)]
    pub temperature: f32,
    #[builder(default = 1.0)]
    pub top_p: f32,
    #[builder(default = 512)]
    pub max_tokens: u32,
    #[builder(default = 512)]
    pub max_completion_tokens: u32,
    #[builder(default = 1)]
    pub n: u8,
    #[builder(default = 0.0)]
    pub presence_penalty: f32,
    #[builder(default = 0.0)]
    pub frequency_penalty: f32,
    #[builder(default = 42)]
    pub seed: i64,
    pub logit_bias: Option<HashMap<String, Value>>,
}

impl Default for LMConfig {
    fn default() -> Self {
        LMConfig::builder().build()
    }
}
