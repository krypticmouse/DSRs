use bon::Builder;
use serde_json::Value;
use std::collections::HashMap;

/// Tunable inference parameters applied to each [`LM::call`].
#[derive(Clone, Debug, Builder)]
pub struct LMConfig {
    /// Default model identifier. Accepts `provider/model` to infer base URL.
    #[builder(default = "gpt-4o-mini".to_string())]
    pub model: String,
    /// Sampling temperature. Higher values increase randomness.
    #[builder(default = 0.7)]
    pub temperature: f32,
    /// Nucleus sampling parameter (`top_p`). Set either temperature or `top_p`.
    #[builder(default = 0.0)]
    pub top_p: f32,
    /// Maximum tokens requested for the completion.
    #[builder(default = 512)]
    pub max_tokens: u32,
    /// Reserved for providers that differentiate prompt vs. completion limits.
    #[builder(default = 512)]
    pub max_completion_tokens: u32,
    /// Number of completions to request per call.
    #[builder(default = 1)]
    pub n: u8,
    /// Presence penalty forwarded to compatible providers.
    #[builder(default = 0.0)]
    pub presence_penalty: f32,
    /// Frequency penalty forwarded to compatible providers.
    #[builder(default = 0.0)]
    pub frequency_penalty: f32,
    /// Optional deterministic seed when the provider supports it.
    #[builder(default = 42)]
    pub seed: i64,
    /// Token-level logit adjustments keyed by provider-specific IDs.
    pub logit_bias: Option<HashMap<String, Value>>,
    /// Enables the shared response cache and history surface.
    #[builder(default = true)]
    pub cache: bool,
}

impl Default for LMConfig {
    fn default() -> Self {
        LMConfig::builder().build()
    }
}
