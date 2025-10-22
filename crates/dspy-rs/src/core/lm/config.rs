use bon::Builder;

/// Tunable inference parameters applied to each [`LM::call`].
#[derive(Clone, Debug, Builder)]
pub struct LMConfig {
    #[builder(default = "openai:gpt-4o-mini".to_string())]
    pub model: String,
    /// Sampling temperature. Higher values increase randomness.
    #[builder(default = 0.7)]
    pub temperature: f32,
    #[builder(default = 512)]
    pub max_tokens: u32,
    #[builder(default = true)]
    pub cache: bool,
}

impl Default for LMConfig {
    fn default() -> Self {
        LMConfig::builder().build()
    }
}
