#![cfg(feature = "rlm")]

use std::marker::PhantomData;

use rig::agent::Agent;
use rig::providers::openai::CompletionModel;

use crate::Signature;

use super::RlmConfig;

/// Typed Recursive Language Model.
///
/// Uses a DSRs Signature to run RLM with typed inputs/outputs.
#[allow(dead_code)]
pub struct TypedRlm<S: Signature> {
    agent: Agent<CompletionModel>,
    config: RlmConfig,
    _marker: PhantomData<S>,
}

impl<S: Signature> TypedRlm<S> {
    /// Create a new TypedRlm with the given agent and config.
    pub fn new(agent: Agent<CompletionModel>, config: RlmConfig) -> Self {
        Self {
            agent,
            config,
            _marker: PhantomData,
        }
    }

    /// Create with default config.
    pub fn with_agent(agent: Agent<CompletionModel>) -> Self {
        Self::new(agent, RlmConfig::default())
    }
}
