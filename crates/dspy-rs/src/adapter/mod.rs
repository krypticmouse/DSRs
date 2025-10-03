pub mod chat;

pub use chat::*;

use crate::{Chat, Example, LM, Message, MetaSignature, Prediction};
use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

#[async_trait]
pub trait Adapter: Send + Sync + 'static {
    fn format(&self, signature: &dyn MetaSignature, inputs: Example) -> Chat;
    fn parse_response(
        &self,
        signature: &dyn MetaSignature,
        response: Message,
    ) -> HashMap<String, Value>;
    async fn call(
        &self,
        lm: Arc<Mutex<LM>>,
        signature: &dyn MetaSignature,
        inputs: Example,
    ) -> Result<Prediction>;
}
