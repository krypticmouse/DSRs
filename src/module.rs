use std::collections::HashMap;

use crate::adapter::chat_adapter::ChatAdapter;
use crate::clients::lm::LM;
use crate::data::prediction::Prediction;

#[allow(async_fn_in_trait)]
pub trait Module<'a> {
    async fn forward(
        &self,
        inputs: HashMap<String, String>,
        lm: Option<LM<'a>>,
        adapter: Option<ChatAdapter>,
    ) -> Prediction;
}
