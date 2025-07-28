use crate::adapter::chat_adapter::ChatAdapter;
use crate::clients::lm::LM;
use crate::data::{example::Example, prediction::Prediction};

#[allow(async_fn_in_trait)]
pub trait Module {
    async fn forward(
        &self,
        inputs: Example,
        lm: Option<LM>,
        adapter: Option<ChatAdapter>,
    ) -> Prediction;
}
