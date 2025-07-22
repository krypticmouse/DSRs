use std::collections::HashMap;

use crate::adapter::base::Adapter;
use crate::adapter::chat_adapter::ChatAdapter;
use crate::clients::lm::LM;
use crate::data::prediction::Prediction;
use crate::module::Module;
use crate::signature::signature::Signature;

pub struct Predict<'a> {
    pub signature: Signature<'a>,
}

impl<'a> Module<'a> for Predict<'a> {
    async fn forward(
        &self,
        inputs: HashMap<String, String>,
        lm: Option<LM<'a>>,
        adapter: Option<ChatAdapter>,
    ) -> Prediction {
        let mut lm = lm.unwrap_or_default();
        let adapter = adapter.unwrap_or_default();

        let messages = adapter.format(&self.signature, inputs);
        let response = lm.call(&messages, self.signature.name).await.unwrap();
        adapter.parse_response(&self.signature, response)
    }
}
