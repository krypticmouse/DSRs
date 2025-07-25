use std::collections::HashMap;

use crate::adapter::base::Adapter;
use crate::adapter::chat_adapter::ChatAdapter;
use crate::clients::dummy_lm::DummyLM;
use crate::data::prediction::Prediction;
use crate::signature::signature::Signature;

pub struct DummyPredict<'a> {
    pub signature: &'a mut Signature<'a>,
}

impl<'a> DummyPredict<'a> {
    pub async fn forward(
        &self,
        inputs: HashMap<String, String>,
        output: &'a str,
        lm: Option<DummyLM<'a>>,
        adapter: Option<ChatAdapter>,
    ) -> Prediction {
        let mut lm = lm.unwrap_or_default();
        let adapter = adapter.unwrap_or_default();

        let messages = adapter.format(self.signature, inputs);
        let response = lm
            .call(&messages, output, self.signature.name)
            .await
            .unwrap();
        adapter.parse_response(self.signature, response)
    }
}
