use crate::adapter::base::Adapter;
use crate::adapter::chat_adapter::ChatAdapter;
use crate::clients::dummy_lm::DummyLM;
use crate::data::example::Example;
use crate::data::prediction::Prediction;
use crate::internal::MetaSignature;

pub struct DummyPredict {
    pub signature: MetaSignature,
}

impl DummyPredict {
    pub async fn forward(
        &self,
        inputs: Example,
        output: &str,
        lm: Option<DummyLM>,
        adapter: Option<ChatAdapter>,
    ) -> Prediction {
        let mut lm = lm.unwrap_or_default();
        let adapter = adapter.unwrap_or_default();

        let messages = adapter.format(&self.signature, inputs);
        let response = lm
            .call(&messages, output, &self.signature.name)
            .await
            .unwrap();
        adapter.parse_response(&self.signature, response)
    }
}
