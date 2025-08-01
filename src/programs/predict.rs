use crate::adapter::{base::Adapter, chat_adapter::ChatAdapter};
use crate::clients::lm::LM;
use crate::data::{example::Example, prediction::Prediction};
use crate::module::Module;
use crate::signature::Signature;
use crate::utils::settings::SETTINGS;

pub struct Predict {
    pub signature: Signature,
}

impl Module for Predict {
    async fn forward(
        &self,
        inputs: Example,
        lm: Option<LM>,
        adapter: Option<ChatAdapter>,
    ) -> Prediction {
        let mut lm = lm.unwrap_or(SETTINGS.lock().unwrap().lm.clone());
        let adapter = adapter.unwrap_or(SETTINGS.lock().unwrap().adapter.clone());

        let messages = adapter.format(self.signature.clone(), inputs);
        let response = lm
            .call(&messages, self.signature.name.clone())
            .await
            .unwrap();

        adapter.parse_response(self.signature.clone(), response)
    }
}
