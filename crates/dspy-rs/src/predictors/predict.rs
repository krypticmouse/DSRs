use crate::{core::{GLOBAL_SETTINGS, Module}, data::{example::Example, prediction::Prediction}};
use crate::core::MetaSignature;

pub struct Predict{
    pub signature: Box<dyn MetaSignature>,
}

impl Module for Predict {
    async fn forward(
        &self,
        inputs: Example
    ) -> anyhow::Result<Prediction> {
        let guard = GLOBAL_SETTINGS.read().unwrap();
        let settings = guard.as_ref().unwrap();
        let adapter = &settings.adapter;
        let lm = &mut settings.lm.clone();
        let result = adapter.call(lm, self.signature.as_ref(), inputs).await;
        result
    }
}
