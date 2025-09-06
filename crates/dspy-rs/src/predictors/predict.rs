use crate::core::MetaSignature;
use crate::{ChatAdapter, Example, GLOBAL_SETTINGS, LM, Prediction, Predictor};

pub struct Predict {
    pub signature: Box<dyn MetaSignature>,
}

impl Predict {
    pub fn new(signature: impl MetaSignature + 'static) -> Self {
        Self {
            signature: Box::new(signature),
        }
    }
}

impl Predictor for Predict {
    async fn forward(&self, inputs: Example) -> anyhow::Result<Prediction> {
        let (adapter, mut lm) = {
            let guard = GLOBAL_SETTINGS.read().unwrap();
            let settings = guard.as_ref().unwrap();
            (settings.adapter.clone(), settings.lm.clone())
        }; // guard is dropped here
        adapter.call(&mut lm, self.signature.as_ref(), inputs).await
    }

    async fn forward_with_config(
        &self,
        inputs: Example,
        lm: &mut LM,
    ) -> anyhow::Result<Prediction> {
        ChatAdapter.call(lm, self.signature.as_ref(), inputs).await
    }
}
