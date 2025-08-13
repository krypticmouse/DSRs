use crate::adapter::chat_adapter::ChatAdapter;
use crate::data::{example::Example, prediction::Prediction};
use crate::field::{In, Out};
use crate::internal::MetaSignature;
use crate::module::Module;
use crate::programs::predict::Predict;
use crate::providers::lm::LM;

pub struct ChainofThought {
    pub signature: MetaSignature,
    pub add_hint: bool,
    pub predictor: Predict,
}

impl ChainofThought {
    pub fn new(signature: &mut MetaSignature, add_hint: bool) -> Self {
        if add_hint {
            let mut hint_field = In::<String>::default();
            hint_field.desc = "A hint to the answer".to_string();

            signature.append("hint".to_string(), hint_field);
        }

        let mut reasoning_field = Out::<String>::default();
        reasoning_field.desc = "A reasoning step".to_string();
        signature.prepend("reasoning".to_string(), reasoning_field);

        Self {
            signature: signature.clone(),
            add_hint,
            predictor: Predict {
                signature: signature.clone(),
            },
        }
    }
}

impl Module for ChainofThought {
    async fn forward(
        &self,
        inputs: Example,
        lm: Option<LM>,
        adapter: Option<ChatAdapter>,
    ) -> Prediction {
        self.predictor.forward(inputs.clone(), lm, adapter).await
    }
}
