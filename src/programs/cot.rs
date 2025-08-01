use crate::adapter::chat_adapter::ChatAdapter;
use crate::clients::lm::LM;
use crate::data::{example::Example, prediction::Prediction};
use crate::field::Out;
use crate::module::Module;
use crate::programs::predict::Predict;
use crate::signature::Signature;

pub struct ChainofThought {
    pub signature: Signature,
    pub add_hint: bool,
    pub predictor: Predict,
}

impl ChainofThought {
    pub fn new(signature: &mut Signature, add_hint: bool) -> Self {
        if add_hint {
            signature.append_output("hint".to_string(), Out::default());
        }

        signature.prepend_output("reasoning".to_string(), Out::default());

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
