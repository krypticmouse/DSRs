use crate::adapter::chat_adapter::ChatAdapter;
use crate::clients::lm::LM;
use crate::data::{example::Example, prediction::Prediction};
use crate::module::Module;
use crate::programs::predict::Predict;
use crate::signature::{field::Field, signature::Signature};

pub struct ChainofThought {
    pub signature: Signature,
    pub add_hint: bool,
    pub predictor: Predict,
}

impl ChainofThought {
    pub fn new(signature: &mut Signature, add_hint: bool) -> Self {
        if add_hint {
            signature.append(
                "hint".to_string(),
                Field::Out("The hint for the answer".to_string()),
            );
        }

        signature.prepend(
            "reasoning".to_string(),
            Field::Out("Let's think step by step.".to_string()),
        );

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
