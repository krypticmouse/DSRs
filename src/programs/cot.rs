use std::collections::HashMap;

use crate::adapter::chat_adapter::ChatAdapter;
use crate::clients::lm::LM;
use crate::data::prediction::Prediction;
use crate::module::Module;
use crate::programs::predict::Predict;
use crate::signature::field::Field;
use crate::signature::signature::Signature;

pub struct ChainofThought<'a> {
    pub signature: &'a Signature<'a>,
    pub add_hint: bool,
    pub predictor: Predict<'a>,
}

impl<'a> ChainofThought<'a> {
    pub fn new(signature: &'a mut Signature<'a>, add_hint: bool) -> Self {
        if add_hint {
            signature.append("hint".to_string(), Field::Out("The hint for the answer"));
        }

        signature.prepend(
            "reasoning".to_string(),
            Field::Out("Let's think step by step."),
        );

        Self {
            signature,
            add_hint,
            predictor: Predict { signature },
        }
    }
}

impl<'a> Module<'a> for ChainofThought<'a> {
    async fn forward(
        &self,
        inputs: HashMap<String, String>,
        lm: Option<LM<'a>>,
        adapter: Option<ChatAdapter>,
    ) -> Prediction {
        self.predictor.forward(inputs, lm, adapter).await
    }
}
