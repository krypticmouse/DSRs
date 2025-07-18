use std::collections::HashMap;

use crate::adapter::base::Adapter;
use crate::adapter::chat_adapter::ChatAdapter;
use crate::clients::lm::LM;
use crate::data::prediction::Prediction;
use crate::module::Module;
use crate::signature::field::Field;
use crate::signature::signature::Signature;

pub struct ChainofThought<'a> {
    pub signature: &'a mut Signature,
    pub add_hint: bool,
}

impl<'a> ChainofThought<'a> {
    pub fn new(signature: &'a mut Signature, add_hint: bool) -> Self {
        if add_hint {
            signature.append(
                "hint".to_string(),
                Field::OutputField {
                    prefix: "".to_string(),
                    format: None,
                    desc: "The hint for the answer".to_string(),
                    output_type: "String".to_string(),
                },
            );
        }

        signature.prepend(
            "reasoning".to_string(),
            Field::OutputField {
                prefix: "".to_string(),
                format: None,
                desc: "Let's think step by step.".to_string(),
                output_type: "String".to_string(),
            },
        );

        Self {
            signature,
            add_hint,
        }
    }
}

impl<'a> Module for ChainofThought<'a> {
    async fn forward(
        &self,
        inputs: HashMap<String, String>,
        lm: Option<LM>,
        adapter: Option<ChatAdapter>,
    ) -> Prediction {
        let mut lm = lm.unwrap_or_default();
        let adapter = adapter.unwrap_or_default();

        let messages = adapter.format(self.signature, inputs);
        let response = lm
            .call(&messages, self.signature.name.clone())
            .await
            .unwrap();
        adapter.parse_response(self.signature, response)
    }
}
