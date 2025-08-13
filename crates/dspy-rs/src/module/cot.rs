use super::Predict;
use crate::core::{Module, Signature, SignatureMetadata};

use schemars::JsonSchema;
use serde::{Deserialize, de::DeserializeOwned};

#[derive(Deserialize, JsonSchema)]
#[serde(bound = "I: DeserializeOwned + JsonSchema")]
pub struct CoTOutputs<I> {
    pub reasoning: String,
    #[serde(flatten)]
    pub inner: I,
}

pub struct CoTSignature<S: Signature> {
    pub inner: S,
    metadata: SignatureMetadata,
}

impl<S> Default for CoTSignature<S>
where
    S: Signature + Default,
{
    fn default() -> Self {
        let inner = S::default();
        let inner_metadata = inner.metadata();

        let output_schema = serde_json::json!(schemars::schema_for!(CoTOutputs<S::Outputs>));

        let metadata = SignatureMetadata {
            instructions: inner_metadata.instructions.clone(),
            input_schema: inner_metadata.input_schema.clone(),
            output_schema: output_schema,
        };

        CoTSignature {
            inner: S::default(),
            metadata,
        }
    }
}

impl<S: Signature> Signature for CoTSignature<S> {
    type Inputs = S::Inputs;
    type Outputs = CoTOutputs<S::Outputs>;

    fn extract_fields(&self, inputs: &Self::Inputs) -> Vec<impl Into<String>> {
        self.inner.extract_fields(inputs)
    }

    fn extract_history(&self, inputs: &Self::Inputs) -> Option<crate::core::History> {
        self.inner.extract_history(inputs)
    }

    fn extract_tools(&self, inputs: &Self::Inputs) -> Option<Vec<crate::core::Tool>> {
        self.inner.extract_tools(inputs)
    }

    fn metadata(&self) -> &crate::core::SignatureMetadata {
        &self.metadata
    }

    fn metadata_mut(&mut self) -> &mut SignatureMetadata {
        &mut self.metadata
    }
}

pub struct CoT<S: Signature> {
    predictor: Predict<CoTSignature<S>>,
}

impl<S: Signature> Module for CoT<S> {
    type SIG = CoTSignature<S>;

    async fn aforward_with_settings(
        &self,
        inputs: &<<Self as Module>::SIG as Signature>::Inputs,
        settings: &mut crate::core::Settings,
    ) -> anyhow::Result<<<Self as Module>::SIG as Signature>::Outputs> {
        let result = self
            .predictor
            .aforward_with_settings(inputs, settings)
            .await;
        result
    }
}
