use super::{History, Tool};
use anyhow::Result;
use schemars::JsonSchema;
use serde::{Serialize, de::DeserializeOwned};

#[derive(Clone)]
pub struct SignatureMetadata {
    pub instructions: String,
    pub input_schema: serde_json::Value,
    pub output_schema: serde_json::Value,
}

impl SignatureMetadata {
    pub fn new(
        instructions: String,
        input_schema: serde_json::Value,
        output_schema: serde_json::Value,
    ) -> Self {
        Self {
            instructions,
            input_schema,
            output_schema,
        }
    }

    pub fn set_input_field_meta(&mut self, field: &str, description: String) -> Result<()> {
        if let Some(properties) = self
            .input_schema
            .get_mut("properties")
            .and_then(|p| p.as_object_mut())
        {
            if let Some(field_obj) = properties.get_mut(field).and_then(|f| f.as_object_mut()) {
                field_obj.insert(
                    "description".to_string(),
                    serde_json::Value::String(description),
                );
                Ok(())
            } else {
                Err(anyhow::anyhow!(
                    "Field '{}' not found in input schema",
                    field
                ))
            }
        } else {
            Err(anyhow::anyhow!("Invalid input schema structure"))
        }
    }

    pub fn set_output_field_meta(&mut self, field: &str, description: String) -> Result<()> {
        if let Some(properties) = self
            .output_schema
            .get_mut("properties")
            .and_then(|p| p.as_object_mut())
        {
            if let Some(field_obj) = properties.get_mut(field).and_then(|f| f.as_object_mut()) {
                field_obj.insert(
                    "description".to_string(),
                    serde_json::Value::String(description),
                );
                Ok(())
            } else {
                Err(anyhow::anyhow!(
                    "Field '{}' not found in output schema",
                    field
                ))
            }
        } else {
            Err(anyhow::anyhow!("Invalid output schema structure"))
        }
    }

    pub fn input_fields_meta(&self) -> Vec<(String, String)> {
        let mut fields = Vec::new();
        if let Some(properties) = self
            .input_schema
            .get("properties")
            .and_then(|p| p.as_object())
        {
            for (field_name, field_def) in properties {
                let description = field_def
                    .get("description")
                    .and_then(|d| d.as_str())
                    .unwrap_or("")
                    .to_string();
                fields.push((field_name.clone(), description));
            }
        }
        fields
    }

    pub fn output_fields_meta(&self) -> Vec<(String, String)> {
        let mut fields = Vec::new();
        if let Some(properties) = self
            .output_schema
            .get("properties")
            .and_then(|p| p.as_object())
        {
            for (field_name, field_def) in properties {
                let description = field_def
                    .get("description")
                    .and_then(|d| d.as_str())
                    .unwrap_or("")
                    .to_string();
                fields.push((field_name.clone(), description));
            }
        }
        fields
    }
}

pub trait Signature: Default {
    // user interface. Modules have associated signature type SIG and forward (&self, &SIG::Inputs) -> SIG::Outputs
    type Inputs: Serialize + JsonSchema;
    type Outputs: DeserializeOwned + JsonSchema;

    // signature metadata
    fn metadata(&self) -> &SignatureMetadata;
    fn metadata_mut(&mut self) -> &mut SignatureMetadata;

    // Special input field extraction methods - default implementations return None
    // If a signature has a type that is handled specially by adapters, they are extracted here

    fn extract_fields(&self, inputs: &Self::Inputs) -> Vec<impl Into<String>>;

    fn extract_history(&self, _inputs: &Self::Inputs) -> Option<History>;

    fn extract_tools(&self, _inputs: &Self::Inputs) -> Option<Vec<Tool>>;
}
