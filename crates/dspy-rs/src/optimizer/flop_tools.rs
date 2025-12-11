use crate::Example;
use crate::core::MetaSignature;
use crate::trace::signature_utils::{fuse_signatures, modify_signature, split_signature};
use anyhow::Result;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt;
use std::sync::Arc;

// ============================================================================
// Error Types
// ============================================================================

#[derive(Debug)]
pub struct SignatureToolError(String);

impl fmt::Display for SignatureToolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Signature tool error: {}", self.0)
    }
}

impl std::error::Error for SignatureToolError {}

// ============================================================================
// Tool Arguments
// ============================================================================

#[derive(Deserialize, Serialize)]
pub struct SplitArgs {
    pub original_signature: String, // JSON representation of signature
    pub split_points: Vec<SplitPoint>,
}

#[derive(Deserialize, Serialize)]
pub struct SplitPoint {
    pub name: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

#[derive(Deserialize, Serialize)]
pub struct FuseArgs {
    pub signatures: Vec<String>, // JSON representations
    pub merged_name: String,
}

#[derive(Deserialize, Serialize)]
pub struct ModifyArgs {
    pub signature: String, // JSON representation
    pub new_instruction: Option<String>,
    pub add_inputs: Option<Vec<FieldDef>>,
    pub add_outputs: Option<Vec<FieldDef>>,
    pub remove_fields: Option<Vec<String>>,
}

#[derive(Deserialize, Serialize)]
pub struct FieldDef {
    pub name: String,
    pub desc: String,
    pub type_name: String,
}

// ============================================================================
// Helper Struct for JSON Serialization of Signatures
// ============================================================================

#[derive(Serialize, Deserialize)]
struct JsonSignature {
    instruction: String,
    inputs: Value,
    outputs: Value,
    demos: Vec<Example>,
}

// Helper struct to convert JSON to MetaSignature
struct JsonSigWrapper {
    instruction: String,
    inputs: Value,
    outputs: Value,
    demos: Vec<Example>,
}

impl MetaSignature for JsonSigWrapper {
    fn demos(&self) -> Vec<Example> {
        self.demos.clone()
    }
    fn set_demos(&mut self, demos: Vec<Example>) -> Result<()> {
        self.demos = demos;
        Ok(())
    }
    fn instruction(&self) -> String {
        self.instruction.clone()
    }
    fn input_fields(&self) -> Value {
        self.inputs.clone()
    }
    fn output_fields(&self) -> Value {
        self.outputs.clone()
    }
    fn update_instruction(&mut self, instruction: String) -> Result<()> {
        self.instruction = instruction;
        Ok(())
    }
    fn append(&mut self, name: &str, value: Value) -> Result<()> {
        if let Some(obj) = self.inputs.as_object_mut() {
            obj.insert(name.to_string(), value);
        }
        Ok(())
    }
}

impl JsonSignature {
    fn from_meta(sig: &dyn MetaSignature) -> Self {
        Self {
            instruction: sig.instruction(),
            inputs: sig.input_fields(),
            outputs: sig.output_fields(),
            demos: sig.demos(),
        }
    }

    fn to_meta(&self) -> Arc<dyn MetaSignature> {
        Arc::new(JsonSigWrapper {
            instruction: self.instruction.clone(),
            inputs: self.inputs.clone(),
            outputs: self.outputs.clone(),
            demos: self.demos.clone(),
        })
    }
}

// ============================================================================
// Tools
// ============================================================================

pub struct SplitSignatureTool;

impl Tool for SplitSignatureTool {
    const NAME: &'static str = "split_signature";

    type Args = SplitArgs;
    type Output = Vec<String>; // List of JSON signatures
    type Error = SignatureToolError;

    async fn definition(&self, _prompt: String) -> rig::completion::ToolDefinition {
        rig::completion::ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Splits a signature into multiple sequential signatures based on specified split points.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "original_signature": { "type": "string", "description": "JSON string of original signature" },
                    "split_points": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": { "type": "string" },
                                "inputs": { "type": "array", "items": { "type": "string" } },
                                "outputs": { "type": "array", "items": { "type": "string" } }
                            }
                        }
                    }
                },
                "required": ["original_signature", "split_points"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let orig_json: JsonSignature = serde_json::from_str(&args.original_signature)
            .map_err(|e| SignatureToolError(e.to_string()))?;
        let orig_sig = orig_json.to_meta();

        let mut results = Vec::new();

        // For simplicity, we implement a sequential split where each split point
        // defines a new signature derived from the original context.
        // A more complex implementation would chain them properly (passing outputs to inputs).
        // Here we just use the split_signature logic from signature_utils but adapted for N-way split?
        // Actually, signature_utils::split_signature does a 2-way split.
        // We can iteratively split or just project fields.

        // Convert split_points to split_metadata
        let split_metadata: Vec<Value> = args
            .split_points
            .iter()
            .map(|point| {
                serde_json::json!({
                    "name": point.name,
                    "inputs": point.inputs,
                    "outputs": point.outputs
                })
            })
            .collect();

        let split_sigs = split_signature(orig_sig.as_ref(), split_metadata)
            .map_err(|e| SignatureToolError(e.to_string()))?;

        for sig in split_sigs {
            let json_sig = JsonSignature::from_meta(sig.as_ref());
            results.push(serde_json::to_string(&json_sig).unwrap());
        }

        Ok(results)
    }
}

pub struct FuseSignatureTool;

impl Tool for FuseSignatureTool {
    const NAME: &'static str = "fuse_signatures";

    type Args = FuseArgs;
    type Output = String; // JSON signature
    type Error = SignatureToolError;

    async fn definition(&self, _prompt: String) -> rig::completion::ToolDefinition {
        rig::completion::ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Fuses multiple signatures into a single merged signature.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "signatures": { "type": "array", "items": { "type": "string" } },
                    "merged_name": { "type": "string" }
                },
                "required": ["signatures", "merged_name"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        if args.signatures.is_empty() {
            return Err(SignatureToolError("No signatures provided".to_string()));
        }

        let mut sigs: Vec<Arc<dyn MetaSignature>> = Vec::new();

        for sig_str in args.signatures {
            let json_sig: JsonSignature =
                serde_json::from_str(&sig_str).map_err(|e| SignatureToolError(e.to_string()))?;
            sigs.push(json_sig.to_meta());
        }

        let sig_refs: Vec<&dyn MetaSignature> = sigs.iter().map(|s| s.as_ref()).collect();
        let merged = fuse_signatures(&sig_refs);
        let final_sig = modify_signature(
            merged.as_ref(),
            Some(format!("{}: {}", args.merged_name, merged.instruction())),
            None,
            None,
            None,
        );

        let json_res = JsonSignature::from_meta(final_sig.as_ref());
        Ok(serde_json::to_string(&json_res).unwrap())
    }
}

pub struct ModifySignatureTool;

impl Tool for ModifySignatureTool {
    const NAME: &'static str = "modify_signature";

    type Args = ModifyArgs;
    type Output = String; // JSON signature
    type Error = SignatureToolError;

    async fn definition(&self, _prompt: String) -> rig::completion::ToolDefinition {
        rig::completion::ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Modifies a signature by changing instruction or adding/removing fields."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "signature": { "type": "string" },
                    "new_instruction": { "type": "string" },
                    "add_inputs": {
                        "type": "array",
                        "items": {
                             "type": "object",
                             "properties": { "name": {"type": "string"}, "desc": {"type": "string"}, "type_name": {"type": "string"} }
                        }
                    },
                    "add_outputs": {
                         "type": "array",
                        "items": {
                             "type": "object",
                             "properties": { "name": {"type": "string"}, "desc": {"type": "string"}, "type_name": {"type": "string"} }
                        }
                    },
                    "remove_fields": { "type": "array", "items": { "type": "string" } }
                },
                "required": ["signature"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let json_sig: JsonSignature =
            serde_json::from_str(&args.signature).map_err(|e| SignatureToolError(e.to_string()))?;
        let sig = json_sig.to_meta();

        // Convert Vec<(String, Value)> to slices for modify_signature
        let add_inputs_slice: Vec<(String, Value)> = args
            .add_inputs
            .as_ref()
            .map(|fields| {
                fields
                    .iter()
                    .map(|f| {
                        (
                            f.name.clone(),
                            serde_json::json!({
                                "type": f.type_name,
                                "desc": f.desc,
                                "__dsrs_field_type": "input"
                            }),
                        )
                    })
                    .collect()
            })
            .unwrap_or_default();

        let add_outputs_slice: Vec<(String, Value)> = args
            .add_outputs
            .as_ref()
            .map(|fields| {
                fields
                    .iter()
                    .map(|f| {
                        (
                            f.name.clone(),
                            serde_json::json!({
                                "type": f.type_name,
                                "desc": f.desc,
                                "__dsrs_field_type": "output"
                            }),
                        )
                    })
                    .collect()
            })
            .unwrap_or_default();

        let modified = modify_signature(
            sig.as_ref(),
            args.new_instruction.clone(),
            if add_inputs_slice.is_empty() {
                None
            } else {
                Some(&add_inputs_slice)
            },
            if add_outputs_slice.is_empty() {
                None
            } else {
                Some(&add_outputs_slice)
            },
            args.remove_fields.as_deref(),
        );

        let json_res = JsonSignature::from_meta(modified.as_ref());
        Ok(serde_json::to_string(&json_res).unwrap())
    }
}
