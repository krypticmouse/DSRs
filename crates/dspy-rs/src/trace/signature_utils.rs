use crate::Example;
use crate::core::MetaSignature;
use anyhow::Result;
use serde_json::Value;
use std::sync::Arc;

// Internal helper struct - not exported, only used for creating MetaSignature instances
struct SignatureImpl {
    instruction: String,
    input_fields: Value,
    output_fields: Value,
    demos: Vec<Example>,
}

impl MetaSignature for SignatureImpl {
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
        self.input_fields.clone()
    }

    fn output_fields(&self) -> Value {
        self.output_fields.clone()
    }

    fn update_instruction(&mut self, instruction: String) -> Result<()> {
        self.instruction = instruction;
        Ok(())
    }

    fn append(&mut self, name: &str, value: Value) -> Result<()> {
        if let Some(obj) = self.input_fields.as_object_mut() {
            obj.insert(name.to_string(), value);
        }
        Ok(())
    }
}

/// Split a signature into multiple signatures based on split metadata
/// split_metadata is a Vec of Value objects, each containing:
/// - "name": String (optional name for the split part)
/// - "inputs": Vec<String> (input field names)
/// - "outputs": Vec<String> (output field names)
pub fn split_signature(
    original: &dyn MetaSignature,
    split_metadata: Vec<Value>,
) -> Result<Vec<Arc<dyn MetaSignature>>> {
    let orig_inputs = original.input_fields();
    let orig_inputs_obj = orig_inputs.as_object().unwrap();

    let orig_outputs = original.output_fields();
    let orig_outputs_obj = orig_outputs.as_object().unwrap();

    let mut results = Vec::new();
    let mut previous_outputs = serde_json::Map::new();

    for (idx, metadata) in split_metadata.iter().enumerate() {
        let name = metadata
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or(&format!("Part {}", idx + 1))
            .to_string();

        let inputs: Vec<String> = metadata
            .get("inputs")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        let outputs: Vec<String> = metadata
            .get("outputs")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        // Build inputs for this split part
        let mut part_inputs = serde_json::Map::new();
        for key in &inputs {
            // Check if it comes from original inputs
            if let Some(val) = orig_inputs_obj.get(key) {
                part_inputs.insert(key.clone(), val.clone());
            }
            // Check if it comes from previous split's outputs
            else if let Some(val) = previous_outputs.get(key) {
                let mut input_schema = val.clone();
                if let Some(obj) = input_schema.as_object_mut() {
                    obj.insert("__dsrs_field_type".to_string(), "input".into());
                }
                part_inputs.insert(key.clone(), input_schema);
            }
        }

        // Build outputs for this split part
        let mut part_outputs = serde_json::Map::new();
        for key in &outputs {
            if let Some(val) = orig_outputs_obj.get(key) {
                part_outputs.insert(key.clone(), val.clone());
            } else {
                // New intermediate output
                part_outputs.insert(
                    key.clone(),
                    serde_json::json!({
                        "type": "String",
                        "desc": "Intermediate output",
                        "__dsrs_field_type": "output"
                    }),
                );
            }
        }

        // Update previous_outputs for next iteration
        previous_outputs = part_outputs.clone();

        let sig: Arc<dyn MetaSignature> = Arc::new(SignatureImpl {
            instruction: format!("{}: {}", name, original.instruction()),
            input_fields: Value::Object(part_inputs),
            output_fields: Value::Object(part_outputs),
            demos: vec![],
        });

        results.push(sig);
    }

    Ok(results)
}

/// Merge multiple signatures into one
pub fn fuse_signatures(signatures: &[&dyn MetaSignature]) -> Arc<dyn MetaSignature> {
    if signatures.is_empty() {
        return Arc::new(SignatureImpl {
            instruction: "Empty signature".to_string(),
            input_fields: Value::Object(serde_json::Map::new()),
            output_fields: Value::Object(serde_json::Map::new()),
            demos: vec![],
        });
    }

    let mut combined_instruction = String::new();
    let mut input_map = serde_json::Map::new();
    let mut output_map = serde_json::Map::new();

    for (i, sig) in signatures.iter().enumerate() {
        if i > 0 {
            combined_instruction.push_str("\n");
        }
        combined_instruction.push_str(&format!("Task {}: {}", i + 1, sig.instruction()));

        // Merge inputs: Union of inputs
        for (k, v) in sig.input_fields().as_object().unwrap() {
            if !input_map.contains_key(k) {
                input_map.insert(k.clone(), v.clone());
            }
        }

        // Merge outputs: Union of outputs
        for (k, v) in sig.output_fields().as_object().unwrap() {
            if !output_map.contains_key(k) {
                output_map.insert(k.clone(), v.clone());
            }
        }
    }

    combined_instruction.push_str("\nCombined Task: Perform all tasks sequentially.");

    Arc::new(SignatureImpl {
        instruction: combined_instruction,
        input_fields: Value::Object(input_map),
        output_fields: Value::Object(output_map),
        demos: vec![],
    })
}

/// Modify a signature by changing instruction or adding/removing fields
pub fn modify_signature(
    original: &dyn MetaSignature,
    new_instruction: Option<String>,
    add_inputs: Option<&[(String, Value)]>,
    add_outputs: Option<&[(String, Value)]>,
    remove_fields: Option<&[String]>,
) -> Arc<dyn MetaSignature> {
    let mut instruction = original.instruction();
    if let Some(inst) = new_instruction {
        instruction = inst;
    }

    let mut inputs = original.input_fields().as_object().unwrap().clone();
    let mut outputs = original.output_fields().as_object().unwrap().clone();

    if let Some(fields) = add_inputs {
        for (name, value) in fields {
            inputs.insert(name.clone(), value.clone());
        }
    }

    if let Some(fields) = add_outputs {
        for (name, value) in fields {
            outputs.insert(name.clone(), value.clone());
        }
    }

    if let Some(fields) = remove_fields {
        for f in fields {
            inputs.remove(f);
            outputs.remove(f);
        }
    }

    Arc::new(SignatureImpl {
        instruction,
        input_fields: Value::Object(inputs),
        output_fields: Value::Object(outputs),
        demos: original.demos(),
    })
}
