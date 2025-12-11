use crate::trace::signature_utils::{fuse_signatures, split_signature};
use crate::{Example, MetaSignature, Prediction};
use anyhow::Result;
use std::fmt;
use std::sync::Arc;

#[derive(Clone)]
pub enum NodeType {
    Root, // Initial input
    Predict {
        signature_name: String,
        signature: Arc<dyn MetaSignature>,
    },
    Operator {
        name: String,
    },
    Map {
        // Describes: for each field in output, where does it come from?
        // Key: output field name
        // Value: (Node Index, input field name)
        mapping: Vec<(String, (usize, String))>,
    },
}

impl fmt::Debug for NodeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Root => write!(f, "Root"),
            Self::Predict { signature_name, .. } => f
                .debug_struct("Predict")
                .field("signature_name", signature_name)
                .finish(),
            Self::Operator { name } => f.debug_struct("Operator").field("name", name).finish(),
            Self::Map { mapping } => f.debug_struct("Map").field("mapping", mapping).finish(),
        }
    }
}

#[derive(Clone)]
pub struct Node {
    pub id: usize,
    pub node_type: NodeType,
    pub inputs: Vec<usize>, // IDs of parent nodes
    pub output: Option<Prediction>,
    pub input_data: Option<Example>,
}

impl fmt::Debug for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Node")
            .field("id", &self.id)
            .field("node_type", &self.node_type)
            .field("inputs", &self.inputs)
            .field("output", &self.output)
            .field("input_data", &self.input_data)
            .finish()
    }
}

#[derive(Debug, Clone, Default)]
pub struct Graph {
    pub nodes: Vec<Node>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_node(
        &mut self,
        node_type: NodeType,
        inputs: Vec<usize>,
        input_data: Option<Example>,
    ) -> usize {
        let id = self.nodes.len();
        self.nodes.push(Node {
            id,
            node_type,
            inputs,
            output: None,
            input_data,
        });
        id
    }

    pub fn set_output(&mut self, id: usize, output: Prediction) {
        if let Some(node) = self.nodes.get_mut(id) {
            node.output = Some(output);
        }
    }

    pub fn get_node(&self, id: usize) -> Option<&Node> {
        self.nodes.get(id)
    }

    pub fn get_prompt(&self) -> String {
        let mut prompt = String::new();
        prompt.push_str("Execution Graph:\n");
        prompt.push_str("===============\n\n");

        for node in &self.nodes {
            prompt.push_str(&format!("Node {}: ", node.id));

            match &node.node_type {
                NodeType::Root => {
                    prompt.push_str("Root (Initial Input)\n");
                    if let Some(input_data) = &node.input_data {
                        let input_keys: Vec<String> = input_data.data.keys().cloned().collect();
                        prompt.push_str(&format!("  Input fields: {}\n", input_keys.join(", ")));
                    }
                }
                NodeType::Predict {
                    signature_name,
                    signature,
                } => {
                    prompt.push_str(&format!("Predict ({})\n", signature_name));

                    // Show input fields
                    let input_fields = signature.input_fields();
                    if let Some(input_obj) = input_fields.as_object() {
                        let input_keys: Vec<String> = input_obj.keys().cloned().collect();
                        prompt.push_str(&format!("  Input fields: {}\n", input_keys.join(", ")));
                    }

                    // Show output fields
                    let output_fields = signature.output_fields();
                    if let Some(output_obj) = output_fields.as_object() {
                        let output_keys: Vec<String> = output_obj.keys().cloned().collect();
                        prompt.push_str(&format!("  Output fields: {}\n", output_keys.join(", ")));
                    }

                    // Show instruction if available
                    let instruction = signature.instruction();
                    if !instruction.is_empty() {
                        prompt.push_str(&format!("  Instruction: {}\n", instruction));
                    }

                    // Show dependencies
                    if !node.inputs.is_empty() {
                        prompt.push_str(&format!(
                            "  Depends on: Node {}\n",
                            node.inputs
                                .iter()
                                .map(|id| id.to_string())
                                .collect::<Vec<_>>()
                                .join(", Node ")
                        ));
                    }
                }
                NodeType::Map { mapping } => {
                    prompt.push_str("Map (Data Transformation)\n");
                    prompt.push_str("  Mapping:\n");
                    for (output_key, (source_node_id, source_key)) in mapping {
                        prompt.push_str(&format!(
                            "    {} <- Node {}[{}]\n",
                            output_key, source_node_id, source_key
                        ));
                    }
                    if !node.inputs.is_empty() {
                        prompt.push_str(&format!(
                            "  Depends on: Node {}\n",
                            node.inputs
                                .iter()
                                .map(|id| id.to_string())
                                .collect::<Vec<_>>()
                                .join(", Node ")
                        ));
                    }
                }
                NodeType::Operator { name } => {
                    prompt.push_str(&format!("Operator ({})\n", name));
                    if !node.inputs.is_empty() {
                        prompt.push_str(&format!(
                            "  Depends on: Node {}\n",
                            node.inputs
                                .iter()
                                .map(|id| id.to_string())
                                .collect::<Vec<_>>()
                                .join(", Node ")
                        ));
                    }
                }
            }

            // Show output if available
            if let Some(output) = &node.output {
                let output_keys: Vec<String> = output.data.keys().cloned().collect();
                if !output_keys.is_empty() {
                    prompt.push_str(&format!("  Output keys: {}\n", output_keys.join(", ")));
                }
            }

            prompt.push('\n');
        }

        // Add execution flow summary
        prompt.push_str("Execution Flow:\n");
        prompt.push_str("===============\n");

        // Show sequential flow with dependencies
        for node in &self.nodes {
            let node_desc = match &node.node_type {
                NodeType::Root => "Root".to_string(),
                NodeType::Predict { signature_name, .. } => format!("Predict({})", signature_name),
                NodeType::Map { .. } => "Map".to_string(),
                NodeType::Operator { name } => format!("Operator({})", name),
            };

            if !node.inputs.is_empty() {
                prompt.push_str(&format!(
                    "Node {} → Node {} ({})\n",
                    node.inputs
                        .iter()
                        .map(|id| id.to_string())
                        .collect::<Vec<_>>()
                        .join(", "),
                    node.id,
                    node_desc
                ));
            } else {
                prompt.push_str(&format!("Node {} ({})\n", node.id, node_desc));
            }
        }
        prompt.push('\n');

        prompt
    }

    /// Split a Predict node into multiple nodes with specified signatures
    ///
    /// # Arguments
    /// * `node_id` - ID of the Predict node to split
    /// * `split_points` - Vector of split configurations (name, inputs, outputs)
    pub fn split_predict_node(
        &mut self,
        node_id: usize,
        split_points: Vec<serde_json::Value>,
    ) -> Result<Vec<usize>> {
        // Capture initial node count to identify new nodes later
        let initial_node_count = self.nodes.len();

        let node = self
            .get_node(node_id)
            .ok_or_else(|| anyhow::anyhow!("Node {} not found", node_id))?;

        let original_signature = match &node.node_type {
            NodeType::Predict { signature, .. } => signature,
            _ => return Err(anyhow::anyhow!("Node {} is not a Predict node", node_id)),
        };

        // Split the signature
        let split_sigs = split_signature(original_signature.as_ref(), split_points.clone())?;
        if split_sigs.is_empty() {
            return Err(anyhow::anyhow!("Split resulted in no signatures"));
        }

        let mut new_node_ids = Vec::new();

        // 1. Update the original node to be the FIRST split node
        let first_split = &split_sigs[0];
        let first_meta = &split_points[0];
        let first_name = first_meta["name"].as_str().unwrap_or("Split_0").to_string();

        if let Some(node) = self.nodes.get_mut(node_id) {
            node.node_type = NodeType::Predict {
                signature_name: first_name.clone(),
                signature: first_split.clone(),
            };
        }
        new_node_ids.push(node_id);

        let mut previous_node_id = node_id;
        let mut previous_outputs: Vec<String> = first_meta["outputs"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .map(|v| v.as_str().unwrap_or("").to_string())
                    .collect()
            })
            .unwrap_or_default();

        // 2. Create subsequent nodes chained sequentially
        for i in 1..split_sigs.len() {
            let sig = &split_sigs[i];
            let meta = &split_points[i];
            let name = meta["name"]
                .as_str()
                .unwrap_or(&format!("Split_{}", i))
                .to_string();
            let outputs: Vec<String> = meta["outputs"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .map(|v| v.as_str().unwrap_or("").to_string())
                        .collect()
                })
                .unwrap_or_default();

            // Create map node connecting previous -> current
            // We assume strict sequential dependency: previous outputs -> current inputs
            // (Or at least available in context)
            let map_mapping: Vec<(String, (usize, String))> = previous_outputs
                .iter()
                .map(|field| (field.clone(), (previous_node_id, field.clone())))
                .collect();

            let map_node_id = self.add_node(
                NodeType::Map {
                    mapping: map_mapping,
                },
                vec![previous_node_id],
                None,
            );

            // Create the new Predict node
            let new_id = self.add_node(
                NodeType::Predict {
                    signature_name: name,
                    signature: sig.clone(),
                },
                vec![map_node_id],
                None,
            );

            new_node_ids.push(new_id);
            previous_node_id = new_id;
            previous_outputs = outputs;
        }

        // 3. Update downstream dependencies
        // All nodes that depended on the original `node_id` should now depend on the LAST new node
        // because that's where the final flow ends up (conceptually).
        // Unless the downstream node needed intermediate outputs?
        // For simplicity in this linear split model, we reroute all children to the tail.
        let last_node_id = *new_node_ids.last().unwrap();

        if last_node_id != node_id {
            // Identify external dependents: nodes that existed before (id < initial_node_count)
            // and depend on node_id, EXCLUDING node_id itself (which is now Split 0).
            for i in 0..initial_node_count {
                if i == node_id {
                    continue;
                }

                if let Some(node) = self.nodes.get_mut(i) {
                    if node.inputs.contains(&node_id) {
                        node.inputs = node
                            .inputs
                            .iter()
                            .map(|&id| if id == node_id { last_node_id } else { id })
                            .collect();
                    }
                }
            }
        }

        Ok(new_node_ids)
    }

    /// Fuse multiple Predict nodes into one with a merged signature
    ///
    /// # Arguments
    /// * `node_ids` - List of IDs of Predict nodes to fuse
    /// * `merged_name` - Name for the merged signature
    pub fn fuse_predict_nodes(&mut self, node_ids: &[usize], merged_name: String) -> Result<usize> {
        if node_ids.is_empty() {
            return Err(anyhow::anyhow!("No nodes to fuse"));
        }

        // Verify all are Predict nodes and gather signatures
        let mut signatures = Vec::new();
        let mut all_inputs = Vec::new();

        for &id in node_ids {
            let node = self
                .get_node(id)
                .ok_or_else(|| anyhow::anyhow!("Node {} not found", id))?;

            match &node.node_type {
                NodeType::Predict { signature, .. } => {
                    signatures.push(signature.clone());
                }
                _ => return Err(anyhow::anyhow!("Node {} is not a Predict node", id)),
            }

            // Collect inputs from all nodes to merge dependencies
            // Logic: The fused node depends on the union of dependencies of all fused nodes,
            // MINUS dependencies that are internal to the group (i.e. if A->B and we fuse A,B, dependency A is internal).
            for &inp in &node.inputs {
                if !node_ids.contains(&inp) && !all_inputs.contains(&inp) {
                    all_inputs.push(inp);
                }
            }
        }

        // Merge signatures
        let sig_refs: Vec<&dyn MetaSignature> = signatures.iter().map(|s| s.as_ref()).collect();
        let merged_sig = fuse_signatures(&sig_refs);

        // Use the first node ID as the ID for the fused node
        let fused_node_id = node_ids[0];

        // Update the first node to be the Fused node
        if let Some(node) = self.nodes.get_mut(fused_node_id) {
            node.node_type = NodeType::Predict {
                signature_name: merged_name,
                signature: merged_sig,
            };
            node.inputs = all_inputs;
        }

        // Mark other nodes as Removed and update dependencies
        for &id in &node_ids[1..] {
            // Reroute anything pointing to this node to point to the fused node
            // We need to do this carefully: iterating over all nodes
            // Use a separate loop to avoid borrow issues if possible, or indices

            // Mark as removed
            if let Some(node) = self.nodes.get_mut(id) {
                node.node_type = NodeType::Operator {
                    name: "Removed".to_string(),
                };
            }
        }

        // Reroute dependencies globally
        for node in &mut self.nodes {
            // If this node is NOT one of the removed ones (or the fused one itself, though it shouldn't point to removed nodes)
            if !node_ids[1..].contains(&node.id) {
                let mut changed = false;
                let new_inputs: Vec<usize> = node
                    .inputs
                    .iter()
                    .map(|&inp| {
                        if node_ids[1..].contains(&inp) {
                            changed = true;
                            fused_node_id
                        } else {
                            inp
                        }
                    })
                    .collect();

                if changed {
                    // Dedup inputs just in case
                    let mut unique_inputs = new_inputs.clone();
                    unique_inputs.sort();
                    unique_inputs.dedup();
                    node.inputs = unique_inputs;
                }
            }
        }

        Ok(fused_node_id)
    }
}
