use crate::trace::dag::{Graph, NodeType};
use crate::{Example, Prediction};
use anyhow::Result;
use std::collections::HashMap;

pub struct Executor {
    pub graph: Graph,
}

impl Executor {
    pub fn new(graph: Graph) -> Self {
        Self { graph }
    }

    pub async fn execute(&self, root_input: Example) -> Result<Vec<Prediction>> {
        // Simple execution: assume graph nodes are in topological order (which they are by construction of trace)
        // Store outputs of each node
        let mut node_outputs: HashMap<usize, Prediction> = HashMap::new();
        // Store input example for root node 0 (if valid)
        // Actually, Root node 0 usually contains the input data from trace.
        // If we want to run with NEW input, we replace Root's data.

        // We will return the output of the *last* node(s), or just all predictions?
        // Usually we want the leaf nodes.

        for node in &self.graph.nodes {
            match &node.node_type {
                NodeType::Root => {
                    // For root, we use the provided root_input
                    // But wait, the graph might have multiple roots or specific inputs?
                    // For simplicity, assume node 0 is the main root and takes root_input.
                    // Or we check if node.id == 0.
                    if node.id == 0 {
                        // Creating a "Prediction" that just holds the input data, so downstream nodes can read it.
                        // Wait, Prediction structure is for outputs.
                        // But Map nodes read from "Prediction" or "Example"?
                        // Map inputs come from `TrackedValue`, which stores (node_id, key).
                        // If node_id points to Root, we need to get data from Root.
                        // We can synthesize a Prediction from Example data for uniform access.

                        let pred = Prediction::from(
                            root_input
                                .data
                                .iter()
                                .map(|(k, v)| (k.clone(), v.clone()))
                                .collect::<Vec<_>>(),
                        );
                        node_outputs.insert(node.id, pred);
                    } else {
                        // Other roots? maybe constants?
                        if let Some(data) = &node.input_data {
                            let pred = Prediction::from(
                                data.data
                                    .iter()
                                    .map(|(k, v)| (k.clone(), v.clone()))
                                    .collect::<Vec<_>>(),
                            );
                            node_outputs.insert(node.id, pred);
                        }
                    }
                }
                NodeType::Predict { signature_name } => {
                    return Err(anyhow::anyhow!(
                        "Cannot execute traced Predict node for {signature_name}: signature data is not stored"
                    ));
                }
                NodeType::Map { mapping } => {
                    // Execute the mapping
                    // We create a new "Prediction" (acting as data container) based on sources.
                    let mut data = HashMap::new();

                    for (output_key, (source_node_id, source_key)) in mapping {
                        if let Some(source_pred) = node_outputs.get(source_node_id) {
                            let val = source_pred.get(source_key, None);
                            data.insert(output_key.clone(), val);
                        }
                    }

                    let result = Prediction::from(
                        data.iter()
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect::<Vec<_>>(),
                    );
                    node_outputs.insert(node.id, result);
                }
                NodeType::Operator { .. } => {
                    // Not implemented yet
                }
            }
        }

        // Return the output of the last node? or all Predict outputs?
        // Let's return the output of the last node in the list.
        if let Some(last_node) = self.graph.nodes.last()
            && let Some(output) = node_outputs.get(&last_node.id)
        {
            return Ok(vec![output.clone()]);
        }

        Ok(vec![])
    }
}
