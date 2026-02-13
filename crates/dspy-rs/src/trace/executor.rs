use crate::trace::dag::{Graph, NodeType};
use crate::{Prediction, RawExample};
use anyhow::Result;
use std::collections::HashMap;

/// Replays a traced execution graph with new input data.
///
/// Takes a [`Graph`] captured by [`trace()`](crate::trace::trace) and re-runs it with
/// a new root input to see how data flows through a pipeline with different inputs.
///
/// Only `Root` and `Map` nodes produce useful output right now — `Predict` nodes
/// can't replay because the signature type isn't stored in the trace (they'll error),
/// and `Operator` nodes are skipped. This covers data-routing inspection but not
/// full program replay. Returns the output of the last node only.
pub struct Executor {
    pub graph: Graph,
}

impl Executor {
    pub fn new(graph: Graph) -> Self {
        Self { graph }
    }

    pub async fn execute(&self, root_input: RawExample) -> Result<Vec<Prediction>> {
        let mut node_outputs: HashMap<usize, Prediction> = HashMap::new();

        for node in &self.graph.nodes {
            match &node.node_type {
                NodeType::Root => {
                    // Node 0 gets the caller-supplied input; other Root nodes use
                    // their captured input_data (constants from the original trace).
                    if node.id == 0 {
                        let pred = Prediction::from(
                            root_input
                                .data
                                .iter()
                                .map(|(k, v)| (k.clone(), v.clone()))
                                .collect::<Vec<_>>(),
                        );
                        node_outputs.insert(node.id, pred);
                    } else if let Some(data) = &node.input_data {
                        let pred = Prediction::from(
                            data.data
                                .iter()
                                .map(|(k, v)| (k.clone(), v.clone()))
                                .collect::<Vec<_>>(),
                        );
                        node_outputs.insert(node.id, pred);
                    }
                }
                NodeType::Predict { signature_name } => {
                    return Err(anyhow::anyhow!(
                        "Cannot execute traced Predict node for {signature_name}: signature data is not stored"
                    ));
                }
                NodeType::Map { mapping } => {
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
                NodeType::Operator { .. } => {}
            }
        }

        if let Some(last_node) = self.graph.nodes.last()
            && let Some(output) = node_outputs.get(&last_node.id)
        {
            return Ok(vec![output.clone()]);
        }

        Ok(vec![])
    }
}
