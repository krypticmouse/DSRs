use crate::trace::dag::{Graph, NodeType};
use crate::{Example, GLOBAL_SETTINGS, Prediction};
use anyhow::{Context, Result};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

pub struct Executor {
    pub graph: Graph,
}

impl Executor {
    pub fn new(graph: Graph) -> Self {
        Self { graph }
    }

    /// Topologically sort nodes to execute in dependency order
    fn topological_sort(&self) -> Vec<usize> {
        let mut visited = HashSet::new();
        let mut result = Vec::new();
        let mut temp_visited = HashSet::new();

        fn visit(
            node_id: usize,
            graph: &Graph,
            visited: &mut HashSet<usize>,
            temp_visited: &mut HashSet<usize>,
            result: &mut Vec<usize>,
        ) {
            if visited.contains(&node_id) {
                return;
            }
            if temp_visited.contains(&node_id) {
                // Cycle detected, but we'll continue anyway
                return;
            }

            temp_visited.insert(node_id);
            if let Some(node) = graph.nodes.iter().find(|n| n.id == node_id) {
                for &dep_id in &node.inputs {
                    visit(dep_id, graph, visited, temp_visited, result);
                }
            }
            temp_visited.remove(&node_id);
            visited.insert(node_id);
            result.push(node_id);
        }

        for node in &self.graph.nodes {
            if !visited.contains(&node.id) {
                visit(
                    node.id,
                    &self.graph,
                    &mut visited,
                    &mut temp_visited,
                    &mut result,
                );
            }
        }

        result
    }

    pub async fn execute(&self, root_input: Example) -> Result<Vec<Prediction>> {
        let mut node_outputs: HashMap<usize, Prediction> = HashMap::new();
        let mut final_predictions = Vec::new();

        // Get execution order (topological sort)
        let execution_order = self.topological_sort();

        for node_id in execution_order {
            let node = self
                .graph
                .nodes
                .iter()
                .find(|n| n.id == node_id)
                .ok_or_else(|| anyhow::anyhow!("Node {} not found", node_id))?;

            match &node.node_type {
                NodeType::Root => {
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
                NodeType::Predict { signature, .. } => {
                    // Predict nodes can depend on Root or Map nodes
                    // We need to gather all inputs from parent nodes
                    let mut input_data = HashMap::new();

                    for &parent_id in &node.inputs {
                        if let Some(parent_pred) = node_outputs.get(&parent_id) {
                            // Merge all data from parent
                            for (key, value) in &parent_pred.data {
                                input_data.insert(key.clone(), value.clone());
                            }
                        }
                    }

                    // If no inputs found, skip this node
                    if input_data.is_empty() {
                        continue;
                    }

                    let example = Example::new(
                        input_data,
                        vec![], // input_keys
                        vec![], // output_keys
                    );

                    let (adapter, lm) = {
                        let guard = GLOBAL_SETTINGS.read().unwrap();
                        let settings = guard.as_ref().unwrap();
                        (settings.adapter.clone(), Arc::clone(&settings.lm))
                    };

                    let tools = vec![];
                    let result = adapter
                        .call(lm, signature.as_ref(), example, tools)
                        .await
                        .with_context(|| format!("Failed to execute Predict node {}", node.id))?;

                    node_outputs.insert(node.id, result.clone());
                    final_predictions.push(result);
                }
                NodeType::Map { mapping } => {
                    let mut data = HashMap::new();

                    for (output_key, (source_node_id, source_key)) in mapping {
                        if let Some(source_pred) = node_outputs.get(source_node_id) {
                            if let Some(val) = source_pred.data.get(source_key) {
                                data.insert(output_key.clone(), val.clone());
                            }
                        }
                    }

                    let result = Prediction::from(
                        data.iter()
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect::<Vec<_>>(),
                    );
                    node_outputs.insert(node.id, result);
                }
                NodeType::Operator { name } => {
                    // Skip removed nodes
                    if name == "Removed" {
                        continue;
                    }
                    // Other operators not implemented yet
                }
            }
        }

        // Return all Predict node outputs, or the last node's output if no predictions
        if !final_predictions.is_empty() {
            Ok(final_predictions)
        } else if let Some(last_node) = self.graph.nodes.last() {
            if let Some(output) = node_outputs.get(&last_node.id) {
                Ok(vec![output.clone()])
            } else {
                Ok(vec![])
            }
        } else {
            Ok(vec![])
        }
    }
}
