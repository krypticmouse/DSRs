use crate::{Prediction, RawExample};
use std::fmt;

/// The kind of operation a trace node represents.
#[derive(Clone)]
pub enum NodeType {
    /// The entry point — holds the initial input data.
    Root,
    /// An LM call through [`Predict`](crate::Predict).
    Predict {
        /// The `type_name::<S>()` of the signature.
        signature_name: String,
    },
    /// A user-defined operation (custom module logic between Predict calls).
    Operator {
        /// Human-readable name for the operation.
        name: String,
    },
    /// A field-level data routing between nodes.
    ///
    /// Each entry maps an output field name to `(source_node_id, source_field_name)`.
    Map {
        mapping: Vec<(String, (usize, String))>,
    },
}

impl fmt::Debug for NodeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Root => write!(f, "Root"),
            Self::Predict { signature_name } => f
                .debug_struct("Predict")
                .field("signature_name", signature_name)
                .finish(),
            Self::Operator { name } => f.debug_struct("Operator").field("name", name).finish(),
            Self::Map { mapping } => f.debug_struct("Map").field("mapping", mapping).finish(),
        }
    }
}

/// A single node in the execution trace graph.
///
/// Nodes are created by [`record_node`](crate::trace::record_node) during a
/// [`trace()`](crate::trace::trace) scope. Each node has a type, links to parent
/// nodes (inputs), and optionally captures the output data.
#[derive(Clone)]
pub struct Node {
    /// Unique ID within this graph (assigned sequentially).
    pub id: usize,
    /// What kind of operation this node represents.
    pub node_type: NodeType,
    /// IDs of parent nodes whose outputs feed into this node.
    pub inputs: Vec<usize>,
    /// The output produced by this node (set after execution completes).
    pub output: Option<Prediction>,
    /// The input data passed to this node (for Root nodes).
    pub input_data: Option<RawExample>,
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

/// A directed acyclic graph of execution trace nodes.
///
/// Built incrementally during a [`trace()`](crate::trace::trace) scope as each
/// [`Predict`](crate::Predict) call records itself. Nodes are stored in insertion
/// order, which is topological order by construction (a node is always recorded
/// after its inputs).
///
/// This is a record of what actually happened, not a mutable program topology.
#[derive(Debug, Clone, Default)]
pub struct Graph {
    /// Nodes in insertion (topological) order.
    pub nodes: Vec<Node>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a node and returns its ID.
    ///
    /// The ID is the node's index in the `nodes` vec. IDs in `inputs` must refer
    /// to previously added nodes (this is not validated — the graph trusts the caller).
    pub fn add_node(
        &mut self,
        node_type: NodeType,
        inputs: Vec<usize>,
        input_data: Option<RawExample>,
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
}
