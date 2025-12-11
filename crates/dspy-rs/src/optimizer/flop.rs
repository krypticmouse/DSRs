use crate as dspy_rs;
use crate::core::lm::LM;
use crate::data::example::Example;
use crate::optimizer::{FieldDef, Optimizer};
use crate::trace;
use crate::trace::dag::{Graph, NodeType};
use crate::trace::signature_utils::modify_signature;
use crate::{Evaluator, Module, Optimizable, Predict, Predictor, example};
use anyhow::Result;
use bon::Builder;
use dsrs_macros::Signature;
use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::Deserialize;
use serde_json::json;
use std::fmt;
use std::sync::{Arc, Mutex};

// ============================================================================
// Signatures
// ============================================================================

#[Signature]
struct MetaPlannerSignature {
    /// You are a Meta Planner for a modular AI system.
    /// Please propose a high-level Reference Plan or dependency graph for this task.
    /// Identify key steps, their inputs/outputs, and how they should be connected.
    /// Output the plan as a clear, numbered list of steps.

    #[input(desc = "Description of the task")]
    pub task_description: String,

    #[output(desc = "Proposed plan")]
    pub reference_plan: String,
}

#[Signature]
struct DagOptimizerSignature {
    /// You are a DAG Optimizer. Your goal is to restructure the Execution Graph to match the Reference Plan.
    ///
    /// CRITICAL: The Reference Plan has MORE STEPS than the current graph. You MUST use split_node to break
    /// down nodes that do too much into smaller, focused nodes.
    ///
    /// Available tools (USE THEM!):
    /// 1. split_node: MOST IMPORTANT - Split a node into multiple sequential nodes. Use this when a node
    ///    does multiple things that should be separate steps (like "retrieve AND process" should be two nodes).
    ///    Example: split_node(node_id=1, split_points=[{name: "Step1", inputs: ["x"], outputs: ["y"]}, {name: "Step2", inputs: ["y"], outputs: ["z"]}])
    /// 2. fuse_nodes: Combine closely related nodes into one.
    /// 3. modify_node: Add/update instructions for a node.
    ///
    /// Strategy:
    /// 1. Compare the number of steps in Reference Plan vs nodes in current graph
    /// 2. If Reference Plan has more steps, use split_node to add more nodes
    /// 3. Each step in Reference Plan should ideally map to one Predict node
    /// 4. After structural changes, use modify_node to refine instructions
    /// 5. Only say 'DONE' when graph structure matches Reference Plan
    ///
    /// DO NOT just modify instructions and say DONE - you must restructure the graph first!

    #[input(desc = "Reference Plan - count the steps and match graph structure to it")]
    pub reference_plan: String,

    #[input(desc = "Current Execution Graph - compare node count to Reference Plan steps")]
    pub current_graph: String,

    #[input(desc = "History of actions taken - avoid repeating, build on previous work")]
    pub history: String,

    #[output(
        desc = "Analysis: How many steps in plan vs nodes in graph? What structural change is needed?"
    )]
    pub thought: String,

    #[output(
        desc = "'DONE' only when graph structure matches plan, otherwise describe action taken"
    )]
    pub status: String,
}

// ============================================================================
// Tool Error
// ============================================================================

#[derive(Debug)]
struct ToolError(String);

impl fmt::Display for ToolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tool error: {}", self.0)
    }
}

impl std::error::Error for ToolError {}

// ============================================================================
// Graph Tools
// ============================================================================

#[derive(Deserialize)]
struct SplitNodeArgs {
    node_id: usize,
    split_points: Vec<SplitPoint>,
}

#[derive(Deserialize)]
struct SplitPoint {
    name: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
}

#[derive(Deserialize)]
struct FuseNodeArgs {
    node_ids: Vec<usize>,
    merged_name: String,
}

#[derive(Deserialize)]
struct ModifyNodeArgs {
    node_id: usize,
    new_instruction: Option<String>,
    add_inputs: Option<Vec<FieldDef>>,
    add_outputs: Option<Vec<FieldDef>>,
    remove_fields: Option<Vec<String>>,
}

struct SplitNodeTool {
    graph: Arc<Mutex<Graph>>,
}

impl Tool for SplitNodeTool {
    const NAME: &'static str = "split_node";
    type Error = ToolError;
    type Args = SplitNodeArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Splits a Predict node into multiple sequential Predict nodes".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "node_id": { "type": "integer", "description": "ID of the node to split" },
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
                "required": ["node_id", "split_points"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let mut graph = self.graph.lock().unwrap();

        let split_metadata: Vec<serde_json::Value> = args
            .split_points
            .iter()
            .map(|sp| {
                json!({
                    "name": sp.name,
                    "inputs": sp.inputs,
                    "outputs": sp.outputs
                })
            })
            .collect();

        let new_ids = graph
            .split_predict_node(args.node_id, split_metadata)
            .map_err(|e| ToolError(e.to_string()))?;

        Ok(format!(
            "Successfully split node {} into nodes {:?}",
            args.node_id, new_ids
        ))
    }
}

struct FuseNodeTool {
    graph: Arc<Mutex<Graph>>,
}

impl Tool for FuseNodeTool {
    const NAME: &'static str = "fuse_nodes";
    type Error = ToolError;
    type Args = FuseNodeArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Fuses multiple Predict nodes into one".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "node_ids": { "type": "array", "items": { "type": "integer" }, "description": "IDs of nodes to fuse" },
                    "merged_name": { "type": "string", "description": "Name for the merged node" }
                },
                "required": ["node_ids", "merged_name"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let mut graph = self.graph.lock().unwrap();
        let id = graph
            .fuse_predict_nodes(&args.node_ids, args.merged_name)
            .map_err(|e| ToolError(e.to_string()))?;
        Ok(format!(
            "Successfully fused nodes {:?} into node {}",
            args.node_ids, id
        ))
    }
}

struct ModifyNodeTool {
    graph: Arc<Mutex<Graph>>,
}

impl Tool for ModifyNodeTool {
    const NAME: &'static str = "modify_node";
    type Error = ToolError;
    type Args = ModifyNodeArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Modifies a Predict node's signature".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "node_id": { "type": "integer", "description": "ID of the node to modify" },
                    "new_instruction": { "type": "string", "description": "New instruction text" },
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
                "required": ["node_id"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let mut graph = self.graph.lock().unwrap();

        // Find the node and check if it's a Predict node
        // We need to extract signature first to avoid holding borrow during modify_signature
        let (signature, signature_name) = {
            let node = graph
                .get_node(args.node_id)
                .ok_or_else(|| ToolError(format!("Node {} not found", args.node_id)))?;

            match &node.node_type {
                NodeType::Predict {
                    signature,
                    signature_name,
                } => (signature.clone(), signature_name.clone()),
                _ => {
                    return Err(ToolError(format!(
                        "Node {} is not a Predict node",
                        args.node_id
                    )));
                }
            }
        };

        // Prepare modification args
        let add_inputs_vec = args.add_inputs.map(|fields| {
            fields
                .into_iter()
                .map(|f| {
                    (
                        f.name,
                        json!({
                            "desc": f.desc,
                            "type": f.type_name,
                            "__dsrs_field_type": "input"
                        }),
                    )
                })
                .collect::<Vec<_>>()
        });

        let add_outputs_vec = args.add_outputs.map(|fields| {
            fields
                .into_iter()
                .map(|f| {
                    (
                        f.name,
                        json!({
                            "desc": f.desc,
                            "type": f.type_name,
                            "__dsrs_field_type": "output"
                        }),
                    )
                })
                .collect::<Vec<_>>()
        });

        // Apply modification
        let modified_sig = modify_signature(
            signature.as_ref(),
            args.new_instruction,
            add_inputs_vec.as_deref(),
            add_outputs_vec.as_deref(),
            args.remove_fields.as_deref(),
        );

        // Update node
        if let Some(node) = graph.nodes.get_mut(args.node_id) {
            node.node_type = NodeType::Predict {
                signature_name: format!("{}_modified", signature_name),
                signature: modified_sig,
            };
        }

        Ok(format!("Successfully modified node {}", args.node_id))
    }
}

// ============================================================================
// FLOP Optimizer
// ============================================================================

#[derive(Builder)]
pub struct FLOP {
    pub lm: LM,
    pub task_description: String,
    #[builder(default = 5)]
    pub max_steps: usize,
}

impl FLOP {
    async fn generate_reference_plan(&self) -> Result<String> {
        let signature = MetaPlannerSignature::new();
        // Instruction is now part of the signature definition

        let predictor = Predict::new(signature);
        let lm = Arc::new(self.lm.clone());

        let input = example! {
            "task_description": "input" => self.task_description.clone()
        };

        let res = predictor.forward_with_config(input, lm).await?;
        Ok(res.get("reference_plan", Some("")).to_string())
    }

    async fn optimize_dag(&self, initial_graph: Graph, reference_plan: &str) -> Result<Graph> {
        let graph = Arc::new(Mutex::new(initial_graph));
        let lm = Arc::new(self.lm.clone());
        let mut history = Vec::new();

        println!(
            "FLOP: Starting DAG optimization loop (max {} steps)",
            self.max_steps
        );

        for i in 0..self.max_steps {
            println!("\n--- FLOP Step {} ---", i + 1);

            // Define tools inside the loop so they're dropped after each iteration
            let tools: Vec<Box<dyn rig::tool::ToolDyn>> = vec![
                Box::new(SplitNodeTool {
                    graph: graph.clone(),
                }),
                Box::new(FuseNodeTool {
                    graph: graph.clone(),
                }),
                Box::new(ModifyNodeTool {
                    graph: graph.clone(),
                }),
            ];

            let signature = DagOptimizerSignature::new();
            let predictor = Predict::new_with_tools(signature, tools);

            // Get current graph state for prompt
            let graph_prompt = {
                let g = graph.lock().unwrap();
                g.get_prompt()
            };

            let history_str = if history.is_empty() {
                "No actions taken yet.".to_string()
            } else {
                history.join("\n")
            };

            let input = example! {
                "reference_plan": "input" => reference_plan.to_string(),
                "current_graph": "input" => graph_prompt,
                "history": "input" => history_str
            };

            // Call Predict
            let response = predictor.forward_with_config(input, lm.clone()).await?;

            let thought = response
                .get("thought", Some(""))
                .as_str()
                .unwrap_or("")
                .to_string();
            let status = response
                .get("status", Some(""))
                .as_str()
                .unwrap_or("")
                .to_string();

            println!("  Thought: {}", thought);
            println!("  Status: {}", status);

            // Check current graph state
            let current_node_count = {
                let g = graph.lock().unwrap();
                g.nodes
                    .iter()
                    .filter(|n| matches!(n.node_type, NodeType::Predict { .. }))
                    .count()
            };
            println!("  Predict nodes in graph: {}", current_node_count);

            // Log history
            history.push(format!(
                "Step {}: Thought: {} | Status: {}",
                i + 1,
                thought,
                status
            ));

            // Check if done
            if status.contains("DONE") {
                // Warn if done too early without structural changes
                if i == 0 && current_node_count <= 2 {
                    println!(
                        "FLOP: WARNING - Optimizer said DONE on first step with only {} Predict nodes",
                        current_node_count
                    );
                    println!(
                        "FLOP: The graph structure may not have been optimized. Continuing..."
                    );
                    continue; // Force another iteration
                }
                println!("FLOP: Optimizer signaled DONE");
                break;
            }

            // Show updated graph
            {
                let g = graph.lock().unwrap();
                println!("  Current graph nodes: {}", g.nodes.len());
            }

            // predictor and tools are dropped here at end of loop iteration
        }

        // Extract the final graph
        // Try to unwrap, or clone if there are still references
        let final_graph = match Arc::try_unwrap(graph) {
            Ok(mutex) => mutex.into_inner().unwrap(),
            Err(arc) => {
                // Still has references, clone the graph instead
                arc.lock().unwrap().clone()
            }
        };

        Ok(final_graph)
    }

    /// Compile and return the optimized graph directly.
    /// Unlike the Optimizer trait's compile(), this returns the Graph
    /// which can then be executed using the Executor.
    pub async fn compile_and_get_graph<M>(
        &self,
        module: &M,
        trainset: Vec<Example>,
    ) -> Result<Graph>
    where
        M: Module + Send + Sync,
    {
        if trainset.is_empty() {
            return Err(anyhow::anyhow!(
                "FLOP requires at least one example in trainset to trace execution"
            ));
        }

        // 1. Trace execution using first example
        let example_input = trainset[0].clone();

        println!("FLOP: Tracing module...");
        let (_, initial_graph) =
            trace::trace(|| async { module.forward(example_input.clone()).await }).await;

        println!("\n=== INITIAL GRAPH (BEFORE optimization) ===");
        println!("{}", initial_graph.get_prompt());

        println!("\nFLOP: Generating reference plan...");
        let reference_plan = self.generate_reference_plan().await?;
        println!("Reference Plan:\n{}", reference_plan);

        println!("FLOP: Optimizing DAG...");
        let optimized_graph = self.optimize_dag(initial_graph, &reference_plan).await?;

        println!("\n=== OPTIMIZED GRAPH (AFTER optimization) ===");
        println!("{}", optimized_graph.get_prompt());
        println!("FLOP: Optimization complete.\n");

        Ok(optimized_graph)
    }
}

impl Optimizer for FLOP {
    async fn compile<M>(&self, module: &mut M, trainset: Vec<Example>) -> Result<()>
    where
        M: Module + Optimizable + Evaluator,
    {
        if trainset.is_empty() {
            return Err(anyhow::anyhow!(
                "FLOP requires at least one example in trainset to trace execution"
            ));
        }

        // 1. Trace execution using first example
        let example_input = trainset[0].clone();

        println!("FLOP: Tracing module...");
        let (_, initial_graph) =
            trace::trace(|| async { module.forward(example_input.clone()).await }).await;

        println!("FLOP: Generating reference plan...");
        let reference_plan = self.generate_reference_plan().await?;
        println!("Reference Plan:\n{}", reference_plan);

        println!("FLOP: Optimizing DAG...");
        let optimized_graph = self.optimize_dag(initial_graph, &reference_plan).await?;

        println!("FLOP: Optimization complete.");
        println!("Optimized Graph:\n{}", optimized_graph.get_prompt());

        // TODO: How to apply this back to M?
        // M is a static Rust struct. We can't change its code.
        // If M supports loading a graph, we would do it here.
        // For now, we just output the graph.

        Ok(())
    }
}
