/*
Example: Tracing, Modifying, and Re-executing a Graph

This example demonstrates:
1. Tracing a module execution to capture the execution graph
2. Modifying the graph (splitting/fusing nodes, modifying signatures)
3. Re-executing the modified graph with new inputs

Run with:
```
cargo run --example 13-graph-modification
```
*/

use anyhow::Result;
use bon::Builder;
use dspy_rs::{
    ChatAdapter, LM, Module, Predict, Prediction, Predictor, Signature, configure, example,
    prediction,
    trace::dag::NodeType,
    trace::{self, IntoTracked},
};

#[Signature]
struct QASignature {
    #[input]
    pub question: String,
    #[output]
    pub answer: String,
}

#[Signature]
struct RateSignature {
    #[input]
    pub question: String,
    #[input]
    pub answer: String,
    #[output]
    pub rating: i8,
}

#[derive(Builder)]
pub struct QARater {
    #[builder(default = Predict::new(QASignature::new()))]
    pub answerer: Predict,
    #[builder(default = Predict::new(RateSignature::new()))]
    pub rater: Predict,
}

impl Module for QARater {
    async fn forward(&self, inputs: dspy_rs::Example) -> Result<Prediction> {
        let answerer_prediction = self.answerer.forward(inputs.clone()).await?;

        // Use tracked values to preserve lineage
        let question = inputs.data.get("question").unwrap().clone().into_tracked();
        let answer = answerer_prediction.get_tracked("answer");

        let inputs = example! {
            "question": "input" => question.clone(),
            "answer": "input" => answer.clone()
        };

        let rating_prediction = self.rater.forward(inputs).await?;

        Ok(prediction! {
            "answer"=> answer.value,
            "question"=> question.value,
            "rating"=> rating_prediction.data.get("rating").unwrap().clone(),
        }
        .set_lm_usage(rating_prediction.lm_usage))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("Graph Modification Example\n");
    println!("==========================\n");

    // Configure LM
    configure(
        LM::builder()
            .model("openai:gpt-4o-mini".to_string())
            .build()
            .await
            .unwrap(),
        ChatAdapter,
    );

    let module = QARater::builder().build();
    let example_input = example! {
        "question": "input" => "What is the capital of France?",
    };

    // Step 1: Trace the execution
    println!("Step 1: Tracing execution...");
    let (result, mut graph) = trace::trace(|| async { module.forward(example_input).await }).await;

    match result {
        Ok(pred) => println!("  Original prediction keys: {:?}\n", pred.data.keys()),
        Err(e) => println!("  Error (expected if no API key/network): {}\n", e),
    }

    println!("Original Graph:");
    println!("{}", graph.get_prompt());

    // Step 2: Inspect and modify the graph
    println!("\nStep 2: Modifying the graph...\n");

    // Find Predict nodes
    let predict_nodes: Vec<(usize, &dspy_rs::trace::dag::Node)> = graph
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| matches!(node.node_type, NodeType::Predict { .. }))
        .collect();

    println!("Found {} Predict nodes", predict_nodes.len());

    if predict_nodes.len() >= 2 {
        let (first_idx, first_node) = predict_nodes[0];
        let (second_idx, second_node) = predict_nodes[1];

        println!("\n  Node {}: {:?}", first_idx, first_node.node_type);
        println!("  Node {}: {:?}", second_idx, second_node.node_type);

        // Example: Fuse the two predict nodes
        println!("\n  Fusing nodes {} and {}...", first_idx, second_idx);

        if let (NodeType::Predict { .. }, NodeType::Predict { .. }) =
            (&first_node.node_type, &second_node.node_type)
        {
            match graph.fuse_predict_nodes(&[first_idx, second_idx], "FusedQARater".to_string()) {
                Ok(fused_id) => {
                    println!("  ✓ Successfully fused into node {}", fused_id);
                }
                Err(e) => {
                    println!("  ✗ Failed to fuse: {}", e);
                }
            }
        }
    }

    // Step 3: Modify a signature (example: add a field)
    println!("\nStep 3: Modifying signatures...\n");

    // Find Predict nodes and collect indices first to avoid borrow issues
    let predict_indices: Vec<usize> = graph
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| matches!(node.node_type, NodeType::Predict { .. }))
        .map(|(idx, _)| idx)
        .collect();

    // Modify signatures
    use dspy_rs::trace::signature_utils::modify_signature;
    use serde_json::json;

    for idx in predict_indices {
        // Clone the signature info first to avoid borrow conflicts
        let (signature, signature_name) = {
            let node = &graph.nodes[idx];
            if let NodeType::Predict {
                signature,
                signature_name,
            } = &node.node_type
            {
                (signature.clone(), signature_name.clone())
            } else {
                continue;
            }
        };

        println!("  Found Predict node {}: {}", idx, signature_name);

        let new_output = json!({
            "type": "f32",
            "desc": "Confidence score",
            "__dsrs_field_type": "output"
        });

        let modified_sig = modify_signature(
            signature.as_ref(),
            None,
            None,
            Some(&[("confidence".to_string(), new_output)]),
            None,
        );

        // Update the node with modified signature
        if let Some(node) = graph.nodes.get_mut(idx) {
            node.node_type = NodeType::Predict {
                signature_name: format!("{}_modified", signature_name),
                signature: modified_sig,
            };
            println!("  ✓ Modified node {} to include confidence output", idx);
        }
    }

    // Step 4: Show modified graph
    println!("\nStep 4: Modified Graph:");
    println!("{}", graph.get_prompt());

    // Step 5: Execute the modified graph
    println!("\nStep 5: Executing modified graph...\n");
    let executor = dspy_rs::trace::Executor::new(graph);
    let new_input = example! {
        "question": "input" => "What is the capital of Germany?",
    };

    match executor.execute(new_input).await {
        Ok(preds) => {
            if preds.is_empty() {
                println!("⚠ Warning: Executor returned empty predictions");
            } else {
                println!("✓ Executor returned {} prediction(s):", preds.len());
                for (idx, pred) in preds.iter().enumerate() {
                    println!("\n  Prediction {}:", idx + 1);
                    println!("    Keys: {:?}", pred.data.keys());
                    for (key, value) in &pred.data {
                        println!("    {}: {}", key, value);
                    }
                }
            }
        }
        Err(e) => {
            println!("✗ Graph Execution Error: {}", e);
            println!("  Error chain:");
            let mut source = e.source();
            let mut depth = 0;
            while let Some(err) = source {
                depth += 1;
                println!("    {}: {}", depth, err);
                source = err.source();
            }
        }
    }

    Ok(())
}
