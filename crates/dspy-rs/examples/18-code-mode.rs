/*
Example: Code Mode — many tools, one `run_js` meta-tool.

Instead of advertising N JSON tool schemas and paying one model round trip per
tool call, `ToolSet::code_mode` collapses the tools into a single sandboxed
`run_js` tool. The model writes one JavaScript script that calls the tools as
global functions and composes their results — several tool calls, one turn.

Run with:
```
cargo run --example 18-code-mode
```
*/

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use dspy_rs::{Chat, LM, Message, SandboxConfig, ToolLoopMode, ToolSet, init_tracing};
use rig::completion::ToolDefinition;
use rig::tool::{Tool, ToolDyn};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;

#[derive(Debug)]
struct ToolFailure(String);

impl fmt::Display for ToolFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for ToolFailure {}

// ---------------------------------------------------------------------------
// Tool 1: keyword search over an embedded FAQ (no I/O)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, Serialize)]
struct FaqArgs {
    query: String,
}

#[derive(Clone)]
struct FaqSearchTool;

impl Tool for FaqSearchTool {
    const NAME: &'static str = "faq_search";

    type Error = ToolFailure;
    type Args = FaqArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Search the Sourdough FAQ. Returns the top matching entries.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "query": { "type": "string" } },
                "required": ["query"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let faq = [
            (
                "starter data after update",
                "Starter history is stored locally and survives app updates. \
                 If the tracker shows 0 days, pull to refresh — the data is intact.",
            ),
            (
                "export csv",
                "Settings > Data > Export sends your full feeding log as CSV.",
            ),
            (
                "refund policy",
                "Refunds are handled by the app store you purchased through.",
            ),
        ];
        let query = args.query.to_lowercase();
        let hits: Vec<String> = faq
            .iter()
            .filter(|(topic, _)| query.split_whitespace().any(|word| topic.contains(word)))
            .map(|(topic, answer)| format!("[{topic}] {answer}"))
            .collect();
        if hits.is_empty() {
            Ok("no FAQ entries matched".to_string())
        } else {
            Ok(hits.join("\n"))
        }
    }
}

// ---------------------------------------------------------------------------
// Tool 2: order lookup over an embedded table (no I/O)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, Serialize)]
struct OrderArgs {
    order_id: String,
}

#[derive(Clone)]
struct OrderLookupTool;

impl Tool for OrderLookupTool {
    const NAME: &'static str = "order_lookup";

    type Error = ToolFailure;
    type Args = OrderArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Look up an order by its number. Returns status, carrier, and ETA."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "order_id": { "type": "string" } },
                "required": ["order_id"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let orders: HashMap<&str, &str> = HashMap::from([
            ("4127", "status: shipped, carrier: UPS, eta: Thursday"),
            ("4099", "status: delivered, carrier: DHL, delivered: Monday"),
        ]);
        orders
            .get(args.order_id.as_str())
            .map(|order| order.to_string())
            .ok_or_else(|| ToolFailure(format!("no order with id {}", args.order_id)))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing()?;

    // Collapse both tools into ONE `run_js` tool. Its description lists them
    // as a JavaScript API (faq_search(args), order_lookup(args)).
    let toolset = ToolSet::code_mode(
        vec![
            Arc::new(FaqSearchTool) as Arc<dyn ToolDyn>,
            Arc::new(OrderLookupTool) as Arc<dyn ToolDyn>,
        ],
        SandboxConfig::default(),
    )
    .await?;

    println!("=== the one tool the model sees ===\n");
    for definition in toolset.definitions() {
        println!("name: {}", definition.name);
        println!("{}\n", definition.description);
    }

    let lm = LM::builder()
        .model("openai:gpt-4o-mini".to_string())
        .build()
        .await?;

    let chat = Chat::new(vec![Message::user(
        "Find the FAQ entry about starter data after an update, and look up \
         order 4127. Report both findings.",
    )]);

    // The model should answer with one script that calls both tools inside a
    // single sandbox execution, instead of two JSON tool-call round trips.
    let response = lm
        .call_with_toolset(chat, &toolset, ToolLoopMode::Auto)
        .await?;

    println!("=== run_js round trips: {} ===", response.tool_calls.len());
    for (idx, execution) in response.tool_executions.iter().enumerate() {
        println!("script result {}: {}", idx + 1, execution);
    }

    println!("\n=== final answer ===\n{}", response.output.content());

    Ok(())
}
