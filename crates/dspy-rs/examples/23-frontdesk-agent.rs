/*
Front Desk, Act II finale: tools, an agent with a leash, and capabilities.

The chapter 8/9 milestone from the Learn book: `#[tool]` functions (plain
Rust, embedded data, no I/O), an `#[agent]` step that may call them inside a
bounded loop, and a `#[module]` that grants exactly the capabilities its
tools declared.

This example shows:
1. `#[tool(caps("..."))]` — ordinary functions the model may ask to run
2. `#[agent(tools(...), max_turns = 6)]` — a loop with a leash
3. `#[module(caps(...))]` — the program's capability ceiling, self-granted
   when you run your own binary, re-checked when the file travels
4. The printed artifact carrying `caps { ... }`, the tool decls, and the
   agent node

Run with:
```
cargo run --example 23-frontdesk-agent
```
*/

use anyhow::Result;
use dspy_rs::{LM, agent, configure, cot, init_tracing, module, tool};

// ---------------------------------------------------------------------------
// Tools: buttons with labels (embedded data, no I/O)
// ---------------------------------------------------------------------------

/// Search the Sourdough FAQ. Returns the top matching entries.
#[tool(caps("kb:read"))]
fn faq_search(query: String) -> String {
    let faq = [
        (
            "starter data after update",
            "Starter history is stored locally and survives app updates. \
             If the tracker shows 0 days, pull to refresh — the data is intact.",
        ),
        (
            "ceramic starter jar backorder",
            "Ceramic starter jars ran a two-week backorder in March; all \
             pending orders have now shipped.",
        ),
        (
            "refund policy",
            "Refunds are handled by the app store you purchased through.",
        ),
    ];
    let query = query.to_lowercase();
    let hits: Vec<String> = faq
        .iter()
        .filter(|(topic, _)| query.split_whitespace().any(|word| topic.contains(word)))
        .map(|(topic, answer)| format!("[{topic}] {answer}"))
        .collect();
    if hits.is_empty() {
        "no FAQ entries matched".to_string()
    } else {
        hits.join("\n")
    }
}

/// Look up an order by its number. Returns status, carrier, and ETA.
#[tool(caps("orders:read"))]
async fn order_lookup(order_id: String) -> Result<String, String> {
    let orders = [
        ("4127", "status: shipped, carrier: UPS, eta: Thursday"),
        ("4099", "status: delivered, carrier: DHL, delivered: Monday"),
    ];
    orders
        .iter()
        .find(|(id, _)| *id == order_id)
        .map(|(_, order)| order.to_string())
        .ok_or_else(|| format!("no order with id {order_id}"))
}

// ---------------------------------------------------------------------------
// The agent: a loop with a leash
// ---------------------------------------------------------------------------

/// Research this ticket: find the relevant FAQ entry and, if an order is
/// referenced, its current status. Report only facts you actually found.
#[agent(tools(faq_search, order_lookup), max_turns = 6)]
fn research(ticket: String) -> String;

/// Write a warm, concrete reply to this Sourdough support ticket.
/// Never promise a refund. Sign off as "The Sourdough Team".
#[cot]
fn draft(ticket: String, summary: String) -> String;

// ---------------------------------------------------------------------------
// The module: grant what the tools declared
// ---------------------------------------------------------------------------

#[dspy_rs::Schema]
#[derive(Debug)]
pub struct DeskOut {
    pub summary: String,
    pub reply: String,
}

#[module(caps("kb:read", "orders:read"))]
async fn frontdesk(ticket: String) -> Result<DeskOut, dspy_rs::ir::RunError> {
    // scrub email addresses before anything leaves the building
    let clean: String = {
        let t: String = ticket; // the type anchor for the incoming value
        t.split_whitespace()
            .map(|w| if w.contains('@') { "[email]" } else { w })
            .collect::<Vec<&str>>()
            .join(" ")
    };

    let facts = research(clean.clone()).await?;
    let drafter = draft(clean.clone(), facts.research.clone()).await?;

    Ok(DeskOut {
        summary: facts.research,
        reply: drafter.draft,
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing()?;

    // Tools stay ordinary functions: callable in a unit test, no model near.
    println!("=== faq_search(\"jar\") as plain Rust ===\n");
    println!("{}\n", faq_search("jar".to_string()));

    // The artifact declares its ceiling, its tools, and the agent loop.
    println!("=== frontdesk.dsrs ===\n");
    println!("{}", frontdesk::program().to_dsrs());

    if std::env::var("OPENAI_API_KEY").is_err() {
        println!("set OPENAI_API_KEY to run the pipeline");
        return Ok(());
    }

    configure(
        LM::builder()
            .model("openai:gpt-4o-mini".to_string())
            .build()
            .await?,
    );

    let out = frontdesk(
        "Where is my order?? I paid for the ceramic starter jar TWO WEEKS ago. \
         Order #4127."
            .to_string(),
    )
    .await?;

    println!("summary: {}\n", out.summary);
    println!("reply:\n{}", out.reply);

    Ok(())
}
