/*
Example: ReAct — the think/act/observe loop as a drop-in strategy.

`ReAct<S>` runs a bounded loop over any signature: the model reads the input,
decides whether it needs a tool, calls it, reads the observation, and repeats
until it finishes (or hits `max_steps`). Same contract as `Predict<S>` /
`ChainOfThought<S>` — the signature never hears about the strategy.

Run with:
```
cargo run --example 19-react
```
*/

use anyhow::Result;
use dspy_rs::{LM, Module, ReAct, Signature, configure, init_tracing};

/// Answer the customer's support question. Use the tools to look up facts;
/// report only what you actually found.
#[derive(Signature, Clone, Debug)]
struct SupportAnswer {
    #[input]
    question: String,

    #[output]
    answer: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing()?;

    configure(
        LM::builder()
            .model("openai:gpt-4o-mini".to_string())
            .build()
            .await?,
    );

    // A tool is a name, a description the model reads, and an async closure.
    // `max_steps` is the leash: the loop always terminates.
    let agent = ReAct::<SupportAnswer>::builder()
        .tool(
            "order_lookup",
            "Look up an order by its number. Returns status, carrier, and ETA.",
            |args: String| async move {
                let order_id: String = args
                    .trim()
                    .trim_matches(|c| c == '"' || c == '\'')
                    .chars()
                    .filter(char::is_ascii_digit)
                    .collect();
                match order_id.as_str() {
                    "4127" => "status: shipped, carrier: UPS, eta: Thursday".to_string(),
                    other => format!("no order with id {other}"),
                }
            },
        )
        .max_steps(4)
        .build();

    let predicted = agent
        .call(SupportAnswerInput {
            question: "Where is my order? I paid for the ceramic starter jar two weeks ago. \
                       Order #4127."
                .to_string(),
        })
        .await?;

    println!("answer: {}", predicted.answer);

    // The trajectory (thoughts, actions, observations) rides in metadata.
    let metadata = predicted.metadata();
    println!("\ntool calls: {}", metadata.tool_calls.len());
    for entry in &metadata.tool_executions {
        println!("---\n{entry}");
    }

    Ok(())
}
