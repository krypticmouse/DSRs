/*
Front Desk, Act II (the `#[module]` lane): one parse, two projections.

The chapter 6/7 milestone from the Learn book: steps as bodyless functions,
an ordinary async fn body compiled into an IR program, the privacy scrub
extracted as a typed host hole, and the map printed as canonical `.dsrs` text.

This example shows:
1. `#[predict]` / `#[cot(model = "@strong")]` steps — the fn IS the signature
2. `#[module]`: the same source projected twice (executable + program)
3. `frontdesk::program().to_dsrs()` — the map, with the hole marked `extern`
4. `frontdesk::OPACITY` — every expression the macro could not lower
5. Binding the `@strong` handle at run time (the runner decides what it means)

Run with:
```
cargo run --example 22-frontdesk-module
```
*/

use std::sync::Arc;

use anyhow::Result;
use dspy_rs::ir::{Budget, Interpreter};
use dspy_rs::trace::JsonMap;
use dspy_rs::{LM, configure, cot, init_tracing, module, predict};

// ---------------------------------------------------------------------------
// Steps: the smallest legible unit
// ---------------------------------------------------------------------------

/// One sentence a busy human reads instead of the ticket.
#[predict]
fn summarize(ticket: String) -> String;

/// Write a warm, concrete reply to this Sourdough support ticket.
/// Never promise a refund. Sign off as "The Sourdough Team".
#[cot(model = "@strong")]
fn draft(ticket: String, summary: String) -> String;

// ---------------------------------------------------------------------------
// The module: an ordinary async fn, read twice
// ---------------------------------------------------------------------------

#[dspy_rs::Schema]
#[derive(Debug)]
pub struct DeskOut {
    pub summary: String,
    pub reply: String,
}

#[module]
async fn frontdesk(ticket: String) -> Result<DeskOut, dspy_rs::ir::RunError> {
    // scrub email addresses before anything leaves the building
    let clean: String = {
        let t: String = ticket; // the type anchor for the incoming value
        t.split_whitespace()
            .map(|w| if w.contains('@') { "[email]" } else { w })
            .collect::<Vec<&str>>()
            .join(" ")
    };

    let sum = summarize(clean.clone()).await?;
    let drafter = draft(clean.clone(), sum.summarize.clone()).await?;

    Ok(DeskOut {
        summary: sum.summarize,
        reply: drafter.draft,
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing()?;

    // Projection two: the map. Built from source at compile time, linked on
    // first use — no API key, no model call.
    println!("=== frontdesk.dsrs ===\n");
    println!("{}", frontdesk::program().to_dsrs());

    // Every dragon confesses: the scrub is a host hole with a typed border.
    println!("=== opacity ===\n");
    for hole in frontdesk::OPACITY {
        println!("{} ({}): {}", hole.name, hole.kind, hole.excerpt);
    }

    if std::env::var("OPENAI_API_KEY").is_err() {
        println!("\nset OPENAI_API_KEY to run the pipeline");
        return Ok(());
    }

    // Projection one: the function. `@strong` is a named handle, not a model
    // id — whoever runs the program decides what it means. `frontdesk::env()`
    // already carries the hole binding and the default model; we add strong.
    configure(
        LM::builder()
            .model("openai:gpt-4o-mini".to_string())
            .build()
            .await?,
    );
    let strong = Arc::new(
        LM::builder()
            .model("openai:gpt-4o".to_string())
            .build()
            .await?,
    );

    let interp = Interpreter::load(
        frontdesk::program().clone(),
        frontdesk::env().bind_model("strong", strong),
    )
    .await?;

    let mut input = JsonMap::new();
    input.insert(
        "ticket".to_string(),
        serde_json::json!(
            "I updated last night and now my starter tracker shows 0 days. \
             Reach me at herman.fan@example.com. Herman is FOUR YEARS OLD."
        ),
    );
    let out = interp.run(input, None, Budget::default()).await?;

    println!("\nsummary: {}", out["summary"].as_str().unwrap_or_default());
    println!("reply:\n{}", out["reply"].as_str().unwrap_or_default());

    Ok(())
}
