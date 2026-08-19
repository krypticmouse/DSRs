/*
Front Desk, chapter 10 ("The Expedition Log"): capture a run, hold it in your
hand as JSONL, then walk it again.

- `capture` records one span per leaf while the pipeline runs live.
- `ReplayMode::Strict` proves the pipeline still behaves exactly as recorded:
  every call is served from the log, zero provider calls, no API key spent.
- `ReplayMode::UntilDivergence` replays the unchanged prefix free and goes
  live only from the first call your change actually touches. Divergence is
  detected from the request hash, never declared.

Run with:
```
cargo run --example 24-frontdesk-replay
```
(the recording run and the post-divergence calls need OPENAI_API_KEY; the
strict replay itself would work without one)
*/

use std::sync::Arc;

use anyhow::Result;
use dspy_rs::ir::{Instruction, Overlay, with_ambient_overlay};
use dspy_rs::trace::capture;
use dspy_rs::{LM, ReplayMode, Trace, configure, init_tracing, module, predict, replay};

/// Rewrite the ticket with names, emails, and order numbers redacted.
#[predict]
fn scrub(ticket: String) -> String;

/// Draft a short, warm reply to the scrubbed ticket.
#[predict]
fn draft(ticket: String) -> String;

#[dspy_rs::Schema]
#[derive(Debug)]
pub struct FrontDeskOut {
    pub reply: String,
}

#[module]
async fn frontdesk(ticket: String) -> Result<FrontDeskOut, dspy_rs::ir::RunError> {
    let scrubber = scrub(ticket.clone()).await?;
    let drafter = draft(scrubber.scrub.clone()).await?;
    Ok(FrontDeskOut {
        reply: drafter.draft,
    })
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

    let herman_ticket = "My sourdough starter Herman looks dead after the update. \
                         Order #4127. Please help."
        .to_string();

    // --- Capture: run live once, get the run back as a value ---------------
    let (out, trace) = capture(|| frontdesk(herman_ticket.clone())).await;
    let out = out?;
    println!("live reply: {}", out.reply);

    let path = std::env::temp_dir().join("frontdesk-herman.jsonl");
    std::fs::write(&path, trace.to_jsonl()?)?;
    println!("recorded {} spans to {}", trace.spans.len(), path.display());

    // --- Strict replay: yesterday, for free --------------------------------
    let trace = Trace::from_jsonl(&std::fs::read_to_string(&path)?)?;

    let (replayed, report) = replay(&trace, ReplayMode::Strict, || {
        frontdesk(herman_ticket.clone())
    })
    .await;
    let replayed = replayed?;

    assert_eq!(report.live, 0, "strict replay never calls a provider");
    assert_eq!(replayed.reply, out.reply);
    println!(
        "strict replay: served {} spans, {} live",
        report.served, report.live
    );

    // --- UntilDivergence: pay only from the turn you took ------------------
    // Change the drafter's instruction through an overlay. The scrub call
    // still renders the same prompt, so it replays free; the drafter's
    // prompt is different at exactly that call, so it (and everything after
    // it) goes live. Nothing was declared -- the hash mismatch found it.
    let program = frontdesk::program();
    let mut candidate = Overlay::new(program);
    let slot = program
        .slot_of::<Instruction>("drafter.instruction")
        .expect("drafter.instruction is an Instruction slot");
    candidate.set_instruction(
        slot,
        "Reply in three short sentences. Never promise a refund.",
    );
    let candidate = Arc::new(candidate);

    let (out, report) = replay(&trace, ReplayMode::UntilDivergence, || {
        with_ambient_overlay(Arc::clone(&candidate), frontdesk(herman_ticket.clone()))
    })
    .await;
    let out = out?;

    println!("served {} free, {} live", report.served, report.live);
    println!("counterfactual reply: {}", out.reply);

    Ok(())
}
