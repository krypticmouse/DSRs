/*
Front Desk, chapters 12 and 14 ("Tracing Paper" / "The Map Leaves Home"):
candidate changes are overlays over a fixed program, runs read through the
sheet without the base map ever getting a mark on it, and the winner bakes
into a self-contained .dsrs artifact with its lineage stamped inside.

- `program.param_id` / `program.slot_of::<Instruction>`: every dial has a
  name and an address.
- `Overlay::new` + `set_instruction`: a candidate is data, addressed to one
  specific program (the overlay remembers the base program's hash).
- `with_ambient_overlay`: run anything under the sheet; no `&mut` anywhere.
- `program.bake(&overlay, Lineage { .. })`: fold the winner in as the new
  defaults, stamp provenance, write the file.

Run with:
```
cargo run --example 26-frontdesk-tune
```
(the run-under-overlay step needs OPENAI_API_KEY; overlay and bake do not)
*/

use std::sync::Arc;

use anyhow::Result;
use dspy_rs::ir::{Instruction, Lineage, Overlay, with_ambient_overlay};
use dspy_rs::{LM, configure, init_tracing, module, predict};

/// Classify the support ticket into Bug, Billing, HowTo, or FeatureRequest.
#[predict]
fn triage(ticket: String) -> String;

/// Draft a short, warm reply to the ticket.
#[predict]
fn draft(ticket: String) -> String;

#[dspy_rs::Schema]
#[derive(Debug)]
pub struct DeskOut {
    pub category: String,
    pub reply: String,
}

#[module]
async fn frontdesk(ticket: String) -> Result<DeskOut, dspy_rs::ir::RunError> {
    let triager = triage(ticket.clone()).await?;
    let drafter = draft(ticket.clone()).await?;
    Ok(DeskOut {
        category: triager.triage,
        reply: drafter.draft,
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing()?;

    // --- The map answers for its dials -------------------------------------
    let program = frontdesk::program();
    assert!(program.param_id("drafter.instruction").is_some());
    assert!(program.param_id("triager.instruction").is_some());

    // --- A candidate is tracing paper: an overlay over this one program ----
    let mut candidate = Overlay::new(program);
    let slot = program
        .slot_of::<Instruction>("drafter.instruction")
        .expect("drafter.instruction is an Instruction slot");
    candidate.set_instruction(
        slot,
        "Reply in three short paragraphs. Open by naming the customer's \
         actual problem. Never promise a refund.",
    );
    let candidate = Arc::new(candidate);

    // --- Run under the sheet; the base program is never mutated ------------
    if std::env::var("OPENAI_API_KEY").is_ok() {
        configure(
            LM::builder()
                .model("openai:gpt-4o-mini".to_string())
                .build()
                .await?,
        );
        let out = with_ambient_overlay(
            Arc::clone(&candidate),
            frontdesk("Order #4127 still not here. It has been TWO WEEKS.".to_string()),
        )
        .await?;
        println!("candidate reply: {}", out.reply);
    }
    assert!(
        frontdesk::program()
            .to_dsrs()
            .contains("Draft a short, warm reply"),
        "the artifact keeps the default instruction"
    );

    // --- Bake: ink the winner ----------------------------------------------
    // `bake` folds the overlay's values in as the new defaults, refuses any
    // overlay minted against a different program, and stamps `parent` and
    // `overlay` itself with the base program's and the candidate's hashes.
    let baked = program.bake(
        &candidate,
        Lineage {
            optimizer: "hand-tuned".into(),
            trainset: "tickets@v2 (60)".into(),
            budget: "1 candidate / 1 rollout".into(),
            parent: None, // bake fills this in
            date: "2026-08-14".into(),
            overlay: None, // this too
        },
    )?;

    let path = std::env::temp_dir().join("frontdesk.dsrs");
    std::fs::write(&path, baked.to_dsrs())?;
    println!("baked program written to {}", path.display());
    // `dsrs check <file>` and `dsrs serve <file>` take it from here. (One
    // caveat specific to numbered example binaries: printed class names embed
    // this crate's name, and `26_frontdesk_tune::...` is not a valid `.dsrs`
    // identifier, so the reparse gate refuses the file. From a normally named
    // crate the printed artifact round-trips.)

    let lineage = baked.meta.lineage.as_ref().expect("bake stamps lineage");
    println!(
        "lineage: optimizer={} parent={} overlay={}",
        lineage.optimizer,
        lineage.parent.as_deref().unwrap_or("-"),
        lineage.overlay.as_deref().unwrap_or("-"),
    );

    // Baking changed the program's hash, so the promoted sheet is now stale
    // by design: it cannot be applied (or re-baked) onto the new program.
    assert_ne!(baked.meta.program_hash, program.meta.program_hash);
    assert!(baked.bake(&candidate, Lineage::default()).is_err());

    Ok(())
}
