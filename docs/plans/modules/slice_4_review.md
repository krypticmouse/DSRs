### Findings
#### Finding 1
Severity: high
Category: Shape compliance
Location: crates/dspy-rs/src/modules/react.rs:53-63
Issue: The new `ReAct` struct is not deriving `facet::Facet`, so the optimizer’s Facet walker never learns about the `action` and `extract` `Predict` leaves. The design doc explicitly says module authors rely on `#[derive(Facet)]` to make their structure “the declaration” (Design Reference §1) and that ReAct must expose two discoverable `Predict` leaves (Design Reference §ReAct). Without the Facet shape, the optimizer cannot reach the leaf predictors, violating F6/F11 and preventing any higher-layer tooling from seeing ReAct internals.
Suggestion: Add `#[derive(facet::Facet)]` (and the necessary `#[facet(skip)]` annotations on `tools`/`max_steps`) so the walker can access `action`/`extract`. Keep the predictor fields public or `pub(crate)` and avoid wrapping them in non-Facet-friendly containers so that the derived shape exposes them as the Optimizer expects.

#### Finding 2
Severity: medium
Category: Spec fidelity
Location: crates/dspy-rs/src/modules/react.rs:82-157
Issue: The ReAct action loop builds prompts by serializing the entire input with `serde_json::to_string`, manually assembling a `trajectory` string, and hard-coding the tool manifest. Design Reference §ReAct explicitly states “Action loop uses adapter building blocks (F7) for dynamic trajectory formatting.” Bypassing `ChatAdapter` / `SignatureSchema` means the action/extract prompts no longer follow the canonical “build system → format input/output → parse sections” pipeline, so the module cannot rely on adapters to handle flattening, instructions, demos, or the `[ [ ## field ## ] ]` framing that every other module uses.
Suggestion: Reuse the existing adapter helpers (`SignatureSchema::of::<ReActActionStep>()`, `ChatAdapter::format_input_typed`, `parse_sections`, etc.) when formatting each action/extract prompt and preserve the canonical prompt text in `trajectory` rather than hand-rolled strings. That keeps ReAct in sync with the rest of the typed path and ensures the module benefits from the same field metadata, instructions, and demo formatting the spec mandates.

### Summary
Severity counts: high=1, medium=1, low=0
Overall assessment: The implementation delivers the operational surfaces, but to satisfy the ground-truth spec we must expose ReAct’s predictors through Facet and rebuild the action loop on the shared adapter helpers so prompts/metadata stay consistent with the rest of the module stack.
