# Implementation Tracker

## Current State
- **Slice**: 1
- **Phase**: Commit

## Active Subagents
| ID | Purpose | Slice | Phase | Status | Notes |
|----|---------|-------|-------|--------|-------|

## Completed Subagents
| ID | Purpose | Slice | Phase | Outcome |
|----|---------|-------|-------|---------|
| `019c41ac-7619-7013-9147-858cc5d57ebe` | Research brief for V1 typed call | 1 | Research | Completed; confirmed V1 chokepoints are `Signature`/adapter/predictor return surfaces and identified flat-`FieldSpec` gaps vs target `SignatureSchema` path model |
| `019c41bd-b436-72e3-a9f6-7be83ad9aafc` | Research brief for Slice 1 (V1 typed call) | 1 | Research | Completed; produced `slice_1_research.md` with code-level inventory and migration path; reviewed and amended for strict Slice 1 scope |
| `019c41c1-87a7-7eb0-8959-4316b2a12033` | Stupidly implementable plan for Slice 1 (V1 typed call) | 1 | Plan | Completed; generated `slice_1.md` draft with file-level steps, but initial review flagged divergence from S1/S6 full-replacement decisions |
| `019c41c5-0229-73f3-9874-4d88971cfc65` | Plan refinery against ground truth for Slice 1 | 1 | Plan Refinery | Completed; produced `slice_1_refinery.md`, corrected plan fidelity issues, and surfaced arbitration points that were resolved in `slice_1.md` |
| `019c41ca-9537-7e01-9ab4-d560308f1cd3` | Implement Slice 1 plan in code/tests | 1 | Implement | Partial; edited core/macro/adapter/predict surfaces but left compile break (`core/module.rs` delimiter), incomplete test migration, and unexpected out-of-scope edits in optimizer files (`optimizer/gepa.rs`, `optimizer/mipro.rs`) |
| `manual` | Implement Slice 1 completion pass | 1 | Implement | Completed; fixed compile break, finalized `CallOutcome`/schema test migration, added `test_signature_schema.rs` + `test_call_outcome.rs` + `test_chat_adapter_schema.rs`, and updated typed integration tests to `Predict::call(...).await.into_result()` |
| `019c41e1-6eb1-76e2-9402-aee1bdb2f20e` | Adversarial review against ground truth | 1 | Adversarial Review | Completed; reported one high finding (`MetaSignature` flatten marker mismatch); finding accepted and fixed by switching legacy field keys to `lm_name` and broadening header parser regex |

## Decisions & Architectural Notes
<!-- Log every non-obvious decision, especially cross-slice implications -->
- Slice definitions for this execution are V1-V3 from `/Users/darin/src/personal/DSRs/docs/specs/modules/breadboard.md` (V1 Typed call, V2 Augmentation + CoT, V3 Module authoring).
- Ground truth hierarchy for arbitration is: breadboard + shapes + design_reference + spikes S1-S8.
- **Locked (2026-02-09):** N8/typed call default return is `CallOutcome<O>` (metadata-first). `call_with_meta` is folded into `call`; there is no separate convenience path like `forward_result`.
- **Calling convention constraint:** single return type + single convention. `CallOutcome` must support ergonomic `?`-style consumption via traits (if feasible on toolchain) without introducing parallel APIs.
- **Error payload constraint:** errors must carry call metadata context (raw response/usage/field parse detail) in the same default return flow.
- **Plan review decision (2026-02-09):** Slice 1 plan must align with S1/S6 Option C replacement direction; broad legacy compatibility strategy in draft plan requires refinery correction or explicit arbitration.
- **Arbitration (2026-02-09): Flatten alias/constraint semantics.** `SignatureSchema` enforces unique LM-visible names per side (input/output). Collisions after flatten are hard errors with path detail. Constraints/format metadata are attached to flattened emitted leaf paths.
- **Arbitration (2026-02-09): `CallOutcome` ergonomics.** Implement `Try`/`FromResidual` on nightly (`try_trait_v2`) and keep `into_result()` explicit conversion API.
- **Implementation decision (2026-02-09):** Keep minimal optimizer file edits in `optimizer/gepa.rs` and `optimizer/mipro.rs` because they are mechanical call-site adaptations required by `Module::forward -> CallOutcome<Prediction>`; no optimizer behavior changes were introduced.
- **Adversarial arbitration (2026-02-09):** Accepted high-severity review finding on legacy flatten marker mismatch. Fixed by (1) emitting `FieldSchema::lm_name` keys in `schema_fields_to_value`, and (2) updating `FIELD_HEADER_PATTERN` to parse non-`\w` marker names (including dotted aliases/paths).
- **Smoke test (2026-02-09):** Real LM call passed end-to-end using `cargo run -p dspy-rs --example _slice1_smoke` with `.env` `OPENAI_API_KEY` and model `openai:gpt-5.2`; typed path returned expected `answer = "smoke-ok"`.
- **Arbitration result (2026-02-09):** Agreed with the single review finding and fixed it in-place (`predict.rs` legacy field-key mapping and `chat.rs` header regex). Post-fix test suite and smoke run passed.

## Stumbling Blocks
<!-- Things that were confusing, ambiguous, or required judgment calls -->
- Existing tracker lacked `Current State` fields from the required template; normalized before continuing to avoid ambiguous phase transitions.
- Initial research draft mixed Slice 1 scope with Slice 2/5 artifacts (augmentation and DynPredictor migration). Corrected to keep Slice 1 deliverables focused on V1 call path while preserving cross-slice constraints.
- Implementation subagent introduced unexpected edits outside assigned ownership (`optimizer/gepa.rs`, `optimizer/mipro.rs`) while attempting to satisfy compile ripple effects from `Module` return type changes.
- `cargo check -p dspy-rs -p dsrs_macros` and both test suites now pass, but `cargo check -p dspy-rs --examples` still fails because examples have not yet been migrated to the new `Module::forward` / `CallOutcome` interfaces.

## Open Questions
<!-- Unresolved issues to revisit -->
- If nightly `try_trait_v2` introduces instability during implementation, decide whether to keep `Try` behind cfg while preserving `into_result()` as non-divergent baseline.
- Whether Slice 1 should include an explicit follow-up example migration pass (`--examples` currently failing on old `Result`-based module signatures and removed `call_with_meta` usage).
