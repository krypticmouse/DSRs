# AGENTS (bamltype)

## Baseline Commands

1. `cargo test -p bamltype --tests`
2. `cargo test -p dspy-rs --test typed_integration --test test_typed_alias --test test_typed_prompt_format`

## Test Layers

1. `integration.rs`: runtime behavior and conversion semantics.
2. `contract_frozen_oracle.rs`: frozen contract fixtures for schema/parse/error behavior.
3. `render_options_parity.rs`: frozen fixtures for render option combinations.
4. `golden_parity.rs`: frozen schema text snapshots for canonical examples.
5. `property_parity.rs`: deterministic property tests plus frozen fixtures for key parse/error outputs.
6. `contract_bridge_ui_messages.rs` + `ui.rs`: compile-fail diagnostics and fixture integrity.

## When Editing Behavior

1. Run target tests first:
   `cargo test -p bamltype --test contract_frozen_oracle --test render_options_parity --test golden_parity --test property_parity --test contract_bridge_ui_messages --test parity_bridge_api --test ui`
2. If a behavior change is intentional, bless fixtures with `UPDATE_EXPECT=1` for the affected tests only.
3. Rerun all `bamltype` tests without `UPDATE_EXPECT`.
4. Record the behavior change and rationale in the commit message.

## Snapshot Stability Rules

1. Keep fixture serialization canonical (sorted map keys, explicit fields, deterministic seed for proptest).
2. Prefer snapshotting semantic payloads (value/checks/flags/explanations/error fields) over raw debug dumps when possible.
3. Avoid snapshotting non-deterministic data (pointer addresses, hash iteration order).
