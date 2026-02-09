# Slices 1-4 Closure Audit

Date: 2026-02-09  
Scope: Breadboard vertical slices `V1`, `V2`, `V3`, `V4` from `docs/specs/modules/breadboard.md`.

## Audit Method
- Re-checked `docs/specs/modules/breadboard.md` slice definitions and `docs/specs/modules/shapes.md` / `docs/specs/modules/design_reference.md` constraints.
- Verified implementation in the live codebase (not docs-only) with file-level evidence.
- Classified each slice requirement as `Implemented` or `Deferred` with explicit follow-up mapping.

## Slices 1-3 Baseline
- Baseline accounting for `V1`-`V3` remains in `docs/plans/modules/slices_1_3_closure_audit.md`.
- This document extends that ledger through `V4` and updates deferred-item routing now that all four slices are implemented.

## Slice 4 (V4 ReAct + Operational) Accounting

| Affordance(s) | Status | Evidence |
|---|---|---|
| `U14` ReAct builder with tools (`.tool("name", "desc", fn)`) | Implemented | `crates/dspy-rs/src/modules/react.rs:53`, `crates/dspy-rs/src/modules/react.rs:273`, `crates/dspy-rs/src/modules/react.rs:325`, `crates/dspy-rs/tests/test_react_builder.rs:93` |
| `U14` ReAct action/extract composition (two `Predict` leaves + loop in `forward`) | Implemented | `crates/dspy-rs/src/modules/react.rs:61`, `crates/dspy-rs/src/modules/react.rs:64`, `crates/dspy-rs/src/modules/react.rs:148`, `crates/dspy-rs/tests/test_react_builder.rs:66` |
| `U14` ReAct trajectory parity without extra API (`CallOutcome` metadata carries trace, no `call_with_trajectory`) | Implemented | `crates/dspy-rs/src/modules/react.rs:85`, `crates/dspy-rs/src/modules/react.rs:135`, `crates/dspy-rs/src/modules/react.rs:236`, `crates/dspy-rs/examples/93-smoke-slice4-react-operational.rs:89` |
| `U48` standalone `forward_all(&module, inputs, concurrency)` | Implemented | `crates/dspy-rs/src/core/module.rs:22`, `crates/dspy-rs/src/evaluate/evaluator.rs:24`, `crates/dspy-rs/tests/test_module_forward_all.rs:21` |
| `U51` module combinators (`.map()`, `.and_then()`) | Implemented | `crates/dspy-rs/src/core/module_ext.rs:5`, `crates/dspy-rs/src/core/module_ext.rs:28`, `crates/dspy-rs/src/core/module_ext.rs:55`, `crates/dspy-rs/tests/test_module_ext.rs:37` |
| `S4` tool storage for operational modules | Implemented | `crates/dspy-rs/src/modules/react.rs:66`, `crates/dspy-rs/src/modules/react.rs:152`, `crates/dspy-rs/src/modules/react.rs:335` |

Slice 4 verdict: **Implemented**.

## Consolidated Deferred Ledger (Post-Implementation Cleanup)

| Deferred item | Why deferred | Target phase | Exit criteria |
|---|---|---|---|
| Strict typed `Module` bounds (`Input/Output: BamlType + Facet`) | Compatibility with legacy/untyped module surfaces still present | **Post-Implementation Cleanup** | `Module` bounds tightened and impacted examples/tests migrated |
| F12 helper generic bounds threading in generated helper structs | Macro helper constraints still use transitional strategy | **Post-Implementation Cleanup** | Generic helper declarations preserve source generic contract with `dsrs-macros` tests green |
| `__phantom` helper-field authoring ergonomics | Generic helper phantom initialization still leaks into same-module literals | **Post-Implementation Cleanup** | No user-facing phantom initialization burden in macro tests/examples |
| Option-C full legacy cutover (`MetaSignature`/`LegacyPredict`) | Legacy compatibility surfaces still active for older flows | **Post-Implementation Cleanup** | Schema-first typed path is sole default path and legacy surfaces are removed/quarantined with migration notes |
| `V5` walker discoverability for additional wrappers/combinators | Deferred by earlier closure audits; only Slice 4 ReAct discoverability addressed now | **Post-Implementation Cleanup** (prep) + **V5 Implement** (completion) | Walker traverses wrapper module trees end-to-end with tests for nested combinator/module stacks |

## Cleanup Kickoff Reference

Phase 4.5 execution planning and decision arbitration checkpoints are now tracked in:

- `docs/plans/modules/phase_4_5_cleanup_kickoff.md`

Use that doc as the active decision matrix for:
- strict typed-bound migration strategy,
- legacy-surface cutover gates,
- optimizer/evaluator contract migration boundaries,
- wrapper/combinator walker completion scope.

## Post-Implementation Cleanup Resolved Items
- `U29` (`ChainOfThought` Facet discoverability) resolved in code: `crates/dspy-rs/src/modules/chain_of_thought.rs:16`.
- `build_system` API/spec mismatch resolved by spec alignment to fallible return (`Result<String>`): `docs/specs/modules/breadboard.md:101`, `docs/specs/modules/design_reference.md:583`.

## Validation During Slice 4 Closure Audit
- `cargo check -p dspy-rs`
- `cargo check -p dspy-rs --examples`
- `cargo test -p dspy-rs --test test_module_forward_all --test test_module_ext --test test_react_builder --test test_chain_of_thought_swap`
- `set -a && source .env && set +a && cargo run -p dspy-rs --example 93-smoke-slice4-react-operational`

Observed smoke output (calculator trajectory parity pass): `tool_calls: 3`, `tool_executions: 5`, trajectory printed with `Step 1..4`, `answer: 70`.
